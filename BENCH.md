# Running lgalloc benchmarks on a target host

## Prerequisites

* Rust toolchain (stable)
* `systemd-run` (systemd 232+, any modern Linux)
* Swap enabled (for paging benchmarks)

## Build

```sh
cargo build --release --bench alloc_bench
```

The binary is at `target/release/deps/alloc_bench-*` (pick the newest non-`.d` file).

## Throughput benchmarks

Measures lgalloc, system allocator, and raw mmap across 1/2/4/8/16 threads.
Limits: 4G RAM, no swap, 16 CPUs.

```sh
BENCH=$(ls -t target/release/deps/alloc_bench-* | grep -v '\.d$' | head -1)
systemd-run --user -p MemoryMax=4G -p MemorySwapMax=0 -p CPUQuota=1600% \
  --wait --pipe "$BENCH"
```

Adjust `CPUQuota` to match the host (100% per core, e.g. 800% for 8 cores).

## Paging benchmarks

Allocates 3x the memory limit, forces swap, measures random read latency (CCDF)
and realloc+touch latency from a swapped pool.

```sh
systemd-run --user -p MemoryMax=512M -p MemorySwapMax=4G -p CPUQuota=1600% \
  --wait --pipe "$BENCH" -- --paging
```

For heavier swap pressure:

```sh
systemd-run --user -p MemoryMax=128M -p MemorySwapMax=4G -p CPUQuota=1600% \
  --wait --pipe "$BENCH" -- --paging
```

## What to look at

* **lgalloc vs sysalloc+touch**: the main comparison.
  lgalloc reuses faulted pages; the system allocator mmaps/munmaps on each cycle.
* **sysalloc+touch vs sysalloc+nohuge+touch**: quantifies THP benefit for the system allocator.
  If the host has THP=`never`, these should be similar.
* **Scaling**: lgalloc should scale linearly (lock-free work-stealing);
  sysalloc and mmap degrade due to kernel mmap_lock contention.
* **Paging CCDF**: bimodal distribution — resident pages (µs) vs swap-in (ms).
  The tail shows swap I/O latency of the target host's storage.
* **realloc+touch from swapped pool**: lgalloc recycles virtual addresses without syscalls,
  so the cost is just page faults, not mmap+fault.

## Ratio sweep

Fixes a working set size and varies the cgroup RAM limit to explore the
relationship between resident-to-swap ratio and latency.

```sh
for RAM in 4096 2048 1024 512 256; do
  SWAP=$((8192 - RAM))
  systemd-run --user \
    -p MemoryMax=${RAM}M -p MemorySwapMax=${SWAP}M -p CPUQuota=1600% \
    --wait --pipe "$BENCH" -- --ratio-sweep --total-mib 4096
done
```

### Findings (16-vCPU, EBS-backed instance)

**Random reads collapse as soon as data spills to swap.**
At 1 thread with a 4 GiB working set:

| RAM    | Ratio | ops/s   | p50    | p99    |
|--------|-------|---------|--------|--------|
| 4096M  | 1:1   | 232,041 | 238ns  | 632ns  |
| 2048M  | 1:2   |     156 | 326µs  | 132ms  |
|  512M  | 1:8   |      93 | 471µs  | 139ms  |
|  256M  | 1:16  |     144 | 470µs  | 132ms  |

Once swapping, throughput is dominated by per-page swap-in latency (~130 ms p99),
and the exact ratio matters little.

**Realloc+touch is unaffected by swap pressure.**
lgalloc recycles the same virtual pages; the kernel keeps the hot working page
resident since it is touched immediately after dealloc:

| RAM    | Ratio | ops/s   | p50   |
|--------|-------|---------|-------|
| 4096M  | 1:1   | 461,636 | 2.2µs |
|  512M  | 1:8   | 461,323 | 2.2µs |

Scaling is near-linear up to 16 threads (~7 M ops/s) at every ratio.

### madvise strategy experiments (1 thread, 512 M RAM / 4 GiB working set)

| Strategy              | ops/s | p50    | p99    | p999   | max    |
|-----------------------|-------|--------|--------|--------|--------|
| baseline              | 2,155 | 468µs  | 777µs  | 2.3ms  | 209ms  |
| MADV_RANDOM           | 2,334 | 497µs  | 738µs  | 1.1ms  | 4.4ms  |
| MADV_SEQUENTIAL       | 2,363 | 498µs  | 729µs  | 925µs  | 2.8ms  |
| prefetch (32 ahead)   | 3,618 | 322µs  | 503µs  | 1.6ms  | 19ms   |
| batch8+WILLNEED       | 3,724 | 314µs  | 562µs  | 949µs  | 2.2ms  |
| batch32+WILLNEED      | 3,605 | 319µs  | 520µs  | 790µs  | 2.9ms  |
| batch128+WILLNEED     | 3,764 | 1.3µs  | 445µs  | 637µs  | 10ms   |
| MADV_RANDOM+pf (8thr) | 7,032 | 377ns  | 15.3ms | 30ms   | 263ms  |

* **MADV_RANDOM/SEQUENTIAL** barely help (~8%). Default readahead is already
  small for random patterns and swap I/O is seek-dominated.
* **MADV_WILLNEED prefetch** is the clear winner: +68% throughput, p50 drops
  from 468 µs → 322 µs. A prefetch thread (or batch WILLNEED) issues swap-in
  requests ahead of time, overlapping I/O with computation.
* **batch128+WILLNEED** achieves a **1.3 µs p50** because most pages are already
  resident by the time the reader reaches them.
* **MADV_RANDOM + prefetch at 8 threads** reaches 7,032 ops/s (2× baseline),
  the best multi-threaded result. Disabling kernel readahead avoids wasted I/O
  while explicit prefetch keeps the pipeline full.
* No strategy degrades the all-fits case (~230K ops/s at 1 thread).

These results motivate the `prefetch_hint` API: callers who know their access
pattern can issue MADV_WILLNEED on specific page ranges ahead of time.

## THP configuration

Check:

```sh
cat /sys/kernel/mm/transparent_hugepage/enabled
```

* `always` — THP on all anonymous mappings (lgalloc and sysalloc both benefit)
* `madvise` — THP only for regions with MADV_HUGEPAGE (lgalloc benefits, sysalloc does not)
* `never` — no THP (lgalloc's MADV_HUGEPAGE hint is a no-op, warns once)

The `sysalloc+nohuge` variants force 4K pages via MADV_NOHUGEPAGE regardless of this setting.
