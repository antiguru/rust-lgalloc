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

### Findings — EBS vs NVMe swap (16-vCPU r6gd, THP=madvise)

#### Random reads (1 thread, 4 GiB working set)

| RAM    | Ratio | EBS ops/s | NVMe ops/s | NVMe p50 | NVMe p99  |
|--------|-------|-----------|------------|----------|-----------|
| 4096M  | 1:1   |   232,041 |  1,545,189 |   181ns  |    533ns  |
| 2048M  | 1:2   |       156 |     23,174 |    75µs  |    241µs  |
| 1024M  | 1:4   |         — |     16,504 |    77µs  |    167µs  |
|  512M  | 1:8   |        93 |     12,996 |    78µs  |    251µs  |
|  256M  | 1:16  |       144 |     12,578 |    77µs  |    242µs  |

On EBS, throughput collapsed to ~100 ops/s once swapping, dominated by
per-page swap-in latency (~130 ms p99). On NVMe, swap-in completes in
~80 µs (p50) with sub-300 µs p99 — a **100–150× throughput improvement**
and **~500× p99 improvement**.

The ratio still matters on NVMe (23K → 13K from 1:2 → 1:8), but the
degradation is gradual rather than catastrophic.

Multi-threaded scaling is near-linear on NVMe:

| RAM    | 1 thr    | 4 thr    | 8 thr     | 16 thr    |
|--------|----------|----------|-----------|-----------|
| 2048M  |  23,174  |  91,094  |  161,827  |  268,920  |
|  512M  |  12,996  |  52,354  |   94,063  |  156,873  |

#### Realloc+touch is unaffected by swap pressure

lgalloc recycles the same virtual pages; the kernel keeps the hot working page
resident since it is touched immediately after dealloc:

| RAM    | Ratio | ops/s   | p50   |
|--------|-------|---------|-------|
| 4096M  | 1:1   | 465,828 | 2.2µs |
|  512M  | 1:8   | 453,985 | 2.2µs |

Scaling is near-linear up to 16 threads (~7.8 M ops/s) at every ratio.

### Paging benchmark (CCDF)

512 MiB RAM, 1536 MiB allocated (3× overcommit), NVMe swap:

| Metric                 | 1 thread |
|------------------------|----------|
| random_read ops/s      |   17,392 |
| random_read p50        |   76.2µs |
| random_read p99        |  240.5µs |
| random_read max        |    3.7ms |
| realloc+touch ops/s    |  447,291 |
| realloc+touch p50      |    2.3µs |

CCDF is bimodal: ~50% of reads hit resident pages (< 300 ns), ~50% swap in
at ~76–87 µs. The tail (p99.9 = 252 µs, max = 3.7 ms) reflects NVMe device
latency under queue depth, not EBS network round-trips.

### madvise strategy experiments

#### EBS (1 thread, 512 M RAM / 4 GiB working set)

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

#### NVMe (1 thread, 512 M RAM / 4 GiB working set)

| Strategy              | ops/s   | p50    | p99    | p999   | max    |
|-----------------------|---------|--------|--------|--------|--------|
| baseline              |  12,996 | 78µs   | 251µs  | 298µs  | 5.8ms  |
| MADV_RANDOM           |  12,725 | 78µs   | 251µs  | 452µs  | 1.9ms  |
| MADV_SEQUENTIAL       |  12,825 | 77µs   | 250µs  | 279µs  | 817µs  |
| prefetch (32 ahead)   | 150,411 | 1.5µs  | 33µs   | 109µs  | 23.9ms |
| batch8+WILLNEED       |  30,362 | 4.4µs  | 162µs  | 219µs  | 402µs  |
| batch32+WILLNEED      |  78,060 | 837ns  | 147µs  | 164µs  | 412µs  |
| batch128+WILLNEED     | 136,947 | 846ns  | 31µs   | 135µs  | 330µs  |
| MADV_RANDOM+pf (8thr) | 241,035 | 1.3µs  | 591µs  | 830µs  | 12ms   |

#### Analysis

* **MADV_RANDOM/SEQUENTIAL** barely help on either storage backend (~2–8%).
  Default readahead is already small for random patterns.
* **MADV_WILLNEED prefetch** is the clear winner on both backends, but the
  benefit is dramatically larger on NVMe: **150K ops/s** (12× baseline) vs
  3.6K (1.7× baseline) on EBS. NVMe's low latency means the prefetch thread
  can keep the I/O pipeline full — most pages are already resident by the time
  the reader reaches them.
* **batch128+WILLNEED** achieves sub-microsecond p50 on both backends, but
  NVMe sustains **137K ops/s** vs 3.8K on EBS.
* **MADV_RANDOM + prefetch at 8 threads** reaches **241K ops/s** on NVMe
  (19× baseline), approaching the all-resident rate. On EBS the same strategy
  managed 7K ops/s with a 263 ms max — the network round-trip dominates.
* No strategy degrades the all-fits case (~1.5M ops/s at 1 thread on NVMe).

**Takeaway**: NVMe instance storage transforms swap from "emergency fallback"
to a viable tier. With prefetch, swapped data on NVMe approaches DRAM
throughput for batch workloads. EBS swap is only practical for cold data
that is rarely accessed.

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
