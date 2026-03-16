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

## THP configuration

Check:

```sh
cat /sys/kernel/mm/transparent_hugepage/enabled
```

* `always` — THP on all anonymous mappings (lgalloc and sysalloc both benefit)
* `madvise` — THP only for regions with MADV_HUGEPAGE (lgalloc benefits, sysalloc does not)
* `never` — no THP (lgalloc's MADV_HUGEPAGE hint is a no-op, warns once)

The `sysalloc+nohuge` variants force 4K pages via MADV_NOHUGEPAGE regardless of this setting.
