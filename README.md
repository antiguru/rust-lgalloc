lgalloc
=======

A memory allocator for large objects backed by anonymous mappings with huge page hints.
Lgalloc stands for large (object) allocator.
We spell it `lgalloc` and pronounce it el-gee-alloc.

```toml
[dependencies]
lgalloc = "0.7"
```

## Example

```rust
use std::mem::ManuallyDrop;
fn main() -> Result<(), lgalloc::AllocError> {
  lgalloc::lgalloc_set_config(
    lgalloc::LgAlloc::new()
      .enable(),
  );

  // Allocate memory
  let (ptr, cap, handle) = lgalloc::allocate::<u8>(2 << 20)?;
  // SAFETY: `allocate` returns a valid memory region and errors otherwise.
  let mut vec = ManuallyDrop::new(unsafe { Vec::from_raw_parts(ptr.as_ptr(), 0, cap) });

  // Write into region, make sure not to reallocate vector.
  vec.extend_from_slice(&[1, 2, 3, 4]);

  // We can read from the vector.
  assert_eq!(&*vec, &[1, 2, 3, 4]);

  // Deallocate after use
  lgalloc::deallocate(handle);

  Ok(())
}
```

## When to use lgalloc

Lgalloc is designed for programs that allocate and recycle many large memory regions (2 MiB+).
It pools regions by size class and reuses them without returning virtual address space to the kernel, which avoids the `mmap`/`munmap` overhead and kernel `mmap_lock` contention that dominate at high thread counts.
On Linux, it requests transparent huge pages via `MADV_HUGEPAGE`, reducing TLB misses for large working sets.

Lgalloc is a low-level API.
Callers get a raw pointer, a capacity, and a handle; they are responsible for building higher-level abstractions (vectors, buffers) on top.

## Usage constraints

* **No fork.**
  Anonymous mappings are shared with child processes after `fork`.
  Two processes writing to the same mapping causes undefined behavior.
  There is no way to mark mappings as non-inheritable.
* **No mlock.**
  Callers must not lock pages (`mlock`) on regions managed by lgalloc, or must unlock them before returning the region.
  The background worker calls `madvise` on returned regions, which fails on locked pages.
* **Do not free with another allocator.**
  Memory obtained from `allocate` must be returned via `deallocate`.
  Passing the pointer to `free`, `Vec::from_raw_parts` without `ManuallyDrop`, or any other allocator is undefined behavior.
* **Minimum allocation is 2 MiB.**
  Size classes range from 2^21 (2 MiB) to 2^36 (64 GiB).
  Requests below 2 MiB return `AllocError::InvalidSizeClass`.
* **Capacity may be rounded up.**
  The returned capacity can be larger than requested because allocations are rounded to power-of-two size classes.

## Thread safety

`Handle` is `Send` and `Sync`.
Allocations can be made on one thread and freed on another.
Each thread maintains a local cache; the global pool uses lock-free work-stealing to redistribute regions.

## How it works

Lgalloc is size-classed: each power-of-two size from 2 MiB to 64 GiB has its own pool.
Within a size class, contiguous *areas* of increasing size back individual *regions*.

* Each thread maintains a bounded local cache of regions.
* On allocation, the thread checks its local cache, then the global dirty pool, then the global clean pool, then steals from other threads.
* On deallocation, the region goes to the local cache or, if full, to the global dirty pool.
  With `eager_return` enabled, `MADV_DONTNEED` is called before pushing to the global pool.
* An optional background worker moves dirty regions to the clean pool by calling `MADV_FREE` (Linux) or `MADV_DONTNEED` (other platforms), which marks pages as lazily reclaimable.
* When all pools are empty, lgalloc creates a new area via `mmap(MAP_ANONYMOUS)` and applies `MADV_HUGEPAGE`.
  Area sizes double on each refill, controlled by the `growth_dampener` config.
* Regions are never unmapped during normal operation.
  This avoids `munmap` syscall overhead but grows virtual address space.
  Areas are unmapped when the global state is dropped (process exit).

## Platform notes

* **Linux**: requests transparent huge pages via `MADV_HUGEPAGE`.
  The kernel uses 2 MiB pages when `/sys/kernel/mm/transparent_hugepage/enabled` is `always` or `madvise`.
  If THP is disabled, the hint is silently ignored (one warning on stderr).
* **macOS ARM**: the kernel does not expose a userspace huge page API.
  Lgalloc uses the base 16 KiB page size.

## To do

* Testing is very limited.
* Allocating areas of doubling sizes seems to stress the `mmap` system call.
  Consider a different strategy, such as constant-sized blocks or a limit on what areas we allocate.
  There's probably a trade-off between area size and number of areas.
* Fixed-size areas could allow us to move areas between size classes.
* Reference-counting can determine when an area isn't referenced anymore, although this is not trivial because it's a lock-free system.

#### License

<sup>
Licensed under either of <a href="LICENSE-APACHE">Apache License, Version
2.0</a> or <a href="LICENSE-MIT">MIT license</a> at your option.
</sup>

<br>

<sub>
Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this crate by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
</sub>
