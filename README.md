lgalloc
=======

A memory allocator for large objects. Lgalloc stands for large (object) allocator.
We spell it `lgalloc` and pronounce it el-gee-alloc.

```toml
[dependencies]
lgalloc = "0.6"
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

## Details

Lgalloc is a memory allocator that backs allocations with anonymous memory mappings and
requests transparent huge pages (THP) via `MADV_HUGEPAGE`.
It is size-classed, meaning that it can only allocate memory in power-of-two sized regions, and each region
is independent of the others. Each region is backed by areas of increasing size.

On Linux, anonymous mappings with THP hints allow the kernel to use 2 MiB huge pages,
reducing TLB pressure for large allocations. The kernel promotes pages to huge pages
transparently when `/sys/kernel/mm/transparent_hugepage/enabled` is set to `always` or
`madvise`. On macOS ARM, the kernel does not expose a userspace huge page API, so lgalloc
uses the base 16 KiB page size.

Lgalloc provides a low-level API, but does not expose a high-level interface. Clients are advised to
implement their own high-level abstractions, such as vectors or other data structures, on top of
lgalloc.

Anonymous mappings have some properties that are not immediately obvious:
* Allocations do not use physical memory until they are touched.
* Returning memory is a two-step process. After deallocation, lgalloc can eagerly return
  physical memory by calling `MADV_DONTNEED`. An optional background worker periodically
  calls `MADV_FREE` on unused memory regions, which marks pages as lazily reclaimable
  without immediate zeroing overhead.
* Interacting with the memory subsystem can cause contention, especially when multiple threads
  try to interact with the virtual address space at the same time. For example, the `mmap` and
  `madvise` system calls can contend with each other and with other parts of the program.

* Lgalloc provides an allocator for power-of-two sized memory regions, with an optional dampener.
* The requested capacity can be rounded up to a larger capacity.
* The memory can be repurposed, for example to back a vector, however, the caller needs to be
  careful never to free the memory using another allocator.
* Memory is not unmapped during normal operation, but can be lazily marked as unused with a
  background thread.
* On Linux with THP enabled, allocations can benefit from 2 MiB huge pages, reducing TLB misses.
  On macOS ARM, huge pages are not available and the base 16 KiB page size is used.
* The library does not consume physical memory when all regions are freed, but pollutes the
  virtual address space because it doesn't unmap regions during normal operation. Mappings are
  unmapped when the global state is dropped.
* Generally, use at your own risk because nobody should write a memory allocator.
* Performance seems to be reasonable, similar to the system allocator when not touching the data,
  and faster when touching the data. The reason is that this library does not unmap its regions.


The allocator tries to minimize contention. It relies on thread-local allocations and a
work-stealing pattern to move allocations between threads. Each size class acts as its own
allocator. However, some system calls can contend on mapping objects, which is why reclamation
can cause contention.

We use the term region for a power-of-two sized allocation, and area for a contiguous allocations.
Each area can back multiple regions.

* Each thread maintains a bounded cache of regions.
* If on allocation the cache is empty, it checks the global pool first, and then other threads.
* The global pool has a dirty and clean variant. Dirty contains allocations that were recently
  recycled, and clean contains allocations that we marked as not needed/removed to the OS.
* An optional background worker periodically moves allocations from dirty to clean.
* Lgalloc makes heavy use of `crossbeam-deque`, which provides a lock-free work stealing API.
* Refilling areas is a synchronous operation. It requires creating an anonymous mapping and
  applying huge page hints. We double the size of the allocation each time a size class is empty.
* Lgalloc reports metrics about allocations, deallocations, and refills.

## To do

* Testing is very limited.
* Allocating areas of doubling sizes seems to stress the `mmap` system call. Consider a different
  strategy, such as constant-sized blocks or a limit on what areas we allocate. There's probably
  a trade-off between area size and number of areas.
* Fixed-size areas could allow us to move areas between size classes.
* Reference-counting can determine when an area isn't referenced anymore, although this is not
  trivial because it's a lock-free system.

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
