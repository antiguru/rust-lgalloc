lgalloc
=======

A memory allocator for large objects. Lgalloc stands for large (object) allocator.
We spell it `lgalloc` and pronounce it el-gee-alloc.

```toml
[dependencies]
lgalloc = "0.2"
```

## Example

```rust
use std::mem::ManuallyDrop;
fn main() -> Result<(), lgalloc::AllocError> {
  lgalloc::lgalloc_set_config(
    lgalloc::LgAlloc::new()
      .enable()
      .with_path(std::env::temp_dir()),
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

- Lgalloc provides an allocator for power-of-two sized memory regions.
- The requested capacity can be rounded up to a larger capacity.
- The memory can be repurposed, for example to back a vector, however, the caller needs to be
  careful never to free the memory using the specific allocator.
- Memory is not unmapped, but can be lazily marked as unused with a background thread. The exact
  options for this still need to be determined.
- The allocations are mapped from a file, which allows the OS to page without using swap.
- On Linux, this means it can only handle regular pages (4KiB), the region cannot be mapped
  with huge pages.
- The library does not consume physical memory when all regions are freed, but pollutes the
  virtual address space because it doesn't unmap regions. This is because the library does
  not keep track what parts of a mapping are still in use.
- Generally, use at your own risk because nobody should write a memory allocator.
- Performance seems to be reasonable, similar to the system allocator when not touching the data,
  and faster when touching the data. The reason is that this library does not unmap its regions.


The allocator tries to minimize contention. It relies on thread-local allocations and a
work-stealing pattern to move allocations between threads. Each size class acts as its own
allocator.

We use the term region for a power-of-two sized allocation, and area for a contiguous allocations.
Each area can back multiple regions.

* Each thread maintains a bounded cache of regions.
* If on allocation the cache is empty, it checks the global pool first, and then other threads.
* The global pool has a dirty and clean variant. Dirty contains allocations that were recently
  recycled, and clean contains allocations that we marked as not needed/removed to the OS.
* An optional background worker periodically moves allocations from dirty to clean.
* Lgalloc makes heavy use of `crossbeam-deque`, which provides a lock-free work stealing API.
* Refilling areas is a synchronous operation. It requires to create a file, allocate space, and
  map its contents. We double the size of the allocation each time a size class is empty.
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
