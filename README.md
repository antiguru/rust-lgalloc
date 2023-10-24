lgalloc
=======

This library provides a memory allocator for large objects.

```toml
[dependencies]
lgalloc = { git = "https://github.com/antiguru/rust-lgalloc" }
```

## Example

```rust
use lgalloc::Region;

fn main() {
    // Create a 2MiB region
    let mut region = Region::new_mmap(2 << 20);
    region.extend_from_slice(&[1, 2, 3, 4]);
}
```

## Details

- Lgalloc provides an allocator for power-of-two sized objects. Regions encapsulate objects
  as a `Vec<T>`.
- Although a region provides mutable access to a `Vec<T>`, the caller has to make sure that
  the vector is never re-allocated, i.e., never push or extend with more data than the remaining
  capacity.
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
