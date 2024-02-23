# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0](https://github.com/antiguru/rust-lgalloc/compare/v0.1.7...v0.2.0) - 2024-02-23

### Other
- Use release-plz to update
- Remove Region API
- numa_maps parsing and reporting
- Add eager_return option to call MADV_DONTNEED on non-cached frees
- Remove bench, require less recent Rust
- Bump version
- Merge pull request [#33](https://github.com/antiguru/rust-lgalloc/pull/33) from antiguru/extend_size_classes
- Add more supported size classes
- Cleaner APIs
- Support zst
- Change public interface to directly allocate memory
- Bump version to 0.1.6
- Threads maintain a byte-size bounded cache
- Merge pull request [#24](https://github.com/antiguru/rust-lgalloc/pull/24) from antiguru/bump_version
- Do not populate when mapping
- Merge pull request [#20](https://github.com/antiguru/rust-lgalloc/pull/20) from antiguru/bgthread_fix
- Merge pull request [#22](https://github.com/antiguru/rust-lgalloc/pull/22) from antiguru/all_targets
- Re-use buffer in tests
- Documentation, bump version, remove unused print_stats
- Store areas in ManuallyDrop
- Expose less access to the region-allocated vector
- Hide the internal state of Region::MMap
- Address comments
- Introduce benchmark
- Clearer implementation around ftruncate
- Remove stale comment
- Make tests faster
- Reuse background worker on configuration update
- Remove diagnostics function
- Remove Region::MIN_MMAP_SIZE
- Update readme
- Relax version requirements
- Bump version
- Fix MADV_DONTNEED_STRATEGY
- Use MADV_REMOVE on Linux, MADV_DONTNEED otherwise
- Allow publishing crate
- Merge pull request [#9](https://github.com/antiguru/rust-lgalloc/pull/9) from antiguru/deps
- Rename crate to lgalloc
- Only require Rust 1.70, chosen to be compatible with other projects
- Make LgAllocStats::size_class pub
- Report stats
- Merge pull request [#4](https://github.com/antiguru/rust-lgalloc/pull/4) from antiguru/enabled_flag
- More focused inline annotations
- cargo format
- Introduce configurations, size class abstraction.
- More usability improvements.
- Split global region into resident and non-resident
- Initial import
