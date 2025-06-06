# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0](https://github.com/antiguru/rust-lgalloc/compare/v0.5.0...v0.6.0) - 2025-05-27

### Other

- Split fast and slow stats ([#62](https://github.com/antiguru/rust-lgalloc/pull/62))

## [0.5.0](https://github.com/antiguru/rust-lgalloc/compare/v0.4.0...v0.5.0) - 2025-02-27

### Other

- Configurable local buffer size ([#60](https://github.com/antiguru/rust-lgalloc/pull/60))
- Assign thread name to background worker ([#58](https://github.com/antiguru/rust-lgalloc/pull/58))
- File growth dampener ([#57](https://github.com/antiguru/rust-lgalloc/pull/57))

## [0.4.0](https://github.com/antiguru/rust-lgalloc/compare/v0.3.1...v0.4.0) - 2024-11-23

### Other

- File stats wrapped in result ([#53](https://github.com/antiguru/rust-lgalloc/pull/53))

## [0.3.1](https://github.com/antiguru/rust-lgalloc/compare/v0.3.0...v0.3.1) - 2024-03-13

### Other
- Try harder to reclaim disk space
- Only run tests on pull request and pushes to main
- Add API to disable lgalloc globally

## [0.3.0](https://github.com/antiguru/rust-lgalloc/compare/v0.2.2...v0.3.0) - 2024-03-06

### Other
- Remove memory based on byte size instead of count

## [0.2.2](https://github.com/antiguru/rust-lgalloc/compare/v0.2.1...v0.2.2) - 2024-02-28

### Other
- Fix the background scheduler to reschedule itself

## [0.2.1](https://github.com/antiguru/rust-lgalloc/compare/v0.2.0...v0.2.1) - 2024-02-25

### Other
- Extract numa_maps into separate crate
- Update documentation

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
