//! A size-classed file-backed large object allocator.
//!
//! This library contains types to allocate memory outside the heap,
//! supporting power-of-two object sizes. Each size class has its own
//! memory pool.
//!
//! # Safety
//!
//! This library is very unsafe on account of `unsafe` and interacting directly
//! with libc, including Linux extension.
//!
//! The library relies on memory-mapped files. Users of this file must not fork the process
//! because otherwise two processes would share the same mappings, causing undefined behavior
//! because the mutable pointers would not be unique anymore. Unfortunately, there is no way
//! to tell the memory subsystem that the shared mappings must not be inherited.
//!
//! Clients must not lock pages (`mlock`), or need to unlock the pages before returning them
//! to lgalloc.

#![deny(missing_docs)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::marker::PhantomData;
use std::mem::{take, ManuallyDrop, MaybeUninit};
use std::ops::{Deref, Range};
use std::os::fd::{AsFd, AsRawFd};
use std::path::PathBuf;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::thread::{JoinHandle, ThreadId};
use std::time::{Duration, Instant};

use crossbeam_deque::{Injector, Steal, Stealer, Worker};
use memmap2::MmapMut;
use numa_maps::NumaMap;
use thiserror::Error;

mod readme {
    #![doc = include_str!("../README.md")]
}

/// Handle to describe allocations.
///
/// Handles represent a leased allocation, which must be explicitly freed. Otherwise, the caller will permanently leak
/// the associated memory.
pub struct Handle {
    /// The actual pointer.
    ptr: NonNull<u8>,
    /// Length of the allocation.
    len: usize,
}

unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}

#[allow(clippy::len_without_is_empty)]
impl Handle {
    /// Construct a new handle from a region of memory
    fn new(ptr: NonNull<u8>, len: usize) -> Self {
        Self { ptr, len }
    }

    /// Construct a dangling handle, which is only suitable for zero-sized types.
    fn dangling() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
        }
    }

    fn is_dangling(&self) -> bool {
        self.ptr == NonNull::dangling()
    }

    /// Length of the memory area in bytes.
    fn len(&self) -> usize {
        self.len
    }

    /// Pointer to memory.
    fn as_non_null(&self) -> NonNull<u8> {
        self.ptr
    }

    /// Indicate that the memory is not in use and that the OS can recycle it.
    fn clear(&mut self) -> std::io::Result<()> {
        // SAFETY: `MADV_DONTNEED_STRATEGY` guaranteed to be a valid argument.
        unsafe { self.madvise(MADV_DONTNEED_STRATEGY) }
    }

    /// Indicate that the memory is not in use and that the OS can recycle it.
    fn fast_clear(&mut self) -> std::io::Result<()> {
        // SAFETY: `libc::MADV_DONTNEED` documented to be a valid argument.
        unsafe { self.madvise(libc::MADV_DONTNEED) }
    }

    /// Call `madvise` on the memory region. Unsafe because `advice` is passed verbatim.
    unsafe fn madvise(&self, advice: libc::c_int) -> std::io::Result<()> {
        // SAFETY: Calling into `madvise`:
        // * The ptr is page-aligned by construction.
        // * The ptr + length is page-aligned by construction (not required but surprising otherwise)
        // * Mapped shared and writable (for MADV_REMOVE),
        // * Pages not locked.
        // * The caller is responsible for passing a valid `advice` parameter.
        let ptr = self.as_non_null().as_ptr().cast();
        let ret = unsafe { libc::madvise(ptr, self.len, advice) };
        if ret != 0 {
            let err = std::io::Error::last_os_error();
            return Err(err);
        }
        Ok(())
    }
}

/// Initial file size
const INITIAL_SIZE: usize = 32 << 20;

/// Range of valid size classes.
pub const VALID_SIZE_CLASS: Range<usize> = 10..37;

/// Strategy to indicate that the OS can reclaim pages
// TODO: On Linux, we want to use MADV_REMOVE, but that's only supported
// on file systems that supports FALLOC_FL_PUNCH_HOLE. We should check
// the return value and retry EOPNOTSUPP with MADV_DONTNEED.
#[cfg(target_os = "linux")]
const MADV_DONTNEED_STRATEGY: libc::c_int = libc::MADV_REMOVE;

#[cfg(not(target_os = "linux"))]
const MADV_DONTNEED_STRATEGY: libc::c_int = libc::MADV_DONTNEED;

type PhantomUnsyncUnsend<T> = PhantomData<*mut T>;

/// Allocation errors
#[derive(Error, Debug)]
pub enum AllocError {
    /// IO error, unrecoverable
    #[error("I/O error")]
    Io(#[from] std::io::Error),
    /// Out of memory, meaning that the pool is exhausted.
    #[error("Out of memory")]
    OutOfMemory,
    /// Size class too large or small
    #[error("Invalid size class")]
    InvalidSizeClass(usize),
    /// Allocator disabled
    #[error("Disabled by configuration")]
    Disabled,
    /// Failed to allocate memory that suits alignment properties.
    #[error("Memory unsuitable for requested alignment")]
    UnalignedMemory,
}

impl AllocError {
    /// Check if this error is [`AllocError::Disabled`].
    #[must_use]
    pub fn is_disabled(&self) -> bool {
        matches!(self, AllocError::Disabled)
    }
}

/// Abstraction over size classes.
#[derive(Clone, Copy)]
struct SizeClass(usize);

impl SizeClass {
    const fn new_unchecked(value: usize) -> Self {
        Self(value)
    }

    const fn index(self) -> usize {
        self.0 - VALID_SIZE_CLASS.start
    }

    /// The size in bytes of this size class.
    const fn byte_size(self) -> usize {
        1 << self.0
    }

    const fn from_index(index: usize) -> Self {
        Self(index + VALID_SIZE_CLASS.start)
    }

    /// Obtain a size class from a size in bytes.
    fn from_byte_size(byte_size: usize) -> Result<Self, AllocError> {
        let class = byte_size.next_power_of_two().trailing_zeros() as usize;
        class.try_into()
    }

    const fn from_byte_size_unchecked(byte_size: usize) -> Self {
        Self::new_unchecked(byte_size.next_power_of_two().trailing_zeros() as usize)
    }
}

impl TryFrom<usize> for SizeClass {
    type Error = AllocError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        if VALID_SIZE_CLASS.contains(&value) {
            Ok(SizeClass(value))
        } else {
            Err(AllocError::InvalidSizeClass(value))
        }
    }
}

#[derive(Default, Debug)]
struct AllocStats {
    allocations: AtomicU64,
    slow_path: AtomicU64,
    refill: AtomicU64,
    deallocations: AtomicU64,
    clear_eager: AtomicU64,
    clear_slow: AtomicU64,
}

/// Handle to the shared global state.
static INJECTOR: OnceLock<GlobalStealer> = OnceLock::new();

/// Enabled switch to turn on or off lgalloc. Off by default.
static LGALLOC_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable eager returning of memory. Off by default.
static LGALLOC_EAGER_RETURN: AtomicBool = AtomicBool::new(false);

/// Dampener in the file growth rate. 0 corresponds to doubling and in general `n` to `1+1/(n+1)`.
///
/// Setting this to 0 results in creating files with doubling capacity.
/// Larger numbers result in more conservative approaches that create more files.
static LGALLOC_FILE_GROWTH_DAMPENER: AtomicUsize = AtomicUsize::new(0);

/// The size of allocations to retain locally, per thread and size class.
static LOCAL_BUFFER_BYTES: AtomicUsize = AtomicUsize::new(32 << 20);

/// Type maintaining the global state for each size class.
struct GlobalStealer {
    /// State for each size class. An entry at position `x` handle size class `x`, which is areas
    /// of size `1<<x`.
    size_classes: Vec<SizeClassState>,
    /// Path to store files
    path: RwLock<Option<PathBuf>>,
    /// Shared token to access background thread.
    background_sender: Mutex<Option<(JoinHandle<()>, Sender<BackgroundWorkerConfig>)>>,
}

/// Per-size-class state
#[derive(Default)]
struct SizeClassState {
    /// Handle to memory-mapped regions.
    ///
    /// We must never dereference the memory-mapped regions stored here.
    areas: RwLock<Vec<ManuallyDrop<(File, MmapMut)>>>,
    /// Injector to distribute memory globally.
    injector: Injector<Handle>,
    /// Injector to distribute memory globally, freed memory.
    clean_injector: Injector<Handle>,
    /// Slow-path lock to refill pool.
    lock: Mutex<()>,
    /// Thread stealers to allow all participating threads to steal memory.
    stealers: RwLock<HashMap<ThreadId, PerThreadState<Handle>>>,
    /// Summed stats for terminated threads.
    alloc_stats: AllocStats,
    /// Total virtual size of all mappings in this size class in bytes.
    total_bytes: AtomicUsize,
    /// Count of areas backing this size class.
    area_count: AtomicUsize,
}

impl GlobalStealer {
    /// Obtain the shared global state.
    fn get_static() -> &'static Self {
        INJECTOR.get_or_init(Self::new)
    }

    /// Obtain the per-size-class global state.
    fn get_size_class(&self, size_class: SizeClass) -> &SizeClassState {
        &self.size_classes[size_class.index()]
    }

    fn new() -> Self {
        let mut size_classes = Vec::with_capacity(VALID_SIZE_CLASS.len());

        for _ in VALID_SIZE_CLASS {
            size_classes.push(SizeClassState::default());
        }

        Self {
            size_classes,
            path: RwLock::default(),
            background_sender: Mutex::default(),
        }
    }
}

impl Drop for GlobalStealer {
    fn drop(&mut self) {
        take(&mut self.size_classes);
    }
}

struct PerThreadState<T> {
    stealer: Stealer<T>,
    alloc_stats: Arc<AllocStats>,
}

/// Per-thread and state, sharded by size class.
struct ThreadLocalStealer {
    /// Per-size-class state
    size_classes: Vec<LocalSizeClass>,
    _phantom: PhantomUnsyncUnsend<Self>,
}

impl ThreadLocalStealer {
    fn new() -> Self {
        let thread_id = std::thread::current().id();
        let size_classes = VALID_SIZE_CLASS
            .map(|size_class| LocalSizeClass::new(SizeClass::new_unchecked(size_class), thread_id))
            .collect();
        Self {
            size_classes,
            _phantom: PhantomData,
        }
    }

    /// Allocate a memory region from a specific size class.
    ///
    /// Returns [`AllocError::Disabled`] if lgalloc is not enabled. Returns other error types
    /// if out of memory, or an internal operation fails.
    fn allocate(&self, size_class: SizeClass) -> Result<Handle, AllocError> {
        if !LGALLOC_ENABLED.load(Ordering::Relaxed) {
            return Err(AllocError::Disabled);
        }
        self.size_classes[size_class.index()].get_with_refill()
    }

    /// Return memory to the allocator. Must have been obtained through [`allocate`].
    fn deallocate(&self, mem: Handle) {
        let size_class = SizeClass::from_byte_size_unchecked(mem.len());

        self.size_classes[size_class.index()].push(mem);
    }
}

thread_local! {
    static WORKER: RefCell<ThreadLocalStealer> = RefCell::new(ThreadLocalStealer::new());
}

/// Per-thread, per-size-class state
///
/// # Safety
///
/// We store parts of areas in this struct. Leaking this struct leaks the areas, which is safe
/// because we will never try to access or reclaim them.
struct LocalSizeClass {
    /// Local memory queue.
    worker: Worker<Handle>,
    /// Size class we're covering
    size_class: SizeClass,
    /// Handle to global size class state
    size_class_state: &'static SizeClassState,
    /// Owning thread's ID
    thread_id: ThreadId,
    /// Shared statistics maintained by this thread.
    stats: Arc<AllocStats>,
    /// Phantom data to prevent sending the type across thread boundaries.
    _phantom: PhantomUnsyncUnsend<Self>,
}

impl LocalSizeClass {
    /// Construct a new local size class state. Registers the worker with the global state.
    fn new(size_class: SizeClass, thread_id: ThreadId) -> Self {
        let worker = Worker::new_lifo();
        let stealer = GlobalStealer::get_static();
        let size_class_state = stealer.get_size_class(size_class);

        let stats = Arc::new(AllocStats::default());

        let mut lock = size_class_state.stealers.write().expect("lock poisoned");
        lock.insert(
            thread_id,
            PerThreadState {
                stealer: worker.stealer(),
                alloc_stats: Arc::clone(&stats),
            },
        );

        Self {
            worker,
            size_class,
            size_class_state,
            thread_id,
            stats,
            _phantom: PhantomData,
        }
    }

    /// Get a memory area. Tries to get a region from the local cache, before obtaining data from
    /// the global state. As a last option, obtains memory from other workers.
    ///
    /// Returns [`AllcError::OutOfMemory`] if all pools are empty.
    #[inline]
    fn get(&self) -> Result<Handle, AllocError> {
        self.worker
            .pop()
            .or_else(|| {
                std::iter::repeat_with(|| {
                    // The loop tries to obtain memory in the following order:
                    // 1. Memory from the global state,
                    // 2. Memory from the global cleaned state,
                    // 3. Memory from other threads.
                    let limit = 1.max(
                        LOCAL_BUFFER_BYTES.load(Ordering::Relaxed)
                            / self.size_class.byte_size()
                            / 2,
                    );

                    self.size_class_state
                        .injector
                        .steal_batch_with_limit_and_pop(&self.worker, limit)
                        .or_else(|| {
                            self.size_class_state
                                .clean_injector
                                .steal_batch_with_limit_and_pop(&self.worker, limit)
                        })
                        .or_else(|| {
                            self.size_class_state
                                .stealers
                                .read()
                                .expect("lock poisoned")
                                .values()
                                .map(|state| state.stealer.steal())
                                .collect()
                        })
                })
                .find(|s| !s.is_retry())
                .and_then(Steal::success)
            })
            .ok_or(AllocError::OutOfMemory)
    }

    /// Like [`Self::get()`] but trying to refill the pool if it is empty.
    fn get_with_refill(&self) -> Result<Handle, AllocError> {
        self.stats.allocations.fetch_add(1, Ordering::Relaxed);
        // Fast-path: Get non-blocking
        match self.get() {
            Err(AllocError::OutOfMemory) => {
                self.stats.slow_path.fetch_add(1, Ordering::Relaxed);
                // Get a slow-path lock
                let _lock = self.size_class_state.lock.lock().expect("lock poisoned");
                // Try again because another thread might have refilled already
                if let Ok(mem) = self.get() {
                    return Ok(mem);
                }
                self.try_refill_and_get()
            }
            r => r,
        }
    }

    /// Recycle memory. Stores it locally or forwards it to the global state.
    fn push(&self, mut mem: Handle) {
        debug_assert_eq!(mem.len(), self.size_class.byte_size());
        self.stats.deallocations.fetch_add(1, Ordering::Relaxed);
        if self.worker.len()
            >= LOCAL_BUFFER_BYTES.load(Ordering::Relaxed) / self.size_class.byte_size()
        {
            if LGALLOC_EAGER_RETURN.load(Ordering::Relaxed) {
                self.stats.clear_eager.fetch_add(1, Ordering::Relaxed);
                mem.fast_clear().expect("clearing successful");
            }
            self.size_class_state.injector.push(mem);
        } else {
            self.worker.push(mem);
        }
    }

    /// Refill the memory pool, and get one area.
    ///
    /// Returns an error if the memory pool cannot be refilled.
    fn try_refill_and_get(&self) -> Result<Handle, AllocError> {
        self.stats.refill.fetch_add(1, Ordering::Relaxed);
        let mut stash = self.size_class_state.areas.write().expect("lock poisoned");

        let initial_capacity = std::cmp::max(1, INITIAL_SIZE / self.size_class.byte_size());

        let last_capacity =
            stash.iter().last().map_or(0, |mmap| mmap.1.len()) / self.size_class.byte_size();
        let growth_dampener = LGALLOC_FILE_GROWTH_DAMPENER.load(Ordering::Relaxed);
        // We would like to grow the file capacity by a factor of `1+1/(growth_dampener+1)`,
        // but at least by `initial_capacity`.
        let next_capacity = last_capacity
            + std::cmp::max(
                initial_capacity,
                last_capacity / (growth_dampener.saturating_add(1)),
            );

        let next_byte_len = next_capacity * self.size_class.byte_size();
        let (file, mut mmap) = Self::init_file(next_byte_len)?;

        self.size_class_state
            .total_bytes
            .fetch_add(next_byte_len, Ordering::Relaxed);
        self.size_class_state
            .area_count
            .fetch_add(1, Ordering::Relaxed);

        // SAFETY: Memory region initialized, so pointers to it are valid.
        let mut chunks = mmap
            .as_mut()
            .chunks_exact_mut(self.size_class.byte_size())
            .map(|chunk| NonNull::new(chunk.as_mut_ptr()).expect("non-null"));

        // Capture first region to return immediately.
        let ptr = chunks.next().expect("At least once chunk allocated.");
        let mem = Handle::new(ptr, self.size_class.byte_size());

        // Stash remaining in the injector.
        for ptr in chunks {
            self.size_class_state
                .clean_injector
                .push(Handle::new(ptr, self.size_class.byte_size()));
        }

        stash.push(ManuallyDrop::new((file, mmap)));
        Ok(mem)
    }

    /// Allocate and map a file of size `byte_len`. Returns an handle, or error if the allocation
    /// fails.
    fn init_file(byte_len: usize) -> Result<(File, MmapMut), AllocError> {
        let file = {
            let path = GlobalStealer::get_static()
                .path
                .read()
                .expect("lock poisoned");
            let Some(path) = &*path else {
                return Err(AllocError::Io(std::io::Error::from(
                    std::io::ErrorKind::NotFound,
                )));
            };
            tempfile::tempfile_in(path)?
        };
        let fd = file.as_fd().as_raw_fd();
        let length = libc::off_t::try_from(byte_len).expect("Must fit");
        // SAFETY: Calling ftruncate on the file, which we just created.
        let ret = unsafe { libc::ftruncate(fd, length) };
        if ret != 0 {
            // file goes out of scope here, so no need for further cleanup.
            return Err(std::io::Error::last_os_error().into());
        }
        // SAFETY: We only map `file` once, and never share it with other processes.
        let mmap = unsafe { memmap2::MmapOptions::new().map_mut(&file)? };
        assert_eq!(mmap.len(), byte_len);
        Ok((file, mmap))
    }
}

impl Drop for LocalSizeClass {
    fn drop(&mut self) {
        // Remove state associated with thread
        if let Ok(mut lock) = self.size_class_state.stealers.write() {
            lock.remove(&self.thread_id);
        }

        // Send memory back to global state
        while let Some(mem) = self.worker.pop() {
            self.size_class_state.injector.push(mem);
        }

        let ordering = Ordering::Relaxed;

        // Update global metrics by moving all worker-local metrics to global state.
        self.size_class_state
            .alloc_stats
            .allocations
            .fetch_add(self.stats.allocations.load(ordering), ordering);
        let global_stats = &self.size_class_state.alloc_stats;
        global_stats
            .refill
            .fetch_add(self.stats.refill.load(ordering), ordering);
        global_stats
            .slow_path
            .fetch_add(self.stats.slow_path.load(ordering), ordering);
        global_stats
            .deallocations
            .fetch_add(self.stats.deallocations.load(ordering), ordering);
        global_stats
            .clear_slow
            .fetch_add(self.stats.clear_slow.load(ordering), ordering);
        global_stats
            .clear_eager
            .fetch_add(self.stats.clear_eager.load(ordering), ordering);
    }
}

/// Access the per-thread context.
fn thread_context<R, F: FnOnce(&ThreadLocalStealer) -> R>(f: F) -> R {
    WORKER.with(|cell| f(&cell.borrow()))
}

/// Allocate a memory area suitable to hold `capacity` consecutive elements of `T`.
///
/// Returns a pointer, a capacity in `T`, and a handle if successful, and an error
/// otherwise. The capacity can be larger than requested.
///
/// The memory must be freed using [`deallocate`], otherwise the memory leaks. The memory can be freed on a different thread.
///
/// # Errors
///
/// Allocate errors if the capacity cannot be supported by one of the size classes,
/// the alignment requirements of `T` cannot be fulfilled, if no more memory can be
/// obtained from the system, or if any syscall fails.
///
/// The function also returns an error if lgalloc is disabled.
///
/// In the case of an error, no memory is allocated, and we maintain the internal
/// invariants of the allocator.
///
/// # Panics
///
/// The function can panic on internal errors, specifically when an allocation returned
/// an unexpected size. In this case, we do not maintain the allocator invariants
/// and the caller should abort the process.
///
/// Panics if the thread local variable has been dropped, see [`std::thread::LocalKey`]
/// for details.
pub fn allocate<T>(capacity: usize) -> Result<(NonNull<T>, usize, Handle), AllocError> {
    if std::mem::size_of::<T>() == 0 {
        return Ok((NonNull::dangling(), usize::MAX, Handle::dangling()));
    } else if capacity == 0 {
        return Ok((NonNull::dangling(), 0, Handle::dangling()));
    }

    // Round up to at least a page.
    let byte_len = std::cmp::max(page_size::get(), std::mem::size_of::<T>() * capacity);
    // With above rounding up to page sizes, we only allocate multiples of page size because
    // we only support powers-of-two sized regions.
    let size_class = SizeClass::from_byte_size(byte_len)?;

    let handle = thread_context(|s| s.allocate(size_class))?;
    debug_assert_eq!(handle.len(), size_class.byte_size());
    let ptr: NonNull<T> = handle.as_non_null().cast();
    // Memory region should be page-aligned, which we assume to be larger than any alignment
    // we might encounter. If this is not the case, bail out.
    if ptr.as_ptr().align_offset(std::mem::align_of::<T>()) != 0 {
        thread_context(move |s| s.deallocate(handle));
        return Err(AllocError::UnalignedMemory);
    }
    let actual_capacity = handle.len() / std::mem::size_of::<T>();
    Ok((ptr, actual_capacity, handle))
}

/// Free the memory referenced by `handle`, which has been obtained from [`allocate`].
///
/// This function cannot fail. The caller must not access the memory after freeing it. The caller is responsible
/// for dropping/forgetting data.
///
/// # Panics
///
/// Panics if the thread local variable has been dropped, see [`std::thread::LocalKey`]
/// for details.
pub fn deallocate(handle: Handle) {
    if handle.is_dangling() {
        return;
    }
    thread_context(|s| s.deallocate(handle));
}

/// A background worker that performs periodic tasks.
struct BackgroundWorker {
    config: BackgroundWorkerConfig,
    receiver: Receiver<BackgroundWorkerConfig>,
    global_stealer: &'static GlobalStealer,
    worker: Worker<Handle>,
}

impl BackgroundWorker {
    fn new(receiver: Receiver<BackgroundWorkerConfig>) -> Self {
        let config = BackgroundWorkerConfig {
            interval: Duration::MAX,
            ..Default::default()
        };
        let global_stealer = GlobalStealer::get_static();
        let worker = Worker::new_fifo();
        Self {
            config,
            receiver,
            global_stealer,
            worker,
        }
    }

    fn run(&mut self) {
        let mut next_cleanup: Option<Instant> = None;
        loop {
            let timeout = next_cleanup.map_or(Duration::MAX, |next_cleanup| {
                next_cleanup.saturating_duration_since(Instant::now())
            });
            match self.receiver.recv_timeout(timeout) {
                Ok(config) => {
                    self.config = config;
                    next_cleanup = None;
                }
                Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => {
                    self.maintenance();
                }
            }
            next_cleanup = next_cleanup
                .unwrap_or_else(Instant::now)
                .checked_add(self.config.interval);
        }
    }

    fn maintenance(&self) {
        for (index, size_class_state) in self.global_stealer.size_classes.iter().enumerate() {
            let size_class = SizeClass::from_index(index);
            let count = self.clear(size_class, size_class_state, &self.worker);
            size_class_state
                .alloc_stats
                .clear_slow
                .fetch_add(count.try_into().expect("must fit"), Ordering::Relaxed);
        }
    }

    fn clear(
        &self,
        size_class: SizeClass,
        state: &SizeClassState,
        worker: &Worker<Handle>,
    ) -> usize {
        // Clear batch size, and at least one element.
        let byte_size = size_class.byte_size();
        let mut limit = (self.config.clear_bytes + byte_size - 1) / byte_size;
        let mut count = 0;
        let mut steal = Steal::Retry;
        while limit > 0 && !steal.is_empty() {
            steal = std::iter::repeat_with(|| state.injector.steal_batch_with_limit(worker, limit))
                .find(|s| !s.is_retry())
                .unwrap_or(Steal::Empty);
            while let Some(mut mem) = worker.pop() {
                match mem.clear() {
                    Ok(()) => count += 1,
                    Err(e) => panic!("Syscall failed: {e:?}"),
                }
                state.clean_injector.push(mem);
                limit -= 1;
            }
        }
        count
    }
}

/// Set or update the configuration for lgalloc.
///
/// The function accepts a configuration, which is then applied on lgalloc. It allows clients to
/// change the path where area files reside, and change the configuration of the background task.
///
/// Updating the area path only applies to new allocations, existing allocations are not moved to
/// the new path.
///
/// Updating the background thread configuration eventually applies the new configuration on the
/// running thread, or starts the background worker.
///
/// # Panics
///
/// Panics if the internal state of lgalloc is corrupted.
pub fn lgalloc_set_config(config: &LgAlloc) {
    let stealer = GlobalStealer::get_static();

    if let Some(enabled) = &config.enabled {
        LGALLOC_ENABLED.store(*enabled, Ordering::Relaxed);
    }

    if let Some(eager_return) = &config.eager_return {
        LGALLOC_EAGER_RETURN.store(*eager_return, Ordering::Relaxed);
    }

    if let Some(path) = &config.path {
        *stealer.path.write().expect("lock poisoned") = Some(path.clone());
    }

    if let Some(file_growth_dampener) = &config.file_growth_dampener {
        LGALLOC_FILE_GROWTH_DAMPENER.store(*file_growth_dampener, Ordering::Relaxed);
    }

    if let Some(local_buffer_bytes) = &config.local_buffer_bytes {
        LOCAL_BUFFER_BYTES.store(*local_buffer_bytes, Ordering::Relaxed);
    }

    if let Some(config) = config.background_config.clone() {
        let mut lock = stealer.background_sender.lock().expect("lock poisoned");

        let config = if let Some((_, sender)) = &*lock {
            match sender.send(config) {
                Ok(()) => None,
                Err(err) => Some(err.0),
            }
        } else {
            Some(config)
        };
        if let Some(config) = config {
            let (sender, receiver) = std::sync::mpsc::channel();
            let mut worker = BackgroundWorker::new(receiver);
            let join_handle = std::thread::Builder::new()
                .name("lgalloc-0".to_string())
                .spawn(move || worker.run())
                .expect("thread started successfully");
            sender.send(config).expect("Receiver exists");
            *lock = Some((join_handle, sender));
        }
    }
}

/// Configuration for lgalloc's background worker.
#[derive(Default, Debug, Clone, Eq, PartialEq)]
pub struct BackgroundWorkerConfig {
    /// How frequently it should tick
    pub interval: Duration,
    /// How many bytes to clear per size class.
    pub clear_bytes: usize,
}

/// Lgalloc configuration
#[derive(Default, Clone, Eq, PartialEq)]
pub struct LgAlloc {
    /// Whether the allocator is enabled or not.
    pub enabled: Option<bool>,
    /// Path where files reside.
    pub path: Option<PathBuf>,
    /// Configuration of the background worker.
    pub background_config: Option<BackgroundWorkerConfig>,
    /// Whether to return physical memory on deallocate
    pub eager_return: Option<bool>,
    /// Dampener in the file growth rate. 0 corresponds to doubling and in general `n` to `1+1/(n+1)`.
    pub file_growth_dampener: Option<usize>,
    /// Size of the per-thread per-size class cache, in bytes.
    pub local_buffer_bytes: Option<usize>,
}

impl LgAlloc {
    /// Construct a new configuration. All values are initialized to their default (None) values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable lgalloc globally.
    pub fn enable(&mut self) -> &mut Self {
        self.enabled = Some(true);
        self
    }

    /// Disable lgalloc globally.
    pub fn disable(&mut self) -> &mut Self {
        self.enabled = Some(false);
        self
    }

    /// Set the background worker configuration.
    pub fn with_background_config(&mut self, config: BackgroundWorkerConfig) -> &mut Self {
        self.background_config = Some(config);
        self
    }

    /// Set the area file path.
    pub fn with_path(&mut self, path: PathBuf) -> &mut Self {
        self.path = Some(path);
        self
    }

    /// Enable eager memory reclamation.
    pub fn eager_return(&mut self, eager_return: bool) -> &mut Self {
        self.eager_return = Some(eager_return);
        self
    }

    /// Set the file growth dampener.
    pub fn file_growth_dampener(&mut self, file_growth_dapener: usize) -> &mut Self {
        self.file_growth_dampener = Some(file_growth_dapener);
        self
    }

    /// Set the local buffer size.
    pub fn local_buffer_bytes(&mut self, local_buffer_bytes: usize) -> &mut Self {
        self.local_buffer_bytes = Some(local_buffer_bytes);
        self
    }
}

/// Determine global statistics per size class.
///
/// This function is supposed to be relatively fast. It causes some syscalls, but they
/// should be cheap (stat on a file descriptor).
///
/// Note that this function take a read lock on various structures. It calls `fstat` while
/// holding a read lock on portions of the global state, which can block refills until the
/// function returns.
///
/// # Panics
///
/// Panics if the internal state of lgalloc is corrupted.
pub fn lgalloc_stats() -> LgAllocStats {
    LgAllocStats::read(None)
}

/// Determine global statistics per size class, and include mapping information.
///
/// This function can be very slow as it needs to read the `numa_maps` file. Depending
/// on the heap size of the program, this can take seconds to minutes, so call this
/// function with care.
///
/// Note that this function take a read lock on various structures. In addition to the locks
/// described on [`lgalloc_stats`], this function reads the `/proc/self/numa_maps` file without
/// holding any locks, but the kernel might block other memory operations while reading this file.
///
/// # Panics
///
/// Panics if the internal state of lgalloc is corrupted.
pub fn lgalloc_stats_with_mapping() -> std::io::Result<LgAllocStats> {
    let mut numa_map = NumaMap::from_file("/proc/self/numa_maps")?;
    Ok(LgAllocStats::read(Some(&mut numa_map)))
}

/// Statistics about lgalloc's internal behavior.
#[derive(Debug)]
pub struct LgAllocStats {
    /// Per size-class statistics.
    pub size_class: Vec<(usize, SizeClassStats)>,
    /// Per size-class and backing file statistics. Each entry identifies the
    /// size class it describes, and there can be multiple entries for each size class.
    pub file: Vec<(usize, std::io::Result<FileStats>)>,
    /// Per size-class and map statistics. Each entry identifies the
    /// size class it describes, and there can be multiple entries for each size class.
    pub map: Option<Vec<(usize, MapStats)>>,
}

impl LgAllocStats {
    /// Read lgalloc statistics.
    ///
    /// Supply a `numa_map` to obtain mapping stats.
    fn read(mut numa_map: Option<&mut NumaMap>) -> Self {
        let global = GlobalStealer::get_static();

        if let Some(numa_map) = numa_map.as_mut() {
            // Normalize numa_maps, and sort by address.
            for entry in &mut numa_map.ranges {
                entry.normalize();
            }
            numa_map.ranges.sort();
        }

        let mut size_class_stats = Vec::with_capacity(VALID_SIZE_CLASS.len());
        let mut file_stats = Vec::default();
        let mut map_stats = Vec::default();
        for (index, state) in global.size_classes.iter().enumerate() {
            let size_class = SizeClass::from_index(index);
            let size_class_bytes = size_class.byte_size();

            size_class_stats.push((size_class_bytes, SizeClassStats::from(state)));

            let areas = state.areas.read().expect("lock poisoned");
            for (file, mmap) in areas.iter().map(Deref::deref) {
                file_stats.push((size_class_bytes, FileStats::extract_from(file)));
                if let Some(numa_map) = numa_map.as_deref() {
                    map_stats.push((size_class_bytes, MapStats::extract_from(mmap, numa_map)));
                }
            }
        }

        Self {
            size_class: size_class_stats,
            file: file_stats,
            map: numa_map.map(|_| map_stats),
        }
    }
}

/// Statistics per size class.
#[derive(Debug)]
pub struct SizeClassStats {
    /// Number of areas backing a size class.
    pub areas: usize,
    /// Total number of bytes summed across all areas.
    pub area_total_bytes: usize,
    /// Free regions
    pub free_regions: usize,
    /// Clean free regions in the global allocator
    pub clean_regions: usize,
    /// Regions in the global allocator
    pub global_regions: usize,
    /// Regions retained in thread-local allocators
    pub thread_regions: usize,
    /// Total allocations
    pub allocations: u64,
    /// Total slow-path allocations (globally out of memory)
    pub slow_path: u64,
    /// Total refills
    pub refill: u64,
    /// Total deallocations
    pub deallocations: u64,
    /// Total times memory has been returned to the OS (eager reclamation) in regions.
    pub clear_eager_total: u64,
    /// Total times memory has been returned to the OS (slow reclamation) in regions.
    pub clear_slow_total: u64,
}

impl From<&SizeClassState> for SizeClassStats {
    fn from(size_class_state: &SizeClassState) -> Self {
        let areas = size_class_state.area_count.load(Ordering::Relaxed);
        let area_total_bytes = size_class_state.total_bytes.load(Ordering::Relaxed);
        let global_regions = size_class_state.injector.len();
        let clean_regions = size_class_state.clean_injector.len();
        let stealers = size_class_state.stealers.read().expect("lock poisoned");
        let mut thread_regions = 0;
        let mut allocations = 0;
        let mut deallocations = 0;
        let mut refill = 0;
        let mut slow_path = 0;
        let mut clear_eager_total = 0;
        let mut clear_slow_total = 0;
        for thread_state in stealers.values() {
            thread_regions += thread_state.stealer.len();
            let thread_stats = &*thread_state.alloc_stats;
            allocations += thread_stats.allocations.load(Ordering::Relaxed);
            deallocations += thread_stats.deallocations.load(Ordering::Relaxed);
            refill += thread_stats.refill.load(Ordering::Relaxed);
            slow_path += thread_stats.slow_path.load(Ordering::Relaxed);
            clear_eager_total += thread_stats.clear_eager.load(Ordering::Relaxed);
            clear_slow_total += thread_stats.clear_slow.load(Ordering::Relaxed);
        }

        let free_regions = thread_regions + global_regions + clean_regions;

        let global_stats = &size_class_state.alloc_stats;
        allocations += global_stats.allocations.load(Ordering::Relaxed);
        deallocations += global_stats.deallocations.load(Ordering::Relaxed);
        refill += global_stats.refill.load(Ordering::Relaxed);
        slow_path += global_stats.slow_path.load(Ordering::Relaxed);
        clear_eager_total += global_stats.clear_eager.load(Ordering::Relaxed);
        clear_slow_total += global_stats.clear_slow.load(Ordering::Relaxed);
        Self {
            areas,
            area_total_bytes,
            free_regions,
            global_regions,
            clean_regions,
            thread_regions,
            allocations,
            deallocations,
            refill,
            slow_path,
            clear_eager_total,
            clear_slow_total,
        }
    }
}

/// Statistics per size class and backing file.
#[derive(Debug)]
pub struct FileStats {
    /// The size of the file in bytes.
    pub file_size: usize,
    /// Size of the file on disk in bytes.
    pub allocated_size: usize,
}

impl FileStats {
    /// Extract file statistics from a file descriptor. Calls `fstat` on the file to obtain
    /// the size and allocated size.
    fn extract_from(file: &File) -> std::io::Result<Self> {
        let mut stat: MaybeUninit<libc::stat> = MaybeUninit::uninit();
        // SAFETY: File descriptor valid, stat object valid.
        let ret = unsafe { libc::fstat(file.as_raw_fd(), stat.as_mut_ptr()) };
        if ret == -1 {
            Err(std::io::Error::last_os_error())
        } else {
            // SAFETY: `stat` is initialized in the fstat non-error case.
            let stat = unsafe { stat.assume_init_ref() };
            let blocks = stat.st_blocks.try_into().unwrap_or(0);
            let file_size = stat.st_size.try_into().unwrap_or(0);
            Ok(FileStats {
                file_size,
                // Documented as multiples of 512
                allocated_size: blocks * 512,
            })
        }
    }
}

/// Statistics per size class and mapping.
#[derive(Debug)]
pub struct MapStats {
    /// Number of mapped bytes, if different from `dirty`. Consult `man 7 numa` for details.
    pub mapped: usize,
    /// Number of active bytes. Consult `man 7 numa` for details.
    pub active: usize,
    /// Number of dirty bytes. Consult `man 7 numa` for details.
    pub dirty: usize,
}

impl MapStats {
    /// Extract memory map stats for `mmap` based on `numa_map`.
    ///
    /// The ranges of in the numa map file must be sorted by address and normalized.
    fn extract_from(mmap: &MmapMut, numa_map: &NumaMap) -> Self {
        // TODO: Use `addr` once our MSRV is 1.84.
        let base = mmap.as_ptr().cast::<()>() as usize;
        let range = match numa_map
            .ranges
            .binary_search_by(|range| range.address.cmp(&base))
        {
            Ok(pos) => Some(&numa_map.ranges[pos]),
            // `numa_maps` only updates periodically, so we might be missing some
            // expected ranges.
            Err(_pos) => None,
        };

        let mut mapped = 0;
        let mut active = 0;
        let mut dirty = 0;
        for property in range.iter().flat_map(|e| e.properties.iter()) {
            match property {
                numa_maps::Property::Dirty(d) => dirty = *d,
                numa_maps::Property::Mapped(m) => mapped = *m,
                numa_maps::Property::Active(a) => active = *a,
                _ => {}
            }
        }

        Self {
            mapped,
            active,
            dirty,
        }
    }
}

#[cfg(test)]
mod test {
    use std::mem::{ManuallyDrop, MaybeUninit};
    use std::ptr::NonNull;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use serial_test::serial;

    use super::*;

    fn initialize() {
        lgalloc_set_config(
            LgAlloc::new()
                .enable()
                .with_background_config(BackgroundWorkerConfig {
                    interval: Duration::from_secs(1),
                    clear_bytes: 4 << 20,
                })
                .with_path(std::env::temp_dir())
                .file_growth_dampener(1),
        );
    }

    struct Wrapper<T> {
        handle: MaybeUninit<Handle>,
        ptr: NonNull<MaybeUninit<T>>,
        cap: usize,
    }

    unsafe impl<T: Send> Send for Wrapper<T> {}
    unsafe impl<T: Sync> Sync for Wrapper<T> {}

    impl<T> Wrapper<T> {
        fn allocate(capacity: usize) -> Result<Self, AllocError> {
            let (ptr, cap, handle) = allocate(capacity)?;
            assert!(cap > 0);
            let handle = MaybeUninit::new(handle);
            Ok(Self { ptr, cap, handle })
        }

        fn as_slice(&mut self) -> &mut [MaybeUninit<T>] {
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.cap) }
        }
    }

    impl<T> Drop for Wrapper<T> {
        fn drop(&mut self) {
            unsafe { deallocate(self.handle.assume_init_read()) };
        }
    }

    #[test]
    #[serial]
    fn test_readme() -> Result<(), AllocError> {
        initialize();

        // Allocate memory
        let (ptr, cap, handle) = allocate::<u8>(2 << 20)?;
        // SAFETY: `allocate` returns a valid memory region and errors otherwise.
        let mut vec = ManuallyDrop::new(unsafe { Vec::from_raw_parts(ptr.as_ptr(), 0, cap) });

        // Write into region, make sure not to reallocate vector.
        vec.extend_from_slice(&[1, 2, 3, 4]);

        // We can read from the vector.
        assert_eq!(&*vec, &[1, 2, 3, 4]);

        // Deallocate after use
        deallocate(handle);
        Ok(())
    }

    #[test]
    #[serial]
    fn test_1() -> Result<(), AllocError> {
        initialize();
        <Wrapper<u8>>::allocate(4 << 20)?.as_slice()[0] = MaybeUninit::new(1);
        Ok(())
    }

    #[test]
    #[serial]
    fn test_3() -> Result<(), AllocError> {
        initialize();
        let until = Arc::new(AtomicBool::new(true));

        let inner = || {
            let until = Arc::clone(&until);
            move || {
                let mut i = 0;
                let until = &*until;
                while until.load(Ordering::Relaxed) {
                    i += 1;
                    let mut r = <Wrapper<u8>>::allocate(4 << 20).unwrap();
                    r.as_slice()[0] = MaybeUninit::new(1);
                }
                println!("repetitions: {i}");
            }
        };
        let handles = [
            std::thread::spawn(inner()),
            std::thread::spawn(inner()),
            std::thread::spawn(inner()),
            std::thread::spawn(inner()),
        ];
        std::thread::sleep(Duration::from_secs(4));
        until.store(false, Ordering::Relaxed);
        for handle in handles {
            handle.join().unwrap();
        }
        // std::thread::sleep(Duration::from_secs(600));
        Ok(())
    }

    #[test]
    #[serial]
    fn test_4() -> Result<(), AllocError> {
        initialize();
        let until = Arc::new(AtomicBool::new(true));

        let inner = || {
            let until = Arc::clone(&until);
            move || {
                let mut i = 0;
                let until = &*until;
                let batch = 64;
                let mut buffer = Vec::with_capacity(batch);
                while until.load(Ordering::Relaxed) {
                    i += 64;
                    buffer.extend((0..batch).map(|_| {
                        let mut r = <Wrapper<u8>>::allocate(2 << 20).unwrap();
                        r.as_slice()[0] = MaybeUninit::new(1);
                        r
                    }));
                    buffer.clear();
                }
                println!("repetitions vec: {i}");
            }
        };
        let handles = [
            std::thread::spawn(inner()),
            std::thread::spawn(inner()),
            std::thread::spawn(inner()),
            std::thread::spawn(inner()),
        ];
        std::thread::sleep(Duration::from_secs(4));
        until.store(false, Ordering::Relaxed);
        for handle in handles {
            handle.join().unwrap();
        }
        std::thread::sleep(Duration::from_secs(1));
        let stats = lgalloc_stats();
        for size_class in &stats.size_class {
            println!("size_class {:?}", size_class);
        }
        for (size_class, file_stats) in &stats.file {
            match file_stats {
                Ok(file_stats) => println!("file_stats {size_class} {file_stats:?}"),
                Err(e) => eprintln!("error: {e}"),
            }
        }
        Ok(())
    }

    #[test]
    #[serial]
    fn leak() -> Result<(), AllocError> {
        lgalloc_set_config(&LgAlloc {
            enabled: Some(true),
            path: Some(std::env::temp_dir()),
            ..Default::default()
        });
        let r = <Wrapper<u8>>::allocate(1000)?;

        let thread = std::thread::spawn(move || drop(r));

        thread.join().unwrap();
        Ok(())
    }

    #[test]
    #[serial]
    fn test_zst() -> Result<(), AllocError> {
        initialize();
        <Wrapper<()>>::allocate(10)?;
        Ok(())
    }

    #[test]
    #[serial]
    fn test_zero_capacity_zst() -> Result<(), AllocError> {
        initialize();
        <Wrapper<()>>::allocate(0)?;
        Ok(())
    }

    #[test]
    #[serial]
    fn test_zero_capacity_nonzst() -> Result<(), AllocError> {
        initialize();
        <Wrapper<()>>::allocate(0)?;
        Ok(())
    }

    #[test]
    #[serial]
    fn test_stats() -> Result<(), AllocError> {
        initialize();
        let (_ptr, _cap, handle) = allocate::<usize>(1024)?;
        deallocate(handle);

        let stats = lgalloc_stats();

        assert!(!stats.size_class.is_empty());

        Ok(())
    }

    #[test]
    #[serial]
    fn test_disable() {
        lgalloc_set_config(&*LgAlloc::new().disable());
        assert!(matches!(allocate::<u8>(1024), Err(AllocError::Disabled)));
    }
}
