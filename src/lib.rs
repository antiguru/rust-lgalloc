//! A size-classed large object allocator backed by anonymous mappings.
//!
//! This library contains types to allocate memory outside the heap,
//! supporting power-of-two object sizes. Each size class has its own
//! memory pool.
//!
//! Allocations use `MAP_ANONYMOUS` with `MADV_HUGEPAGE` hints to benefit
//! from transparent huge pages when available.
//!
//! # Safety
//!
//! This library is very unsafe on account of `unsafe` and interacting directly
//! with libc, including Linux extension.
//!
//! The library relies on anonymous memory mappings. Users must not fork the process
//! because otherwise two processes would share the same mappings, causing undefined behavior
//! because the mutable pointers would not be unique anymore. Unfortunately, there is no way
//! to tell the memory subsystem that the shared mappings must not be inherited.
//!
//! Clients must not lock pages (`mlock`), or need to unlock the pages before returning them
//! to lgalloc.

#![deny(missing_docs)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::mem::{take, ManuallyDrop};
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::thread::{JoinHandle, ThreadId};
use std::time::{Duration, Instant};

use crossbeam_deque::{Injector, Steal, Stealer, Worker};
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

    /// Indicate that the memory is not in use and that the OS can lazily recycle it.
    ///
    /// Uses `MADV_FREE` on Linux (lazy reclaim, avoids immediate page zeroing) and
    /// `MADV_DONTNEED` elsewhere.
    fn clear(&mut self) -> std::io::Result<()> {
        // SAFETY: `MADV_CLEAR_STRATEGY` guaranteed to be a valid argument.
        unsafe { self.madvise(MADV_CLEAR_STRATEGY) }
    }

    /// Indicate that the memory is not in use and that the OS should immediately recycle it.
    fn fast_clear(&mut self) -> std::io::Result<()> {
        // SAFETY: `libc::MADV_DONTNEED` documented to be a valid argument.
        unsafe { self.madvise(libc::MADV_DONTNEED) }
    }

    /// Hint to the kernel that a byte range within this allocation will be needed soon.
    ///
    /// Issues `MADV_WILLNEED` to initiate asynchronous page-in for the range
    /// `[offset, offset + len)`, which is especially useful when pages may reside in
    /// swap. The kernel begins reading pages in the background; a subsequent access to
    /// a prefetched page will either find it already resident or wait for a shorter I/O.
    ///
    /// `offset` and `len` do not need to be page-aligned — the kernel rounds to page
    /// boundaries internally.
    ///
    /// This is a performance hint and never affects correctness. The kernel may ignore
    /// it under memory pressure. Calling it on already-resident pages is a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError::OutOfMemory`] if `offset + len` exceeds the allocation length.
    pub fn prefetch(&self, offset: usize, len: usize) -> Result<(), AllocError> {
        if len == 0 || self.is_dangling() {
            return Ok(());
        }
        if offset.saturating_add(len) > self.len {
            return Err(AllocError::OutOfMemory);
        }
        // SAFETY: MADV_WILLNEED is a hint that never modifies memory contents.
        // The pointer arithmetic is in-bounds per the check above.
        unsafe {
            let ptr = self.as_non_null().as_ptr().add(offset);
            libc::madvise(ptr.cast(), len, libc::MADV_WILLNEED);
        }
        Ok(())
    }

    /// Call `madvise` on the memory region. Unsafe because `advice` is passed verbatim.
    unsafe fn madvise(&self, advice: libc::c_int) -> std::io::Result<()> {
        // SAFETY: Calling into `madvise`:
        // * The ptr is page-aligned by construction.
        // * The ptr + length is page-aligned by construction (not required but surprising otherwise)
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

/// Initial area size
const INITIAL_SIZE: usize = 32 << 20;

/// Range of valid size classes.
pub const VALID_SIZE_CLASS: Range<usize> = 20..37;

/// Strategy for background worker clear: `MADV_FREE` on Linux (lazy reclaim), `MADV_DONTNEED` elsewhere.
#[cfg(target_os = "linux")]
const MADV_CLEAR_STRATEGY: libc::c_int = libc::MADV_FREE;

#[cfg(not(target_os = "linux"))]
const MADV_CLEAR_STRATEGY: libc::c_int = libc::MADV_DONTNEED;

/// Whether we have already warned about `MADV_HUGEPAGE` failure.
#[cfg(target_os = "linux")]
static MADV_HUGEPAGE_WARNED: AtomicBool = AtomicBool::new(false);

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

/// Dampener in the area growth rate. 0 corresponds to doubling and in general `n` to `1+1/(n+1)`.
///
/// Setting this to 0 results in creating areas with doubling capacity.
/// Larger numbers result in more conservative approaches that create more areas.
static LGALLOC_GROWTH_DAMPENER: AtomicUsize = AtomicUsize::new(0);

/// The size of allocations to retain locally, per thread and size class.
static LOCAL_BUFFER_BYTES: AtomicUsize = AtomicUsize::new(32 << 20);

/// Type maintaining the global state for each size class.
struct GlobalStealer {
    /// State for each size class. An entry at position `x` handle size class `x`, which is areas
    /// of size `1<<x`.
    size_classes: Vec<SizeClassState>,
    /// Shared token to access background thread.
    background_sender: Mutex<Option<(JoinHandle<()>, Sender<BackgroundWorkerConfig>)>>,
}

/// Per-size-class state
#[derive(Default)]
struct SizeClassState {
    /// Handle to anonymous memory-mapped regions.
    ///
    /// We must never dereference the memory-mapped regions stored here.
    areas: RwLock<Vec<ManuallyDrop<(usize, usize)>>>,
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
            background_sender: Mutex::default(),
        }
    }
}

impl Drop for GlobalStealer {
    fn drop(&mut self) {
        // Unmap all areas to return virtual address space.
        for size_class_state in &mut self.size_classes {
            let mut areas = size_class_state.areas.write().expect("lock poisoned");
            for area in areas.drain(..) {
                let (addr, len) = ManuallyDrop::into_inner(area);
                // SAFETY: `addr` and `len` were returned by `mmap` during `try_refill_and_get`.
                unsafe {
                    libc::munmap(addr as *mut libc::c_void, len);
                }
            }
        }
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
            stash.iter().last().map_or(0, |mmap| mmap.1) / self.size_class.byte_size();
        let growth_dampener = LGALLOC_GROWTH_DAMPENER.load(Ordering::Relaxed);
        // We would like to grow the area capacity by a factor of `1+1/(growth_dampener+1)`,
        // but at least by `initial_capacity`.
        let next_capacity = last_capacity
            + std::cmp::max(
                initial_capacity,
                last_capacity / (growth_dampener.saturating_add(1)),
            );

        let next_byte_len = next_capacity * self.size_class.byte_size();

        let (mmap_ptr, slice) = mmap_anonymous(next_byte_len)?;

        self.size_class_state
            .total_bytes
            .fetch_add(next_byte_len, Ordering::Relaxed);
        self.size_class_state
            .area_count
            .fetch_add(1, Ordering::Relaxed);

        // SAFETY: Memory region initialized, so pointers to it are valid.
        let mut chunks = slice
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

        stash.push(ManuallyDrop::new((mmap_ptr, next_byte_len)));
        Ok(mem)
    }
}

/// Create an anonymous memory mapping with huge page hints.
///
/// Returns a tuple of `(address, mutable slice)` on success.
fn mmap_anonymous(len: usize) -> Result<(usize, &'static mut [u8]), AllocError> {
    // SAFETY: Creating an anonymous private mapping with no file descriptor.
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error().into());
    }

    // Hint to the kernel to use transparent huge pages. This is a performance hint,
    // not a correctness requirement — THP may be disabled system-wide.
    #[cfg(target_os = "linux")]
    {
        // SAFETY: `ptr` is a valid mapping returned by `mmap` above.
        let ret = unsafe { libc::madvise(ptr, len, libc::MADV_HUGEPAGE) };
        if ret == -1 && !MADV_HUGEPAGE_WARNED.swap(true, Ordering::Relaxed) {
            eprintln!(
                "lgalloc: MADV_HUGEPAGE failed: {}. Transparent huge pages may be disabled.",
                std::io::Error::last_os_error()
            );
        }
    }

    // SAFETY: `ptr` is a valid mapping of `len` bytes.
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr.cast::<u8>(), len) };
    Ok((ptr as usize, slice))
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
/// change the configuration of the background task.
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

    if let Some(growth_dampener) = &config.growth_dampener {
        LGALLOC_GROWTH_DAMPENER.store(*growth_dampener, Ordering::Relaxed);
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
    /// Configuration of the background worker.
    pub background_config: Option<BackgroundWorkerConfig>,
    /// Whether to return physical memory on deallocate
    pub eager_return: Option<bool>,
    /// Dampener in the area growth rate. 0 corresponds to doubling and in general `n` to `1+1/(n+1)`.
    pub growth_dampener: Option<usize>,
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

    /// Enable eager memory reclamation.
    pub fn eager_return(&mut self, eager_return: bool) -> &mut Self {
        self.eager_return = Some(eager_return);
        self
    }

    /// Set the area growth dampener.
    pub fn growth_dampener(&mut self, growth_dampener: usize) -> &mut Self {
        self.growth_dampener = Some(growth_dampener);
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
/// should be cheap.
///
/// Note that this function takes a read lock on various structures, which can block refills
/// until the function returns.
///
/// # Panics
///
/// Panics if the internal state of lgalloc is corrupted.
pub fn lgalloc_stats() -> LgAllocStats {
    let global = GlobalStealer::get_static();

    let mut size_class_stats = Vec::with_capacity(VALID_SIZE_CLASS.len());
    for (index, state) in global.size_classes.iter().enumerate() {
        let size_class = SizeClass::from_index(index);
        let size_class_bytes = size_class.byte_size();

        size_class_stats.push((size_class_bytes, SizeClassStats::from(state)));
    }

    LgAllocStats {
        size_class: size_class_stats,
    }
}

/// Statistics about lgalloc's internal behavior.
#[derive(Debug)]
pub struct LgAllocStats {
    /// Per size-class statistics.
    pub size_class: Vec<(usize, SizeClassStats)>,
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
