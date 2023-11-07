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

use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::mem::{take, ManuallyDrop};
use std::ops::{Add, Deref, Range};
use std::os::fd::{AsFd, AsRawFd};
use std::path::PathBuf;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::thread::{spawn, JoinHandle, ThreadId};
use std::time::{Duration, Instant};

use crossbeam_deque::{Injector, Steal, Stealer, Worker};
use memmap2::MmapMut;
use thiserror::Error;

/// Pointer to a region of memory.
struct Mem {
    ptr: NonNull<[u8]>,
}

unsafe impl Send for Mem {}

impl Mem {
    fn len(&self) -> usize {
        self.ptr.len()
    }

    fn clear(&mut self) -> std::io::Result<()> {
        // SAFETY: Calling into `madvise`:
        // * The ptr is page-aligned by construction.
        // * The ptr + length is page-aligned by construction (not required but surprising otherwise)
        // * Mapped shared and writable (for MADV_REMOVE),
        // * Pages not locked.
        let ret = unsafe {
            libc::madvise(
                self.ptr.as_ptr().cast(),
                self.ptr.len(),
                MADV_DONTNEED_STRATEGY,
            )
        };
        if ret != 0 {
            let err = std::io::Error::last_os_error();
            eprintln!("madvise failed: {ret} {err:?}",);
            return Err(err);
        }
        Ok(())
    }
}

impl From<&mut [u8]> for Mem {
    fn from(value: &mut [u8]) -> Self {
        Self {
            ptr: NonNull::new(value).expect("Mapped memory ptr not null"),
        }
    }
}

/// The number of allocations to retain locally, per thread and size class.
const LOCAL_BUFFER: usize = 32;

// Initial file size
const INITIAL_SIZE: usize = 32 << 20;

/// Range of valid size classes.
pub const VALID_SIZE_CLASS: Range<usize> = 16..33;

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
    #[error("Memory unsuitable for requested alignment")]
    UnalignedMemory,
}

impl AllocError {
    pub fn is_disabled(&self) -> bool {
        matches!(self, AllocError::Disabled)
    }
}

/// Abstraction over size classes.
#[derive(Clone, Copy)]
struct SizeClass(usize);

impl SizeClass {
    /// Smallest supported size class
    const MIN: SizeClass = SizeClass::new_unchecked(VALID_SIZE_CLASS.start);
    /// Largest supported size class
    const MAX: SizeClass = SizeClass::new_unchecked(VALID_SIZE_CLASS.end);

    const fn new_unchecked(value: usize) -> Self {
        Self(value)
    }

    const fn index(&self) -> usize {
        self.0 - VALID_SIZE_CLASS.start
    }

    const fn byte_size(&self) -> usize {
        1 << self.0
    }

    const fn from_index(index: usize) -> Self {
        Self(index + VALID_SIZE_CLASS.start)
    }

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
}

/// Handle to the shared global state.
static INJECTOR: OnceLock<GlobalStealer> = OnceLock::new();

/// Enabled switch to turn on or off lgalloc. Off by default.
static LGALLOC_ENABLED: AtomicBool = AtomicBool::new(false);

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
    areas: RwLock<Vec<MmapMut>>,
    /// Injector to distribute memory globally.
    injector: Injector<Mem>,
    /// Injector to distribute memory globally, freed memory.
    clean_injector: Injector<Mem>,
    /// Slow-path lock to refill pool.
    lock: Mutex<()>,
    /// Thread stealers to allow all participating threads to steal memory.
    stealers: RwLock<HashMap<ThreadId, PerThreadState<Mem>>>,
    /// Summed stats for terminated threads.
    alloc_stats: AllocStats,
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
            path: Default::default(),
            background_sender: Default::default(),
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

    fn get(&mut self, size_class: SizeClass) -> Result<Mem, AllocError> {
        if !LGALLOC_ENABLED.load(Ordering::Relaxed) {
            return Err(AllocError::Disabled);
        }
        self.size_classes[size_class.index()].get_with_refill()
    }

    fn push(&self, mem: Mem) {
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
    worker: Worker<Mem>,
    /// Size class we're covering
    size_class: SizeClass,
    /// Handle to global size class state
    size_class_state: &'static SizeClassState,
    /// Owning thread's ID
    thread_id: ThreadId,
    /// Shared statistics maintained by this thread.
    stats: Arc<AllocStats>,
    _phantom: PhantomUnsyncUnsend<Self>,
}

impl LocalSizeClass {
    /// Construct a new local size class state. Registers the worker with the global state.
    fn new(size_class: SizeClass, thread_id: ThreadId) -> Self {
        let worker = Worker::new_lifo();
        let stealer = GlobalStealer::get_static();
        let size_class_state = stealer.get_size_class(size_class);

        let stats = Arc::new(AllocStats::default());

        let mut lock = size_class_state.stealers.write().unwrap();
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
    #[inline(always)]
    fn get(&self) -> Result<Mem, AllocError> {
        self.worker
            .pop()
            .or_else(|| {
                std::iter::repeat_with(|| {
                    // The loop tries to obtain memory in the following order:
                    // 1. Memory from the global state,
                    // 2. Memory from the global cleaned state,
                    // 3. Memory from other threads.

                    self.size_class_state
                        .injector
                        .steal_batch_with_limit_and_pop(&self.worker, LOCAL_BUFFER / 2)
                        .or_else(|| {
                            self.size_class_state
                                .clean_injector
                                .steal_batch_with_limit_and_pop(&self.worker, LOCAL_BUFFER / 2)
                        })
                        .or_else(|| {
                            self.size_class_state
                                .stealers
                                .read()
                                .unwrap()
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
    fn get_with_refill(&self) -> Result<Mem, AllocError> {
        self.stats.allocations.fetch_add(1, Ordering::Relaxed);
        // Fast-path: Get non-blocking
        match self.get() {
            Ok(mem) => Ok(mem),
            Err(AllocError::OutOfMemory) => {
                self.stats.slow_path.fetch_add(1, Ordering::Relaxed);
                // Get a a slow-path lock
                let _lock = self.size_class_state.lock.lock().unwrap();
                // Try again because another thread might have refilled already
                if let Ok(mem) = self.get() {
                    return Ok(mem);
                }
                self.try_refill_and_get()
            }
            e => e,
        }
    }

    /// Recycle memory. Stores it locally or forwards it to the global state.
    fn push(&self, mem: Mem) {
        debug_assert_eq!(mem.len(), self.size_class.byte_size());
        self.stats.deallocations.fetch_add(1, Ordering::Relaxed);
        if self.worker.len() >= LOCAL_BUFFER {
            self.size_class_state.injector.push(mem);
        } else {
            self.worker.push(mem);
        }
    }

    /// Refill the memory pool, and get one area.
    ///
    /// Returns an error if the memory pool cannot be refilled.
    fn try_refill_and_get(&self) -> Result<Mem, AllocError> {
        self.stats.refill.fetch_add(1, Ordering::Relaxed);
        let mut stash = self.size_class_state.areas.write().unwrap();

        let byte_len = stash.iter().last().map_or_else(
            || std::cmp::max(INITIAL_SIZE, self.size_class.byte_size()),
            |mmap| mmap.len() * 2,
        );

        let mut mmap = Self::init_file(byte_len)?;
        let mut chunks = mmap.as_mut().chunks_mut(self.size_class.byte_size());
        let mem = chunks
            .next()
            .expect("At least once chunk allocated.")
            .into();
        for slice in chunks {
            self.size_class_state.injector.push(slice.into());
        }
        stash.push(mmap);
        Ok(mem)
    }

    /// Allocate and map a file of size `byte_len`. Returns an handle, or error if the allocation
    /// fails.
    fn init_file(byte_len: usize) -> Result<MmapMut, AllocError> {
        let path = GlobalStealer::get_static().path.read().unwrap().clone();
        let Some(path) = path else {
            return Err(AllocError::Io(std::io::Error::from(
                std::io::ErrorKind::NotFound,
            )));
        };
        let file = tempfile::tempfile_in(path)?;
        // SAFETY: Calling ftruncate on the file, which we just created.
        unsafe {
            let ret = libc::ftruncate(
                file.as_fd().as_raw_fd(),
                libc::off_t::try_from(byte_len).expect("Must fit"),
            );
            if ret != 0 {
                return Err(std::io::Error::last_os_error().into());
            }
        }
        // SAFETY: We only map `file` once, and never share it with other processes.
        let mmap = unsafe { memmap2::MmapOptions::new().populate().map_mut(&file)? };
        assert_eq!(mmap.len(), byte_len);
        Ok(mmap)
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

        self.size_class_state.alloc_stats.allocations.fetch_add(
            self.stats.allocations.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        let global_stats = &self.size_class_state.alloc_stats;
        global_stats
            .refill
            .fetch_add(self.stats.refill.load(Ordering::Relaxed), Ordering::Relaxed);
        global_stats.slow_path.fetch_add(
            self.stats.slow_path.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        global_stats.deallocations.fetch_add(
            self.stats.deallocations.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }
}

fn with_stealer<R, F: FnMut(&mut ThreadLocalStealer) -> R>(mut f: F) -> R {
    WORKER.with(|cell| f(&mut cell.borrow_mut()))
}

struct BackgroundWorker {
    config: BackgroundWorkerConfig,
    receiver: Receiver<BackgroundWorkerConfig>,
    global_stealer: &'static GlobalStealer,
    worker: Worker<Mem>,
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
        let mut next_cleanup = Instant::now();
        loop {
            match self
                .receiver
                .recv_timeout(next_cleanup.saturating_duration_since(Instant::now()))
            {
                Ok(config) => self.config = config,
                Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => {
                    next_cleanup = next_cleanup.add(self.config.interval);
                    self.maintenance();
                }
            }
        }
    }

    fn maintenance(&self) {
        for size_class in &self.global_stealer.size_classes {
            let _ = self.clear(size_class, &self.worker);
        }
    }

    fn clear(&self, size_class: &SizeClassState, worker: &Worker<Mem>) -> usize {
        let _ = size_class
            .injector
            .steal_batch_with_limit(worker, self.config.batch);
        let mut count = 0;
        while let Some(mut mem) = worker.pop() {
            match mem.clear() {
                Ok(()) => count += 1,
                Err(e) => panic!("Syscall failed: {e:?}"),
            }
            size_class.clean_injector.push(mem);
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
pub fn lgalloc_set_config(config: &LgAlloc) {
    let stealer = GlobalStealer::get_static();

    if let Some(enabled) = &config.enabled {
        LGALLOC_ENABLED.store(*enabled, Ordering::Relaxed);
    }

    if let Some(path) = &config.path {
        *stealer.path.write().unwrap() = Some(path.clone());
    }

    if let Some(config) = config.background_config.clone() {
        let mut lock = stealer.background_sender.lock().unwrap();

        match &*lock {
            Some((_, sender)) => sender.send(config).expect("Receiver exists"),
            None => {
                let (sender, receiver) = std::sync::mpsc::channel();
                let mut worker = BackgroundWorker::new(receiver);
                let join_handle = spawn(move || worker.run());
                sender.send(config).expect("Receiver exists");
                *lock = Some((join_handle, sender));
            }
        }
    }
}

/// Configuration for lgalloc's background worker.
#[derive(Default, Clone, Eq, PartialEq)]
pub struct BackgroundWorkerConfig {
    /// How frequently it should tick
    pub interval: Duration,
    /// How many allocations to clear per size class.
    pub batch: usize,
    /// Enable debug stat printing.
    pub print_stats: bool,
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
}

impl LgAlloc {
    /// Construct a new configuration. All values are initialized to their default (None) values.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enable(&mut self) -> &mut Self {
        self.enabled = Some(true);
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
}

/// Determine global statistics per size class
///
/// Note that this function take a read lock on various structures.
pub fn lgalloc_stats(stats: &mut LgAllocStats) {
    stats.size_class.clear();

    let global = GlobalStealer::get_static();

    for (index, size_class_state) in global.size_classes.iter().enumerate() {
        let size_class = SizeClass::from_index(index);

        let areas_lock = size_class_state.areas.read().unwrap();

        let areas = areas_lock.len();
        if areas == 0 {
            continue;
        }

        let size_class = size_class.byte_size();
        let area_total_bytes = areas_lock.iter().map(|area| area.len()).sum();
        let global_regions = size_class_state.injector.len();
        let clean_regions = size_class_state.clean_injector.len();
        let stealers = size_class_state.stealers.read().unwrap();
        let mut thread_regions = 0;
        let mut allocations = 0;
        let mut deallocations = 0;
        let mut refill = 0;
        let mut slow_path = 0;
        for thread_state in stealers.values() {
            thread_regions += thread_state.stealer.len();
            let thread_stats = &*thread_state.alloc_stats;
            allocations += thread_stats.allocations.load(Ordering::Relaxed);
            deallocations += thread_stats.deallocations.load(Ordering::Relaxed);
            refill += thread_stats.refill.load(Ordering::Relaxed);
            slow_path += thread_stats.slow_path.load(Ordering::Relaxed);
        }

        let free_regions = thread_regions + global_regions + clean_regions;

        let global_stats = &size_class_state.alloc_stats;
        allocations += global_stats.allocations.load(Ordering::Relaxed);
        deallocations += global_stats.deallocations.load(Ordering::Relaxed);
        refill += global_stats.refill.load(Ordering::Relaxed);
        slow_path += global_stats.slow_path.load(Ordering::Relaxed);

        stats.size_class.push(SizeClassStats {
            size_class,
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
        });
    }
}

#[derive(Debug, Default)]
pub struct LgAllocStats {
    pub size_class: Vec<SizeClassStats>,
}

#[derive(Debug)]
pub struct SizeClassStats {
    /// Size class in bytes
    pub size_class: usize,
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
}

/// An abstraction over different kinds of allocated regions.
pub enum Region<T> {
    /// A possibly empty heap-allocated region, represented as a vector.
    Heap(Vec<T>),
    /// A mmaped region, represented by a vector and its backing memory mapping.
    MMap(MMapRegion<T>),
}

pub struct MMapRegion<T> {
    inner: ManuallyDrop<Vec<T>>,
    mem: Option<Mem>,
}

impl<T> MMapRegion<T> {
    unsafe fn clear(&mut self) {
        self.inner.set_len(0);
    }
}

impl<T> Deref for MMapRegion<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> Default for Region<T> {
    #[inline]
    fn default() -> Self {
        Self::new_empty()
    }
}

impl<T> Region<T> {
    /// Create a new empty region.
    #[inline]
    #[must_use]
    pub fn new_empty() -> Region<T> {
        Region::Heap(Vec::new())
    }

    /// Create a new heap-allocated region of a specific capacity.
    #[inline]
    #[must_use]
    pub fn new_heap(capacity: usize) -> Region<T> {
        Region::Heap(Vec::with_capacity(capacity))
    }

    /// Create a new file-based mapped region of a specific capacity. The capacity of the
    /// returned region can be larger than requested to accommodate page sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if the memory allocation fails.
    #[inline(always)]
    pub fn new_mmap(capacity: usize) -> Result<Region<T>, AllocError> {
        if std::mem::size_of::<T>() == 0 || capacity == 0 {
            // Handle zero-sized types.
            return Ok(Region::new_heap(capacity));
        }

        // Round up to at least a page.
        // TODO: This assumes 4k pages.
        let byte_len = std::cmp::max(0x1000, std::mem::size_of::<T>() * capacity);
        let size_class = SizeClass::from_byte_size(byte_len)?;

        with_stealer(|s| s.get(size_class)).and_then(|mem| {
            debug_assert_eq!(mem.len(), size_class.byte_size());
            let actual_capacity = mem.len() / std::mem::size_of::<T>();
            let ptr: *mut T = mem.ptr.as_ptr().cast();
            // Memory region should be page-aligned, which we assume to be larger than any alignment
            // we might encounter. If this is not the case, bail out.
            if ptr.align_offset(std::mem::align_of::<T>()) != 0 {
                return Err(AllocError::UnalignedMemory);
            }
            // SAFETY: memory points to suitable memory.
            let new_local = unsafe { Vec::from_raw_parts(ptr, 0, actual_capacity) };
            debug_assert!(std::mem::size_of::<T>() * new_local.len() <= mem.len());
            Ok(Region::MMap(MMapRegion {
                inner: ManuallyDrop::new(new_local),
                mem: Some(mem),
            }))
        })
    }

    /// Create a region depending on the capacity.
    ///
    /// The capacity of the returned region must be at least as large as the requested capacity,
    /// but can be larger if the implementation requires it.
    ///
    /// Crates a [`Region::Nil`] for empty capacities, a [`Region::Heap`] for allocations up to 2
    /// Mib, and [`Region::MMap`] for larger capacities.
    #[must_use]
    pub fn new_auto(capacity: usize) -> Region<T> {
        if std::mem::size_of::<T>() == 0 || capacity == 0 {
            // Handle zero-sized types.
            return Region::new_heap(capacity);
        }
        let bytes = std::mem::size_of::<T>() * capacity;
        if bytes < SizeClass::MIN.byte_size() || bytes > SizeClass::MAX.byte_size() {
            Region::new_heap(capacity)
        } else {
            Region::new_mmap(capacity).unwrap_or_else(|err| {
                if !err.is_disabled() {
                    eprintln!("Mmap pool exhausted, falling back to heap: {err}");
                }
                Region::new_heap(capacity)
            })
        }
    }

    /// Clears the contents of the region, without dropping its elements.
    ///
    /// # Safety
    ///
    /// Discards all contends. Elements are not dropped.
    #[inline]
    pub unsafe fn clear(&mut self) {
        match self {
            Region::Heap(vec) => vec.set_len(0),
            Region::MMap(inner) => inner.clear(),
        }
    }

    /// Returns the capacity of the underlying allocation.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        match self {
            Region::Heap(vec) => vec.capacity(),
            Region::MMap(inner) => inner.inner.capacity(),
        }
    }

    /// Returns the number of elements in the allocation.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Region::Heap(vec) => vec.len(),
            Region::MMap(inner) => inner.len(),
        }
    }

    /// Returns true if the region does not contain any elements.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match self {
            Region::Heap(vec) => vec.is_empty(),
            Region::MMap(inner) => inner.is_empty(),
        }
    }

    /// Dereference to the contained vector
    #[inline]
    #[must_use]
    pub fn as_vec(&self) -> &Vec<T> {
        match self {
            Region::Heap(vec) => vec,
            Region::MMap(inner) => &inner.inner,
        }
    }
}

impl<T> AsMut<Vec<T>> for Region<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut Vec<T> {
        match self {
            Region::Heap(vec) => vec,
            Region::MMap(inner) => &mut inner.inner,
        }
    }
}

impl<T: Clone> Region<T> {
    #[inline]
    pub fn extend_from_slice(&mut self, slice: &[T]) {
        self.as_mut().extend_from_slice(slice);
    }
}

impl<T> Extend<T> for Region<T> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.as_mut().extend(iter);
    }
}

impl<T> std::ops::Deref for Region<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_vec()
    }
}

impl<T> std::ops::DerefMut for Region<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}

impl<T> Drop for Region<T> {
    #[inline]
    fn drop(&mut self) {
        match self {
            Region::Heap(vec) => {
                // SAFETY: Don't drop the elements, drop the vec.
                unsafe { vec.set_len(0) }
            }
            Region::MMap(_) => {}
        }
    }
}

impl<T> Drop for MMapRegion<T> {
    fn drop(&mut self) {
        // Forget reasoning: The vector points to the mapped region, which frees the
        // allocation. Don't drop elements, don't drop vec.
        with_stealer(|s| s.push(take(&mut self.mem).unwrap()));
    }
}

#[cfg(test)]
mod test {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use serial_test::serial;

    use crate::{AllocError, BackgroundWorkerConfig, LgAlloc, Region};

    fn initialize() {
        crate::lgalloc_set_config(
            LgAlloc::new()
                .enable()
                .with_background_config(BackgroundWorkerConfig {
                    interval: Duration::from_secs(1),
                    batch: 32,
                    print_stats: false,
                })
                .with_path(std::env::temp_dir()),
        );
    }

    #[test]
    #[serial]
    fn test_1() -> Result<(), AllocError> {
        initialize();
        let mut r: Region<u8> = Region::new_auto(4 << 20);
        r.as_mut().push(1);
        drop(r);
        Ok(())
    }

    #[test]
    #[serial]
    fn test_2() -> Result<(), AllocError> {
        initialize();
        let mut r: Region<u8> = Region::new_auto(4 << 20);
        r.as_mut().push(1);
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
                    let mut r: Region<u8> = std::hint::black_box(Region::new_auto(4 << 20));
                    // r.as_mut().extend(std::iter::repeat(0).take(2 << 20));
                    r.as_mut().push(1);
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
                while until.load(Ordering::Relaxed) {
                    i += 64;
                    let _ = (0..64)
                        .map(|_| {
                            let mut r: Region<u8> = std::hint::black_box(Region::new_auto(2 << 20));
                            // r.as_mut().extend(std::iter::repeat(0).take(2 << 20));
                            r.as_mut().push(1);
                            r
                        })
                        .collect::<Vec<_>>();
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
        let mut stats = Default::default();
        crate::lgalloc_stats(&mut stats);
        println!("stats: {stats:?}");
        Ok(())
    }

    #[test]
    #[serial]
    fn leak() {
        crate::lgalloc_set_config(&crate::LgAlloc {
            enabled: Some(true),
            path: Some(std::env::temp_dir()),
            background_config: None,
        });
        let r = Region::<i32>::new_mmap(10000).unwrap();

        let thread = std::thread::spawn(move || drop(r));

        thread.join().unwrap();
    }
}
