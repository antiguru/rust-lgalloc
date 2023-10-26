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

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::mem::take;
use std::ops::Range;
use std::os::fd::{AsFd, AsRawFd};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{Mutex, MutexGuard, OnceLock, RwLock};
use std::thread::{spawn, ThreadId};
use std::time::Duration;

use crossbeam_deque::{Injector, Steal, Stealer, Worker};
use memmap2::MmapMut;
use thiserror::Error;

/// Pointer to a region of memory.
type Mem = &'static mut [u8];

/// The number of allocations to retain locally, per thread and size class.
const LOCAL_BUFFER: usize = 32;

// Initial file size
const INITIAL_SIZE: usize = 32 << 20;

/// Range of valid size classes.
pub const VALID_SIZE_CLASS: Range<usize> = 16..33;

type PhantomUnsync = PhantomData<Cell<()>>;
type PhantomUnsend = PhantomData<MutexGuard<'static, ()>>;

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
    InvalidSizeClass,
}

#[derive(Clone, Copy)]
struct SizeClass(usize);

impl SizeClass {
    fn new_unchecked(value: usize) -> Self {
        Self(value)
    }

    fn index(&self) -> usize {
        self.0 - VALID_SIZE_CLASS.start
    }

    fn byte_size(&self) -> usize {
        1 << self.0
    }

    fn from_index(index: usize) -> Self {
        Self(index + VALID_SIZE_CLASS.start)
    }

    fn from_byte_size(byte_size: usize) -> Result<Self, AllocError> {
        let class = byte_size.next_power_of_two().trailing_zeros() as usize;
        class.try_into()
    }

    fn from_byte_size_unchecked(byte_size: usize) -> Self {
        Self::new_unchecked(byte_size.next_power_of_two().trailing_zeros() as usize)
    }
}

impl TryFrom<usize> for SizeClass {
    type Error = AllocError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        if VALID_SIZE_CLASS.contains(&value) {
            Ok(SizeClass(value))
        } else {
            Err(AllocError::InvalidSizeClass)
        }
    }
}

/// Handle to the shared global state.
static INJECTOR: OnceLock<GlobalStealer> = OnceLock::new();

/// Type maintaining the global state for each size class.
struct GlobalStealer {
    /// State for each size class. An entry at position `x` handle size class `x`, which is areas
    /// of size `1<<x`.
    size_classes: Vec<SizeClassState>,
    /// Path to store files
    path: RwLock<Option<PathBuf>>,
    /// Shared token to access background thread.
    background_sender: Mutex<Option<std::sync::mpsc::Sender<BackgroundWorkerConfig>>>,
}

/// Per-size-class state
#[derive(Default)]
struct SizeClassState {
    /// Handle to memory-mapped regions.
    areas: RwLock<Vec<MmapMut>>,
    /// Injector to distribute memory globally.
    injector: Injector<Mem>,
    /// Injector to distribute memory globally, freed memory.
    clean_injector: Injector<Mem>,
    /// Slow-path lock to refill pool.
    lock: Mutex<()>,
    /// Thread stealers to allow all participating threads to steal memory.
    stealers: RwLock<HashMap<ThreadId, Stealer<Mem>>>,
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

    /// Print diagnostics about the state of the allocator.
    #[allow(unused)]
    fn diagnostics(&self) {
        for (index, size_class_state) in self.size_classes.iter().enumerate() {
            let size_class = SizeClass::from_index(index);
            let areas = size_class_state.areas.read().unwrap();
            if !areas.is_empty() {
                let total: usize = areas.iter().map(|area| area.len()).sum();
                let injector_len = size_class_state.injector.len();
                let clean_len = size_class_state.clean_injector.len();
                let mut free_areas = injector_len + clean_len;
                eprintln!(
                    "Size class {index} ({}) areas: {} global: {injector_len} chunks -clean: {clean_len} chunks",
                    size_class.byte_size(),
                    areas.len()
                );
                let stealers = size_class_state.stealers.read().unwrap();
                for (thread, stealer) in &*stealers {
                    let stealer_len = stealer.len();
                    if stealer_len == 0 {
                        continue;
                    }
                    free_areas += stealer_len;
                    eprintln!("  {thread:?} {stealer_len}");
                }
                eprintln!(
                    "  free info: {}/{} bytes, {free_areas}/{} chunks",
                    free_areas * size_class.byte_size() / 1024,
                    total / 1024,
                    total / size_class.byte_size(),
                );
            }
        }
    }
}

impl Drop for GlobalStealer {
    fn drop(&mut self) {
        take(&mut self.size_classes);
    }
}

/// Per-thread and state, sharded by size class.
struct ThreadLocalStealer {
    /// Per-size-class state
    size_classes: Vec<LocalSizeClass>,
    _phantom: (PhantomUnsend, PhantomUnsync),
}

impl ThreadLocalStealer {
    fn new() -> Self {
        let thread_id = std::thread::current().id();
        let size_classes = VALID_SIZE_CLASS
            .map(|size_class| LocalSizeClass::new(SizeClass::new_unchecked(size_class), thread_id))
            .collect();
        Self {
            size_classes,
            _phantom: (PhantomData, PhantomData),
        }
    }

    fn get(&mut self, size_class: SizeClass) -> Result<Mem, AllocError> {
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
struct LocalSizeClass {
    /// Local memory queue.
    worker: Worker<Mem>,
    /// Size class we're covering
    size_class: SizeClass,
    /// Handle to global size class state
    size_class_state: &'static SizeClassState,
    /// Owning thread's ID
    thread_id: ThreadId,
    _phantom: (PhantomUnsend, PhantomUnsync),
}

impl LocalSizeClass {
    /// Construct a new local size class state. Registers the worker with the global state.
    fn new(size_class: SizeClass, thread_id: ThreadId) -> Self {
        let worker = Worker::new_lifo();
        let stealer = GlobalStealer::get_static();
        let size_class_state = stealer.get_size_class(size_class);

        let mut lock = size_class_state.stealers.write().unwrap();
        lock.insert(thread_id, worker.stealer());

        Self {
            worker,
            size_class,
            size_class_state,
            thread_id,
            _phantom: (PhantomData, PhantomData),
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
                                .map(Stealer::steal)
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
        // Fast-path: Get non-blocking
        match self.get() {
            Ok(mem) => Ok(mem),
            Err(AllocError::OutOfMemory) => {
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
        let mut stash = self.size_class_state.areas.write().unwrap();

        let byte_len = stash.iter().last().map_or_else(
            || std::cmp::max(INITIAL_SIZE, self.size_class.byte_size()),
            |mmap| mmap.len() * 2,
        );

        let mut mmap = Self::init_file(byte_len)?;
        // SAFETY: Changing the lifetime of the mapping to static.
        let area: Mem = unsafe { std::mem::transmute(&mut *mmap) };
        let mut chunks = area.chunks_mut(self.size_class.byte_size());
        let mem = chunks.next().expect("At least once chunk allocated.");
        for slice in chunks {
            self.size_class_state.injector.push(slice);
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
        // SAFETY: Calling ftruncate on the file.
        unsafe {
            match libc::ftruncate(
                file.as_fd().as_raw_fd(),
                libc::off_t::try_from(byte_len).expect("Must fit"),
            ) {
                0 => Ok(memmap2::MmapOptions::new().populate().map_mut(&file)?),
                _ => Err(std::io::Error::last_os_error().into()),
            }
        }
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
    }
}

fn with_stealer<R, F: FnMut(&mut ThreadLocalStealer) -> R>(mut f: F) -> R {
    WORKER.with(|cell| f(&mut cell.borrow_mut()))
}

struct BackgroundWorker {
    config: BackgroundWorkerConfig,
    receiver: Receiver<BackgroundWorkerConfig>,
}

impl BackgroundWorker {
    fn new(config: BackgroundWorkerConfig, receiver: Receiver<BackgroundWorkerConfig>) -> Self {
        Self { config, receiver }
    }

    fn run(&mut self) {
        let global = GlobalStealer::get_static();

        let worker = Worker::new_fifo();
        let mut diagnostics = 0;
        loop {
            match self.receiver.try_recv() {
                Ok(config) => self.config = config,
                Err(TryRecvError::Disconnected) => break,
                Err(TryRecvError::Empty) => {}
            }
            for size_class in &global.size_classes {
                let _ = self.clear(size_class, &worker);
            }
            if diagnostics >= 10 {
                if self.config.print_stats {
                    global.diagnostics();
                }
                diagnostics = 0;
            }
            diagnostics += 1;
            std::thread::sleep(self.config.interval);
        }
    }

    fn clear(&self, size_class: &SizeClassState, worker: &Worker<Mem>) -> usize {
        let _ = size_class
            .injector
            .steal_batch_with_limit(worker, self.config.batch);
        let mut count = 0;
        while let Some(mut mem) = worker.pop() {
            match Self::clear_area(&mut mem) {
                Ok(()) => count += 1,
                Err(e) => panic!("Syscall failed: {e:?}"),
            }
            size_class.clean_injector.push(mem);
        }
        count
    }

    fn clear_area(mem: &mut Mem) -> std::io::Result<()> {
        // SAFETY: Calling into `madvise`
        let ret = unsafe { libc::madvise(mem.as_mut_ptr().cast(), mem.len(), libc::MADV_REMOVE) };
        if ret != 0 {
            let err = std::io::Error::last_os_error();
            eprintln!("madvise failed: {ret} {err:?}",);
            return Err(err);
        }
        Ok(())
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

    if let Some(path) = &config.path {
        *stealer.path.write().unwrap() = Some(path.clone());
    }

    if let Some(config) = &config.background_config {
        let (sender, receiver) = std::sync::mpsc::channel();
        *stealer.background_sender.lock().unwrap() = Some(sender);
        let mut worker = BackgroundWorker::new(config.clone(), receiver);
        spawn(move || worker.run());
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

/// An abstraction over different kinds of allocated regions.
pub enum Region<T> {
    /// A possibly empty heap-allocated region, represented as a vector.
    Heap(Vec<T>),
    /// A mmaped region, represented by a vector and its backing memory mapping.
    MMap(Vec<T>, Option<Mem>),
}

impl<T> Default for Region<T> {
    #[inline]
    fn default() -> Self {
        Self::new_empty()
    }
}

impl<T> Region<T> {
    const MIN_MMAP_SIZE: usize = 1 << 16;

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
        // return Ok(Self::new_heap(capacity));
        // Round up to at least a page.
        let byte_len = std::cmp::max(0x1000, std::mem::size_of::<T>() * capacity);
        let size_class = SizeClass::from_byte_size(byte_len)?;
        with_stealer(|s| s.get(size_class)).map(|mem| {
            debug_assert_eq!(mem.len(), size_class.byte_size());
            let actual_capacity = mem.len() / std::mem::size_of::<T>();
            let ptr = mem.as_mut_ptr().cast();
            // SAFETY: memory points to suitable memory.
            let new_local = unsafe { Vec::from_raw_parts(ptr, 0, actual_capacity) };
            debug_assert!(std::mem::size_of::<T>() * new_local.len() <= mem.len());
            Region::MMap(new_local, Some(mem))
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
        if bytes < Self::MIN_MMAP_SIZE {
            Region::new_heap(capacity)
        } else {
            Region::new_mmap(capacity).unwrap_or_else(|err| {
                eprintln!("Mmap pool exhausted, falling back to heap: {err}");
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
            Region::Heap(vec) | Region::MMap(vec, _) => vec.set_len(0),
        }
    }

    /// Returns the capacity of the underlying allocation.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        match self {
            Region::Heap(vec) | Region::MMap(vec, _) => vec.capacity(),
        }
    }

    /// Returns the number of elements in the allocation.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Region::Heap(vec) | Region::MMap(vec, _) => vec.len(),
        }
    }

    /// Returns true if the region does not contain any elements.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match self {
            Region::Heap(vec) | Region::MMap(vec, _) => vec.is_empty(),
        }
    }

    /// Dereference to the contained vector
    #[inline]
    #[must_use]
    pub fn as_vec(&self) -> &Vec<T> {
        match self {
            Region::Heap(vec) | Region::MMap(vec, _) => vec,
        }
    }
}

impl<T> AsMut<Vec<T>> for Region<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut Vec<T> {
        match self {
            Region::Heap(vec) | Region::MMap(vec, _) => vec,
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
            Region::MMap(vec, mmap) => {
                // Forget reasoning: The vector points to the mapped region, which frees the
                // allocation. Don't drop elements, don't drop vec.
                std::mem::forget(std::mem::take(vec));
                with_stealer(|s| s.push(std::mem::take(mmap).unwrap()));
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{AllocError, BackgroundWorkerConfig, LgAlloc, Region};
    use std::sync::{Arc, OnceLock, RwLock};
    use std::time::Duration;

    static INIT: OnceLock<()> = OnceLock::new();

    fn initialize() {
        INIT.get_or_init(|| {
            crate::lgalloc_set_config(
                LgAlloc::new()
                    .with_background_config(BackgroundWorkerConfig {
                        interval: Duration::from_secs(1),
                        batch: 32,
                        print_stats: false,
                    })
                    .with_path(std::env::temp_dir()),
            );
        });
    }

    #[test]
    fn test_1() -> Result<(), AllocError> {
        initialize();
        let mut r: Region<u8> = Region::new_mmap(4 << 20)?;
        r.as_mut().push(1);
        drop(r);
        Ok(())
    }

    #[test]
    fn test_2() -> Result<(), AllocError> {
        initialize();
        let mut r: Region<u8> = Region::new_mmap(4 << 20)?;
        r.as_mut().push(1);
        Ok(())
    }

    #[test]
    fn test_3() -> Result<(), AllocError> {
        initialize();
        let until = Arc::new(RwLock::new(true));

        let inner = || {
            let until = Arc::clone(&until);
            move || {
                let mut i = 0;
                while *until.read().unwrap() {
                    i += 1;
                    let mut r: Region<u8> =
                        std::hint::black_box(Region::new_mmap(4 << 20)).unwrap();
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
        *until.write().unwrap() = false;
        for handle in handles {
            handle.join().unwrap();
        }
        // std::thread::sleep(Duration::from_secs(600));
        Ok(())
    }

    #[test]
    fn test_4() -> Result<(), AllocError> {
        initialize();
        let until = Arc::new(RwLock::new(true));

        let inner = || {
            let until = Arc::clone(&until);
            move || {
                let mut i = 0;
                while *until.read().unwrap() {
                    i += 64;
                    let _ = (0..64)
                        .map(|_| {
                            let mut r: Region<u8> =
                                std::hint::black_box(Region::new_mmap(2 << 20)).unwrap();
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
        *until.write().unwrap() = false;
        for handle in handles {
            handle.join().unwrap();
        }
        // std::thread::sleep(Duration::from_secs(600));
        Ok(())
    }
}
