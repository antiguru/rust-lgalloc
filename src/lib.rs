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

use std::cell::RefCell;
use std::collections::HashMap;
use std::mem::take;
use std::ops::{Deref, DerefMut};
use std::os::fd::{AsFd, AsRawFd};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};
use std::thread::{spawn, JoinHandle, ThreadId};
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

/// Largest supported object size class.
pub const MAX_SIZE_CLASS: usize = 32;

/// Smallest supported object size class. Corresponds to 4KiB.
pub const MIN_SIZE_CLASS: usize = 12;

/// Allocation errors
#[derive(Error, Debug)]
pub enum AllocError {
    /// IO error, unrecoverable
    #[error("I/O error")]
    Io(#[from] std::io::Error),
    /// Out of memory, meaning that the pool is exhausted.
    #[error("Out of memory")]
    OutOfMemory,
}

/// Handle to the shared global state.
static INJECTOR: OnceLock<GlobalStealer> = OnceLock::new();

/// Type maintaining the global state for each size class.
struct GlobalStealer {
    /// State for each size class. An entry at position `x` handle size class `x`, which is areas
    /// of size `1<<x`.
    size_classes: Vec<SizeClassState>,
}

/// Per-size-class state
#[derive(Default)]
struct SizeClassState {
    /// Handle to memory-mapped regions.
    areas: RwLock<Vec<MmapMut>>,
    /// Injector to distribute memory globally.
    injector: Injector<Mem>,
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
    fn get_size_class(&self, size_class: usize) -> &SizeClassState {
        &self.size_classes[size_class]
    }

    fn new() -> Self {
        let mut size_classes = Vec::with_capacity(MAX_SIZE_CLASS + 1);

        for _ in 0..=MAX_SIZE_CLASS {
            size_classes.push(SizeClassState::default());
        }

        Self { size_classes }
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
    size_class: Vec<LocalSizeClass>,
    /// The owning thread ID, used for cleaning up global state.
    thread_id: ThreadId,
}

impl ThreadLocalStealer {
    fn new() -> Self {
        let thread_id = std::thread::current().id();
        Self {
            size_class: vec![],
            thread_id,
        }
    }

    #[inline]
    fn get(&mut self, size_class: usize) -> Result<Mem, AllocError> {
        while self.size_class.len() <= size_class {
            self.size_class
                .push(LocalSizeClass::new(self.size_class.len(), self.thread_id));
        }

        self.size_class[size_class].get_with_refill()
    }

    #[inline]
    fn push(&self, mem: Mem) {
        let size_class = mem.len().next_power_of_two().trailing_zeros() as usize;
        self.size_class[size_class].push(mem);
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
    size_class: usize,
    /// Handle to global size class state
    size_class_state: &'static SizeClassState,
    /// Owning thread's ID
    thread_id: ThreadId,
}

impl LocalSizeClass {
    /// Construct a new local size class state. Registers the worker with the global state.
    fn new(size_class: usize, thread_id: ThreadId) -> Self {
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
        }
    }

    /// Get a memory area. Tries to get a region from the local cache, before obtaining data from
    /// the global state. As a last option, obtains memory from other workers.
    ///
    /// Returns [`AllcError::OutOfMemory`] if all pools are empty.
    #[inline]
    fn get(&self) -> Result<Mem, AllocError> {
        self.worker
            .pop()
            .or_else(|| {
                std::iter::repeat_with(|| {
                    self.size_class_state
                        .injector
                        .steal_batch_with_limit_and_pop(&self.worker, LOCAL_BUFFER / 2)
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
    #[inline]
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
    #[inline]
    fn push(&self, mem: Mem) {
        debug_assert_eq!(mem.len(), 1 << self.size_class);
        if self.worker.len() > LOCAL_BUFFER {
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
            || std::cmp::max(INITIAL_SIZE, 1 << self.size_class),
            |mmap| mmap.len() * 2,
        );

        let mut mmap = Self::init_file(byte_len)?;
        // SAFETY: Changing the lifetime of the mapping to static.
        let area: Mem = unsafe { std::mem::transmute(&mut *mmap) };
        let mut chunks = area.chunks_mut(1 << self.size_class);
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
        // TODO: Make the directory configurable.
        let file = tempfile::tempfile()?;
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

#[inline]
fn with_stealer<R, F: FnMut(&mut ThreadLocalStealer) -> R>(mut f: F) -> R {
    WORKER.with(|cell| f(&mut cell.borrow_mut()))
}

/// Get the system's page size.
fn page_size() -> usize {
    static PAGE_SIZE: AtomicUsize = AtomicUsize::new(0);

    let mut page_size = PAGE_SIZE.load(Ordering::Relaxed);
    if page_size == 0 {
        // SAFETY: Calling into sysconf
        page_size =
            usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).expect("Must fit");
        PAGE_SIZE.store(page_size, Ordering::Relaxed);
    }
    page_size
}

static BACKGROUND_WORKER: OnceLock<JoinHandle<()>> = OnceLock::new();

struct BackgroundWorker {
    config: BackgroundWorkerConfig,
}

impl BackgroundWorker {
    fn new(config: BackgroundWorkerConfig) -> Self {
        Self { config }
    }

    fn run(&self) {
        let global = GlobalStealer::get_static();
        let worker = Worker::new_fifo();
        loop {
            std::thread::sleep(self.config.interval);
            for size_class in &global.size_classes {
                let _ = size_class
                    .injector
                    .steal_batch_with_limit(&worker, self.config.batch);
                while let Some(mut mem) = worker.pop() {
                    Self::clear(&mut mem);
                    size_class.injector.push(mem);
                }
            }
        }
    }

    fn clear(mem: &mut Mem) -> bool {
        // From the documentation: We need a byte for every page, and the length is rounded up
        // to the full page.
        let mut bitmap: Vec<u8> = std::iter::repeat(0u8)
            .take((mem.len() + page_size() - 1) / page_size())
            .collect();

        // SAFETY: Calling into `mincore`
        let ret = unsafe {
            libc::mincore(
                mem.as_mut_ptr().cast(),
                mem.len(),
                bitmap.as_mut_ptr().cast(),
            )
        };
        if ret != 0 {
            eprintln!(
                "mincore failed: {ret} {:?}",
                std::io::Error::last_os_error()
            );
            return false;
        }

        if bitmap.iter().any(|x| *x != 0) {
            // SAFETY: Calling into `madvise`
            let ret =
                unsafe { libc::madvise(mem.as_mut_ptr().cast(), mem.len(), libc::MADV_REMOVE) };
            if ret != 0 {
                eprintln!(
                    "madvise failed: {ret} {:?}",
                    std::io::Error::last_os_error()
                );
            }
            true
        } else {
            false
        }
    }
}

pub fn background_worker(config: BackgroundWorkerConfig) -> &'static JoinHandle<()> {
    BACKGROUND_WORKER.get_or_init(|| {
        let worker = BackgroundWorker::new(config);
        spawn(move || worker.run())
    })
}

pub struct BackgroundWorkerConfig {
    pub interval: Duration,
    pub batch: usize,
}

/// An abstraction over different kinds of allocated regions.
pub enum Region<T> {
    /// An empty region, not backed by anything.
    Nil,
    /// A heap-allocated region, represented as a vector.
    Heap(Vec<T>),
    /// A mmaped region, represented by a vector and its backing memory mapping.
    MMap(Vec<T>, Option<Mem>),
}

impl<T> Default for Region<T> {
    fn default() -> Self {
        Self::new_nil()
    }
}

impl<T> Region<T> {
    const MMAP_SIZE: usize = 2 << 20;

    /// Create a new empty region.
    #[must_use]
    pub fn new_nil() -> Region<T> {
        Region::Nil
    }

    /// Create a new heap-allocated region of a specific capacity.
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
    pub fn new_mmap(capacity: usize) -> Result<Region<T>, AllocError> {
        // return Ok(Self::new_heap(capacity));
        // Round up to at least a page.
        let byte_len = std::cmp::max(0x1000, std::mem::size_of::<T>() * capacity);
        let byte_len = byte_len.next_power_of_two();
        let size_class = byte_len.trailing_zeros() as usize;
        let actual_capacity = byte_len / std::mem::size_of::<T>();
        with_stealer(|s| s.get(size_class)).map(|mem| {
            debug_assert_eq!(mem.len(), byte_len);
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
        if std::mem::size_of::<T>() == 0 {
            // Handle zero-sized types.
            Region::new_heap(capacity)
        } else {
            let bytes = std::mem::size_of::<T>() * capacity;
            match bytes {
                0 => Region::new_nil(),
                Self::MMAP_SIZE => Region::new_mmap(capacity).unwrap_or_else(|err| {
                    eprintln!("Mmap pool exhausted, falling back to heap: {err}");
                    Region::new_heap(capacity)
                }),
                _ => Region::new_heap(capacity),
            }
        }
    }

    /// Clears the contents of the region, without dropping its elements.
    ///
    /// # Safety
    ///
    /// Discards all contends. Elements are not dropped.
    pub unsafe fn clear(&mut self) {
        match self {
            Region::Nil => {}
            Region::Heap(vec) | Region::MMap(vec, _) => vec.set_len(0),
        }
    }

    /// Returns the capacity of the underlying allocation.
    #[must_use]
    pub fn capacity(&self) -> usize {
        match self {
            Region::Nil => 0,
            Region::Heap(vec) | Region::MMap(vec, _) => vec.capacity(),
        }
    }

    /// Returns the number of elements in the allocation.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Region::Nil => 0,
            Region::Heap(vec) | Region::MMap(vec, _) => vec.len(),
        }
    }

    /// Returns true if the region does not contain any elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match self {
            Region::Nil => true,
            Region::Heap(vec) | Region::MMap(vec, _) => vec.is_empty(),
        }
    }
}

impl<T, D> AsMut<D> for Region<T>
where
    <Region<T> as Deref>::Target: AsMut<D>,
{
    fn as_mut(&mut self) -> &mut D {
        self.deref_mut().as_mut()
    }
}

impl<T> Deref for Region<T> {
    type Target = Vec<T>;

    /// Obtain a vector of the allocation. Panics for [`Region::Nil`].
    fn deref(&self) -> &Self::Target {
        match self {
            Region::Nil => panic!("Cannot represent Nil region as vector"),
            Region::Heap(vec) | Region::MMap(vec, _) => vec,
        }
    }
}

impl<T> DerefMut for Region<T> {
    /// Obtain a mutable vector of the allocation. Panics for [`Region::Nil`].
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Region::Nil => panic!("Cannot represent Nil region as vector"),
            Region::Heap(vec) | Region::MMap(vec, _) => vec,
        }
    }
}

impl<T: Clone> Region<T> {
    pub fn extend_from_slice(&mut self, slice: &[T]) {
        self.deref_mut().extend_from_slice(slice);
    }
}

impl<T> Drop for Region<T> {
    #[inline]
    fn drop(&mut self) {
        match self {
            Region::Nil => {}
            Region::Heap(vec) => {
                // SAFETY: Don't drop the elements.
                unsafe { vec.set_len(0) }
            }
            Region::MMap(vec, mmap) => {
                // Forget reasoning: The vector points to the mapped region, which frees the
                // allocation
                std::mem::forget(std::mem::take(vec));
                with_stealer(|s| s.push(std::mem::take(mmap).unwrap()));
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{AllocError, BackgroundWorkerConfig, Region};
    use std::ops::DerefMut;
    use std::sync::{Arc, RwLock};
    use std::time::Duration;

    #[test]
    fn test_1() -> Result<(), AllocError> {
        let mut r: Region<u8> = Region::new_mmap(4 << 20)?;
        r.deref_mut().push(1);
        drop(r);
        Ok(())
    }

    #[test]
    fn test_2() -> Result<(), AllocError> {
        let mut r: Region<u8> = Region::new_mmap(4 << 20)?;
        r.deref_mut().push(1);
        Ok(())
    }

    #[test]
    fn test_3() -> Result<(), AllocError> {
        crate::background_worker(BackgroundWorkerConfig {
            interval: Duration::from_secs(1),
            batch: 32,
        });
        let until = Arc::new(RwLock::new(true));

        let inner = || {
            let until = Arc::clone(&until);
            move || {
                let mut i = 0;
                while *until.read().unwrap() {
                    i += 1;
                    let mut r: Region<u8> =
                        std::hint::black_box(Region::new_mmap(4 << 20)).unwrap();
                    // r.deref_mut().extend(std::iter::repeat(0).take(2 << 20));
                    r.deref_mut().push(1);
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
        let until = Arc::new(RwLock::new(true));

        let inner = || {
            let until = Arc::clone(&until);
            move || {
                let mut i = 0;
                while *until.read().unwrap() {
                    i += 64;
                    let _ = (0..64)
                        .into_iter()
                        .map(|_| {
                            let mut r: Region<u8> =
                                std::hint::black_box(Region::new_mmap(2 << 20)).unwrap();
                            // r.deref_mut().extend(std::iter::repeat(0).take(2 << 20));
                            r.deref_mut().push(1);
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
