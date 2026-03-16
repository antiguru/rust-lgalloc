use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use lgalloc::{
    allocate, deallocate, lgalloc_set_config, lgalloc_stats, AllocError, BackgroundWorkerConfig,
    Handle, LgAlloc,
};

fn initialize() {
    lgalloc_set_config(
        LgAlloc::new()
            .enable()
            .with_background_config(BackgroundWorkerConfig {
                interval: Duration::from_secs(1),
                clear_bytes: 4 << 20,
            })
            .growth_dampener(1),
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
fn concurrent_single_alloc() -> Result<(), AllocError> {
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
    Ok(())
}

#[test]
fn concurrent_batch_alloc() -> Result<(), AllocError> {
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
    Ok(())
}
