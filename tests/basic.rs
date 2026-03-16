use std::mem::{ManuallyDrop, MaybeUninit};
use std::ptr::NonNull;
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
fn allocate_and_write() -> Result<(), AllocError> {
    initialize();
    <Wrapper<u8>>::allocate(4 << 20)?.as_slice()[0] = MaybeUninit::new(1);
    Ok(())
}

#[test]
fn cross_thread_dealloc() -> Result<(), AllocError> {
    lgalloc_set_config(&LgAlloc {
        enabled: Some(true),
        ..Default::default()
    });
    let r = <Wrapper<u8>>::allocate(1 << 20)?;

    let thread = std::thread::spawn(move || drop(r));

    thread.join().unwrap();
    Ok(())
}

#[test]
fn zst() -> Result<(), AllocError> {
    initialize();
    <Wrapper<()>>::allocate(10)?;
    Ok(())
}

#[test]
fn zero_capacity_zst() -> Result<(), AllocError> {
    initialize();
    <Wrapper<()>>::allocate(0)?;
    Ok(())
}

#[test]
fn zero_capacity_nonzst() -> Result<(), AllocError> {
    initialize();
    <Wrapper<()>>::allocate(0)?;
    Ok(())
}

#[test]
fn stats() -> Result<(), AllocError> {
    initialize();
    let (_ptr, _cap, handle) = allocate::<usize>(1 << 17)?;
    deallocate(handle);

    let stats = lgalloc_stats();

    assert!(!stats.size_class.is_empty());

    Ok(())
}
