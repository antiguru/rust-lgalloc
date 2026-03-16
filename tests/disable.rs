use lgalloc::{allocate, lgalloc_set_config, AllocError, LgAlloc};

#[test]
fn disabled_allocator_returns_error() {
    lgalloc_set_config(&*LgAlloc::new().disable());
    assert!(matches!(allocate::<u8>(1 << 20), Err(AllocError::Disabled)));
}
