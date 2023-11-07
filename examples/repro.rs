use lgalloc::Region;

fn main() {
    lgalloc::lgalloc_set_config(&lgalloc::LgAlloc {
        enabled: Some(true),
        path: Some(std::env::temp_dir()),
        background_config: None,
    });
    if let Region::MMap(ref mut v, _) = Region::<i32>::new_mmap(10000).unwrap() {
        let mut mine = Vec::<i32>::new();
        std::mem::swap(v, &mut mine);
        drop(mine);
    }
}
