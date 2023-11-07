use lgalloc::Region;

fn main() {
    lgalloc::lgalloc_set_config(&lgalloc::LgAlloc {
        enabled: Some(true),
        path: Some(std::env::temp_dir()),
        background_config: None,
    });
    let r = Region::<i32>::new_mmap(10000).unwrap();

    let thread = std::thread::spawn(move || drop(r));

    thread.join().unwrap();
}
