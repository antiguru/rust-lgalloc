fn main() {
    let mut stats = lgalloc::LgAllocStats::default();
    let buffer_size = 32 << 20;

    let buffers = 32;

    let mut config = lgalloc::LgAlloc::new();
    config.enable();
    config.eager_return(true);
    config.with_path(std::env::temp_dir());
    lgalloc::lgalloc_set_config(&config);

    println!("Allocating {buffers} regions of {buffer_size} size...");
    let mut regions: Vec<_> = (0..32)
        .map(|_| lgalloc::allocate::<u8>(32 << 20).unwrap())
        .collect();
    print_stats(&mut stats);

    for (ptr, cap, _handle) in &regions {
        println!("Setting region at {ptr:?}...");
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), *cap) };
        for i in slice {
            *i = 1;
        }
    }
    print_stats(&mut stats);

    let mut s = String::new();
    let stdin = std::io::stdin();

    println!("Enter to continue");
    stdin.read_line(&mut s).unwrap();
    print_stats(&mut stats);

    println!("Dropping regions");
    for (_ptr, _cap, handle) in regions.drain(..) {
        lgalloc::deallocate(handle);
    }

    println!("Enter to continue");
    stdin.read_line(&mut s).unwrap();
    print_stats(&mut stats);
}

fn print_stats(stats: &mut lgalloc::LgAllocStats) {
    lgalloc::lgalloc_stats(stats);

    for file_stat in &stats.file_stats {
        println!("{file_stat:?}");
    }
}
