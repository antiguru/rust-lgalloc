//! Example that shows the disk usage for lgalloc.
fn main() {
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
    print_stats();

    for (ptr, cap, _handle) in &regions {
        println!("Setting region at {ptr:?}...");
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), *cap) };
        for i in slice {
            *i = 1;
        }
    }
    print_stats();

    let mut s = String::new();
    let stdin = std::io::stdin();

    println!("Enter to continue");
    stdin.read_line(&mut s).unwrap();
    print_stats();

    println!("Dropping regions");
    for (_ptr, _cap, handle) in regions.drain(..) {
        lgalloc::deallocate(handle);
    }

    println!("Enter to continue");
    stdin.read_line(&mut s).unwrap();
    print_stats();
}

fn print_stats() {
    let stats = lgalloc::lgalloc_stats_with_mapping().unwrap();

    for (size_class, file_stats) in &stats.file {
        match file_stats {
            Ok(file_stats) => println!("file_stats {size_class} {file_stats:?}"),
            Err(e) => eprintln!("Failed to read file stats for size class {size_class}: {e}"),
        }
    }
    for (size_class, map_stats) in stats.map.iter().flatten() {
        println!("map_stats {size_class} {map_stats:?}");
    }
}
