//! Example that demonstrates lgalloc's anonymous memory allocations with THP hints.
fn main() {
    let buffer_size = 32 << 20;

    let buffers = 32;

    let mut config = lgalloc::LgAlloc::new();
    config.enable();
    config.eager_return(true);
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
    let stats = lgalloc::lgalloc_stats();

    for (size_class, stats) in &stats.size_class {
        if stats.areas > 0 {
            println!(
                "size_class {size_class}: areas={}, total_bytes={}, free={}, clean={}, global={}, thread={}",
                stats.areas, stats.area_total_bytes, stats.free_regions, stats.clean_regions,
                stats.global_regions, stats.thread_regions,
            );
        }
    }
}
