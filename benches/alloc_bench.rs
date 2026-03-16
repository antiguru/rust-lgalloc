//! Benchmarks comparing lgalloc pooling vs direct mmap/munmap.
//!
//! Measures allocation throughput and latency across thread counts.
//!
//! Run with:
//!   cargo bench --bench alloc_bench                # throughput benchmarks
//!   cargo bench --bench alloc_bench -- --paging    # paging scenario (needs cgroup memory limit)

use std::hint::black_box;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::time::{Duration, Instant};

use hdrhistogram::Histogram as HdrHistogram;

const REGION_SIZE: usize = 2 << 20; // 2 MiB
const WARMUP: Duration = Duration::from_secs(1);
const MEASURE: Duration = Duration::from_secs(5);
const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8, 16];

/// Wrapper to make `*mut u8` sendable across threads.
/// SAFETY: The underlying memory is a valid mmap region that outlives all threads.
#[derive(Clone, Copy)]
struct SendPtr(*mut u8);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let run_paging = args.iter().any(|a| a == "--paging");

    // Initialize lgalloc once.
    lgalloc::lgalloc_set_config(
        lgalloc::LgAlloc::new()
            .enable()
            .growth_dampener(0)
            .eager_return(false)
            .with_background_config(lgalloc::BackgroundWorkerConfig {
                interval: Duration::from_millis(100),
                clear_bytes: 64 << 20,
            }),
    );

    if run_paging {
        bench_paging();
        return;
    }

    print_header();

    for &threads in THREAD_COUNTS {
        run_bench("lgalloc", threads, bench_lgalloc);
        run_bench("lgalloc+touch", threads, bench_lgalloc_touch);
        run_bench("sysalloc", threads, bench_sysalloc);
        run_bench("sysalloc+touch", threads, bench_sysalloc_touch);
        run_bench("sysalloc+nohuge", threads, bench_sysalloc_nohuge);
        run_bench("sysalloc+nohuge+touch", threads, bench_sysalloc_nohuge_touch);
        run_bench("mmap/munmap", threads, bench_mmap);
        run_bench("mmap/munmap+touch", threads, bench_mmap_touch);
        run_bench("mmap/madvise_dontneed", threads, bench_mmap_madvise_dontneed);
        run_bench("mmap/madvise_free", threads, bench_mmap_madvise_free);
        println!();
    }
}

fn print_header() {
    println!(
        "{:<30} {:>6} {:>12} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "benchmark", "thr", "ops/sec", "avg", "p50", "p99", "p99.9", "max"
    );
    println!("{}", "-".repeat(110));
}

fn format_ns(ns: u64) -> String {
    if ns < 1_000 {
        format!("{ns}ns")
    } else if ns < 1_000_000 {
        format!("{:.1}us", ns as f64 / 1_000.0)
    } else if ns < 1_000_000_000 {
        format!("{:.1}ms", ns as f64 / 1_000_000.0)
    } else {
        format!("{:.2}s", ns as f64 / 1_000_000_000.0)
    }
}

/// Thread-safe histogram backed by hdrhistogram, sharded per-thread then merged.
struct SharedHistogram {
    shards: Mutex<Vec<HdrHistogram<u64>>>,
}

impl SharedHistogram {
    fn new() -> Self {
        Self {
            shards: Mutex::new(Vec::new()),
        }
    }

    /// Create a thread-local recorder. Must be merged back via `merge`.
    fn recorder() -> HdrHistogram<u64> {
        // Range: 1ns to 60s, 3 significant digits.
        HdrHistogram::new_with_bounds(1, 60_000_000_000, 3).unwrap()
    }

    /// Merge a thread-local histogram into the shared state.
    fn merge(&self, h: HdrHistogram<u64>) {
        self.shards.lock().unwrap().push(h);
    }

    /// Combine all shards into one histogram.
    fn combined(&self) -> HdrHistogram<u64> {
        let shards = self.shards.lock().unwrap();
        let mut combined = Self::recorder();
        for s in shards.iter() {
            combined.add(s).unwrap();
        }
        combined
    }

    /// Reset all shards.
    fn reset(&self) {
        let mut shards = self.shards.lock().unwrap();
        for s in shards.iter_mut() {
            s.reset();
        }
    }
}

/// Print a summary line for a benchmark.
fn print_summary(name: &str, threads: usize, elapsed: Duration, hist: &HdrHistogram<u64>) {
    let total = hist.len();
    let ops_per_sec = total as f64 / elapsed.as_secs_f64();
    let avg_ns = hist.mean() as u64;

    println!(
        "{:<30} {:>6} {:>12.0} {:>10} {:>10} {:>10} {:>10} {:>10}",
        name,
        threads,
        ops_per_sec,
        format_ns(avg_ns),
        format_ns(hist.value_at_quantile(0.5)),
        format_ns(hist.value_at_quantile(0.99)),
        format_ns(hist.value_at_quantile(0.999)),
        format_ns(hist.max()),
    );
}

/// Print a CCDF table for a histogram.
fn print_ccdf(name: &str, hist: &HdrHistogram<u64>) {
    let total = hist.len();
    if total == 0 {
        return;
    }
    println!("\n  CCDF for {name} ({total} samples):");
    println!("  {:>12}  {:>10}  {:>10}", "latency", "P(X>x)", "count>=");

    // Walk percentiles to produce CCDF: P(X > x) = 1 - CDF(x)
    let quantiles = [
        0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 0.9999, 0.99999, 1.0,
    ];
    for &q in &quantiles {
        let v = hist.value_at_quantile(q);
        let count_above = total - hist.count_between(0, v);
        let frac_above = count_above as f64 / total as f64;
        println!(
            "  {:>12}  {:>10.6}  {:>10}",
            format_ns(v),
            frac_above,
            count_above,
        );
    }
}

/// Run a benchmark with the given thread count and print results.
fn run_bench(name: &str, threads: usize, f: fn(&AtomicBool, &mut HdrHistogram<u64>)) {
    let running = Arc::new(AtomicBool::new(true));
    let shared_hist = Arc::new(SharedHistogram::new());
    let barrier = Arc::new(Barrier::new(threads + 1));

    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let running = Arc::clone(&running);
            let shared_hist = Arc::clone(&shared_hist);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let mut hist = SharedHistogram::recorder();
                barrier.wait();
                f(&running, &mut hist);
                shared_hist.merge(hist);
            })
        })
        .collect();

    // Warmup phase
    barrier.wait();
    std::thread::sleep(WARMUP);
    shared_hist.reset();

    // Measurement phase
    let start = Instant::now();
    std::thread::sleep(MEASURE);
    running.store(false, Ordering::SeqCst);

    for h in handles {
        h.join().unwrap();
    }
    let elapsed = start.elapsed();

    let hist = shared_hist.combined();
    print_summary(name, threads, elapsed, &hist);
}

// --- Benchmark functions ---

fn bench_lgalloc(running: &AtomicBool, hist: &mut HdrHistogram<u64>) {
    while running.load(Ordering::Relaxed) {
        let start = Instant::now();
        let (_ptr, _cap, handle) = lgalloc::allocate::<u8>(REGION_SIZE).unwrap();
        black_box(&handle);
        lgalloc::deallocate(handle);
        let _ = hist.record(start.elapsed().as_nanos() as u64);
    }
}

fn bench_lgalloc_touch(running: &AtomicBool, hist: &mut HdrHistogram<u64>) {
    while running.load(Ordering::Relaxed) {
        let start = Instant::now();
        let (ptr, cap, handle) = lgalloc::allocate::<u8>(REGION_SIZE).unwrap();
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), cap) };
        for i in (0..slice.len()).step_by(4096) {
            unsafe { std::ptr::write_volatile(&mut slice[i], 1) };
        }
        black_box(slice);
        lgalloc::deallocate(handle);
        let _ = hist.record(start.elapsed().as_nanos() as u64);
    }
}

/// System allocator (glibc malloc). For 2 MiB, glibc uses mmap internally.
/// With THP=always, khugepaged may coalesce pages into huge pages in the background.
fn bench_sysalloc(running: &AtomicBool, hist: &mut HdrHistogram<u64>) {
    let layout = std::alloc::Layout::from_size_align(REGION_SIZE, 4096).unwrap();
    while running.load(Ordering::Relaxed) {
        let start = Instant::now();
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null());
        black_box(ptr);
        unsafe { std::alloc::dealloc(ptr, layout) };
        let _ = hist.record(start.elapsed().as_nanos() as u64);
    }
}

fn bench_sysalloc_touch(running: &AtomicBool, hist: &mut HdrHistogram<u64>) {
    let layout = std::alloc::Layout::from_size_align(REGION_SIZE, 4096).unwrap();
    while running.load(Ordering::Relaxed) {
        let start = Instant::now();
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null());
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, REGION_SIZE) };
        for i in (0..REGION_SIZE).step_by(4096) {
            unsafe { std::ptr::write_volatile(&mut slice[i], 1) };
        }
        black_box(slice);
        unsafe { std::alloc::dealloc(ptr, layout) };
        let _ = hist.record(start.elapsed().as_nanos() as u64);
    }
}

/// System allocator with MADV_NOHUGEPAGE to force 4K pages and prevent
/// khugepaged background coalescing.
fn bench_sysalloc_nohuge(running: &AtomicBool, hist: &mut HdrHistogram<u64>) {
    let layout = std::alloc::Layout::from_size_align(REGION_SIZE, 4096).unwrap();
    while running.load(Ordering::Relaxed) {
        let start = Instant::now();
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null());
        #[cfg(target_os = "linux")]
        unsafe {
            libc::madvise(ptr.cast(), REGION_SIZE, libc::MADV_NOHUGEPAGE);
        }
        black_box(ptr);
        unsafe { std::alloc::dealloc(ptr, layout) };
        let _ = hist.record(start.elapsed().as_nanos() as u64);
    }
}

fn bench_sysalloc_nohuge_touch(running: &AtomicBool, hist: &mut HdrHistogram<u64>) {
    let layout = std::alloc::Layout::from_size_align(REGION_SIZE, 4096).unwrap();
    while running.load(Ordering::Relaxed) {
        let start = Instant::now();
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null());
        #[cfg(target_os = "linux")]
        unsafe {
            libc::madvise(ptr.cast(), REGION_SIZE, libc::MADV_NOHUGEPAGE);
        }
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, REGION_SIZE) };
        for i in (0..REGION_SIZE).step_by(4096) {
            unsafe { std::ptr::write_volatile(&mut slice[i], 1) };
        }
        black_box(slice);
        unsafe { std::alloc::dealloc(ptr, layout) };
        let _ = hist.record(start.elapsed().as_nanos() as u64);
    }
}

fn bench_mmap(running: &AtomicBool, hist: &mut HdrHistogram<u64>) {
    while running.load(Ordering::Relaxed) {
        let start = Instant::now();
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                REGION_SIZE,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(ptr, libc::MAP_FAILED);
        black_box(ptr);
        unsafe { libc::munmap(ptr, REGION_SIZE) };
        let _ = hist.record(start.elapsed().as_nanos() as u64);
    }
}

fn bench_mmap_touch(running: &AtomicBool, hist: &mut HdrHistogram<u64>) {
    while running.load(Ordering::Relaxed) {
        let start = Instant::now();
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                REGION_SIZE,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(ptr, libc::MAP_FAILED);
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr.cast::<u8>(), REGION_SIZE) };
        for i in (0..REGION_SIZE).step_by(4096) {
            unsafe { std::ptr::write_volatile(&mut slice[i], 1) };
        }
        black_box(slice);
        unsafe { libc::munmap(ptr, REGION_SIZE) };
        let _ = hist.record(start.elapsed().as_nanos() as u64);
    }
}

fn bench_mmap_madvise_dontneed(running: &AtomicBool, hist: &mut HdrHistogram<u64>) {
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            REGION_SIZE,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    assert_ne!(ptr, libc::MAP_FAILED);

    while running.load(Ordering::Relaxed) {
        let start = Instant::now();
        unsafe { libc::madvise(ptr, REGION_SIZE, libc::MADV_DONTNEED) };
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr.cast::<u8>(), REGION_SIZE) };
        for i in (0..REGION_SIZE).step_by(4096) {
            unsafe { std::ptr::write_volatile(&mut slice[i], 1) };
        }
        black_box(slice);
        let _ = hist.record(start.elapsed().as_nanos() as u64);
    }
    unsafe { libc::munmap(ptr, REGION_SIZE) };
}

fn bench_mmap_madvise_free(running: &AtomicBool, hist: &mut HdrHistogram<u64>) {
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            REGION_SIZE,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    assert_ne!(ptr, libc::MAP_FAILED);

    while running.load(Ordering::Relaxed) {
        let start = Instant::now();
        unsafe { libc::madvise(ptr, REGION_SIZE, libc::MADV_FREE) };
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr.cast::<u8>(), REGION_SIZE) };
        for i in (0..REGION_SIZE).step_by(4096) {
            unsafe { std::ptr::write_volatile(&mut slice[i], 1) };
        }
        black_box(slice);
        let _ = hist.record(start.elapsed().as_nanos() as u64);
    }
    unsafe { libc::munmap(ptr, REGION_SIZE) };
}

// --- Paging benchmark ---
// Allocates more memory than available RAM, touches all of it, then measures
// random access latency as pages get swapped in/out.

fn bench_paging() {
    // Read cgroup memory limit to determine how much RAM we have.
    let mem_limit = read_cgroup_memory_limit().unwrap_or(512 << 20);
    // Allocate 3x the memory limit to guarantee swap pressure.
    let total_bytes = mem_limit * 3;
    let num_regions = total_bytes / REGION_SIZE;

    println!("=== Paging benchmark ===");
    println!(
        "cgroup memory limit: {} MiB, allocating: {} MiB ({} x {} MiB regions)",
        mem_limit >> 20,
        total_bytes >> 20,
        num_regions,
        REGION_SIZE >> 20,
    );

    // Phase 1: Allocate all regions via lgalloc.
    let alloc_start = Instant::now();
    let mut regions: Vec<_> = (0..num_regions)
        .map(|_| lgalloc::allocate::<u8>(REGION_SIZE).unwrap())
        .collect();
    let alloc_elapsed = alloc_start.elapsed();
    println!(
        "allocation: {} regions in {:.1}ms",
        num_regions,
        alloc_elapsed.as_secs_f64() * 1000.0
    );

    // Phase 2: Touch all pages sequentially to fault them in (and push to swap).
    let touch_start = Instant::now();
    for (ptr, cap, _handle) in &regions {
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), *cap) };
        for i in (0..slice.len()).step_by(4096) {
            unsafe { std::ptr::write_volatile(&mut slice[i], 1) };
        }
    }
    let touch_elapsed = touch_start.elapsed();
    println!(
        "sequential touch: {} MiB in {:.1}ms ({:.0} MiB/s)",
        total_bytes >> 20,
        touch_elapsed.as_secs_f64() * 1000.0,
        (total_bytes as f64 / (1 << 20) as f64) / touch_elapsed.as_secs_f64(),
    );

    // Phase 3: Random-access reads across all regions.
    println!("\n--- random page reads (bimodal: resident vs swapped) ---");
    print_header();

    for &threads in &[1, 4] {
        let running = Arc::new(AtomicBool::new(true));
        let shared_hist = Arc::new(SharedHistogram::new());
        let barrier = Arc::new(Barrier::new(threads + 1));

        // Build a shared index of all page addresses.
        let page_addrs: Arc<Vec<SendPtr>> = {
            let mut addrs = Vec::new();
            for (ptr, cap, _) in &regions {
                let base = ptr.as_ptr();
                for offset in (0..*cap).step_by(4096) {
                    addrs.push(SendPtr(unsafe { base.add(offset) }));
                }
            }
            Arc::new(addrs)
        };

        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let running = Arc::clone(&running);
                let shared_hist = Arc::clone(&shared_hist);
                let barrier = Arc::clone(&barrier);
                let page_addrs = Arc::clone(&page_addrs);
                std::thread::spawn(move || {
                    let mut hist = SharedHistogram::recorder();
                    barrier.wait();
                    let n = page_addrs.len();
                    // Large prime stride for pseudo-random traversal.
                    let stride = 104_729;
                    let mut idx = (t * 7919) % n;
                    while running.load(Ordering::Relaxed) {
                        let start = Instant::now();
                        let val = unsafe { std::ptr::read_volatile(page_addrs[idx].0) };
                        black_box(val);
                        let _ = hist.record(start.elapsed().as_nanos() as u64);
                        idx = (idx + stride) % n;
                    }
                    shared_hist.merge(hist);
                })
            })
            .collect();

        barrier.wait();
        std::thread::sleep(Duration::from_millis(500));
        shared_hist.reset();

        let start = Instant::now();
        std::thread::sleep(Duration::from_secs(10));
        running.store(false, Ordering::SeqCst);

        for h in handles {
            h.join().unwrap();
        }
        let elapsed = start.elapsed();

        let hist = shared_hist.combined();
        print_summary("random_read", threads, elapsed, &hist);
        if threads == 1 {
            print_ccdf("random_read (1 thread)", &hist);
        }
    }

    // Phase 4: Deallocate half, then alloc+touch in a loop (recycled-but-swapped regions).
    println!("\n--- deallocate half, reallocate+touch ---");
    print_header();

    let half = regions.len() / 2;
    for (_ptr, _cap, handle) in regions.drain(..half) {
        lgalloc::deallocate(handle);
    }

    let running = Arc::new(AtomicBool::new(true));
    let shared_hist = Arc::new(SharedHistogram::new());

    let r2 = Arc::clone(&running);
    let sh2 = Arc::clone(&shared_hist);
    let handle = std::thread::spawn(move || {
        let mut hist = SharedHistogram::recorder();
        while r2.load(Ordering::Relaxed) {
            let start = Instant::now();
            let (ptr, cap, handle) = lgalloc::allocate::<u8>(REGION_SIZE).unwrap();
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), cap) };
            for i in (0..slice.len()).step_by(4096) {
                unsafe { std::ptr::write_volatile(&mut slice[i], 1) };
            }
            black_box(slice);
            lgalloc::deallocate(handle);
            let _ = hist.record(start.elapsed().as_nanos() as u64);
        }
        sh2.merge(hist);
    });

    std::thread::sleep(Duration::from_millis(500));
    shared_hist.reset();

    let start = Instant::now();
    std::thread::sleep(Duration::from_secs(10));
    running.store(false, Ordering::SeqCst);
    handle.join().unwrap();

    let elapsed = start.elapsed();
    let hist = shared_hist.combined();
    print_summary("realloc+touch (swapped)", 1, elapsed, &hist);
    print_ccdf("realloc+touch (swapped, 1 thread)", &hist);

    // Cleanup
    for (_ptr, _cap, handle) in regions.drain(..) {
        lgalloc::deallocate(handle);
    }
}

/// Read cgroup v2 memory.max, falling back to cgroup v1.
fn read_cgroup_memory_limit() -> Option<usize> {
    // cgroup v2
    if let Ok(content) = std::fs::read_to_string("/sys/fs/cgroup/memory.max") {
        if let Ok(limit) = content.trim().parse::<usize>() {
            return Some(limit);
        }
    }
    // cgroup v1
    if let Ok(content) = std::fs::read_to_string("/sys/fs/cgroup/memory/memory.limit_in_bytes") {
        if let Ok(limit) = content.trim().parse::<usize>() {
            if limit < (1usize << 50) {
                return Some(limit);
            }
        }
    }
    None
}
