use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lgalloc::Region;
use std::sync::OnceLock;
use std::time::Duration;

static INIT: OnceLock<()> = OnceLock::new();

fn initialize() {
    INIT.get_or_init(|| {
        lgalloc::lgalloc_set_config(
            lgalloc::LgAlloc::new()
                .enable()
                .with_background_config(lgalloc::BackgroundWorkerConfig {
                    interval: Duration::from_secs(1),
                    batch: 32,
                    print_stats: false,
                })
                .with_path(std::env::temp_dir()),
        );
    });
}

fn allocate(size: usize, count: usize, storage: &mut Vec<Region<usize>>) {
    for _ in 0..count {
        storage.push(Region::new_auto(size));
    }
    storage.clear();
}

fn bench_1(c: &mut Criterion) {
    initialize();
    let mut storage = Vec::with_capacity(1);
    for size_class in [1, 2, 4, 8, 16] {
        c.bench_function(&format!("allocate 1, {size_class} << 20"), |b| {
            b.iter(|| allocate(black_box(size_class << 20), 1, &mut storage));
            storage.clear();
        });
    }
}

fn bench_many(c: &mut Criterion) {
    initialize();
    let mut group = c.benchmark_group("allocate");
    for count in [8, 16, 24, 32, 48, 64, 96, 128] {
        let mut storage = Vec::with_capacity(count);
        for size_class in [2, 16] {
            group.throughput(Throughput::Elements(count as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("size_class {size_class} count {count}")),
                &(count, size_class),
                |b, &(count, size_class)| {
                    b.iter(|| allocate(black_box(size_class << 20), count, &mut storage));
                    storage.clear();
                },
            );
        }
    }
}

criterion_group!(benches, bench_1, bench_many);
criterion_main!(benches);
