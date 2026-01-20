//! Tracing overhead benchmarks.
//!
//! Measures the cost of tracing instrumentation on hot paths.
//! Run with and without `tracing-integration` feature to compare.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use asupersync::runtime::RuntimeState;
use asupersync::types::Budget;

fn bench_region_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tracing_overhead");

    group.bench_function("create_root_region", |b| {
        b.iter(|| {
            let mut state = RuntimeState::new();
            // This triggers RegionRecord::new which has the span creation
            black_box(state.create_root_region(Budget::INFINITE))
        })
    });

    group.finish();
}

fn bench_task_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tracing_overhead");

    group.bench_function("create_task", |b| {
        b.iter(|| {
            let mut state = RuntimeState::new();
            let region = state.create_root_region(Budget::INFINITE);
            // This triggers Scope::spawn which has tracing
            // But we can't easily use Scope here without Cx.
            // RuntimeState::create_task also has tracing (debug!).
            black_box(state.create_task(region, Budget::INFINITE, async { 42 }))
        })
    });

    group.finish();
}

criterion_group!(benches, bench_region_creation, bench_task_creation);
criterion_main!(benches);
