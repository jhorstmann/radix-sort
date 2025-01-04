use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use radix_sort::{sort_slice_radix, sort_slice_radix_avx512, TotalOrder};

#[inline(never)]
pub fn standard_sort_slice<T: TotalOrder + Copy + Clone>(slice: &[T]) -> Vec<T> {
    let mut data = slice.to_vec();
    data.sort_unstable_by_key(TotalOrder::to_total_order);
    data
}

const BATCH_SIZE: usize = 1_000_000;

pub fn bench_sort(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);

    let integers = (0..BATCH_SIZE)
        .map(|_| rng.gen_range(0..u64::MAX) as i64)
        .collect::<Vec<i64>>();

    let small_integers = (0..BATCH_SIZE)
        .map(|_| rng.gen_range(0..10_000_i64))
        .collect::<Vec<i64>>();

    let floats = (0..BATCH_SIZE)
        .map(|_| f64::from_bits(rng.gen_range(0..u64::MAX)))
        // .map(|_| rng.gen_range(0..100) as f64 / 4.0)
        .collect::<Vec<f64>>();

    {
        let mut group = c.benchmark_group("integers");
        group.throughput(Throughput::Bytes((BATCH_SIZE * size_of::<i64>()) as u64));

        group
            .bench_function("standard", |b| b.iter(|| standard_sort_slice(&integers)))
            .bench_function("radix_scalar", |b| b.iter(|| sort_slice_radix(&integers)))
            .bench_function("radix_avx512", |b| b.iter(|| sort_slice_radix_avx512(&integers)));
    }

    {
        let mut group = c.benchmark_group("small_integers");
        group.throughput(Throughput::Bytes((BATCH_SIZE * size_of::<i64>()) as u64));

        group
            .bench_function("standard", |b| b.iter(|| standard_sort_slice(&small_integers)))
            .bench_function("radix_scalar", |b| b.iter(|| sort_slice_radix(&small_integers)))
            .bench_function("radix_avx512", |b| b.iter(|| sort_slice_radix_avx512(&small_integers)));
    }

    {
        let mut group = c.benchmark_group("floats");
        group.throughput(Throughput::Bytes((BATCH_SIZE * size_of::<f64>()) as u64));

        group
            .bench_function("standard", |b| b.iter(|| standard_sort_slice(&floats)))
            .bench_function("radix_scalar", |b| b.iter(|| sort_slice_radix(&floats)))
            .bench_function("radix_avx512", |b| b.iter(|| sort_slice_radix_avx512(&floats)));
    }
}

criterion_group!(benches, bench_sort);
criterion_main!(benches);
