use radix_sort::{sort_slice_radix, sort_slice_radix_avx512, TotalOrder};
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};

const BATCH_SIZE: usize = 1_000_000;

#[test]
fn integer_radix_scalar() {
    let mut rng = StdRng::seed_from_u64(42);

    let integers = (0..BATCH_SIZE)
        .map(|_| rng.gen_range(0..u64::MAX) as i64)
        .collect::<Vec<i64>>();

    let sorted = sort_slice_radix(&integers);

    assert!(sorted.is_sorted_by_key(TotalOrder::to_total_order));
}

#[test]
fn integer_radix_avx512() {
    let mut rng = StdRng::seed_from_u64(42);

    let integers = (0..BATCH_SIZE)
        .map(|_| rng.gen_range(0..u64::MAX) as i64)
        .collect::<Vec<i64>>();

    let sorted = sort_slice_radix_avx512(&integers);

    assert!(sorted.is_sorted_by_key(TotalOrder::to_total_order));
}

#[test]
fn float_radix_scalar() {
    let mut rng = StdRng::seed_from_u64(42);

    let floats = (0..BATCH_SIZE)
        .map(|_| f64::from_bits(rng.gen_range(0..u64::MAX)))
        .collect::<Vec<f64>>();

    let sorted = sort_slice_radix(&floats);

    assert!(sorted.is_sorted_by_key(TotalOrder::to_total_order));
}

#[test]
fn float_radix_avx512() {
    let mut rng = StdRng::seed_from_u64(42);

    let floats = (0..BATCH_SIZE)
        .map(|_| f64::from_bits(rng.gen_range(0..u64::MAX)) as f64)
        .collect::<Vec<f64>>();

    let sorted = sort_slice_radix_avx512(&floats);

    assert!(sorted.is_sorted_by_key(TotalOrder::to_total_order));
}
