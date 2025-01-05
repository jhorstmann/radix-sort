// LSB radix sort implementation, based on
// https://crates.io/crates/radsort
// http://codercorner.com/RadixSortRevisited.htm
// http://stereopsis.com/radix.html
// See https://travisdowns.github.io/blog/2019/05/22/sorting.html

use crate::{TotalOrder, TotalOrderAvx512, HIST_PER_U64, RADIX_BITS, RADIX_HIST_LEN};
use std::arch::x86_64::*;

/// Fill the histogram for a digit indicated by the given `shift` amount.
///
/// Returns true when all values belong to the same bucket.
#[inline(never)]
fn fill_histogram<T: TotalOrder>(values: &[T], shift: usize, histogram: &mut [u32; RADIX_HIST_LEN]) -> bool {
    histogram.fill(0);

    let mut last_bucket = 0;
    values.iter().for_each(|v| {
        let bucket = (v.to_total_order() >> shift) & (RADIX_HIST_LEN as u64 - 1);
        histogram[bucket as usize] += 1;
        last_bucket = bucket;
    });

    // if every item is in the same bucket then no sorting is necessary
    histogram[last_bucket as usize] as usize == values.len()
}

/// Fill multiple histograms, one for each digit of the input `values`.
///
/// Returns an array indicating whether the values for a digit all belong to the same bucket.
#[inline(never)]
fn fill_histogram_multipass<T: TotalOrder>(
    values: &[T],
    histogram: &mut [[u32; RADIX_HIST_LEN]; HIST_PER_U64],
) -> [bool; HIST_PER_U64] {
    for h in histogram.iter_mut() {
        h.fill(0);
    }

    // should be const but that gives error that outer generic parameter can not be referenced
    let digits: usize = (size_of::<T>() * 8).div_ceil(RADIX_BITS);

    let mut last_bucket = [0; HIST_PER_U64];
    values.iter().for_each(|v| {
        let ord = v.to_total_order();
        for i in 0..digits {
            let shift = i * RADIX_BITS;
            let bucket = (ord >> shift) & (RADIX_HIST_LEN as u64 - 1);
            histogram[i][bucket as usize] += 1;
            last_bucket[i] = bucket;
        }
    });

    let mut already_sorted = [false; HIST_PER_U64];
    for i in 0..HIST_PER_U64 {
        // mask again to avoid bounds check
        let bucket = last_bucket[i] as usize & (RADIX_HIST_LEN - 1);
        // if every item is in the same bucket then no sorting is necessary
        already_sorted[i] = histogram[i][bucket] as usize == values.len();
    }
    already_sorted
}

#[inline(always)]
unsafe fn total_order_buckets<const MASKED: bool, T: TotalOrderAvx512>(
    values: &[T],
    shift: usize,
    i: usize,
) -> (__m512i, __m256i, __mmask8) {
    debug_assert!(i < values.len());

    let (v, mask) = if MASKED {
        debug_assert!(values.len() - i < 8);
        let mask = (1 << (values.len() - i)) - 1;
        (_mm512_maskz_loadu_epi64(mask, values.as_ptr().add(i) as *const _), mask)
    } else {
        debug_assert!(i + 8 <= values.len());
        (_mm512_loadu_si512(values.as_ptr().add(i) as *const _), 0xFF)
    };

    let digit_mask = _mm512_set1_epi64((RADIX_HIST_LEN as u64 - 1) as i64);
    let digit_shift = _mm512_set1_epi64(shift as i64);

    let order = T::to_total_order_avx512(v);
    let buckets = _mm512_cvtepi64_epi32(_mm512_and_epi64(_mm512_srlv_epi64(order, digit_shift), digit_mask));

    (v, buckets, mask)
}

#[inline(always)]
unsafe fn histogram_avx512_step<const MASKED: bool, T: TotalOrderAvx512>(
    values: &[T],
    histogram: &mut [[u32; RADIX_HIST_LEN]; 8],
    shift: usize,
    i: usize,
) {
    let histogram = histogram.as_flattened_mut();

    let zero = _mm256_set1_epi32(0);
    let one = _mm256_set1_epi32(1);
    let offsets = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32(RADIX_HIST_LEN as _));

    let (_, buckets, mask) = total_order_buckets::<MASKED, T>(values, shift, i);

    let indices = _mm256_add_epi32(buckets, offsets);

    if cfg!(debug_assertions) {
        let indices = std::mem::transmute::<_, [u32; 8]>(indices);
        for i in indices {
            assert!((i as usize) < histogram.len(), "index out of bounds {}, {}", i, histogram.len());
        }
    }

    let old = _mm256_mmask_i32gather_epi32::<4>(zero, mask, indices, histogram.as_ptr() as *const u8);
    let new = _mm256_add_epi32(old, one);
    _mm256_mask_i32scatter_epi32::<4>(histogram.as_mut_ptr() as *mut u8, mask, indices, new)
}

#[inline(always)]
fn histogram_avx512_final(histogram: &mut [[u32; RADIX_HIST_LEN]; 8]) {
    let (first, rest) = histogram.split_first_mut().unwrap();
    rest.iter().for_each(|h| {
        first.iter_mut().zip(h).for_each(|(a, b)| *a += *b);
    })
}

#[inline(never)]
fn fill_histogram_avx512<T: TotalOrderAvx512>(
    values: &[T],
    histogram: &mut [[u32; RADIX_HIST_LEN]; 8],
    shift: usize,
) -> bool {
    histogram.iter_mut().for_each(|h| h.fill(0));

    let mut i = 0;

    while i + 16 <= values.len() {
        unsafe {
            histogram_avx512_step::<false, T>(values, histogram, shift, i);
            i += 8;
            histogram_avx512_step::<false, T>(values, histogram, shift, i);
            i += 8;
        }
    }
    while i + 8 <= values.len() {
        unsafe {
            histogram_avx512_step::<false, T>(values, histogram, shift, i);
            i += 8;
        }
    }
    if i < values.len() {
        unsafe {
            histogram_avx512_step::<true, T>(values, histogram, shift, i);
        }
    }

    histogram_avx512_final(histogram);

    // if every item is in the same bucket then no sorting is necessary
    histogram[0].iter().any(|c| *c as usize == values.len())
}

/// Calculate the prefix sum of the histogram, resulting in the starting indices to the output for each bucket.
#[inline(never)]
pub(crate) fn cumulative_histogram<const N: usize>(histogram: &mut [u32; N]) {
    let mut sum = 0_u32;
    histogram.iter_mut().for_each(|count| {
        let tmp = *count;
        *count = sum;
        sum += tmp;
    });
}

#[inline(never)]
fn reorder_values<T: TotalOrder + Copy>(
    values: &[T],
    output: &mut [T],
    histogram: &mut [u32; RADIX_HIST_LEN],
    shift: usize,
) {
    let chunks = values.chunks_exact(8);
    let remainder = chunks.remainder();
    chunks.into_iter().for_each(|chunk| {
        chunk.iter().for_each(|value| {
            let bucket = (value.to_total_order() >> shift) & (RADIX_HIST_LEN as u64 - 1);
            let output_idx = histogram[bucket as usize];
            unsafe { *output.get_unchecked_mut(output_idx as usize) = *value };
            let next_output_idx = output_idx + 1;
            histogram[bucket as usize] = next_output_idx;
        });
    });
    remainder.iter().for_each(|value| {
        let bucket = (value.to_total_order() >> shift) & (RADIX_HIST_LEN as u64 - 1);
        let output_idx = histogram[bucket as usize];
        unsafe { *output.get_unchecked_mut(output_idx as usize) = *value };
        let next_output_idx = output_idx + 1;
        histogram[bucket as usize] = next_output_idx;
    });
}

#[inline(never)]
pub fn sort_slice_radix<T: TotalOrder + Default + Clone + Copy>(values: &[T]) -> Vec<T> {
    let len = values.len();
    let mut values: Vec<T> = values.to_owned();
    let mut output: Vec<T> = vec![T::default(); len];
    let mut histogram = [0_u32; RADIX_HIST_LEN];

    for i in 0..HIST_PER_U64 {
        let shift = i * RADIX_BITS;
        let all_values_equal = fill_histogram(&values, shift, &mut histogram);
        if !all_values_equal {
            cumulative_histogram(&mut histogram);

            reorder_values(&values, &mut output, &mut histogram, shift);

            std::mem::swap(&mut values, &mut output);
        }
    }

    values
}

#[cfg(not(target_feature = "avx512vpopcntdq"))]
#[inline(always)]
pub(crate) unsafe fn popcount_epi32_lo8(conflicts: __m256i) -> __m256i {
    let table = _mm_set_epi8(4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);
    let table = _mm256_set_m128i(table, table);

    _mm256_add_epi32(
        _mm256_shuffle_epi8(table, _mm256_srli_epi32::<4>(conflicts)),
        _mm256_shuffle_epi8(table, _mm256_and_si256(conflicts, _mm256_set1_epi32(0x0F))),
    )
}

unsafe fn radix_avx512_reorder_step<const MASKED: bool, T: TotalOrder + TotalOrderAvx512 + Copy>(
    values: &[T],
    output: &mut [T],
    histogram: &mut [u32; RADIX_HIST_LEN],
    shift: usize,
    i: usize,
) {
    debug_assert!(values.len() == output.len());

    let zero = _mm256_set1_epi32(0);
    let one = _mm256_set1_epi32(1);

    let (values, buckets, mask) = total_order_buckets::<MASKED, T>(values, shift, i);

    let output_indices = _mm256_mmask_i32gather_epi32::<4>(zero, mask, buckets, histogram.as_ptr() as *const _);
    let conflicts = _mm256_mask_conflict_epi32(zero, mask, output_indices);

    let popcnt = {
        #[cfg(target_feature = "avx512vpopcntdq")]
        {
            _mm256_popcnt_epi32(conflicts)
        }
        #[cfg(not(target_feature = "avx512vpopcntdq"))]
        {
            popcount_epi32_lo8(conflicts)
        }
    };

    let distinct_output_indices = _mm256_add_epi32(output_indices, popcnt);
    _mm512_mask_i32scatter_epi64::<8>(output.as_mut_ptr() as *mut _, mask, distinct_output_indices, values);

    let next_indices = _mm256_add_epi32(distinct_output_indices, one);
    _mm256_mask_i32scatter_epi32::<4>(histogram.as_mut_ptr() as *mut _, mask, buckets, next_indices);
}

#[inline(never)]
pub fn sort_slice_radix_avx512<T: TotalOrder + TotalOrderAvx512 + Copy + Default>(values: &[T]) -> Vec<T> {
    let len = values.len();
    let mut values: Vec<T> = values.to_vec();
    let mut output: Vec<T> = vec![T::default(); len];
    let mut histogram = [[0_u32; RADIX_HIST_LEN]; 8];

    for i in 0..HIST_PER_U64 {
        let shift = i * RADIX_BITS;
        let all_values_equal = fill_histogram_avx512(&values, &mut histogram, shift);
        if !all_values_equal {
            let histogram = &mut histogram[0];
            cumulative_histogram(histogram);

            unsafe {
                let mut i = 0;

                while i + 16 <= len {
                    radix_avx512_reorder_step::<false, _>(&values, &mut output, histogram, shift, i);
                    i += 8;
                    radix_avx512_reorder_step::<false, _>(&values, &mut output, histogram, shift, i);
                    i += 8;
                }
                while i + 8 <= len {
                    radix_avx512_reorder_step::<false, _>(&values, &mut output, histogram, shift, i);
                    i += 8;
                }
                if i < len {
                    radix_avx512_reorder_step::<true, _>(&values, &mut output, histogram, shift, i);
                }
            }
            std::mem::swap(&mut values, &mut output);
        }
    }

    values
}

#[cfg(test)]
mod tests {
    use crate::sort::fill_histogram_avx512;
    use crate::RADIX_HIST_LEN;

    #[test]
    fn test_histogram_avx512_distinct() {
        let mut histogram = [[0_u32; RADIX_HIST_LEN]; 8];
        let values = [1, 2, 3, 4, 5];
        let all_same = fill_histogram_avx512(&values, &mut histogram, 0);
        assert!(!all_same);
        assert_eq!(&histogram[0][0..6], &[0, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_histogram_avx512_repeated() {
        let mut histogram = [[0_u32; RADIX_HIST_LEN]; 8];
        let values = [1, 1, 2, 2, 3];
        let all_same = fill_histogram_avx512(&values, &mut histogram, 0);
        assert!(!all_same);
        assert_eq!(&histogram[0][0..4], &[0, 2, 2, 1]);
    }

    #[test]
    fn test_histogram_avx512_shifted() {
        let mut histogram = [[0_u32; RADIX_HIST_LEN]; 8];
        let values = [1 << 13, 1 << 13, 2 << 13, 2 << 13, 3 << 13];
        let all_same = fill_histogram_avx512(&values, &mut histogram, 13);
        assert!(!all_same);
        assert_eq!(&histogram[0][0..4], &[0, 2, 2, 1]);
    }
}
