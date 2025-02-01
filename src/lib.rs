#![feature(stdarch_x86_avx512)]
mod sort;
mod total_order;

pub use sort::*;
pub use total_order::*;

// 8 -> 8 passes per u64
// 11 -> 6 passes per u64, 16k memory, easily fits into L1 cache
// 13 -> 5 passes per u64, 64k memory, and probably no longer fits into L1 cache
// 16 -> 4 passes per u64, 0.5m memory, should be even worse
pub(crate) const RADIX_BITS: usize = 11;
pub(crate) const RADIX_HIST_LEN: usize = 1 << RADIX_BITS;
pub(crate) const RADIX_HIST_MASK: u64 = (RADIX_HIST_LEN - 1) as u64;
pub(crate) const HIST_PER_U64: usize = 64_usize.div_ceil(RADIX_BITS);
