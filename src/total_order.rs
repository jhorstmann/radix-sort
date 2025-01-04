pub trait ToBits {
    fn to_bits(&self) -> u64;
    fn bit_width(&self) -> u8;
}

impl ToBits for bool {
    #[inline(always)]
    fn to_bits(&self) -> u64 {
        *self as u64
    }

    #[inline(always)]
    fn bit_width(&self) -> u8 {
        1
    }
}

impl ToBits for i64 {
    #[inline(always)]
    fn to_bits(&self) -> u64 {
        *self as u64
    }

    #[inline(always)]
    fn bit_width(&self) -> u8 {
        64
    }
}

impl ToBits for u64 {
    #[inline(always)]
    fn to_bits(&self) -> u64 {
        *self
    }

    #[inline(always)]
    fn bit_width(&self) -> u8 {
        64
    }
}

impl ToBits for f64 {
    #[inline(always)]
    fn to_bits(&self) -> u64 {
        f64::to_bits(*self)
    }

    #[inline(always)]
    fn bit_width(&self) -> u8 {
        64
    }
}

impl ToBits for u32 {
    #[inline(always)]
    fn to_bits(&self) -> u64 {
        *self as u64
    }

    #[inline(always)]
    fn bit_width(&self) -> u8 {
        32
    }
}

impl ToBits for u16 {
    #[inline(always)]
    fn to_bits(&self) -> u64 {
        *self as u64
    }

    #[inline(always)]
    fn bit_width(&self) -> u8 {
        16
    }
}

pub trait TotalOrder: 'static {
    fn to_total_order(&self) -> u64;
}

impl TotalOrder for f64 {
    #[inline(always)]
    fn to_total_order(&self) -> u64 {
        // see f64::total_cmp
        // let bits = self.to_bits();
        // reading via integer pointer seems to avoid a move from xmm to gp reg
        let bits = unsafe { (self as *const f64 as *const u64).read() };
        (bits ^ ((bits as i64 >> 63) as u64 >> 1)) ^ (1 << 63)
    }
}

impl TotalOrder for f32 {
    #[inline(always)]
    fn to_total_order(&self) -> u64 {
        // see f32::total_cmp
        // let bits = self.to_bits();
        // reading via integer pointer seems to avoid a move from xmm to gp reg
        let bits = unsafe { (self as *const f32 as *const u32).read() };
        ((bits ^ ((bits as i32 >> 31) as u32 >> 1)) ^ (1 << 31)) as u64
    }
}

impl TotalOrder for u64 {
    #[inline(always)]
    fn to_total_order(&self) -> u64 {
        *self
    }
}

impl TotalOrder for u32 {
    #[inline(always)]
    fn to_total_order(&self) -> u64 {
        *self as u64
    }
}

impl TotalOrder for u16 {
    #[inline(always)]
    fn to_total_order(&self) -> u64 {
        *self as u64
    }
}

impl TotalOrder for u8 {
    #[inline(always)]
    fn to_total_order(&self) -> u64 {
        *self as u64
    }
}

impl TotalOrder for i64 {
    #[inline(always)]
    fn to_total_order(&self) -> u64 {
        (*self as u64) ^ (1 << 63)
    }
}

impl TotalOrder for i32 {
    #[inline(always)]
    fn to_total_order(&self) -> u64 {
        (*self as u64) ^ (1 << 31)
    }
}

impl TotalOrder for i16 {
    #[inline(always)]
    fn to_total_order(&self) -> u64 {
        (*self as u64) ^ (1 << 15)
    }
}

impl TotalOrder for i8 {
    #[inline(always)]
    fn to_total_order(&self) -> u64 {
        (*self as u64) ^ (1 << 7)
    }
}

impl TotalOrder for bool {
    #[inline(always)]
    fn to_total_order(&self) -> u64 {
        *self as u64
    }
}

pub trait FromTotalOrder {
    fn from_total_order(value: u64) -> Self;
}

impl FromTotalOrder for f64 {
    fn from_total_order(value: u64) -> Self {
        // (bits ^ ((bits as i64 >> 63) as u64 >> 1)) ^ (1 << 63)
        let sign = value >> 63;
        f64::from_bits((value ^ ((value as i64) >> 63) as u64) ^ (sign << 63) ^ u64::MAX)
    }
}

impl FromTotalOrder for i64 {
    fn from_total_order(value: u64) -> Self {
        (value ^ (1 << 63)) as i64
    }
}

pub trait TotalOrderAvx512 {
    fn to_total_order_avx512(values: std::arch::x86_64::__m512i) -> std::arch::x86_64::__m512i
    where
        Self: Sized;
}

impl TotalOrderAvx512 for f64 {
    #[inline(always)]
    fn to_total_order_avx512(values: std::arch::x86_64::__m512i) -> std::arch::x86_64::__m512i
    where
        Self: Sized,
    {
        unsafe {
            //(bits ^ ((bits as i64 >> 63) as u64 >> 1)) ^ (1 << 63)
            std::arch::x86_64::_mm512_xor_epi64(
                std::arch::x86_64::_mm512_xor_epi64(
                    values,
                    std::arch::x86_64::_mm512_srli_epi64(std::arch::x86_64::_mm512_srai_epi64(values, 63), 1),
                ),
                std::arch::x86_64::_mm512_set1_epi64(1 << 63),
            )
        }
    }
}

impl TotalOrderAvx512 for i64 {
    #[inline(always)]
    fn to_total_order_avx512(values: std::arch::x86_64::__m512i) -> std::arch::x86_64::__m512i
    where
        Self: Sized,
    {
        unsafe { std::arch::x86_64::_mm512_xor_epi64(values, std::arch::x86_64::_mm512_set1_epi64(1 << 63)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_total_order() {
        let floats = vec![-f64::NAN, f64::NEG_INFINITY, -0.0, 0.0, 1.0, f64::INFINITY, f64::NAN];
        assert!(floats[0].to_total_order() < floats[1].to_total_order());
        assert!(floats[1].to_total_order() < floats[2].to_total_order());
        assert!(floats[2].to_total_order() < floats[3].to_total_order());
        assert!(floats[3].to_total_order() < floats[4].to_total_order());
        assert!(floats[4].to_total_order() < floats[5].to_total_order());
        assert!(floats[5].to_total_order() < floats[6].to_total_order());
    }

    #[test]
    fn test_total_order_roundtrip_f64() {
        let values = vec![
            0.0,
            1.0,
            -1.0,
            999999999.125,
            f64::MIN,
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -f64::NAN,
            f64::NAN,
            f64::EPSILON,
        ];
        let total_order = values.iter().map(|f| f.to_total_order()).collect::<Vec<_>>();
        let roundtrip = total_order
            .iter()
            .map(|u| f64::from_total_order(*u))
            .collect::<Vec<_>>();

        for (left, right) in values.iter().zip(roundtrip.iter()) {
            assert_eq!(
                left.to_bits(),
                right.to_bits(),
                "{:064b} != {:064b}",
                left.to_bits(),
                right.to_bits()
            );
        }
    }
}
