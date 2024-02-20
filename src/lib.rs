use std::ops::{Add, AddAssign, Mul};

pub use einsum_impl;
use num_traits::Zero;

pub trait ArrayNumericType
where Self: Copy + Clone + Zero + Mul<Self, Output = Self> + Add<Self, Output = Self> + AddAssign<Self> {}
impl <T> ArrayNumericType for T
where T: Copy + Clone + Zero + Mul<T, Output = Self> + Add<Self, Output = Self> + AddAssign<Self> {}

#[macro_export]
macro_rules! einsum {
    ($($in: expr),+ => $out: expr ; $($axis: ident $dim: literal),+) => {
        $crate::einsum_impl::einsum_impl!(($crate,), ($($in,)+), ($out,), ($(($axis, $dim),)+))
    };
    ($($in: expr),+ => . $out: ident ; $($axis: ident $dim: literal),+) => {
        $crate::einsum_impl::einsum_impl!(($crate,), ($($in,)+), (_.$out,), ($(($axis, $dim),)+))
    }
    // TODO
    /*($($in: expr),+ => $out: expr) => {
        ::$crate::einsum_impl::einsum_impl_not_optimized!(::$crate, ($($in),+), ($out))
    }*/
}