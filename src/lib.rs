#[macro_export]
macro_rules! einsum {
    ($($in: expr),+ => $out: expr ; $($axis: ident $dim: literal),+) => {
        einsum_impl::einsum_impl!(($($in,)+), ($out,), ($(($axis, $dim),)+))
    };
    ($($in: expr),+ => $out: expr) => {
        einsum_impl::einsum_impl_not_optimized!(($($in),+), ($out))
    }
}