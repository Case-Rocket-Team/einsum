#[cfg(test)]
mod tests {

    #[test]
    fn check_tree() {
        let a = 1;

        nalgebra_einsum::einsum!(a.ij, b.jk => c.kj; a 1, b 2,)
    }
}