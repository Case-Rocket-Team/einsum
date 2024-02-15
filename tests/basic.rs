#[cfg(test)]
mod tests {

    #[test]
    fn check_tree() {
        let a = 1;

        einsum::einsum!(a.ij, b.jk => c.kj; i 1, j 2, k 3)
    }
}