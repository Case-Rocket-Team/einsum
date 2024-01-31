#[macro_export]

macro_rules! einsum {
    ( $( $x:literal )* ) => ()
        // text processing stuff here
    ( $( $x:expr )* ) => ()
        //matrix processing stuff here
}
