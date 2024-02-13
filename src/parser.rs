use syn::Expr;

pub struct RootToken {
    input: Vec<InputMat>,
    output: Vec<OutputMat>,
}

pub struct InputMat {
    expr: Expr,

}

pub struct OutputMat {
    
}