
use std::{collections::HashMap, fmt::format, iter::Map, process::{Command, Stdio}};

use pyo3::types::{IntoPyDict, PyDict};
use proc_macro::TokenStream;
use syn::{parse::{self, Parse, ParseStream}, parse_macro_input, punctuated::Punctuated, spanned::Spanned, Error, Expr, ExprField, ExprTuple, Ident, Member, Token};
use pyo3::prelude::*;

macro_rules! expect_token_err {
    ($token:expr, $types:expr) => {
        Err(Error::new($token.span(), format!("Expected a token of type(s) {}; got {:#?}", $types, $token)))
    };
}

macro_rules! cast_expr {
    ($token:expr, $ty:tt) => {
        {
            match $token {
                Expr::$ty(expr) => Ok(expr),
                Expr::Group(group) => {
                    let expr = group.expr;
                    match *expr {
                        Expr::$ty(expr) => Ok(expr),
                        _ => expect_token_err!(expr, stringify!(Expr::$ty))
                    }
                },
                _ => expect_token_err!($token, stringify!(Expr::$ty))
            }
        }
    };
}

macro_rules! cast_expr_ref {
    ($token:expr, $ty:tt) => {
        {
            match $token {
                Expr::$ty(expr) => Ok(expr),
                Expr::Group(group) => {
                    match &*group.expr {
                        Expr::$ty(expr) => Ok(expr),
                        _ => expect_token_err!(group.expr, stringify!(Expr::$ty))
                    }
                },
                _ => expect_token_err!($token, stringify!(Expr::$ty))
            }
        }
    };
}

fn expr_ident_string(expr: &Expr) -> Result<String, Error> {
    match expr {
        Expr::Path(path) => {
            Ok(path.path.segments.first().unwrap().ident.to_string())
        }
        _ => expect_token_err!(expr, "Expr::Path"),
    }
}

struct Mat {
    pub expr: Expr,
    pub axes: String,
    pub id: Option<usize>,
}

impl Mat {
    pub fn new(expr: Expr, axis: String) -> Self {
        Self {
            expr,
            axes: axis,
            id: None,
        }
    }

    pub fn from_expr(expr: Expr) -> Result<Self, Error> {
        let field = cast_expr!(expr, Field)?;

        let axes = if let Member::Named(ident) = &field.member {
            ident.to_string()
        } else {
            return expect_token_err!(field.member, "Member::Named")
        };

        let expr = *field.base;


        Ok(Self {
            expr,
            axes,
            id: None,
        })
    }
}

struct EinsumArgs {
    input: Vec<Mat>,
    output: Vec<Mat>,
    dims: HashMap<char, u128>,
}

fn parse_mat_tuple(tuple: ExprTuple) -> Result<Vec<Mat>, Error> {
    tuple.elems.into_iter().map(|x| Mat::from_expr(x)).collect::<Result<Vec<Mat>, Error>>()
}

impl Parse for EinsumArgs {
    fn parse(input: ParseStream) -> parse::Result<Self> {
        let punct: Punctuated<ExprTuple, syn::token::Comma> = Punctuated::parse_terminated(input)?;
        let mut iter = punct.into_iter();
        let err = Error::new(input.span(), "Not enough args");
        let [input_expr, output_expr, dims_expr] = [
            iter.next().ok_or(err.clone())?,
            iter.next().ok_or(err.clone())?,
            iter.next().ok_or(err)?
        ];

        Ok(Self {
            input: parse_mat_tuple(input_expr)?,
            output: parse_mat_tuple(output_expr)?,
            dims: dims_expr.elems.iter().map(|x| {
                let x = match x {
                    Expr::Tuple(x) => x,
                    _ => return expect_token_err!(x, "Expr::Tuple")
                };
                let axis = expr_ident_string(&x.elems[0])?;

                let dim_expr = cast_expr_ref!(&x.elems[1], Lit)?;
                let dim = match &dim_expr.lit {
                    syn::Lit::Int(int) => int.base10_parse::<u128>()?,
                    _ => return Err(Error::new(dim_expr.lit.span(), format!("Expected an integer, got {:?}", dim_expr.lit)))
                };

                Ok((axis.chars().next().unwrap(), dim))
            }).collect::<Result<HashMap<char, u128>, Error>>()?,

        })
    }
}

// Reference: einsum!(a.ij, b.jk => c.kj; a 1, b 2)
//              => ((a.ij, b.jk), (c.kj), ((a, 1), (b, 2)))

#[proc_macro]
pub fn einsum_impl(stream: TokenStream) -> TokenStream {
    let EinsumArgs { input, output, dims, .. } = parse_macro_input!(stream);

    let str_input = input.iter().map(|x| x.axes.clone()).collect::<Vec<String>>().join(",");
    let str_output = output.iter().map(|x| x.axes.clone()).collect::<Vec<String>>().join(",");
    let opt_einsum_input = format!("{str_input}->{str_output}");

    let mut dim_str = String::new();

    println!("Dims: {:#?}", dims);

    for mat in input {
        dim_str.push_str("(");
        for axis in mat.axes.chars() {
            let Some(dim) = dims.get(&axis) else {
                panic!("Axis {} not found in dims", axis);
            };

            dim_str.push_str(format!("{},", dim).as_str());
        }
        dim_str.push_str("), ");
    }

    pyo3::prepare_freethreaded_python();

    let path: Vec<(usize, usize)> = Python::with_gil(|py| {
        let np = match PyModule::import(py, "numpy") {
            Ok(np) => np,
            Err(e) => {
                e.print(py);
                panic!();
            }
        };
        let oe = match PyModule::import(py, "opt_einsum") {
            Ok(oe) => oe,
            Err(e) => {
                e.print(py);
                panic!();
            }
        };
        let locals = [("np", np), ("oe", oe)].into_py_dict(py);

        let res = py.run(format!("
contraction_list = oe.contract_expression(\"{opt_einsum_input}\", {dim_str})
path = [contraction[0] for contraction in contraction_list]
").as_str(),
        None, Some(locals));

        match res {
            Ok(_) => todo!(),//locals.get_item("path").unwrap().unwrap().extract().unwrap(),
            Err(e) => {
                e.print(py);
                panic!();
            }
        }
    });


    println!("{:?}", path);

    return TokenStream::new();
}

//https://optimized-einsum.readthedocs.io/en/stable/autosummary/opt_einsum.contract.ContractExpression.html#opt_einsum.contract.ContractExpression"