
use core::fmt;
use std::{collections::HashMap, fmt::format, io::{Read, Write}, iter::Map, process::{Command, Stdio}, vec, fmt::Display};

use proc_macro::{Span, TokenStream};
use proc_macro2::TokenStream as TokenStream2;
use syn::{parse::{self, Parse, ParseStream}, parse_macro_input, punctuated::Punctuated, spanned::Spanned, Error, Expr, ExprField, ExprTuple, Ident, Item, Member, Token};
use quote::{format_ident, quote, ToTokens};

macro_rules! expect_token_err {
    ($token:expr, $types:expr) => {
        Err(Error::new($token.span(), format!("Expected a token of type(s) {}; got {:#?}", $types, $token)))
    };
}

macro_rules! err {
    ($($tt:tt),+) => {
        Error::new(Span::call_site().into(), format!($($tt),+))
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

// Get the name of this crate.
macro_rules! thiscrate {
    () => {
        $crate
    };
}

fn expr_ident_string(expr: &Expr) -> Result<String, Error> {
    match expr {
        Expr::Path(path) => {
            Ok(path.path.segments
                .first()
                .ok_or(Error::new_spanned(expr, "Couldn't get first item of path for ident string"))?
                .ident.to_string()
            )
        }
        _ => expect_token_err!(expr, "Expr::Path"),
    }
}

#[derive(Debug)]
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

#[derive(Debug)]
struct Axis {
    char: char,
    size: usize,
    ident: Ident,
}

impl Axis {
    pub fn new(char: char, size: usize) -> Self {
        Self {
            char,
            size,
            ident: Ident::new(&format!("axis_{}", char), Span::call_site().into()),
        }
    }

}

impl ToTokens for Axis {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        self.ident.to_tokens(tokens);
    }
}

impl Display for Axis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.char, self.size)
    }
}

fn parse_mat_tuple(tuple: ExprTuple) -> Result<Vec<Mat>, Error> {
    tuple.elems.into_iter().map(|x| Mat::from_expr(x)).collect::<Result<Vec<Mat>, Error>>()
}

#[derive(Debug)]
struct EinsumArgs {
    crate_expr: Expr,
    input: Vec<Mat>,
    output: Vec<Mat>,
    axes: HashMap<char, Axis>,
}

impl Parse for EinsumArgs {
    fn parse(input: ParseStream) -> parse::Result<Self> {
        let punct: Punctuated<ExprTuple, syn::token::Comma> = Punctuated::parse_terminated(input)?;
        let mut iter = punct.into_iter();
        let err = Error::new(input.span(), "Not enough args");
        let [crate_expr, input_expr, output_expr, dims_expr] = [
            iter.next().ok_or(err.clone())?,
            iter.next().ok_or(err.clone())?,
            iter.next().ok_or(err.clone())?,
            iter.next().ok_or(err)?
        ];

        Ok(Self {
            crate_expr: crate_expr.elems.first().ok_or(Error::new(crate_expr.span(), "Couldn't get first item of crate_expr"))?.clone(),
            input: parse_mat_tuple(input_expr)?,
            output: parse_mat_tuple(output_expr)?,
            axes: dims_expr.elems.iter().map(|x| {
                let x = match x {
                    Expr::Tuple(x) => x,
                    _ => return expect_token_err!(x, "Expr::Tuple")
                };
                let axis = expr_ident_string(&x.elems[0])?;

                let dim_expr = cast_expr_ref!(&x.elems[1], Lit)?;
                let dim = match &dim_expr.lit {
                    syn::Lit::Int(int) => int.base10_parse::<usize>()?,
                    _ => return Err(Error::new(dim_expr.lit.span(), format!("Expected an integer, got {:?}", dim_expr.lit)))
                };

                let char = axis.chars().next().ok_or(Error::new(input.span(), "Couldn't read axis chars"))?;
                Ok((char, Axis::new(char, dim)))
            }).collect::<Result<HashMap<_, _>, Error>>()?,
        })
    }
}


//https://optimized-einsum.readthedocs.io/en/stable/autosummary/opt_einsum.contract.ContractExpression.html#opt_einsum.contract.ContractExpression"


// Reference: einsum!(a.ij, b.jk => c.kj; a 1, b 2)
//              => ((a.ij, b.jk), (c.kj), ((a, 1), (b, 2)))

#[proc_macro]
pub fn einsum_impl(stream: TokenStream) -> TokenStream {
    println!("Start");
    let args = parse_macro_input!(stream);

    let res = handle_errors(do_einsum(&args));
    
    quote!{{ #res }}.into()
}

fn handle_errors(result: Result<TokenStream2, Error>) -> TokenStream2 {
    match result {
        Ok(res) => res,
        Err(err) => err.to_compile_error()
    }
}

fn do_einsum(args: &EinsumArgs) -> Result<TokenStream2, Error> {
    // Get the optimized contraction order.
    let opt = get_opt(&args)?;

    let EinsumArgs { crate_expr, input, output, axes: dims, .. } = args;

    let mut tokens: Vec<TokenStream2> = vec![];

    struct MatInfo {
        ident: Ident,
        axes: String,
        id: usize,
    }

    impl ToTokens for MatInfo {
        fn to_tokens(&self, tokens: &mut TokenStream2) {
            self.ident.to_tokens(tokens);
        }
    }

    let mut idents = vec![];
    let mut exprs = vec![];

    let mut mats = vec![];
    for (i, mat) in input.iter().enumerate() {
        let ident = Ident::new(&format!("mat_{}", i), mat.expr.span());
        let expr = &mat.expr;
        idents.push(ident.clone());
        exprs.push(expr);
        
        mats.push(MatInfo { ident, axes: mat.axes.clone(), id: i });
    }

    tokens.push(quote!{
        // Do this all on one line to avoid accidentally shadowing the variables.
    });

    let mut lhs = mats.iter().map(|x| x.id).collect::<Vec<usize>>();
    let mut out_dim = vec![];

    for (mut i, mut j, contraction) in opt {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        }

        let out = MatInfo {
            ident: Ident::new(&format!("mat_{}", mats.len()), Span::call_site().into()),
            axes: contraction.split("->").nth(1)
                .ok_or(err!("Where is the second part of the contraction? Implicit contractions aren't allowed"))?
                .to_string(),
            id: mats.len(),
        };

        let mut dim_tuple = vec![];

        println!("Contraction: {}", contraction);
        println!("---");

        for axis in out.axes.chars() {
            let size = dims.get(&axis).expect(format!("Axis {} not found in dims", axis).as_str()).size;
            dim_tuple.push(quote! {
                #size,
            });
        }

        tokens.push(quote! {
            let mut #out = ndarray::Array::<T, _>::zeros((#(#dim_tuple)*));
        });

        mats.push(out);

        // The two matrices to contract
        let a = &mats[lhs.remove(i)];
        let b = &mats[lhs.remove(j - 1)];
        let out = &mats.last().unwrap();
        let out_axes = out.axes.chars().map(|x| dims.get(&x).expect("Internal error: no dim?"));
        out_dim = out.axes.chars().map(|x| dims.get(&x).expect("Internal error: no dim?").size).collect();

        lhs.push(out.id);

        // For usage inside quote!{}
        let a_axes = a.axes.chars().map(|x| dims.get(&x));
        let b_axes = b.axes.chars().map(|x| dims.get(&x));

        let mut all_axes: Vec<char> = vec![];

        for axis in (a.axes.clone() + &b.axes + &out.axes).chars() {
            if !all_axes.contains(&axis) {
                all_axes.push(axis);
            }   
        }

        // Now we actually do the math.

        // This is the inner body of the loop. We will build this,
        // then wrap it in the next loop, and so on.
        let mut body = quote! {
            #out[(#(#out_axes),*)] += #a[(#(#a_axes),*)] * #b[(#(#b_axes),*)];
        };

        // Reverse the iterator, since we are doing C ordering (as opposed to Fortran)
        // TODO: Support Fortran ordering
        for axis in all_axes.iter().rev() {
            let axis = dims.get(axis).expect("Internal error: no dim?");
            let size = axis.size;

            body = quote! {
                for #axis in 0..#size {
                    #body
                }
            }
        }

        tokens.push(body);
    }

    let out = &mats[lhs[0]];

    let mut input_generics_defs = vec![];
    let mut input_generics = vec![];
    for i in 0..idents.len() {
        let ident = format_ident!("I{}", i);
        input_generics.push(ident.clone());
        input_generics_defs.push(quote! {
            #ident: ndarray::Dimension
        });
    }

    let dim_len = out_dim.len();
    let input_index_tys = input.iter().map(|x| (0..x.axes.len()).map(|_| quote!{usize}).collect::<Vec<_>>()).collect::<Vec<_>>();

    let final_expr = quote! {
        // Use a function for type inference.
        #[inline]
        fn __einsum_impl<T: #crate_expr::ArrayNumericType, #(#input_generics_defs),*>
        (#(#idents: &ndarray::Array<T, #input_generics>),*) -> ndarray::Array<T, ndarray::Dim<[usize; #dim_len]>>
        where #((#(#input_index_tys),*): ndarray::NdIndex<#input_generics>),* {
            #(#tokens)*
            #out
        }
        __einsum_impl(#(&#exprs),*)
    };

    Ok(final_expr.into())
}

fn get_opt(args: &EinsumArgs) -> Result<Vec<(usize, usize, String)>, Error> {
    let EinsumArgs { input, output, axes: dims, .. } = args;
    let str_input = input.iter().map(|x| x.axes.clone()).collect::<Vec<String>>().join(",");
    let str_output = output.iter().map(|x| x.axes.clone()).collect::<Vec<String>>().join(",");
    let opt_einsum_input = format!("{str_input}->{str_output}");

    let mut dim_str = String::new();

    for mat in input {
        dim_str.push_str("(");
        for axis in mat.axes.chars() {
            let Some(axis) = dims.get(&axis) else {
                return Err(Error::new(Span::call_site().into(), format!("Axis {} not found in dims", axis)));
            };

            dim_str.push_str(format!("{},", axis.size).as_str());
        }
        dim_str.push_str("), ");
    }

    fn pyerr(pretext: &str) -> impl Fn(std::io::Error) -> Error {
        let pretext = pretext.to_string();
        move |err| Error::new(Span::call_site().into(), format!("{}: {}", pretext, err))
    }

    let py = Command::new("python")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(pyerr("Error while trying to spawn Python process"))?;

    let code = format!(r#"
import opt_einsum as oe
expr = oe.contract_expression("{opt_einsum_input}", {dim_str})
print("\n".join([";".join([str(contraction[0][0]), str(contraction[0][1]), contraction[2]]) for contraction in expr.contraction_list]))
"#);
    println!("{}", code);
    
    let mut stdin = py.stdin.as_ref().ok_or(err!("Couldn't get stdin for Python process"))?;
    stdin.write(code.as_bytes()).map_err(pyerr("Error while writing to Python process"))?;

    let output = py.wait_with_output()
        .map_err(pyerr("Couldn't wait on Python process"))?;

    if !output.status.success() {
        let code = output.status.code().unwrap_or(-1);
        let err = String::from_utf8(output.stderr).unwrap_or("Error while reading Python process stderr".to_string());
        let out = String::from_utf8(output.stdout).unwrap_or("Error while reading Python process stdout".to_string());
        return Err(err!("Python process failed with non-zero exit code: {}\nstdout:\n{}\nstderr: {}", code, err, out));
    }

    let out = String::from_utf8(
        output.stdout
    ).map_err(|x| Error::new(
        Span::call_site().into(),
        format!("Error while parsing Python output as utf8: {}", x)
    ))?;

    let mut list = Vec::new();

    println!("{}", out);

    let int_err = |err| Error::new(Span::call_site().into(), format!("Error while parsing integer from Python opt_einsum: {}", err));
    let not_enough_err = Error::new(Span::call_site().into(), "Not enough items in contraction list returned from Python opt_einsum");

    for line in out.lines() {
        let line = line.trim();

        let mut iter = line.split(";").peekable();
        while iter.peek().is_some() {
            list.push((
                iter.next().ok_or(not_enough_err.clone())?.parse().map_err(int_err)?,
                iter.next().ok_or(not_enough_err.clone())?.parse().map_err(int_err)?,
                iter.next().ok_or(not_enough_err.clone())?.to_string()
            ));
        }   
    }

    Ok(list)
}