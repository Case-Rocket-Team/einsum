
use proc_macro::TokenStream;
use syn::{punctuated::Punctuated, Expr, Token, parse_macro_input, parse::{Parse, ParseStream}, Ident, ExprField, Member, custom_punctuation};
//#[macro_export]

mod rightarrow;
mod parser;

// TODO: Opt-einsum w expression

use rightarrow::RightArrow;

#[derive(Debug)]
struct EinsumList {
    pub tokens: Punctuated<ExprField, Token![,]>,
    //pub mats: Vec<Mat>
}

impl Parse for EinsumList {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(EinsumList {
            tokens: Punctuated::parse_separated_nonempty(input)?
        })
    }
}

#[derive(Debug)]
struct EinsumInput {
    pub tokens: Punctuated<EinsumList, RightArrow>
}

impl Parse for EinsumInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(EinsumInput {
            tokens: Punctuated::<EinsumList, RightArrow>::parse_terminated(input)?
        })
    }
}

#[derive(Debug)]
struct Mat {
    pub expr: Expr,
    pub indices: String,
}

impl Mat {
    pub fn new(expr: Expr, indices: String) -> Mat {
        Mat {
            expr, indices
        }
    }
}

#[proc_macro]
pub fn einsum(input: TokenStream) -> TokenStream {
    let mut _input = parse_macro_input!(input as EinsumInput).tokens.into_iter();

    todo!();

    /*let mut input = _input.tokens.into_iter();
    let input_tokens = input.next().unwrap().token.into_iter();
    let output_tokens = input.next().unwrap().token.into_iter();

    let mut input_mats: Vec<Mat> = vec![];

    for mat_token in input_tokens {
        let string = match mat_token.member {
            Member::Named(ident) => {
                ident.to_string()
            },
            _ => {
                panic!()
            }
        };

        input_mats.push(Mat::new(*mat_token.base, string));
    }*/

    // TODO: parse output_tokens like:
    // -> ijk, jk

    

    todo!()
}