extern crate nalgebra as na;
extern crate proc_macro;
use proc_macro::TokenStream;
use syn::{punctuated::Punctuated, Expr, Token, parse_macro_input, parse::{Parse, ParseStream}, ExprField, Member};
//#[macro_export]


// #[proc_macro]
// //TODO: implement throwing error if the data types suck
// // let $s:expr = whatever is in the first slot
// macro_rules! einsum {
//     ( $s:expr, $( $x:expr ),* ) => {
//         {
//             //get information on what to do string
//             println!("string:");
//             println!($s);
//             let mut info = String::from($s);

//             //get each matrix
//             let mut mat_vec: Vec<Matrix> = Vec::new();
//             $(
//                 println!("matrix:");
//                 println!("{:?}",$x);
//                 mat_vec.push($x);
//             )*
            
//             //initiate 
//             let subs_vec: Vec<&str> = info.split(',').collect();
            
//             println!("subscripts");
//             println!("{:?}",subs_vec[0]);
//             println!("{:?}",subs_vec[1]);
            

//             //get all the different unique axises we have to deal with
//             let mut char_vec2:Vec<Vec<char>> = Vec::new();
//             let mut dex = 0;
//             let mut char_num_dex = 0;
//             for subs in subs_vec{
//                 let len = subs.chars().count()-1;
//                 let total_len = info.chars().count()-1;
//                 for char_num in 0..len{
//                     let current = subs.chars().nth(char_num).unwrap();
//                     let check_dex = 0;
//                     for check_char_num in 0..total_len{
//                         let test_char = info.chars().nth(check_dex).unwrap();
//                         if current != test_char{
//                             char_vec2[dex][char_num_dex] = current;
//                             char_num_dex = char_num_dex + 1;
//                         }
//                     }
//                 }
//                 dex = dex + 1

//             }




//         }
//     };
// }

struct EinsumList {
    pub token: Punctuated<ExprField, Token![,]>
}

impl Parse for EinsumList {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(EinsumList {
            token: Punctuated::parse_separated_nonempty(input)?
        })
    }
}

struct EinsumInput {
    pub token: Punctuated<EinsumList, Token![->]>
}

impl Parse for EinsumInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(EinsumInput {
            token: Punctuated::<EinsumList, Token![->]>::parse_terminated(input)?
        })
    }
}

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
    let mut input = parse_macro_input!(input as EinsumInput).token.into_iter();

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
    }

    // TODO: parse output_tokens like:
    // -> ijk, jk

    todo!()
}