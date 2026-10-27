use std::path::PathBuf;
use std::{env, fs};

use proc_macro::TokenStream as TokenStream1;
use proc_macro2::TokenStream;
use quote::quote;

#[proc_macro]
pub fn load_weights(input: TokenStream1) -> TokenStream1 {
    if !input.is_empty() {
        panic!("input should be empty")
    }
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let mut combined = TokenStream::new();
    for entry in fs::read_dir(PathBuf::from(manifest_dir).join("./src/weights")).unwrap() {
        let entry = entry.unwrap();
        let entry_name = entry.file_name();
        let entry_name = entry_name.to_str().unwrap();
        if !entry_name.ends_with(".bin") {
            continue;
        }
        let entry_name = entry_name.strip_suffix(".bin").unwrap();
        let path = entry.path().canonicalize().unwrap();
        let size = fs::read_to_string(path.parent().unwrap().join(format!("{entry_name}.size")))
            .unwrap()
            .trim()
            .split(",")
            .map(|s| s.parse::<usize>().unwrap())
            .collect::<Vec<_>>();
        let path = path.to_str().unwrap();
        combined.extend(quote! {
            #entry_name => (&include_bytes!(#path)[..], &[#(#size),*][..]),
        });
    }

    quote! {
        pub fn load_weight(name: &str) -> (Vec<f32>, Vec<usize>) {
            let (weight, shape) = match name {
                #combined
                _ => panic!("no weight exist")
            };
            (
                weight.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                shape.to_vec()
            )
        }
    }
    .into()
}
