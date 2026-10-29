pub mod global;
mod math;
mod model;
mod quant;
pub use global::{LANGS, MAX_LEN, encode};
pub use model::{Model, Scratch};
mod ckpt;

#[cfg(feature = "rust")]
pub static WEIGHTS: &[u8] = include_bytes!("./weight.safetensors");
