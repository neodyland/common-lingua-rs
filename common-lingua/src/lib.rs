pub mod fp32;
pub mod global;
mod weights;
pub use fp32::model::{Model as ModelFp32, Scratch as ScratchFp32};
pub use global::{LANGS, MAX_LEN, encode};
