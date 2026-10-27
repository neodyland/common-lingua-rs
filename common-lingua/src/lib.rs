pub mod fp32;
pub mod global;
pub mod int8;
mod weights;
pub use fp32::model::{Model as ModelFp32, Scratch as ScratchFp32};
pub use global::{LANGS, MAX_LEN, encode};
pub use int8::model::{Model as ModelInt8, Scratch as ScratchInt8};
