#![forbid(unsafe_code)]
#![doc(test(attr(forbid(unsafe_code))))]
#![doc = include_str!("../README.md")]

mod error;
mod ops;
mod shape;
mod tensor;

pub use error::TensorError;
pub use tensor::{Tensor, TensorLimits};
