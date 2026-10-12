//! Pure Rust foundations for machine learning, deep learning, and reinforcement learning.
//!
//! The `ml`, `nn`, and `rl` features enable their respective modules (all on by default).

pub use eta_core as core;
pub use eta_core::{Tensor, TensorError};
#[cfg(feature = "ml")]
pub use eta_ml as ml;
#[cfg(feature = "nn")]
pub use eta_nn as nn;
#[cfg(feature = "rl")]
pub use eta_rl as rl;
