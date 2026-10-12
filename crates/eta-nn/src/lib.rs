//! Interfaces for composable neural network modules.

/// A neural network module with an explicit training or inference mode.
pub trait Module {
  /// Input representation accepted by the module.
  type Input: ?Sized;
  /// Output representation produced by the module.
  type Output;
  /// Errors produced by the forward pass.
  type Error: std::error::Error;

  /// Evaluates the module; `training` enables behavior such as dropout.
  ///
  /// # Errors
  /// Returns the implementation's error for incompatible inputs or failed computation.
  fn forward(&self, input: &Self::Input, training: bool) -> Result<Self::Output, Self::Error>;
}
