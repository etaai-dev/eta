//! Interfaces for traditional machine learning estimators.

/// A supervised estimator that learns from features and targets.
pub trait Estimator {
  /// Input representation accepted during fitting and prediction.
  type Input: ?Sized;
  /// Target representation accepted during fitting.
  type Target: ?Sized;
  /// Prediction representation produced by the estimator.
  type Prediction;
  /// Errors produced by fitting or prediction.
  type Error: std::error::Error;

  /// Fits the estimator to the supplied features and targets.
  ///
  /// # Errors
  /// Returns the implementation's error for invalid data or failed fitting.
  fn fit(&mut self, input: &Self::Input, target: &Self::Target) -> Result<(), Self::Error>;

  /// Predicts outputs for the supplied features.
  ///
  /// # Errors
  /// Returns the implementation's error for invalid input or an unfitted estimator.
  fn predict(&self, input: &Self::Input) -> Result<Self::Prediction, Self::Error>;
}
