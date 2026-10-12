//! Dense CPU storage and shared numerical foundations.

use std::fmt;

/// Invalid tensor storage or shape.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TensorError {
  /// The shape's element count cannot be represented by `usize`.
  ShapeOverflow,
  /// The shape does not match the number of stored elements.
  ElementCountMismatch {
    /// Number of elements required by the shape.
    expected: usize,
    /// Number of elements supplied in storage.
    actual: usize,
  },
}

impl fmt::Display for TensorError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::ShapeOverflow => write!(f, "tensor shape overflows usize"),
      Self::ElementCountMismatch { expected, actual } => {
        write!(f, "tensor shape requires {expected} elements, got {actual}")
      }
    }
  }
}

impl std::error::Error for TensorError {}

/// An owned, contiguous, row-major CPU tensor with a validated shape.
///
/// An empty shape represents a scalar; any zero dimension represents an empty tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T> {
  data: Vec<T>,
  shape: Vec<usize>,
}

impl<T> Tensor<T> {
  /// Constructs a tensor without copying its storage.
  ///
  /// # Errors
  /// Returns an error if the shape's element count overflows or differs from `data.len()`.
  ///
  /// # Examples
  /// ```
  /// use eta_core::Tensor;
  /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
  /// assert_eq!(tensor.shape(), &[2, 2]);
  /// # Ok::<(), eta_core::TensorError>(())
  /// ```
  pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Result<Self, TensorError> {
    let expected = if shape.contains(&0) {
      0
    } else {
      shape.iter().try_fold(1_usize, |count, &dim| {
        count.checked_mul(dim).ok_or(TensorError::ShapeOverflow)
      })?
    };
    if expected != data.len() {
      return Err(TensorError::ElementCountMismatch {
        expected,
        actual: data.len(),
      });
    }
    Ok(Self { data, shape })
  }

  /// Returns the dimensions in storage order.
  pub fn shape(&self) -> &[usize] {
    &self.shape
  }

  /// Returns the contiguous elements in row-major order.
  pub fn as_slice(&self) -> &[T] {
    &self.data
  }

  /// Returns mutable elements without changing the shape or element count.
  pub fn as_mut_slice(&mut self) -> &mut [T] {
    &mut self.data
  }
}

#[cfg(test)]
mod tests {
  use super::{Tensor, TensorError};

  #[test]
  fn mismatched_storage_is_rejected() {
    assert_eq!(
      Tensor::from_vec(vec![1], vec![2]),
      Err(TensorError::ElementCountMismatch { expected: 2, actual: 1 })
    );
  }

  #[test]
  fn overflowing_shape_is_rejected() {
    assert_eq!(
      Tensor::<f32>::from_vec(vec![], vec![usize::MAX, 2]),
      Err(TensorError::ShapeOverflow)
    );
  }

  #[test]
  fn scalar_requires_one_element() {
    assert_eq!(Tensor::from_vec(vec![7], vec![]).unwrap().as_slice(), &[7]);
  }

  #[test]
  fn zero_dimension_is_empty_even_after_large_dimensions() {
    assert!(Tensor::<f32>::from_vec(vec![], vec![usize::MAX, 2, 0]).is_ok());
  }
}
