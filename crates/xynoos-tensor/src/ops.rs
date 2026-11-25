use crate::shape::{require_rank, require_same_shape};
use crate::{Tensor, TensorError, TensorLimits};

impl Tensor {
    /// Element-wise binary32 addition on identical shapes.
    pub fn add(&self, other: &Self, limits: TensorLimits) -> Result<Self, TensorError> {
        self.elementwise(other, limits, |a, b| a + b)
    }

    /// Element-wise binary32 subtraction on identical shapes.
    pub fn subtract(&self, other: &Self, limits: TensorLimits) -> Result<Self, TensorError> {
        self.elementwise(other, limits, |a, b| a - b)
    }

    /// Element-wise binary32 multiplication on identical shapes.
    pub fn multiply(&self, other: &Self, limits: TensorLimits) -> Result<Self, TensorError> {
        self.elementwise(other, limits, |a, b| a * b)
    }

    /// Multiply every element by a binary32 scalar.
    pub fn scale(&self, scalar: f32, limits: TensorLimits) -> Result<Self, TensorError> {
        let mut output = Self::zeros(self.shape(), limits)?;
        for (out, &input) in output.values_mut().iter_mut().zip(self.data()) {
            *out = input * scalar;
        }
        Ok(output)
    }

    /// Dot product of equal-length rank-one tensors.
    ///
    /// Products and sequential accumulation use binary64, followed by one
    /// conversion to binary32. Empty inputs give positive zero.
    pub fn dot(&self, other: &Self) -> Result<f32, TensorError> {
        require_rank(self.rank(), 1)?;
        require_rank(other.rank(), 1)?;
        require_same_shape(self.shape(), other.shape())?;
        Ok(dot_values(self.data().iter(), other.data().iter()))
    }

    /// Row-major matrix product of rank-two tensors with equal inner dimensions.
    ///
    /// Each output cell uses the dot product accumulation policy in increasing
    /// inner-index order. A zero inner dimension produces positive zeros.
    pub fn matmul(&self, other: &Self, limits: TensorLimits) -> Result<Self, TensorError> {
        let &[rows, inner] = self.shape() else {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: self.rank(),
            });
        };
        let &[other_inner, columns] = other.shape() else {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: other.rank(),
            });
        };
        if inner != other_inner {
            return Err(TensorError::MatmulDimensionMismatch {
                left: inner,
                right: other_inner,
            });
        }

        let mut output = Self::zeros(&[rows, columns], limits)?;
        if output.numel() == 0 || inner == 0 {
            return Ok(output);
        }

        let left_rows = self.data().chunks_exact(inner);
        for (out_row, left_row) in output.values_mut().chunks_exact_mut(columns).zip(left_rows) {
            for (column, out) in out_row.iter_mut().enumerate() {
                let right_column = other.data().iter().skip(column).step_by(columns);
                *out = dot_values(left_row.iter(), right_column);
            }
        }
        Ok(output)
    }

    /// Copy a rank-two tensor with its two axes exchanged.
    pub fn transpose(&self, limits: TensorLimits) -> Result<Self, TensorError> {
        let &[rows, columns] = self.shape() else {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: self.rank(),
            });
        };
        let mut output = Self::zeros(&[columns, rows], limits)?;
        if output.numel() == 0 {
            return Ok(output);
        }

        for (column, out_row) in output.values_mut().chunks_exact_mut(rows).enumerate() {
            let input_column = self.data().iter().skip(column).step_by(columns);
            for (out, &input) in out_row.iter_mut().zip(input_column) {
                *out = input;
            }
        }
        Ok(output)
    }

    /// Sum all values in flat row-major order using binary64 accumulation and
    /// one final conversion to binary32. Empty tensors give positive zero.
    pub fn sum(&self) -> f32 {
        let mut total = 0.0f64;
        for &value in self.data() {
            total += f64::from(value);
        }
        total as f32
    }

    fn elementwise(
        &self,
        other: &Self,
        limits: TensorLimits,
        operation: impl Fn(f32, f32) -> f32,
    ) -> Result<Self, TensorError> {
        require_same_shape(self.shape(), other.shape())?;
        let mut output = Self::zeros(self.shape(), limits)?;
        for (out, (&a, &b)) in output
            .values_mut()
            .iter_mut()
            .zip(self.data().iter().zip(other.data()))
        {
            *out = operation(a, b);
        }
        Ok(output)
    }
}

fn dot_values<'a>(
    left: impl Iterator<Item = &'a f32>,
    right: impl Iterator<Item = &'a f32>,
) -> f32 {
    let mut total = 0.0f64;
    for (&a, &b) in left.zip(right) {
        total += f64::from(a) * f64::from(b);
    }
    total as f32
}
