use crate::shape::{checked_elements, offset};
use crate::TensorError;

/// Per-allocation tensor policy. An explicit lower limit is useful for callers
/// with tighter budgets and for testing rejection without large allocations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorLimits {
    max_bytes: usize,
}

impl TensorLimits {
    /// Maximum logical payload per tensor (64 MiB).
    pub const MAX_BYTES: usize = 64 * 1024 * 1024;
    /// Maximum number of dimensions per tensor.
    pub const MAX_RANK: usize = 32;

    /// Set a byte limit. Requests above the project ceiling are clamped to it.
    pub const fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes: if max_bytes < Self::MAX_BYTES {
                max_bytes
            } else {
                Self::MAX_BYTES
            },
        }
    }

    /// Effective logical payload limit, in bytes.
    pub const fn max_bytes(self) -> usize {
        self.max_bytes
    }
}

impl Default for TensorLimits {
    fn default() -> Self {
        Self::new(Self::MAX_BYTES)
    }
}

/// Owned, contiguous, row-major array with dynamic rank and binary32 values.
#[derive(Debug)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    /// Copy the shape and take ownership of the supplied data without copying it.
    ///
    /// The caller may have allocated the buffer before this method checks its
    /// length, the shape and the limits. Supply data in row-major order.
    pub fn from_vec(
        shape: &[usize],
        data: Vec<f32>,
        limits: TensorLimits,
    ) -> Result<Self, TensorError> {
        validate_data(shape, data.len(), limits)?;
        Ok(Self {
            shape: copy_slice(shape)?,
            data,
        })
    }

    /// Copy a shape and its row-major data after validating length and limits.
    pub fn from_data(
        shape: &[usize],
        data: &[f32],
        limits: TensorLimits,
    ) -> Result<Self, TensorError> {
        validate_data(shape, data.len(), limits)?;
        Ok(Self {
            shape: copy_slice(shape)?,
            data: copy_slice(data)?,
        })
    }

    /// Allocate a tensor of the requested shape initialized to positive zero.
    pub fn zeros(shape: &[usize], limits: TensorLimits) -> Result<Self, TensorError> {
        Self::filled(shape, 0.0, limits)
    }

    /// Allocate a tensor of the requested shape initialized to the supplied value.
    pub fn filled(shape: &[usize], value: f32, limits: TensorLimits) -> Result<Self, TensorError> {
        let elements = checked_elements(shape, limits)?;
        let shape = copy_slice(shape)?;
        let mut data = Vec::new();
        data.try_reserve_exact(elements)?;
        data.resize(elements, value);
        Ok(Self { shape, data })
    }

    /// Borrow the dimension lengths in axis order.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of dimensions. A scalar has rank zero.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Number of values. A scalar has one value.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Borrow all values in row-major order.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Borrow one value by flat row-major index.
    pub fn get_flat(&self, index: usize) -> Result<&f32, TensorError> {
        self.data
            .get(index)
            .ok_or_else(|| self.flat_bounds_error(index))
    }

    /// Mutably borrow one value by flat row-major index.
    pub fn get_flat_mut(&mut self, index: usize) -> Result<&mut f32, TensorError> {
        let error = self.flat_bounds_error(index);
        self.data.get_mut(index).ok_or(error)
    }

    /// Borrow one value by multidimensional index. Pass an empty slice for a scalar.
    pub fn get(&self, indices: &[usize]) -> Result<&f32, TensorError> {
        self.get_flat(offset(&self.shape, indices)?)
    }

    /// Mutably borrow one value by multidimensional index.
    pub fn get_mut(&mut self, indices: &[usize]) -> Result<&mut f32, TensorError> {
        self.get_flat_mut(offset(&self.shape, indices)?)
    }

    pub(crate) fn values_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    fn flat_bounds_error(&self, index: usize) -> TensorError {
        TensorError::IndexOutOfBounds {
            axis: None,
            index,
            len: self.numel(),
        }
    }
}

fn validate_data(shape: &[usize], actual: usize, limits: TensorLimits) -> Result<(), TensorError> {
    let expected = checked_elements(shape, limits)?;
    if actual != expected {
        return Err(TensorError::DataLengthMismatch { expected, actual });
    }
    Ok(())
}

fn copy_slice<T: Copy>(source: &[T]) -> Result<Vec<T>, TensorError> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(source.len())?;
    copy.extend_from_slice(source);
    Ok(copy)
}
