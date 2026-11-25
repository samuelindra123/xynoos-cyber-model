use crate::shape::{checked_elements, offset};
use crate::TensorError;
use std::{iter, mem::size_of};

/// Per-tensor limits on logical f32 payload and retained f32 buffer capacity.
///
/// The byte budget excludes shape metadata, allocator overhead and other
/// allocations in the process. Shape rank has its own limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorLimits {
    max_bytes: usize,
}

impl TensorLimits {
    /// Maximum logical payload and retained numerical capacity per tensor (64 MiB).
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

    /// Effective limit, in bytes, for both payload and retained numerical capacity.
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
    /// Copy the shape and adopt the supplied row-major buffer without reallocating it.
    ///
    /// The length must match the shape. Both the logical payload and
    /// `data.capacity() * size_of::<f32>()` must fit the effective byte limit.
    /// Excess capacity is rejected with [`TensorError::RetainedCapacityLimitExceeded`].
    ///
    /// The caller's allocation exists before validation. This call consumes the
    /// buffer and drops it on any error, including shape allocation failure.
    pub fn from_vec(
        shape: &[usize],
        data: Vec<f32>,
        limits: TensorLimits,
    ) -> Result<Self, TensorError> {
        validate_data(shape, data.len(), limits)?;
        validate_retained_capacity(data.capacity(), limits)?;
        Ok(Self {
            shape: copy_shape(shape)?,
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
            shape: copy_shape(shape)?,
            data: collect_data(data.iter().copied(), limits)?,
        })
    }

    /// Allocate a tensor of the requested shape initialized to positive zero.
    pub fn zeros(shape: &[usize], limits: TensorLimits) -> Result<Self, TensorError> {
        Self::filled(shape, 0.0, limits)
    }

    /// Allocate a tensor of the requested shape initialized to the supplied value.
    pub fn filled(shape: &[usize], value: f32, limits: TensorLimits) -> Result<Self, TensorError> {
        let elements = checked_elements(shape, limits)?;
        Ok(Self {
            shape: copy_shape(shape)?,
            data: collect_data(iter::repeat_n(value, elements), limits)?,
        })
    }

    /// Fallibly copy the shape and logical values under the supplied tensor limits.
    pub fn try_clone(&self, limits: TensorLimits) -> Result<Self, TensorError> {
        Self::from_data(self.shape(), self.data(), limits)
    }

    /// Checked total bytes of the retained shape and numerical vector capacities.
    ///
    /// This includes spare capacity and excludes the inline tensor and allocator overhead.
    pub fn retained_bytes(&self) -> Result<usize, TensorError> {
        let overflow = || TensorError::RetainedCapacityByteCountOverflow {
            capacity: self.data.capacity(),
        };
        let shape_bytes = self
            .shape
            .capacity()
            .checked_mul(size_of::<usize>())
            .ok_or_else(overflow)?;
        let data_bytes = self
            .data
            .capacity()
            .checked_mul(size_of::<f32>())
            .ok_or_else(overflow)?;
        shape_bytes.checked_add(data_bytes).ok_or_else(overflow)
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

fn copy_shape(shape: &[usize]) -> Result<Vec<usize>, TensorError> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(shape.len())?;
    for &dim in shape {
        copy.push(dim);
    }
    Ok(copy)
}

fn collect_data(
    values: impl ExactSizeIterator<Item = f32>,
    limits: TensorLimits,
) -> Result<Vec<f32>, TensorError> {
    let mut data = Vec::new();
    data.try_reserve_exact(values.len())?;
    validate_retained_capacity(data.capacity(), limits)?;
    // Individual pushes cannot reallocate while the reserved capacity suffices.
    for value in values {
        data.push(value);
    }
    Ok(data)
}

fn validate_retained_capacity(capacity: usize, limits: TensorLimits) -> Result<(), TensorError> {
    let bytes = capacity
        .checked_mul(size_of::<f32>())
        .ok_or(TensorError::RetainedCapacityByteCountOverflow { capacity })?;
    let limit = limits.max_bytes();
    if bytes > limit {
        return Err(TensorError::RetainedCapacityLimitExceeded { bytes, limit });
    }
    Ok(())
}

#[cfg(test)]
mod retained_capacity_tests {
    use super::*;

    const F32_BYTES: usize = size_of::<f32>();

    fn one_value_with_spare_capacity() -> Vec<f32> {
        let mut data = Vec::with_capacity(2);
        data.push(3.0);
        data
    }

    #[test]
    fn clone_copies_values_and_respects_its_own_limits() {
        let original = Tensor::from_vec(
            &[1],
            one_value_with_spare_capacity(),
            TensorLimits::default(),
        )
        .unwrap();
        assert_eq!(
            original.retained_bytes().unwrap(),
            original.shape.capacity() * size_of::<usize>() + original.data.capacity() * F32_BYTES,
        );
        let mut copied = original.try_clone(TensorLimits::new(F32_BYTES)).unwrap();
        *copied.get_flat_mut(0).unwrap() = -1.0;
        assert_eq!(original.data(), [3.0]);
        assert_eq!(copied.shape(), original.shape());
        assert!(matches!(
            original.try_clone(TensorLimits::new(0)),
            Err(TensorError::PayloadLimitExceeded { .. })
        ));
    }

    #[test]
    fn accepts_capacity_below_limit_without_copying_or_shrinking() {
        let data = one_value_with_spare_capacity();
        let capacity = data.capacity();
        let pointer = data.as_ptr();
        let limits = TensorLimits::new(capacity * F32_BYTES + 1);
        let tensor = Tensor::from_vec(&[1], data, limits).unwrap();
        assert_eq!(tensor.data(), [3.0]);
        assert_eq!(tensor.data().as_ptr(), pointer);
        assert_eq!(tensor.data.capacity(), capacity);
    }

    #[test]
    fn accepts_capacity_at_limit() {
        let data = one_value_with_spare_capacity();
        let capacity = data.capacity();
        let limits = TensorLimits::new(capacity * F32_BYTES);
        let tensor = Tensor::from_vec(&[1], data, limits).unwrap();
        assert_eq!(tensor.data.capacity() * F32_BYTES, limits.max_bytes());
    }

    #[test]
    fn rejects_over_budget_capacity_with_payload_at_limit() {
        let data = one_value_with_spare_capacity();
        let bytes = data.capacity() * F32_BYTES;
        let limits = TensorLimits::new(F32_BYTES);
        assert_eq!(
            Tensor::from_vec(&[1], data, limits).unwrap_err(),
            TensorError::RetainedCapacityLimitExceeded {
                bytes,
                limit: limits.max_bytes(),
            }
        );
    }

    #[test]
    fn rejects_empty_vector_with_spare_capacity_at_zero_budget() {
        let data = Vec::with_capacity(1);
        let bytes = data.capacity() * F32_BYTES;
        assert_eq!(
            Tensor::from_vec(&[0], data, TensorLimits::new(0)).unwrap_err(),
            TensorError::RetainedCapacityLimitExceeded { bytes, limit: 0 }
        );
    }

    #[test]
    fn accepts_empty_vector_without_capacity_at_zero_budget() {
        let tensor = Tensor::from_vec(&[2, 0, 3], Vec::new(), TensorLimits::new(0)).unwrap();
        assert_eq!(tensor.shape(), [2, 0, 3]);
        assert_eq!(tensor.numel(), 0);
        assert_eq!(tensor.data.capacity(), 0);
    }

    #[test]
    fn rejects_scalar_capacity_one_byte_over_limit() {
        let data = one_value_with_spare_capacity();
        let bytes = data.capacity() * F32_BYTES;
        let limit = bytes - 1;
        assert_eq!(
            Tensor::from_vec(&[], data, TensorLimits::new(limit)).unwrap_err(),
            TensorError::RetainedCapacityLimitExceeded { bytes, limit }
        );
    }

    #[test]
    fn distinguishes_capacity_byte_overflow_from_limit_rejection() {
        let largest_representable = usize::MAX / F32_BYTES;
        let limits = TensorLimits::new(0);
        assert_eq!(
            validate_retained_capacity(largest_representable, limits),
            Err(TensorError::RetainedCapacityLimitExceeded {
                bytes: largest_representable * F32_BYTES,
                limit: 0,
            })
        );
        let capacity = largest_representable + 1;
        assert_eq!(
            validate_retained_capacity(capacity, limits),
            Err(TensorError::RetainedCapacityByteCountOverflow { capacity })
        );
    }

    #[test]
    fn enforces_the_project_capacity_ceiling_without_large_allocations() {
        let limit = TensorLimits::MAX_BYTES;
        let max_capacity = limit / F32_BYTES;
        let clamped = TensorLimits::new(usize::MAX);
        assert_eq!(clamped, TensorLimits::default());
        assert_eq!(validate_retained_capacity(max_capacity, clamped), Ok(()));
        assert_eq!(
            validate_retained_capacity(max_capacity + 1, clamped),
            Err(TensorError::RetainedCapacityLimitExceeded {
                bytes: limit + F32_BYTES,
                limit,
            })
        );
    }

    #[test]
    fn checks_length_before_retained_capacity() {
        let data = one_value_with_spare_capacity();
        let limits = TensorLimits::new(0);
        assert_eq!(
            Tensor::from_vec(&[0], data, limits).unwrap_err(),
            TensorError::DataLengthMismatch {
                expected: 0,
                actual: 1,
            }
        );
    }

    #[test]
    fn constructors_and_allocating_operations_keep_capacity_within_budget() {
        let limits = TensorLimits::new(1024);
        let row = Tensor::from_data(&[1, 2], &[1.0, 2.0], limits).unwrap();
        let zero = Tensor::zeros(&[1, 2], limits).unwrap();
        let filled = Tensor::filled(&[1, 2], 3.0, limits).unwrap();
        let column = row.transpose(limits).unwrap();
        let product = row.matmul(&column, limits).unwrap();
        let added = row.add(&filled, limits).unwrap();
        let subtracted = row.subtract(&filled, limits).unwrap();
        let multiplied = row.multiply(&filled, limits).unwrap();
        let scaled = row.scale(2.0, limits).unwrap();
        for (tensor, expected) in [
            (&row, [1.0, 2.0]),
            (&zero, [0.0, 0.0]),
            (&filled, [3.0, 3.0]),
            (&column, [1.0, 2.0]),
            (&added, [4.0, 5.0]),
            (&subtracted, [-2.0, -1.0]),
            (&multiplied, [3.0, 6.0]),
            (&scaled, [2.0, 4.0]),
        ] {
            assert_eq!(tensor.data(), expected);
            assert!(tensor.data.capacity() * F32_BYTES <= limits.max_bytes());
        }
        assert_eq!(column.shape(), [2, 1]);
        assert_eq!(product.shape(), [1, 1]);
        assert_eq!(product.data(), [5.0]);
        assert!(product.data.capacity() * F32_BYTES <= limits.max_bytes());
    }
}
