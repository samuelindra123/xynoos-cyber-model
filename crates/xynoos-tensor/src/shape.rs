use crate::{TensorError, TensorLimits};
use std::mem::size_of;

pub(crate) fn checked_elements(
    shape: &[usize],
    limits: TensorLimits,
) -> Result<usize, TensorError> {
    if shape.len() > TensorLimits::MAX_RANK {
        return Err(TensorError::RankLimitExceeded {
            rank: shape.len(),
            limit: TensorLimits::MAX_RANK,
        });
    }

    // Each intermediate suffix product is the stride for the next axis.
    // A zero sets the product to zero only after strides to its right are checked.
    let mut elements = 1usize;
    for (axis, &dim) in shape.iter().enumerate().rev() {
        elements = elements
            .checked_mul(dim)
            .ok_or(TensorError::ShapeOverflow { axis })?;
    }

    let bytes = elements
        .checked_mul(size_of::<f32>())
        .ok_or(TensorError::ByteCountOverflow { elements })?;
    let limit = limits.max_bytes();
    if bytes > limit {
        return Err(TensorError::PayloadLimitExceeded { bytes, limit });
    }
    Ok(elements)
}

pub(crate) fn require_rank(actual: usize, expected: usize) -> Result<(), TensorError> {
    if actual == expected {
        Ok(())
    } else {
        Err(TensorError::RankMismatch { expected, actual })
    }
}

pub(crate) fn require_same_shape(left: &[usize], right: &[usize]) -> Result<(), TensorError> {
    if left.len() != right.len() {
        return Err(TensorError::ShapeMismatch {
            axis: None,
            left: left.len(),
            right: right.len(),
        });
    }
    for (axis, (&left, &right)) in left.iter().zip(right).enumerate() {
        if left != right {
            return Err(TensorError::ShapeMismatch {
                axis: Some(axis),
                left,
                right,
            });
        }
    }
    Ok(())
}

pub(crate) fn offset(shape: &[usize], indices: &[usize]) -> Result<usize, TensorError> {
    require_rank(indices.len(), shape.len())?;

    // Check every axis first. In particular, a zero axis must return a bounds
    // error even when a prefix of an empty shape would overflow in Horner form.
    for (axis, (&len, &index)) in shape.iter().zip(indices).enumerate() {
        if index >= len {
            return Err(TensorError::IndexOutOfBounds {
                axis: Some(axis),
                index,
                len,
            });
        }
    }
    let mut flat = 0usize;
    for (axis, (&dim, &index)) in shape.iter().zip(indices).enumerate() {
        flat = flat
            .checked_mul(dim)
            .and_then(|base| base.checked_add(index))
            .ok_or(TensorError::OffsetOverflow { axis })?;
    }
    Ok(flat)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_boundary_validates_without_allocating() {
        let bytes = TensorLimits::MAX_BYTES;
        let elements = bytes / size_of::<f32>();
        assert_eq!(
            checked_elements(&[elements], TensorLimits::default()),
            Ok(elements)
        );
        assert_eq!(
            checked_elements(&[elements + 1], TensorLimits::default()),
            Err(TensorError::PayloadLimitExceeded {
                bytes: bytes + 4,
                limit: bytes
            })
        );
    }

    #[test]
    fn unchecked_shape_cannot_wrap_offset_multiplication_or_addition() {
        assert_eq!(
            offset(&[3, usize::MAX], &[2, 0]),
            Err(TensorError::OffsetOverflow { axis: 1 })
        );
        assert_eq!(
            offset(&[2, usize::MAX], &[1, 1]),
            Err(TensorError::OffsetOverflow { axis: 1 })
        );
    }
}
