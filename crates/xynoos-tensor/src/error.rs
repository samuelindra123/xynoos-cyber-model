use std::collections::TryReserveError;
use std::fmt;

/// Validation or reservation failure in a tensor operation.
#[derive(Debug, PartialEq, Eq)]
pub enum TensorError {
    /// Rank exceeds the fixed metadata limit.
    RankLimitExceeded { rank: usize, limit: usize },
    /// A dimension times its canonical stride does not fit in usize.
    ShapeOverflow { axis: usize },
    /// The size in bytes does not fit in usize.
    ByteCountOverflow { elements: usize },
    /// Logical f32 payload exceeds the effective per-tensor byte limit.
    PayloadLimitExceeded { bytes: usize, limit: usize },
    /// Fallible reservation failed for the shape or payload.
    AllocationFailed,
    /// The supplied data length differs from the checked element count.
    DataLengthMismatch { expected: usize, actual: usize },
    /// An operation or index requires a different rank.
    RankMismatch { expected: usize, actual: usize },
    /// Shapes differ in rank (axis is None) or first differing dimension.
    ShapeMismatch {
        axis: Option<usize>,
        left: usize,
        right: usize,
    },
    /// Matrix inner dimensions differ.
    MatmulDimensionMismatch { left: usize, right: usize },
    /// A flat (axis is None) or multidimensional index is out of bounds.
    IndexOutOfBounds {
        axis: Option<usize>,
        index: usize,
        len: usize,
    },
    /// Checked row-major offset arithmetic overflowed.
    OffsetOverflow { axis: usize },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::RankLimitExceeded { rank, limit } => {
                write!(f, "rank {rank} exceeds limit {limit}")
            }
            Self::ShapeOverflow { axis } => write!(f, "shape product overflow at axis {axis}"),
            Self::ByteCountOverflow { elements } => {
                write!(f, "{elements} f32 elements overflow byte count")
            }
            Self::PayloadLimitExceeded { bytes, limit } => {
                write!(f, "{bytes} payload bytes exceed limit {limit}")
            }
            Self::AllocationFailed => f.write_str("tensor allocation reservation failed"),
            Self::DataLengthMismatch { expected, actual } => {
                write!(f, "expected {expected} data elements, got {actual}")
            }
            Self::RankMismatch { expected, actual } => {
                write!(f, "expected rank {expected}, got {actual}")
            }
            Self::ShapeMismatch {
                axis: Some(axis),
                left,
                right,
            } => write!(f, "shapes differ at axis {axis}: {left} vs {right}"),
            Self::ShapeMismatch {
                axis: None,
                left,
                right,
            } => write!(f, "shape ranks differ: {left} vs {right}"),
            Self::MatmulDimensionMismatch { left, right } => {
                write!(f, "matrix inner dimensions differ: {left} vs {right}")
            }
            Self::IndexOutOfBounds {
                axis: Some(axis),
                index,
                len,
            } => write!(
                f,
                "index {index} exceeds bounds of axis {axis} with length {len}"
            ),
            Self::IndexOutOfBounds {
                axis: None,
                index,
                len,
            } => write!(f, "flat index {index} exceeds bounds of length {len}"),
            Self::OffsetOverflow { axis } => write!(f, "index offset overflow at axis {axis}"),
        }
    }
}

impl std::error::Error for TensorError {}

impl From<TryReserveError> for TensorError {
    fn from(_: TryReserveError) -> Self {
        Self::AllocationFailed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservation_error_maps_without_exhausting_memory() {
        let error = Vec::<u8>::new().try_reserve_exact(usize::MAX).unwrap_err();
        assert_eq!(TensorError::from(error), TensorError::AllocationFailed);
    }
}
