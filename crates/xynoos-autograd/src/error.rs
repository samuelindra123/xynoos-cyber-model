use std::{error::Error, fmt};
use xynoos_tensor::TensorError;

/// Work-accounting boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pass {
    ParameterInsertion,
    Forward,
    Backward,
    GradientTransfer,
    Accumulation,
    ZeroGrad,
    Sgd,
}

/// Numerical validation boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Input,
    Parameter,
    Scale,
    Forward,
    Seed,
    Contribution,
    Gradient,
    Candidate,
}

/// Validation, lifecycle, arithmetic, or reservation failure.
#[derive(Debug, PartialEq, Eq)]
pub enum AutogradError {
    Tensor(TensorError),
    InvalidValueId,
    InvalidParameterId,
    IdentityExhausted,
    RevisionExhausted,
    WrongParameterSet,
    StaleGradientBatch {
        batch: u64,
        current: u64,
    },
    InvalidGradientInventory,
    NonScalarRoot {
        rank: usize,
    },
    BackwardAlreadyRun,
    BackwardRequired,
    TapeSealed,
    EmptyReduction,
    InvalidReferenceShape,
    InvalidCandidateInventory,
    InvalidCandidateBatchShape {
        actual: [usize; 2],
    },
    InvalidToken {
        index: usize,
        target: bool,
    },
    InvalidMask {
        index: usize,
    },
    InvalidTargetShift {
        index: usize,
    },
    NoSupervisedTargets,
    ConstantTargetRequired,
    NonFinite {
        stage: Stage,
        index: usize,
    },
    InvalidLearningRate,
    MissingGradient {
        parameter: usize,
    },
    MemoryLimit {
        requested: usize,
        limit: usize,
    },
    ValueLimit {
        limit: usize,
    },
    OperationLimit {
        limit: usize,
    },
    WorkLimit {
        pass: Pass,
        requested: usize,
        limit: usize,
    },
    ArithmeticOverflow,
    AllocationFailed,
}

impl fmt::Display for AutogradError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tensor(error) => error.fmt(f),
            other => write!(f, "{other:?}"),
        }
    }
}

impl Error for AutogradError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Tensor(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TensorError> for AutogradError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}

pub(crate) fn finite(value: f32, stage: Stage, index: usize) -> Result<f32, AutogradError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(AutogradError::NonFinite { stage, index })
    }
}

pub(crate) fn check_finite(data: &[f32], stage: Stage) -> Result<(), AutogradError> {
    for (index, &value) in data.iter().enumerate() {
        finite(value, stage, index)?;
    }
    Ok(())
}

pub(crate) fn same_shape(left: &[usize], right: &[usize]) -> Result<(), AutogradError> {
    if left.len() != right.len() {
        return Err(TensorError::ShapeMismatch {
            axis: None,
            left: left.len(),
            right: right.len(),
        }
        .into());
    }
    for (axis, (&left, &right)) in left.iter().zip(right).enumerate() {
        if left != right {
            return Err(TensorError::ShapeMismatch {
                axis: Some(axis),
                left,
                right,
            }
            .into());
        }
    }
    Ok(())
}
