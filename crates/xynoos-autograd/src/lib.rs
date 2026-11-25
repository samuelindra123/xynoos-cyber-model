#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

mod budget;
mod error;
mod kernels;
mod parameters;
mod reference;
mod tape;

pub use budget::ResourceLimits;
pub use error::{AutogradError, Pass, Stage};
pub use parameters::{GradientBatch, ParameterId, ParameterSet, Sgd};
pub use reference::ReferenceBatch;
pub use tape::{
    CandidateReferenceExperimentPolicy, ReferenceTransformerBackward,
    ReferenceTransformerEvaluation, Tape, TapeMetrics, ValueId,
};

use std::sync::atomic::{AtomicU64, Ordering};

fn identity() -> Result<u64, AutogradError> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| AutogradError::IdentityExhausted)
}
