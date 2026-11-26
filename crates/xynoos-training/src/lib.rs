//! Bounded native training, checkpoint, and detached-runtime support.
#![forbid(unsafe_code)]

mod checkpoint;
mod config;
mod data;
mod error;
mod evaluation;
mod journal;
mod memory;
mod ownership;
mod provenance;
mod runtime;
mod schedule;
mod trainer;

pub use checkpoint::{
    CheckpointCodec, CheckpointCounters, CheckpointData, CheckpointPublication, CheckpointSnapshot,
};
pub use config::{MODEL_ID, RunIdentity, RunLimits, TrainingConfig};
pub use data::{BatchSource, VecBatchSource};
pub use error::{Result, TrainingError};
pub use evaluation::{
    EvaluationLimits, EvaluationReport, EvaluationStage, evaluate_batches,
    evaluate_standard_dataset_split,
};
pub use journal::{JournalProgress, RunJournal};
pub use memory::{MemoryPhase, TrainingMemoryPlan, TrainingRunBudget};
pub use ownership::{AdvisoryLock, LockError};
pub use provenance::{SOURCE_REPOSITORY_ID, SourceIdentity, attest_source_checkout};
pub use runtime::{RunHandoff, RunLogWriter, StopToken, process_vm_hwm_bytes};
pub use schedule::LinearWarmupDecay;
pub use trainer::{StopReason, TimeSummary, Trainer, TrainingReport};
