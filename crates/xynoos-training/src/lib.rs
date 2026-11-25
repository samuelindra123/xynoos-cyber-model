//! Bounded native training, checkpoint, and detached-runtime support.
#![forbid(unsafe_code)]

mod checkpoint;
mod config;
mod data;
mod error;
mod journal;
mod memory;
mod runtime;
mod schedule;
mod trainer;

pub use checkpoint::{
    CheckpointCodec, CheckpointCounters, CheckpointData, CheckpointPublication, CheckpointSnapshot,
};
pub use config::{MODEL_ID, RunIdentity, RunLimits, TrainingConfig};
pub use data::{BatchSource, VecBatchSource};
pub use error::{Result, TrainingError};
pub use journal::{JournalProgress, RunJournal};
pub use memory::{MemoryPhase, TrainingMemoryPlan, TrainingRunBudget};
pub use runtime::{RunHandoff, RunLogWriter, StopToken, process_vm_hwm_bytes};
pub use schedule::LinearWarmupDecay;
pub use trainer::{StopReason, TimeSummary, Trainer, TrainingReport};
