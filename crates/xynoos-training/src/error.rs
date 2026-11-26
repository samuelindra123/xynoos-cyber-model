use std::{error::Error, fmt, io, time::Duration};

pub type Result<T> = std::result::Result<T, TrainingError>;

#[derive(Debug, PartialEq, Eq)]
pub enum TrainingError {
    InvalidConfiguration,
    InvalidIdentity,
    EvaluationSplitMismatch,
    EvaluationTokenLimit,
    EvaluationWorkLimit,
    NonFiniteEvaluation,
    InvalidSchedule,
    ScheduleComplete,
    ResourceLimit {
        requested: usize,
        limit: usize,
    },
    BudgetExhausted(&'static str),
    ArithmeticOverflow,
    AllocationFailed,
    Io(io::ErrorKind),
    Autograd(xynoos_autograd::AutogradError),
    Transformer(xynoos_transformer::Error),
    Dataset(xynoos_data::DatasetError),
    Checkpoint(&'static str),
    /// A required runtime telemetry event could not be written or flushed.
    Logging(&'static str),
    /// A required runtime-owned artifact could not be published.
    RuntimeArtifact(&'static str),
    PublicationDurabilityUncertain,
    /// Another process currently owns this experiment lineage.
    RunLockContended,
    /// The OS lock could not be opened or recovered safely.
    RunLockRecoveryFailure(io::ErrorKind),
    StopRequested,
    RuntimeLimit {
        limit: Duration,
    },
}

impl fmt::Display for TrainingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Error for TrainingError {}

impl From<io::Error> for TrainingError {
    fn from(value: io::Error) -> Self {
        Self::Io(value.kind())
    }
}

impl From<xynoos_autograd::AutogradError> for TrainingError {
    fn from(value: xynoos_autograd::AutogradError) -> Self {
        Self::Autograd(value)
    }
}

impl From<xynoos_transformer::Error> for TrainingError {
    fn from(value: xynoos_transformer::Error) -> Self {
        Self::Transformer(value)
    }
}

impl From<xynoos_data::DatasetError> for TrainingError {
    fn from(value: xynoos_data::DatasetError) -> Self {
        Self::Dataset(value)
    }
}
