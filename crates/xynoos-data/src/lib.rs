//! Governed local datasets, integrity-checked streaming, and next-token batches.
#![forbid(unsafe_code)]

mod batch;
mod manifest;
mod schema;

pub use batch::{Batch, BatchConfig, BatchStream, StreamConfig, StreamStats, Tail};
pub use manifest::{
    Dataset, DatasetReader, Document, FitRecord, FittedTokenizer, ManifestValidator, assign_split,
};
pub use schema::{
    CleaningStatus, DatasetManifest, DocumentRecord, Domain, RedistributionPermission,
    ReviewStatus, SourceRecord, Split, TokenizerId, TrainingPermission, Transformation,
    canonical_line,
};

use std::{fmt, io, mem::size_of};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetError {
    InvalidLimits,
    InvalidConfiguration,
    InvalidSchema,
    UnsupportedVersion,
    InvalidJson,
    NonCanonical,
    InvalidProvenance,
    InvalidPath,
    IntegrityMismatch,
    InvalidUtf8,
    InvalidPayload,
    DuplicateId,
    DuplicateLeakage,
    InvalidSplit,
    HeldOutFit,
    InvalidFit,
    IncompatibleTokenizer,
    RecordLimit,
    MetadataLimit,
    DocumentLimit,
    BufferLimit,
    TruncatedRecord,
    InvalidTokens,
    Overflow,
    Allocation,
    Io(io::ErrorKind),
    Tokenizer(xynoos_tokenizer::TokenizerError),
}

impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for DatasetError {}

impl From<io::Error> for DatasetError {
    fn from(error: io::Error) -> Self {
        Self::Io(error.kind())
    }
}

impl From<xynoos_tokenizer::TokenizerError> for DatasetError {
    fn from(error: xynoos_tokenizer::TokenizerError) -> Self {
        Self::Tokenizer(error)
    }
}

type Result<T> = std::result::Result<T, DatasetError>;
use DatasetError::*;

#[derive(Debug, Clone, Copy)]
pub struct DatasetLimits {
    pub document_bytes: usize,
    pub reader_bytes: usize,
    pub record_bytes: usize,
    pub entries: usize,
    pub metadata_bytes: usize,
    pub token_buffer_bytes: usize,
    pub sequence_length: usize,
    pub batch_size: usize,
    pub batch_positions: usize,
}

impl DatasetLimits {
    pub const MAX: Self = Self {
        document_bytes: 1 << 20,
        reader_bytes: 64 << 10,
        record_bytes: 1 << 20,
        entries: 100_000,
        metadata_bytes: 64 << 20,
        token_buffer_bytes: 16 << 20,
        sequence_length: 2048,
        batch_size: 64,
        batch_positions: 65_536,
    };

    fn validate(self) -> Result<()> {
        let max = Self::MAX;
        if self.document_bytes > max.document_bytes
            || self.reader_bytes == 0
            || self.reader_bytes > max.reader_bytes
            || self.record_bytes > max.record_bytes
            || self.entries > max.entries
            || self.metadata_bytes > max.metadata_bytes
            || self.token_buffer_bytes > max.token_buffer_bytes
            || self.sequence_length > max.sequence_length
            || self.batch_size > max.batch_size
            || self.batch_positions > max.batch_positions
        {
            return Err(InvalidLimits);
        }
        Ok(())
    }
}

impl Default for DatasetLimits {
    fn default() -> Self {
        Self::MAX
    }
}

fn add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right).ok_or(Overflow)
}

fn mul(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right).ok_or(Overflow)
}

fn capacity<T>(items: &Vec<T>) -> Result<usize> {
    mul(items.capacity(), size_of::<T>())
}

fn reserve<T>(items: &mut Vec<T>, additional: usize) -> Result<()> {
    items.try_reserve_exact(additional).map_err(|_| Allocation)
}

fn allocate<T>(count: usize) -> Result<Vec<T>> {
    let mut items = Vec::new();
    reserve(&mut items, count)?;
    Ok(items)
}

fn check(used: usize, limit: usize, error: DatasetError) -> Result<()> {
    if used > limit { Err(error) } else { Ok(()) }
}
