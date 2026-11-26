use std::mem::size_of;

use xynoos_data::{Batch, BatchStream, Split};

use crate::{Result, TrainingError};

/// Deterministic bounded source used by the trainer and resume tests.
pub trait BatchSource {
    fn next_batch(&mut self) -> Result<Option<Batch>>;
    fn cursor(&self) -> u64;
    fn seek_to(&mut self, cursor: u64) -> Result<()>;
    /// Declared dataset role; the trainer rejects held-out sources.
    fn split(&self) -> Split;
    fn order_identity(&self) -> [u8; 32];
    /// Retained or conservative source-side bytes charged to the data budget.
    fn reserved_memory_bytes(&self) -> Result<usize>;

    /// Conservative peak source bytes while one emitted batch is live.
    fn evaluation_reserved_bytes(&self, batch_bytes: usize) -> Result<usize> {
        self.reserved_memory_bytes()?
            .checked_add(batch_bytes)
            .ok_or(TrainingError::ArithmeticOverflow)
    }
}

impl BatchSource for BatchStream<'_> {
    fn next_batch(&mut self) -> Result<Option<Batch>> {
        Ok(BatchStream::next_batch(self)?)
    }

    fn cursor(&self) -> u64 {
        BatchStream::cursor(self)
    }

    fn seek_to(&mut self, cursor: u64) -> Result<()> {
        Ok(BatchStream::seek_batches(self, cursor)?)
    }

    fn split(&self) -> Split {
        BatchStream::split(self)
    }

    fn order_identity(&self) -> [u8; 32] {
        BatchStream::order_identity(self)
    }

    fn reserved_memory_bytes(&self) -> Result<usize> {
        Ok(BatchStream::reserved_memory_bytes(self)?)
    }

    fn evaluation_reserved_bytes(&self, _batch_bytes: usize) -> Result<usize> {
        // BatchStream's reservation already includes the output batch area.
        Ok(BatchStream::reserved_memory_bytes(self)?)
    }
}

impl<S: BatchSource + ?Sized> BatchSource for &mut S {
    fn next_batch(&mut self) -> Result<Option<Batch>> {
        (**self).next_batch()
    }

    fn cursor(&self) -> u64 {
        (**self).cursor()
    }

    fn seek_to(&mut self, cursor: u64) -> Result<()> {
        (**self).seek_to(cursor)
    }

    fn split(&self) -> Split {
        (**self).split()
    }

    fn order_identity(&self) -> [u8; 32] {
        (**self).order_identity()
    }

    fn reserved_memory_bytes(&self) -> Result<usize> {
        (**self).reserved_memory_bytes()
    }

    fn evaluation_reserved_bytes(&self, batch_bytes: usize) -> Result<usize> {
        (**self).evaluation_reserved_bytes(batch_bytes)
    }
}

#[derive(Debug)]
pub struct VecBatchSource {
    batches: Vec<Batch>,
    cursor: u64,
    split: Split,
    order_identity: [u8; 32],
}

impl VecBatchSource {
    pub fn new(batches: Vec<Batch>, order_identity: [u8; 32]) -> Self {
        Self::for_split(batches, order_identity, Split::Train)
    }

    pub fn for_split(batches: Vec<Batch>, order_identity: [u8; 32], split: Split) -> Self {
        Self {
            batches,
            cursor: 0,
            split,
            order_identity,
        }
    }

    pub fn from_shared(batches: Vec<Batch>, order_identity: [u8; 32]) -> Self {
        Self::new(batches, order_identity)
    }
}

impl BatchSource for VecBatchSource {
    fn next_batch(&mut self) -> Result<Option<Batch>> {
        let index = usize::try_from(self.cursor).map_err(|_| TrainingError::ArithmeticOverflow)?;
        let Some(batch) = self.batches.get(index) else {
            return Ok(None);
        };
        let copied = Batch::from_parts(
            batch.shape()[0],
            batch.shape()[1],
            batch.input(),
            batch.target(),
            batch.valid_mask(),
            batch.loss_mask(),
        )?;
        self.cursor = self
            .cursor
            .checked_add(1)
            .ok_or(TrainingError::ArithmeticOverflow)?;
        Ok(Some(copied))
    }

    fn cursor(&self) -> u64 {
        self.cursor
    }

    fn seek_to(&mut self, cursor: u64) -> Result<()> {
        if cursor > self.batches.len() as u64 {
            return Err(TrainingError::InvalidConfiguration);
        }
        self.cursor = cursor;
        Ok(())
    }

    fn split(&self) -> Split {
        self.split
    }

    fn order_identity(&self) -> [u8; 32] {
        self.order_identity
    }

    fn reserved_memory_bytes(&self) -> Result<usize> {
        self.batches.iter().try_fold(
            self.batches
                .capacity()
                .checked_mul(size_of::<Batch>())
                .ok_or(TrainingError::ArithmeticOverflow)?,
            |total, batch| {
                total
                    .checked_add(batch.retained_bytes()?)
                    .ok_or(TrainingError::ArithmeticOverflow)
            },
        )
    }
}
