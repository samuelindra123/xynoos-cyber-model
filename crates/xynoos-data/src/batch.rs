use sha2::{Digest, Sha256};
use xynoos_tokenizer::{Boundaries, SpecialToken, TokenId, Tokenizer, TokenizerLimits};

use crate::{
    Dataset, DatasetError, DatasetError::*, DatasetLimits, DatasetReader, FittedTokenizer, Result,
    Split, add, allocate, capacity, check, mul,
};

#[derive(Debug, Clone, Copy, Default)]
pub enum Tail {
    #[default]
    Drop,
    Pad,
}

#[derive(Debug, Clone, Copy)]
pub struct BatchConfig {
    pub sequence_length: usize,
    pub stride: usize,
    pub batch_size: usize,
    pub tail: Tail,
    pub pad_final_batch: bool,
}

impl BatchConfig {
    pub const fn per_document(sequence_length: usize, batch_size: usize) -> Self {
        Self {
            sequence_length,
            stride: sequence_length,
            batch_size,
            tail: Tail::Drop,
            pad_final_batch: false,
        }
    }

    fn positions(self, limits: DatasetLimits) -> Result<usize> {
        let positions = mul(self.sequence_length, self.batch_size)?;
        if self.sequence_length == 0
            || self.sequence_length > limits.sequence_length
            || self.stride == 0
            || self.stride > self.sequence_length
            || self.batch_size == 0
            || self.batch_size > limits.batch_size
            || positions > limits.batch_positions
        {
            return Err(InvalidConfiguration);
        }
        Ok(positions)
    }

    fn masks(self) -> bool {
        matches!(self.tail, Tail::Pad) || self.pad_final_batch
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StreamConfig {
    pub batch: BatchConfig,
    pub boundaries: Boundaries,
    pub tokenizer_limits: TokenizerLimits,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct StreamStats {
    pub documents: usize,
    pub samples: usize,
    pub batches: usize,
    pub dropped_tails: usize,
    pub dropped_positions: usize,
    pub padded_tails: usize,
    pub padded_positions: usize,
    pub padded_rows: usize,
    /// Highest measured combined batch and document token vector capacity in bytes.
    pub peak_token_buffer_bytes: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Batch {
    rows: usize,
    width: usize,
    input: Vec<TokenId>,
    target: Vec<TokenId>,
    valid: Option<Vec<u8>>,
    loss: Option<Vec<u8>>,
}

impl Batch {
    /// Construct a bounded batch from already tokenized row-major arrays.
    /// Semantic token, shift, padding, and mask rules are validated by the
    /// consuming Transformer; this method enforces only storage shape/caps.
    pub fn from_parts(
        rows: usize,
        width: usize,
        input: &[TokenId],
        target: &[TokenId],
        valid: Option<&[u8]>,
        loss: Option<&[u8]>,
    ) -> Result<Self> {
        let positions = mul(rows, width)?;
        if rows == 0
            || width == 0
            || positions > DatasetLimits::MAX.batch_positions
            || input.len() != positions
            || target.len() != positions
            || valid.is_some_and(|mask| mask.len() != positions)
            || loss.is_some_and(|mask| mask.len() != positions)
        {
            return Err(InvalidTokens);
        }
        let masks = usize::from(valid.is_some()) + usize::from(loss.is_some());
        let required = mul(positions, 2 * size_of::<TokenId>() + masks)?;
        check(required, DatasetLimits::MAX.token_buffer_bytes, BufferLimit)?;
        let mut input_copy = allocate(positions)?;
        input_copy.extend_from_slice(input);
        let mut target_copy = allocate(positions)?;
        target_copy.extend_from_slice(target);
        let valid_copy = valid
            .map(|values| {
                let mut copy = allocate(positions)?;
                copy.extend_from_slice(values);
                Ok::<_, DatasetError>(copy)
            })
            .transpose()?;
        let loss_copy = loss
            .map(|values| {
                let mut copy = allocate(positions)?;
                copy.extend_from_slice(values);
                Ok::<_, DatasetError>(copy)
            })
            .transpose()?;
        Ok(Self {
            rows,
            width,
            input: input_copy,
            target: target_copy,
            valid: valid_copy,
            loss: loss_copy,
        })
    }

    pub fn shape(&self) -> [usize; 2] {
        [self.rows, self.width]
    }

    pub fn input(&self) -> &[TokenId] {
        &self.input
    }

    pub fn target(&self) -> &[TokenId] {
        &self.target
    }

    pub fn valid_mask(&self) -> Option<&[u8]> {
        self.valid.as_deref()
    }

    pub fn loss_mask(&self) -> Option<&[u8]> {
        self.loss.as_deref()
    }

    fn new(config: BatchConfig, positions: usize) -> Result<Self> {
        let with_masks = || config.masks().then(|| allocate(positions)).transpose();
        Ok(Self {
            rows: 0,
            width: config.sequence_length,
            input: allocate(positions)?,
            target: allocate(positions)?,
            valid: with_masks()?,
            loss: with_masks()?,
        })
    }

    /// Retained vector capacity for inputs, targets, and masks.
    pub fn retained_bytes(&self) -> Result<usize> {
        let mut used = add(capacity(&self.input)?, capacity(&self.target)?)?;
        for mask in [&self.valid, &self.loss].into_iter().flatten() {
            used = add(used, capacity(mask)?)?;
        }
        Ok(used)
    }

    fn push(&mut self, input: &[TokenId], target: &[TokenId]) -> Result<()> {
        if input.len() != target.len()
            || input.len() > self.width
            || (input.len() != self.width && self.valid.is_none())
        {
            return Err(InvalidTokens);
        }
        let end = add(self.input.len(), self.width)?;
        if end > self.input.capacity() || end > self.target.capacity() {
            return Err(BufferLimit);
        }
        self.input.extend_from_slice(input);
        self.target.extend_from_slice(target);
        self.input.resize(end, SpecialToken::Pad.id());
        self.target.resize(end, SpecialToken::Pad.id());
        for mask in [&mut self.valid, &mut self.loss].into_iter().flatten() {
            let valid_end = add(mask.len(), input.len())?;
            if end > mask.capacity() {
                return Err(BufferLimit);
            }
            mask.resize(valid_end, 1);
            mask.resize(end, 0);
        }
        self.rows = add(self.rows, 1)?;
        Ok(())
    }
}

pub struct BatchStream<'a> {
    reader: DatasetReader<'a>,
    model: &'a Tokenizer,
    config: StreamConfig,
    positions: usize,
    buffer_limit: usize,
    tokens: Vec<TokenId>,
    start: usize,
    stats: StreamStats,
    done: bool,
    failure: Option<DatasetError>,
    order_identity: [u8; 32],
}

impl Dataset {
    /// Stream with the fixed byte tokenizer.
    pub fn batches<'a>(
        &'a self,
        split: Split,
        model: &'a Tokenizer,
        config: StreamConfig,
    ) -> Result<BatchStream<'a>> {
        if !matches!(model, Tokenizer::Byte(_)) {
            return Err(InvalidFit);
        }
        self.batch_stream(split, model, config)
    }

    /// Stream with a BPE fit bound to this dataset and its train manifest.
    pub fn fitted_batches<'a>(
        &'a self,
        split: Split,
        fit: &'a FittedTokenizer,
        config: StreamConfig,
    ) -> Result<BatchStream<'a>> {
        if !self.is_bound(fit) {
            return Err(InvalidFit);
        }
        self.batch_stream(split, fit.model(), config)
    }

    fn batch_stream<'a>(
        &'a self,
        split: Split,
        model: &'a Tokenizer,
        config: StreamConfig,
    ) -> Result<BatchStream<'a>> {
        if !self.permits(split, model) {
            return Err(IncompatibleTokenizer);
        }
        config.tokenizer_limits.validate()?;
        let limits = self.limits();
        let positions = config.batch.positions(limits)?;
        check(
            batch_bytes(positions, config.batch.masks())?,
            limits.token_buffer_bytes,
            BufferLimit,
        )?;
        let order_identity = stream_identity(self.manifest_sha256(), split, model, config)?;
        Ok(BatchStream {
            reader: self.reader(split),
            model,
            config,
            positions,
            buffer_limit: limits.token_buffer_bytes,
            tokens: Vec::new(),
            start: 0,
            stats: StreamStats::default(),
            done: false,
            failure: None,
            order_identity,
        })
    }
}

fn batch_bytes(positions: usize, masks: bool) -> Result<usize> {
    mul(positions, 2 * (size_of::<TokenId>() + usize::from(masks)))
}

impl BatchStream<'_> {
    pub fn order_identity(&self) -> [u8; 32] {
        self.order_identity
    }

    pub fn stats(&self) -> StreamStats {
        self.stats
    }

    /// Conservative source-owned memory reservation: retained dataset index,
    /// maximum document and line buffers, and the configured token/batch area.
    pub fn reserved_memory_bytes(&self) -> Result<usize> {
        let bytes = self
            .reader
            .reserved_memory_bytes()?
            .checked_add(self.buffer_limit)
            .ok_or(Overflow)?;
        Ok(bytes)
    }

    /// Number of fully yielded batches in the deterministic stream.
    pub fn cursor(&self) -> u64 {
        self.stats.batches as u64
    }

    /// Rebuild stream state from its beginning and position at a committed
    /// batch boundary. Replayed reads are real work and callers must account
    /// for them in their run policy.
    pub fn seek_batches(&mut self, target: u64) -> Result<()> {
        self.reader.reset();
        self.tokens = Vec::new();
        self.start = 0;
        self.stats = StreamStats::default();
        self.done = false;
        self.failure = None;
        let mut skipped = 0_u64;
        while skipped < target {
            if self.next_batch()?.is_none() {
                return Err(InvalidConfiguration);
            }
            skipped = skipped.checked_add(1).ok_or(Overflow)?;
        }
        Ok(())
    }

    pub fn next_batch(&mut self) -> Result<Option<Batch>> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        if self.done {
            return Ok(None);
        }
        let stats = self.stats;
        let result = self.build_batch();
        if let Err(error) = result {
            self.tokens = Vec::new();
            self.stats = stats;
            self.failure = Some(error);
        }
        result
    }

    fn build_batch(&mut self) -> Result<Option<Batch>> {
        let config = self.config.batch;
        check(
            add(
                capacity(&self.tokens)?,
                batch_bytes(self.positions, config.masks())?,
            )?,
            self.buffer_limit,
            BufferLimit,
        )?;
        let mut batch = Batch::new(config, self.positions)?;
        let batch_bytes = batch.retained_bytes()?;
        self.observe_buffers(batch_bytes)?;
        while batch.rows < config.batch_size {
            let pairs = self
                .tokens
                .len()
                .saturating_sub(1)
                .saturating_sub(self.start);
            if pairs != 0 {
                if pairs >= config.sequence_length {
                    self.push_window(&mut batch, config.sequence_length)?;
                    self.start = add(self.start, config.stride)?;
                    continue;
                }
                match config.tail {
                    Tail::Drop => {
                        self.stats.dropped_tails = add(self.stats.dropped_tails, 1)?;
                        self.stats.dropped_positions = add(self.stats.dropped_positions, pairs)?;
                    }
                    Tail::Pad => {
                        self.push_window(&mut batch, pairs)?;
                        self.stats.padded_tails = add(self.stats.padded_tails, 1)?;
                        self.stats.padded_positions =
                            add(self.stats.padded_positions, config.sequence_length - pairs)?;
                    }
                }
            }
            self.tokens = Vec::new();
            self.start = 0;
            if batch.rows == config.batch_size {
                break;
            }
            let Some(document) = self.reader.next_document()? else {
                self.done = true;
                break;
            };
            let owned_budget = self
                .buffer_limit
                .checked_sub(batch_bytes)
                .ok_or(BufferLimit)?;
            let mut tokenizer_limits = self.config.tokenizer_limits;
            // Tokenizer charges its borrowed table too, so add it back when
            // translating the iterator-owned allowance into tokenizer limits.
            let allowed = add(owned_budget, self.model.retained_bytes()?)?;
            tokenizer_limits.buffer_bytes = tokenizer_limits.buffer_bytes.min(allowed);
            self.tokens = self.model.encode_with_specials(
                &document.bytes,
                self.config.boundaries,
                tokenizer_limits,
            )?;
            check_document_tokens(&self.tokens, self.model.vocabulary_size())?;
            self.stats.documents = add(self.stats.documents, 1)?;
            self.observe_buffers(batch_bytes)?;
        }
        if batch.rows == 0 {
            return Ok(None);
        }
        if config.pad_final_batch {
            let missing = config.batch_size - batch.rows;
            self.stats.padded_rows = add(self.stats.padded_rows, missing)?;
            self.stats.padded_positions = add(
                self.stats.padded_positions,
                mul(missing, config.sequence_length)?,
            )?;
            for _ in 0..missing {
                batch.push(&[], &[])?;
            }
        }
        self.stats.batches = add(self.stats.batches, 1)?;
        Ok(Some(batch))
    }

    fn observe_buffers(&mut self, batch_bytes: usize) -> Result<()> {
        let used = add(batch_bytes, capacity(&self.tokens)?)?;
        check(used, self.buffer_limit, BufferLimit)?;
        self.stats.peak_token_buffer_bytes = self.stats.peak_token_buffer_bytes.max(used);
        Ok(())
    }

    fn push_window(&mut self, batch: &mut Batch, pairs: usize) -> Result<()> {
        let end = add(self.start, pairs)?;
        batch.push(
            &self.tokens[self.start..end],
            &self.tokens[add(self.start, 1)?..add(end, 1)?],
        )?;
        self.stats.samples = add(self.stats.samples, 1)?;
        Ok(())
    }
}

fn stream_identity(
    manifest: [u8; 32],
    split: Split,
    model: &Tokenizer,
    config: StreamConfig,
) -> Result<[u8; 32]> {
    let artifact = xynoos_tokenizer::TokenizerArtifact::serialize(model, config.tokenizer_limits)?;
    let tokenizer: [u8; 32] = Sha256::digest(artifact).into();
    let mut hasher = Sha256::new();
    hasher.update(b"XYNOOS-DATA-ORDER-V1\0");
    hasher.update(manifest);
    hasher.update([match split {
        Split::Train => 0,
        Split::Validation => 1,
        Split::Test => 2,
    }]);
    hasher.update(tokenizer);
    for value in [
        config.batch.sequence_length as u64,
        config.batch.stride as u64,
        config.batch.batch_size as u64,
    ] {
        hasher.update(value.to_le_bytes());
    }
    hasher.update([match config.batch.tail {
        Tail::Drop => 0,
        Tail::Pad => 1,
    }]);
    hasher.update([u8::from(config.batch.pad_final_batch)]);
    hasher.update([
        u8::from(config.boundaries.bos),
        u8::from(config.boundaries.eos),
    ]);
    Ok(hasher.finalize().into())
}

fn check_document_tokens(tokens: &[TokenId], vocab: usize) -> Result<()> {
    for (index, &id) in tokens.iter().enumerate() {
        let valid = match id {
            256 => index == 0,
            257 => index == tokens.len() - 1,
            258 | 259 => false,
            _ => (id as usize) < vocab,
        };
        if !valid {
            return Err(InvalidTokens);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_product_and_invalid_document_controls() {
        assert_eq!(
            BatchConfig::per_document(usize::MAX, 2).positions(DatasetLimits::MAX),
            Err(Overflow)
        );
        for tokens in [[259, 7], [258, 7], [7, 256], [257, 7], [260, 7]] {
            assert_eq!(check_document_tokens(&tokens, 260), Err(InvalidTokens));
        }
    }
}
