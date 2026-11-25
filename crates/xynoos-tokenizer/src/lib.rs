//! Exact byte tokenization and a bounded deterministic byte-level BPE reference.
//!
//! ```
//! use xynoos_tokenizer::{Tokenizer, TokenizerLimits};
//!
//! let tokenizer = Tokenizer::byte();
//! let input = b"data\t\xff\r\n";
//! let limits = TokenizerLimits::default();
//! let ids = tokenizer.encode_bytes(input, limits)?;
//! assert_eq!(tokenizer.decode_bytes(&ids, limits)?, input);
//! # Ok::<(), xynoos_tokenizer::TokenizerError>(())
//! ```
#![forbid(unsafe_code)]

mod artifact;
mod bpe;

pub use artifact::TokenizerArtifact;
pub use bpe::{BpeModel, BpeTrainer, Merge};

use std::{fmt, mem::size_of};

/// The common byte, special, and learned token ID type.
pub type TokenId = u32;
pub const BASE_VOCABULARY: usize = 260;
pub const FORMAT_VOCABULARY_MAX: usize = 65_536;
pub const REFERENCE_MERGES_MAX: usize = 512;
pub const REFERENCE_VOCABULARY_MAX: usize = BASE_VOCABULARY + REFERENCE_MERGES_MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SpecialToken {
    Bos = 256,
    Eos = 257,
    Pad = 258,
    DocSep = 259,
}

impl SpecialToken {
    pub const fn id(self) -> TokenId {
        self as TokenId
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Boundaries {
    pub bos: bool,
    pub eos: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub enum SpecialDecodePolicy {
    #[default]
    Reject,
    Omit,
    RenderRegisteredSpelling,
}

#[derive(Debug, Clone, Copy)]
pub struct TokenizerLimits {
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub output_tokens: usize,
    pub corpus_bytes: usize,
    pub corpus_documents: usize,
    pub merges: usize,
    pub expansion_bytes: usize,
    pub artifact_bytes: usize,
    pub training_work: usize,
    pub encoding_work: usize,
    pub buffer_bytes: usize,
}

impl TokenizerLimits {
    pub const MAX: Self = Self {
        input_bytes: 1 << 20,
        output_bytes: 1 << 20,
        output_tokens: 1 << 20,
        corpus_bytes: 128 << 10,
        corpus_documents: 4096,
        merges: REFERENCE_MERGES_MAX,
        expansion_bytes: 4096,
        artifact_bytes: 8 << 20,
        training_work: 64 << 20,
        encoding_work: 64 << 20,
        buffer_bytes: 16 << 20,
    };

    pub fn validate(self) -> Result<()> {
        let max = Self::MAX;
        if self.input_bytes > max.input_bytes
            || self.output_bytes > max.output_bytes
            || self.output_tokens > max.output_tokens
            || self.corpus_bytes > max.corpus_bytes
            || self.corpus_documents > max.corpus_documents
            || self.merges > max.merges
            || self.expansion_bytes > max.expansion_bytes
            || self.artifact_bytes > max.artifact_bytes
            || self.training_work > max.training_work
            || self.encoding_work > max.encoding_work
            || self.buffer_bytes > max.buffer_bytes
        {
            return Err(TokenizerError::InvalidLimits);
        }
        Ok(())
    }
}

impl Default for TokenizerLimits {
    fn default() -> Self {
        Self::MAX
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerError {
    InvalidLimits,
    InputTooLarge,
    OutputTooLarge,
    CorpusTooLarge,
    TooManyDocuments,
    WorkLimitExceeded,
    VocabularyLimitExceeded,
    ExpansionLimitExceeded,
    BufferLimitExceeded,
    InvalidTokenId,
    InvalidSpecialSequence,
    InvalidUtf8,
    MalformedArtifact,
    ArtifactTooLarge,
    UnsupportedVersion,
    UnsupportedAlgorithm,
    ChecksumMismatch,
    ArithmeticOverflow,
    AllocationFailed,
    Io(std::io::ErrorKind),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for TokenizerError {}

type Result<T> = std::result::Result<T, TokenizerError>;

#[derive(Debug, Default)]
pub struct ByteTokenizer;

#[derive(Debug)]
pub enum Tokenizer {
    Byte(ByteTokenizer),
    Bpe(BpeModel),
}

impl Tokenizer {
    pub const fn byte() -> Self {
        Self::Byte(ByteTokenizer)
    }

    pub fn algorithm(&self) -> &'static str {
        match self {
            Self::Byte(_) => "byte-v1",
            Self::Bpe(_) => "byte-bpe-v1",
        }
    }

    pub fn vocabulary_size(&self) -> usize {
        BASE_VOCABULARY + self.merges().len()
    }

    pub fn merges(&self) -> &[Merge] {
        match self {
            Self::Byte(_) => &[],
            Self::Bpe(model) => model.merges(),
        }
    }

    pub fn retained_bytes(&self) -> Result<usize> {
        match self {
            Self::Byte(_) => Ok(0),
            Self::Bpe(model) => model.retained_bytes(),
        }
    }

    pub fn encode_text(&self, text: &str, limits: TokenizerLimits) -> Result<Vec<TokenId>> {
        self.encode_bytes(text.as_bytes(), limits)
    }

    pub fn encode_bytes(&self, bytes: &[u8], limits: TokenizerLimits) -> Result<Vec<TokenId>> {
        self.encode_with_specials(bytes, Boundaries::default(), limits)
    }

    pub fn encode_with_specials(
        &self,
        bytes: &[u8],
        boundaries: Boundaries,
        limits: TokenizerLimits,
    ) -> Result<Vec<TokenId>> {
        use TokenizerError::*;
        limits.validate()?;
        if bytes.len() > limits.input_bytes {
            return Err(InputTooLarge);
        }
        let extra = usize::from(boundaries.bos) + usize::from(boundaries.eos);
        let is_byte = matches!(self, Self::Byte(_));
        let capacity = if is_byte {
            add(bytes.len(), extra)?
        } else {
            bytes.len()
        };
        let retained = self.retained_bytes()?;
        check_capacity(
            add(retained, mul::<TokenId>(capacity)?)?,
            limits.buffer_bytes,
        )?;
        if is_byte && capacity > limits.output_tokens {
            return Err(OutputTooLarge);
        }
        let mut tokens = allocate(capacity)?;
        check_capacity(add(retained, bytes_used(&tokens)?)?, limits.buffer_bytes)?;
        tokens.extend(bytes.iter().map(|&byte| TokenId::from(byte)));
        if let Self::Bpe(model) = self {
            model.encode(&mut tokens, limits.encoding_work)?;
        }
        let output = add(tokens.len(), extra)?;
        if output > limits.output_tokens {
            return Err(OutputTooLarge);
        }
        if output > tokens.capacity() {
            check_capacity(add(retained, mul::<TokenId>(output)?)?, limits.buffer_bytes)?;
            reserve(&mut tokens, extra)?;
            check_capacity(add(retained, bytes_used(&tokens)?)?, limits.buffer_bytes)?;
        }
        if boundaries.bos {
            tokens.insert(0, SpecialToken::Bos.id());
        }
        if boundaries.eos {
            tokens.push(SpecialToken::Eos.id());
        }
        Ok(tokens)
    }

    pub fn decode_bytes(&self, tokens: &[TokenId], limits: TokenizerLimits) -> Result<Vec<u8>> {
        self.decode(tokens, SpecialDecodePolicy::Reject, limits)
    }

    pub fn decode_text(
        &self,
        tokens: &[TokenId],
        policy: SpecialDecodePolicy,
        limits: TokenizerLimits,
    ) -> Result<String> {
        String::from_utf8(self.decode(tokens, policy, limits)?)
            .map_err(|_| TokenizerError::InvalidUtf8)
    }

    /// Decode a single document with optional leading BOS and trailing EOS.
    /// PAD and DOC_SEP are outside the document grammar.
    pub fn decode(
        &self,
        tokens: &[TokenId],
        policy: SpecialDecodePolicy,
        limits: TokenizerLimits,
    ) -> Result<Vec<u8>> {
        use TokenizerError::*;
        limits.validate()?;
        if tokens.len() > limits.output_tokens {
            return Err(InputTooLarge);
        }
        let expansion = |position: usize, id: TokenId| match id {
            0..=255 => Ok(&BYTE_VALUES[id as usize..=id as usize]),
            256..=259 => {
                if matches!(policy, SpecialDecodePolicy::Reject) {
                    return Err(InvalidSpecialSequence);
                }
                let spelling: &[u8] = match id {
                    256 if position == 0 => b"<|bos|>",
                    257 if position == tokens.len() - 1 => b"<|eos|>",
                    _ => return Err(InvalidSpecialSequence),
                };
                Ok(if matches!(policy, SpecialDecodePolicy::Omit) {
                    b""
                } else {
                    spelling
                })
            }
            _ => {
                let bytes = self
                    .merges()
                    .get(id as usize - BASE_VOCABULARY)
                    .map(Merge::expansion)
                    .ok_or(InvalidTokenId)?;
                if bytes.len() > limits.expansion_bytes {
                    return Err(ExpansionLimitExceeded);
                }
                Ok(bytes)
            }
        };
        let mut size = 0;
        for (position, &id) in tokens.iter().enumerate() {
            size = add(size, expansion(position, id)?.len())?;
            if size > limits.output_bytes {
                return Err(OutputTooLarge);
            }
        }
        let retained = self.retained_bytes()?;
        check_capacity(add(retained, size)?, limits.buffer_bytes)?;
        let mut result = allocate(size)?;
        check_capacity(add(retained, bytes_used(&result)?)?, limits.buffer_bytes)?;
        for (position, &id) in tokens.iter().enumerate() {
            result.extend_from_slice(expansion(position, id)?);
        }
        Ok(result)
    }
}

pub(crate) static BYTE_VALUES: [u8; 256] = {
    let mut bytes = [0; 256];
    let mut index = 0;
    while index < bytes.len() {
        bytes[index] = index as u8;
        index += 1;
    }
    bytes
};

fn add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or(TokenizerError::ArithmeticOverflow)
}

fn mul<T>(count: usize) -> Result<usize> {
    count
        .checked_mul(size_of::<T>())
        .ok_or(TokenizerError::ArithmeticOverflow)
}

fn bytes_used<T>(vector: &Vec<T>) -> Result<usize> {
    mul::<T>(vector.capacity())
}

fn allocate<T>(capacity: usize) -> Result<Vec<T>> {
    let mut vector = Vec::new();
    reserve(&mut vector, capacity)?;
    Ok(vector)
}

fn reserve<T>(vector: &mut Vec<T>, count: usize) -> Result<()> {
    vector
        .try_reserve_exact(count)
        .map_err(|_| TokenizerError::AllocationFailed)
}

fn check_capacity(used: usize, limit: usize) -> Result<()> {
    if used > limit {
        return Err(TokenizerError::BufferLimitExceeded);
    }
    Ok(())
}

fn charge(remaining: &mut usize, amount: usize) -> Result<()> {
    *remaining = remaining
        .checked_sub(amount)
        .ok_or(TokenizerError::WorkLimitExceeded)?;
    Ok(())
}
