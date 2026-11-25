use std::{
    fmt::{self, Write as _},
    io::{self, Read},
};

use sha2::{Digest, Sha256};

use crate::{
    BASE_VOCABULARY, BpeModel, FORMAT_VOCABULARY_MAX, Result, Tokenizer, TokenizerError,
    TokenizerLimits, add, bytes_used, check_capacity, reserve,
};

use TokenizerError::*;

const HEADER: &str = "XYNOOS-TOKENIZER\t1";
const SPECIALS: [&str; 4] = [
    "special\t256\tBOS\t<|bos|>",
    "special\t257\tEOS\t<|eos|>",
    "special\t258\tPAD\t<|pad|>",
    "special\t259\tDOC_SEP\t<|doc_sep|>",
];
const CHECKSUM_PREFIX: &str = "sha256\t";

/// Canonical version-1 TSV codec. Loading succeeds only with a complete valid model.
pub struct TokenizerArtifact;

impl TokenizerArtifact {
    pub fn serialize(model: &Tokenizer, limits: TokenizerLimits) -> Result<Vec<u8>> {
        limits.validate()?;
        if model.merges().len() > limits.merges {
            return Err(VocabularyLimitExceeded);
        }
        let available = limits
            .buffer_bytes
            .checked_sub(model.retained_bytes()?)
            .ok_or(BufferLimitExceeded)?;
        let mut out = BoundedBytes::new(limits.artifact_bytes, available);
        out.formatted(format_args!(
            "{HEADER}\nalgorithm\t{}\nvocab_size\t{}\n",
            model.algorithm(),
            model.vocabulary_size()
        ))?;
        for entry in SPECIALS {
            out.formatted(format_args!("{entry}\n"))?;
        }
        for (rank, merge) in model.merges().iter().enumerate() {
            if merge.expansion().len() > limits.expansion_bytes {
                return Err(ExpansionLimitExceeded);
            }
            let (left, right) = merge.pair();
            let id = add(BASE_VOCABULARY, rank)?;
            out.formatted(format_args!("merge\t{rank}\t{left}\t{right}\t{id}\t"))?;
            out.hex(merge.expansion())?;
            out.append(b"\n")?;
        }
        let digest = Sha256::digest(&out.bytes);
        out.append(CHECKSUM_PREFIX.as_bytes())?;
        out.hex(&digest)?;
        out.append(b"\n")?;
        Ok(out.bytes)
    }

    /// Bounded read to EOF, followed by complete checksum and semantic validation.
    pub fn read(mut input: impl Read, limits: TokenizerLimits) -> Result<Tokenizer> {
        limits.validate()?;
        // The staging buffer is charged along with the owned artifact bytes.
        let mut staging = [0; 4096];
        let available = limits
            .buffer_bytes
            .checked_sub(staging.len())
            .ok_or(BufferLimitExceeded)?;
        let mut data = BoundedBytes::new(limits.artifact_bytes, available);
        loop {
            let remaining = limits.artifact_bytes - data.bytes.len();
            let chunk = add(remaining, 1)?.min(staging.len());
            match input.read(&mut staging[..chunk]) {
                Ok(0) => break,
                Ok(n) => data.append(&staging[..n])?,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(Io(error.kind())),
            }
        }
        let owned = add(bytes_used(&data.bytes)?, staging.len())?;
        let mut parse_limits = limits;
        parse_limits.buffer_bytes -= owned;
        Self::parse(&data.bytes, parse_limits)
    }

    pub fn parse(bytes: &[u8], limits: TokenizerLimits) -> Result<Tokenizer> {
        limits.validate()?;
        if bytes.len() > limits.artifact_bytes {
            return Err(ArtifactTooLarge);
        }
        let text = std::str::from_utf8(bytes).map_err(|_| MalformedArtifact)?;
        let text = text.strip_suffix('\n').ok_or(MalformedArtifact)?;
        let (body, trailer) = text.rsplit_once('\n').ok_or(MalformedArtifact)?;
        let checksum = trailer
            .strip_prefix(CHECKSUM_PREFIX)
            .ok_or(MalformedArtifact)?;
        if checksum.len() != 64 || !checksum.bytes().all(is_lower_hex) {
            return Err(MalformedArtifact);
        }
        let digest = Sha256::digest(&bytes[..add(body.len(), 1)?]);
        if digest
            .iter()
            .copied()
            .flat_map(hex_pair)
            .ne(checksum.bytes())
        {
            return Err(ChecksumMismatch);
        }
        let mut lines = body.split('\n');
        match lines.next() {
            Some(HEADER) => (),
            Some(line) if line.starts_with("XYNOOS-TOKENIZER\t") => return Err(UnsupportedVersion),
            _ => return Err(MalformedArtifact),
        }
        let bpe = match lines.next() {
            Some("algorithm\tbyte-v1") => false,
            Some("algorithm\tbyte-bpe-v1") => true,
            Some(line) if line.starts_with("algorithm\t") => return Err(UnsupportedAlgorithm),
            _ => return Err(MalformedArtifact),
        };
        let vocab = decimal(lines.next().and_then(|s| s.strip_prefix("vocab_size\t")))?;
        if vocab > FORMAT_VOCABULARY_MAX {
            return Err(VocabularyLimitExceeded);
        }
        let count = vocab
            .checked_sub(BASE_VOCABULARY)
            .ok_or(MalformedArtifact)?;
        if count > limits.merges {
            return Err(VocabularyLimitExceeded);
        }
        for exact in SPECIALS {
            if lines.next() != Some(exact) {
                return Err(MalformedArtifact);
            }
        }
        if !bpe {
            return if count == 0 && lines.next().is_none() {
                Ok(Tokenizer::byte())
            } else {
                Err(MalformedArtifact)
            };
        }
        let mut model = BpeModel::new(count, limits.buffer_bytes)?;
        for rank in 0..count {
            let mut parts = lines.next().ok_or(MalformedArtifact)?.split('\t');
            if parts.next() != Some("merge") || decimal(parts.next())? != rank {
                return Err(MalformedArtifact);
            }
            let left = decimal(parts.next())?;
            let right = decimal(parts.next())?;
            let id = add(BASE_VOCABULARY, rank)?;
            if left >= id || right >= id || decimal(parts.next())? != id {
                return Err(MalformedArtifact);
            }
            let hex = parts.next().ok_or(MalformedArtifact)?;
            if parts.next().is_some() {
                return Err(MalformedArtifact);
            }
            if hex.len() > limits.expansion_bytes * 2 {
                return Err(ExpansionLimitExceeded);
            }
            model.append(
                (left as u32, right as u32),
                Some(hex.as_bytes()),
                limits.expansion_bytes,
                limits.buffer_bytes,
            )?;
        }
        if lines.next().is_some() {
            return Err(MalformedArtifact);
        }
        Ok(Tokenizer::Bpe(model))
    }
}

fn decimal(text: Option<&str>) -> Result<usize> {
    let text = text.ok_or(MalformedArtifact)?;
    if text.is_empty()
        || (text.starts_with('0') && text.len() > 1)
        || !text.bytes().all(|b| b.is_ascii_digit())
    {
        return Err(MalformedArtifact);
    }
    text.parse().map_err(|_| MalformedArtifact)
}

fn is_lower_hex(byte: u8) -> bool {
    byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
}

pub(crate) fn hex_pair(byte: u8) -> [u8; 2] {
    const HEX: &[u8] = b"0123456789abcdef";
    [HEX[(byte >> 4) as usize], HEX[(byte & 15) as usize]]
}

struct BoundedBytes {
    bytes: Vec<u8>,
    artifact: usize,
    buffer: usize,
    error: Option<TokenizerError>,
}

impl BoundedBytes {
    fn new(artifact: usize, buffer: usize) -> Self {
        Self {
            bytes: Vec::new(),
            artifact,
            buffer,
            error: None,
        }
    }

    fn append(&mut self, bytes: &[u8]) -> Result<()> {
        let needed = add(self.bytes.len(), bytes.len())?;
        if needed > self.artifact {
            return Err(ArtifactTooLarge);
        }
        check_capacity(needed, self.buffer)?;
        if needed > self.bytes.capacity() {
            let growth = self
                .bytes
                .capacity()
                .saturating_mul(2)
                .min(self.buffer.min(self.artifact))
                .max(needed);
            let additional = growth - self.bytes.len();
            reserve(&mut self.bytes, additional)?;
        }
        check_capacity(bytes_used(&self.bytes)?, self.buffer.min(self.artifact))?;
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    fn hex(&mut self, bytes: &[u8]) -> Result<()> {
        for &byte in bytes {
            self.append(&hex_pair(byte))?;
        }
        Ok(())
    }

    fn formatted(&mut self, args: fmt::Arguments<'_>) -> Result<()> {
        let status = self.write_fmt(args);
        status.map_err(|_| self.error.take().unwrap_or(MalformedArtifact))
    }
}

impl fmt::Write for BoundedBytes {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.append(text.as_bytes()).map_err(|error| {
            self.error = Some(error);
            fmt::Error
        })
    }
}
