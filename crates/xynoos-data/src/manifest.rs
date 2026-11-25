use std::{
    fs::{self, File},
    io::{self, Read},
    mem::size_of,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use xynoos_tokenizer::{BpeTrainer, Tokenizer, TokenizerArtifact, TokenizerLimits};

use crate::{
    DatasetError::*,
    DatasetLimits, DatasetManifest, DocumentRecord, Result, SourceRecord, Split, add, allocate,
    capacity, check, mul, reserve,
    schema::{digest_text, identifier, parse_line, valid_path, version},
};

const IDENTITY_TAG: &[u8] = b"XYNOOS-DATASET-V1\0";

/// Version 1 uses the exact payload SHA-256 hex as the duplicate-group ID.
pub fn assign_split(seed: &str, group_id: &str) -> Result<Split> {
    if !identifier(seed) || !digest_text(group_id) {
        return Err(InvalidSplit);
    }
    let mut hasher = Sha256::new();
    hasher.update(b"1\0");
    hasher.update(seed.as_bytes());
    hasher.update(b"\0");
    hasher.update(group_id.as_bytes());
    let digest = hasher.finalize();
    let mut prefix = [0; 8];
    prefix.copy_from_slice(&digest[..8]);
    let scaled = u128::from(u64::from_be_bytes(prefix)) * 100;
    Ok(if scaled < (80u128 << 64) {
        Split::Train
    } else if scaled < (90u128 << 64) {
        Split::Validation
    } else {
        Split::Test
    })
}

/// A completely validated manifest and ordered metadata index. Selected payloads
/// are separately rechecked at point of use before any bytes are yielded.
pub struct Dataset {
    root: PathBuf,
    limits: DatasetLimits,
    manifest: DatasetManifest,
    manifest_sha256: [u8; 32],
    sources: Vec<SourceRecord>,
    records: Vec<DocumentRecord>,
    train_hash: [u8; 32],
    retained_metadata: usize,
}

pub struct ManifestValidator;

impl ManifestValidator {
    pub fn open(root: &Path, limits: DatasetLimits) -> Result<Dataset> {
        limits.validate()?;
        if !fs::symlink_metadata(root)?.is_dir() {
            return Err(InvalidPath);
        }
        let root = root.canonicalize()?;
        let mut registry = Lines::open(&root, "manifest.json", limits)?;
        let line = registry.next_line()?.ok_or(TruncatedRecord)?;
        let manifest = parse_line::<DatasetManifest>(line)?;
        let manifest_sha256 = Sha256::digest(line).into();
        if registry.next_line()?.is_some() {
            return Err(NonCanonical);
        }
        drop(registry);
        manifest.validate(limits)?;

        let mut identity = Sha256::new();
        identity.update(IDENTITY_TAG);
        let mut owned = add(manifest.heap_bytes()?, root.capacity())?;
        check(owned, limits.metadata_bytes, MetadataLimit)?;
        let mut sources = load_records::<SourceRecord>(
            &root,
            "sources.jsonl",
            &manifest.sources_sha256,
            limits,
            &mut identity,
            &mut owned,
            |row, _| {
                row.validate()?;
                row.heap_bytes()
            },
        )?;
        sources.sort_unstable_by(|a, b| a.source_id.cmp(&b.source_id));
        if sources
            .windows(2)
            .any(|pair| pair[0].source_id == pair[1].source_id)
        {
            return Err(DuplicateId);
        }
        let mut train = Sha256::new();
        let records = load_records::<DocumentRecord>(
            &root,
            "documents.jsonl",
            &manifest.documents_sha256,
            limits,
            &mut identity,
            &mut owned,
            |row, line| {
                row.validate(limits)?;
                let index = sources
                    .binary_search_by(|src| src.source_id.cmp(&row.source_id))
                    .map_err(|_| InvalidProvenance)?;
                row.check_source(&sources[index])?;
                if manifest.tokenizer_id != crate::TokenizerId::Unassigned
                    && row.tokenizer_compatibility != crate::TokenizerId::Unassigned
                    && row.tokenizer_compatibility != manifest.tokenizer_id
                {
                    return Err(IncompatibleTokenizer);
                }
                if row.split == Split::Train {
                    train.update(line);
                }
                row.heap_bytes()
            },
        )?;
        if records.len() != manifest.record_count
            || !matches_digest(&manifest.dataset_id, identity.finalize().as_ref())
        {
            return Err(IntegrityMismatch);
        }
        validate_order_and_split(
            &records,
            &manifest.split_seed,
            limits.metadata_bytes - owned,
        )?;
        for row in &records {
            safe_file_path(&root, &row.payload_path)?;
        }
        Ok(Dataset {
            root,
            limits,
            manifest,
            manifest_sha256,
            sources,
            records,
            train_hash: train.finalize().into(),
            retained_metadata: owned,
        })
    }
}

fn load_records<T: serde::de::DeserializeOwned + Serialize>(
    root: &Path,
    name: &str,
    expected: &str,
    limits: DatasetLimits,
    identity: &mut Sha256,
    owned: &mut usize,
    mut validate: impl FnMut(&T, &[u8]) -> Result<usize>,
) -> Result<Vec<T>> {
    let mut lines = Lines::open(root, name, limits)?;
    let length = lines.length;
    identity.update(length.to_be_bytes());
    let mut hash = Sha256::new();
    let mut output = Vec::new();
    let mut consumed = 0u64;
    let mut nested = 0;
    let prior = *owned;
    while let Some(line) = lines.next_line()? {
        if output.len() >= limits.entries {
            return Err(MetadataLimit);
        }
        consumed = consumed
            .checked_add(u64::try_from(line.len()).map_err(|_| Overflow)?)
            .ok_or(Overflow)?;
        if consumed > length {
            return Err(IntegrityMismatch);
        }
        let record = parse_line(line)?;
        let row_bytes = validate(&record, line)?;
        nested = add(nested, row_bytes)?;
        let already_owned = add(prior, nested)?;
        let needed = add(output.len(), 1)?;
        let max_vec = limits
            .metadata_bytes
            .checked_sub(already_owned)
            .ok_or(MetadataLimit)?
            / size_of::<T>();
        check(needed.max(output.capacity()), max_vec, MetadataLimit)?;
        if needed > output.capacity() {
            let wanted = output
                .capacity()
                .saturating_mul(2)
                .max(needed)
                .min(limits.entries.min(max_vec));
            let additional = wanted - output.len();
            reserve(&mut output, additional)?;
            check(output.capacity(), max_vec, MetadataLimit)?;
        }
        identity.update(line);
        hash.update(line);
        output.push(record);
    }
    if consumed != length || !matches_digest(expected, hash.finalize().as_ref()) {
        return Err(IntegrityMismatch);
    }
    *owned = add(add(prior, nested)?, capacity(&output)?)?;
    Ok(output)
}

fn validate_order_and_split(rows: &[DocumentRecord], seed: &str, available: usize) -> Result<()> {
    check(
        mul(rows.len(), size_of::<usize>())?,
        available,
        MetadataLimit,
    )?;
    let mut order = allocate(rows.len())?;
    check(capacity(&order)?, available, MetadataLimit)?;
    order.extend(0..rows.len());
    order.sort_unstable_by(|&a, &b| rows[a].document_id.cmp(&rows[b].document_id));
    if order
        .windows(2)
        .any(|w| rows[w[0]].document_id == rows[w[1]].document_id)
    {
        return Err(DuplicateId);
    }
    order.sort_unstable_by(|&a, &b| rows[a].content_sha256.cmp(&rows[b].content_sha256));
    for pair in order.windows(2) {
        let (a, b) = (&rows[pair[0]], &rows[pair[1]]);
        if a.content_sha256 == b.content_sha256
            && (a.split != b.split || a.byte_length != b.byte_length)
        {
            return Err(DuplicateLeakage);
        }
    }
    order.sort_unstable_by(|&a, &b| rows[a].payload_path.cmp(&rows[b].payload_path));
    for pair in order.windows(2) {
        let (a, b) = (&rows[pair[0]], &rows[pair[1]]);
        if a.payload_path == b.payload_path && a.content_sha256 != b.content_sha256 {
            return Err(IntegrityMismatch);
        }
    }
    for row in rows {
        if assign_split(seed, &row.content_sha256)? != row.split {
            return Err(InvalidSplit);
        }
    }
    Ok(())
}

impl Dataset {
    pub fn manifest(&self) -> &DatasetManifest {
        &self.manifest
    }

    pub fn manifest_sha256(&self) -> [u8; 32] {
        self.manifest_sha256
    }

    pub fn train_manifest_sha256(&self) -> [u8; 32] {
        self.train_hash
    }

    pub fn sources(&self) -> &[SourceRecord] {
        &self.sources
    }

    pub fn records(&self) -> &[DocumentRecord] {
        &self.records
    }

    pub fn retained_metadata_bytes(&self) -> usize {
        self.retained_metadata
    }

    pub fn reader(&self, split: Split) -> DatasetReader<'_> {
        DatasetReader {
            dataset: self,
            split,
            next: 0,
            done: false,
            failure: None,
        }
    }

    pub(crate) fn limits(&self) -> DatasetLimits {
        self.limits
    }

    pub(crate) fn permits(&self, split: Split, model: &Tokenizer) -> bool {
        self.manifest.tokenizer_id.permits(model.algorithm())
            && self
                .records
                .iter()
                .filter(|row| row.split == split)
                .all(|row| row.tokenizer_compatibility.permits(model.algorithm()))
    }

    /// Fit from the validated train selection, with integrity checked before each push.
    pub fn fit_bpe(
        &self,
        split: Split,
        merges: usize,
        limits: TokenizerLimits,
    ) -> Result<FittedTokenizer> {
        if split != Split::Train {
            return Err(HeldOutFit);
        }
        if !self.manifest.tokenizer_id.permits("byte-bpe-v1") {
            return Err(IncompatibleTokenizer);
        }
        let mut trainer = BpeTrainer::new(merges, limits)?;
        let mut reader = self.reader(split);
        let mut train_records = 0usize;
        while let Some(document) = reader.next_document()? {
            if !document
                .record
                .tokenizer_compatibility
                .permits("byte-bpe-v1")
            {
                return Err(IncompatibleTokenizer);
            }
            trainer.push_document(&document.bytes)?;
            train_records = add(train_records, 1)?;
        }
        let model = trainer.finish()?;
        let artifact = TokenizerArtifact::serialize(&model, limits)?;
        let record = FitRecord {
            schema_version: 1,
            dataset_id: copy_ascii(&self.manifest.dataset_id)?,
            training_split: Split::Train,
            training_manifest_sha256: hex_string(&self.train_hash)?,
            artifact_sha256: hex_string(&Sha256::digest(&artifact))?,
            requested_merges: merges,
            train_records,
        };
        Ok(FittedTokenizer {
            model,
            artifact,
            record,
        })
    }

    /// Check the canonical fit record's claims against the selected dataset and
    /// artifact. This verifies binding, without authenticating the producer.
    pub fn verify_fit(
        &self,
        fit_line: &[u8],
        artifact: &[u8],
        limits: TokenizerLimits,
    ) -> Result<FittedTokenizer> {
        check(fit_line.len(), self.limits.record_bytes, RecordLimit)?;
        let fit: FitRecord = parse_line(fit_line)?;
        version(fit.schema_version)?;
        if fit.training_split != Split::Train {
            return Err(HeldOutFit);
        }
        limits.validate()?;
        check(
            artifact.len(),
            limits.artifact_bytes,
            Tokenizer(xynoos_tokenizer::TokenizerError::ArtifactTooLarge),
        )?;
        if fit.dataset_id != self.manifest.dataset_id
            || !matches_digest(&fit.training_manifest_sha256, &self.train_hash)
            || !matches_digest(&fit.artifact_sha256, &Sha256::digest(artifact))
            || fit.requested_merges > limits.merges
            || fit.train_records
                != self
                    .records
                    .iter()
                    .filter(|doc| doc.split == Split::Train)
                    .count()
        {
            return Err(InvalidFit);
        }
        check(artifact.len(), limits.buffer_bytes, BufferLimit)?;
        let mut owned = allocate(artifact.len())?;
        let mut parse_limits = limits;
        parse_limits.buffer_bytes = limits
            .buffer_bytes
            .checked_sub(capacity(&owned)?)
            .ok_or(BufferLimit)?;
        let model = TokenizerArtifact::parse(artifact, parse_limits)?;
        if model.algorithm() != "byte-bpe-v1"
            || model.merges().len() > fit.requested_merges
            || !self.permits(Split::Train, &model)
        {
            return Err(InvalidFit);
        }
        owned.extend_from_slice(artifact);
        Ok(FittedTokenizer {
            model,
            artifact: owned,
            record: fit,
        })
    }

    pub(crate) fn is_bound(&self, fit: &FittedTokenizer) -> bool {
        fit.record.dataset_id == self.manifest.dataset_id
            && matches_digest(&fit.record.training_manifest_sha256, &self.train_hash)
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FitRecord {
    pub schema_version: u32,
    pub dataset_id: String,
    pub training_split: Split,
    pub training_manifest_sha256: String,
    pub artifact_sha256: String,
    pub requested_merges: usize,
    pub train_records: usize,
}

pub struct FittedTokenizer {
    model: Tokenizer,
    artifact: Vec<u8>,
    record: FitRecord,
}

impl FittedTokenizer {
    pub fn model(&self) -> &Tokenizer {
        &self.model
    }

    pub fn artifact(&self) -> &[u8] {
        &self.artifact
    }

    pub fn record(&self) -> &FitRecord {
        &self.record
    }
}

pub struct Document<'a> {
    pub record: &'a DocumentRecord,
    pub bytes: Vec<u8>,
}

/// Synchronous reader with sticky errors and a distinct clean EOF.
pub struct DatasetReader<'a> {
    dataset: &'a Dataset,
    split: Split,
    next: usize,
    done: bool,
    failure: Option<crate::DatasetError>,
}

impl<'a> DatasetReader<'a> {
    pub fn next_document(&mut self) -> Result<Option<Document<'a>>> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        if self.done {
            return Ok(None);
        }
        while let Some(row) = self.dataset.records.get(self.next) {
            self.next = add(self.next, 1)?;
            if row.split == self.split {
                let payload = read_payload(&self.dataset.root, row, self.dataset.limits);
                self.failure = payload.as_ref().err().copied();
                return payload.map(|bytes| Some(Document { record: row, bytes }));
            }
        }
        self.done = true;
        Ok(None)
    }
}

fn read_payload(root: &Path, row: &DocumentRecord, limits: DatasetLimits) -> Result<Vec<u8>> {
    check(row.byte_length, limits.document_bytes, DocumentLimit)?;
    let file = File::open(safe_file_path(root, &row.payload_path)?)?;
    let meta = file.metadata()?;
    if !meta.is_file() || meta.len() != u64::try_from(row.byte_length).map_err(|_| Overflow)? {
        return Err(IntegrityMismatch);
    }
    let mut bytes = allocate(row.byte_length)?;
    check(capacity(&bytes)?, limits.document_bytes, DocumentLimit)?;
    bytes.resize(row.byte_length, 0);
    let mut file = file;
    file.read_exact(&mut bytes).map_err(|error| {
        if error.kind() == io::ErrorKind::UnexpectedEof {
            IntegrityMismatch
        } else {
            error.into()
        }
    })?;
    let mut probe = [0];
    if read_some(&mut file, &mut probe)? != 0
        || !matches_digest(&row.content_sha256, &Sha256::digest(&bytes))
    {
        return Err(IntegrityMismatch);
    }
    std::str::from_utf8(&bytes).map_err(|_| InvalidUtf8)?;
    if bytes.starts_with(b"\xef\xbb\xbf") || bytes.contains(&0) {
        return Err(InvalidPayload);
    }
    Ok(bytes)
}

fn safe_file_path(root: &Path, relative: &str) -> Result<PathBuf> {
    if !valid_path(relative) {
        return Err(InvalidPath);
    }
    let mut path = PathBuf::new();
    path.try_reserve(add(add(root.as_os_str().len(), relative.len())?, 1)?)
        .map_err(|_| Allocation)?;
    path.push(root);
    let mut parts = relative.split('/').peekable();
    while let Some(part) = parts.next() {
        path.push(part);
        let kind = path.symlink_metadata()?.file_type();
        if kind.is_symlink()
            || (parts.peek().is_some() && !kind.is_dir())
            || (parts.peek().is_none() && !kind.is_file())
        {
            return Err(InvalidPath);
        }
    }
    Ok(path)
}

fn read_some(mut reader: impl Read, bytes: &mut [u8]) -> Result<usize> {
    loop {
        match reader.read(bytes) {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            result => return Ok(result?),
        }
    }
}

struct Lines<R = File> {
    file: R,
    length: u64,
    scratch: Vec<u8>,
    cursor: usize,
    end: usize,
    line: Vec<u8>,
    line_max: usize,
}

impl Lines<File> {
    fn open(root: &Path, name: &str, limits: DatasetLimits) -> Result<Self> {
        let file = File::open(safe_file_path(root, name)?)?;
        let meta = file.metadata()?;
        if !meta.is_file() {
            return Err(InvalidPath);
        }
        let length = meta.len();
        if length > u64::try_from(limits.metadata_bytes).map_err(|_| Overflow)? {
            return Err(MetadataLimit);
        }
        Self::new(file, length, limits)
    }
}

impl<R: Read> Lines<R> {
    fn new(file: R, length: u64, limits: DatasetLimits) -> Result<Self> {
        let mut scratch = allocate(limits.reader_bytes)?;
        check(capacity(&scratch)?, limits.reader_bytes, BufferLimit)?;
        scratch.resize(limits.reader_bytes, 0);
        Ok(Self {
            file,
            length,
            scratch,
            cursor: 0,
            end: 0,
            line: Vec::new(),
            line_max: limits.record_bytes,
        })
    }

    fn next_line(&mut self) -> Result<Option<&[u8]>> {
        self.line.clear();
        loop {
            if self.cursor == self.end {
                self.end = read_some(&mut self.file, &mut self.scratch)?;
                self.cursor = 0;
                if self.end == 0 {
                    return if self.line.is_empty() {
                        Ok(None)
                    } else {
                        Err(TruncatedRecord)
                    };
                }
            }
            let remaining = &self.scratch[self.cursor..self.end];
            let lf = remaining.iter().position(|&b| b == b'\n');
            let count = lf.map_or(remaining.len(), |i| i + 1);
            let needed = add(self.line.len(), count)?;
            check(needed, self.line_max, RecordLimit)?;
            if needed > self.line.capacity() {
                let wanted = self
                    .line
                    .capacity()
                    .saturating_mul(2)
                    .max(needed)
                    .min(self.line_max);
                let additional = wanted - self.line.len();
                reserve(&mut self.line, additional)?;
                check(capacity(&self.line)?, self.line_max, RecordLimit)?;
            }
            self.line.extend_from_slice(&remaining[..count]);
            self.cursor = add(self.cursor, count)?;
            if lf.is_some() {
                return Ok(Some(&self.line));
            }
        }
    }
}

fn copy_ascii(text: &str) -> Result<String> {
    let mut output = String::new();
    output
        .try_reserve_exact(text.len())
        .map_err(|_| Allocation)?;
    output.push_str(text);
    Ok(output)
}

fn hex_bytes(bytes: &[u8]) -> impl Iterator<Item = u8> + '_ {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    bytes
        .iter()
        .flat_map(|&byte| [HEX[(byte >> 4) as usize], HEX[(byte & 15) as usize]])
}

fn hex_string(bytes: &[u8]) -> Result<String> {
    let mut result = allocate(mul(bytes.len(), 2)?)?;
    result.extend(hex_bytes(bytes));
    String::from_utf8(result).map_err(|_| InvalidUtf8)
}

fn matches_digest(text: &str, digest: &[u8]) -> bool {
    text.bytes().eq(hex_bytes(digest))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_reads_eof_and_mid_record_io_are_distinct() {
        struct Broken(bool);

        impl Read for Broken {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                if std::mem::replace(&mut self.0, true) {
                    Err(io::ErrorKind::PermissionDenied.into())
                } else {
                    buffer[0] = b'{';
                    Ok(1)
                }
            }
        }

        let limits = DatasetLimits {
            reader_bytes: 1,
            record_bytes: 3,
            ..DatasetLimits::MAX
        };
        let mut valid = Lines::new(&b"{}\n"[..], 3, limits).unwrap();
        assert_eq!(valid.next_line(), Ok(Some(&b"{}\n"[..])));
        assert_eq!(valid.next_line(), Ok(None));
        let mut truncated = Lines::new(&b"{}"[..], 2, limits).unwrap();
        assert_eq!(truncated.next_line(), Err(TruncatedRecord));
        let mut broken = Lines::new(Broken(false), 3, limits).unwrap();
        assert_eq!(broken.next_line(), Err(Io(io::ErrorKind::PermissionDenied)));
        let mut exceeded = Lines::new(&b" {}\n"[..], 4, limits).unwrap();
        assert_eq!(exceeded.next_line(), Err(RecordLimit));
    }
}
