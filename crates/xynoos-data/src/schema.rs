use std::{
    fmt,
    io::{self, Write},
};

use serde::{
    Deserialize, Serialize,
    de::{DeserializeOwned, IgnoredAny, SeqAccess, Visitor},
};

use crate::{DatasetError::*, DatasetLimits, Result, add, capacity, check, reserve};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "kebab-case")]
pub enum Domain {
    English,
    Indonesian,
    Linux,
    Networking,
    Programming,
    Unicode,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Split {
    Train,
    Validation,
    Test,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum TokenizerId {
    Unassigned,
    ByteV1,
    ByteBpeV1,
}

impl TokenizerId {
    pub(crate) fn permits(self, algorithm: &str) -> bool {
        match self {
            Self::Unassigned => true,
            Self::ByteV1 => algorithm == "byte-v1",
            Self::ByteBpeV1 => algorithm == "byte-bpe-v1",
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TrainingPermission {
    Approved,
    Prohibited,
    Unknown,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RedistributionPermission {
    Approved,
    Conditional,
    Prohibited,
    Unknown,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReviewStatus {
    Approved,
    Pending,
    Rejected,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CleaningStatus {
    Original,
    Transformed,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Transformation {
    pub id: String,
    pub version: u32,
    pub parameters: String,
    pub responsible_process: String,
    pub input_sha256: String,
    pub output_sha256: String,
}

impl Transformation {
    fn valid(&self) -> bool {
        // Version 1 recognizes a logged UTF-8 BOM removal only.
        self.id == "remove-utf8-bom"
            && self.version == 1
            && self.parameters == "none"
            && identifier(&self.responsible_process)
            && digest_text(&self.input_sha256)
            && digest_text(&self.output_sha256)
            && self.input_sha256 != self.output_sha256
    }

    fn heap_bytes(&self) -> Result<usize> {
        sum([
            &self.id,
            &self.parameters,
            &self.responsible_process,
            &self.input_sha256,
            &self.output_sha256,
        ])
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceRecord {
    pub schema_version: u32,
    pub source_id: String,
    pub source_reference: String,
    pub source_revision: String,
    pub acquired_utc_date: String,
    pub source_sha256: String,
    pub license_id: String,
    pub license_version: String,
    pub terms_reference: String,
    pub training_permission: TrainingPermission,
    pub redistribution_permission: RedistributionPermission,
    pub redistribution_conditions: String,
    #[serde(deserialize_with = "bounded_list::<_, _, 16>")]
    pub transformation_history: Vec<Transformation>,
    pub domain: Domain,
    pub responsible_maintainer: String,
    pub review_status: ReviewStatus,
    pub reviewed_utc_date: String,
}

impl SourceRecord {
    pub(crate) fn validate(&self) -> Result<()> {
        version(self.schema_version)?;
        let redistribution_valid = match self.redistribution_permission {
            RedistributionPermission::Approved | RedistributionPermission::Conditional => {
                reference(&self.redistribution_conditions)
            }
            RedistributionPermission::Prohibited => self.redistribution_conditions == "prohibited",
            RedistributionPermission::Unknown => false,
        };
        if !identifier(&self.source_id)
            || !reference(&self.source_reference)
            || !identifier(&self.source_revision)
            || !date(&self.acquired_utc_date)
            || !date(&self.reviewed_utc_date)
            || self.reviewed_utc_date < self.acquired_utc_date
            || !digest_text(&self.source_sha256)
            || !identifier(&self.license_id)
            || !identifier(&self.license_version)
            || !reference(&self.terms_reference)
            || !identifier(&self.responsible_maintainer)
            || !matches!(self.training_permission, TrainingPermission::Approved)
            || !matches!(self.review_status, ReviewStatus::Approved)
            || !redistribution_valid
            || !self
                .transformation_history
                .iter()
                .all(Transformation::valid)
        {
            return Err(InvalidProvenance);
        }
        for (i, transform) in self.transformation_history.iter().enumerate() {
            if self.transformation_history[..i]
                .iter()
                .any(|old| old.id == transform.id && old.output_sha256 == transform.output_sha256)
            {
                return Err(InvalidProvenance);
            }
        }
        Ok(())
    }

    pub(crate) fn heap_bytes(&self) -> Result<usize> {
        let strings = sum([
            &self.source_id,
            &self.source_reference,
            &self.source_revision,
            &self.acquired_utc_date,
            &self.source_sha256,
            &self.license_id,
            &self.license_version,
            &self.terms_reference,
            &self.redistribution_conditions,
            &self.responsible_maintainer,
            &self.reviewed_utc_date,
        ])?;
        self.transformation_history.iter().try_fold(
            add(strings, capacity(&self.transformation_history)?)?,
            |total, entry| add(total, entry.heap_bytes()?),
        )
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DocumentRecord {
    pub schema_version: u32,
    pub document_id: String,
    pub source_id: String,
    pub source_revision: String,
    pub payload_path: String,
    pub content_sha256: String,
    pub byte_length: usize,
    pub encoding: String,
    #[serde(deserialize_with = "bounded_list::<_, _, 8>")]
    pub domain_labels: Vec<Domain>,
    pub split: Split,
    pub cleaning_status: CleaningStatus,
    #[serde(deserialize_with = "bounded_list::<_, _, 16>")]
    pub transformation_ids: Vec<String>,
    pub tokenizer_compatibility: TokenizerId,
    pub duplicate_group_id: String,
}

impl DocumentRecord {
    pub(crate) fn validate(&self, limits: DatasetLimits) -> Result<()> {
        version(self.schema_version)?;
        if !identifier(&self.document_id)
            || !identifier(&self.source_id)
            || !identifier(&self.source_revision)
            || !digest_text(&self.content_sha256)
            || self.encoding != "utf-8"
            || self.domain_labels.is_empty()
            || !self.domain_labels.windows(2).all(|w| w[0] < w[1])
            || !matches!(
                (&self.cleaning_status, self.transformation_ids.as_slice()),
                (CleaningStatus::Original, []) | (CleaningStatus::Transformed, [_])
            )
        {
            return Err(InvalidSchema);
        }
        if self.duplicate_group_id != self.content_sha256 {
            return Err(InvalidSplit);
        }
        if self.byte_length == 0 {
            return Err(InvalidPayload);
        }
        check(self.byte_length, limits.document_bytes, DocumentLimit)?;
        if !valid_path(&self.payload_path) {
            return Err(InvalidPath);
        }
        Ok(())
    }

    pub(crate) fn check_source(&self, source: &SourceRecord) -> Result<()> {
        if self.source_revision != source.source_revision {
            return Err(InvalidProvenance);
        }
        if let [id] = self.transformation_ids.as_slice()
            && !source
                .transformation_history
                .iter()
                .any(|entry| entry.id == *id && entry.output_sha256 == self.content_sha256)
        {
            return Err(InvalidProvenance);
        }
        Ok(())
    }

    pub(crate) fn heap_bytes(&self) -> Result<usize> {
        let strings = sum([
            &self.document_id,
            &self.source_id,
            &self.source_revision,
            &self.payload_path,
            &self.content_sha256,
            &self.encoding,
            &self.duplicate_group_id,
        ])?;
        let lists = add(
            capacity(&self.domain_labels)?,
            capacity(&self.transformation_ids)?,
        )?;
        add(add(strings, lists)?, sum(&self.transformation_ids)?)
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetManifest {
    pub schema_version: u32,
    pub dataset_id: String,
    pub dataset_version: String,
    pub tokenizer_id: TokenizerId,
    pub split_algorithm: String,
    pub split_version: u32,
    pub split_seed: String,
    pub group_key_policy: String,
    pub record_count: usize,
    pub sources_sha256: String,
    pub documents_sha256: String,
}

impl DatasetManifest {
    pub(crate) fn validate(&self, limits: DatasetLimits) -> Result<()> {
        version(self.schema_version)?;
        version(self.split_version)?;
        if !digest_text(&self.dataset_id)
            || !digest_text(&self.sources_sha256)
            || !digest_text(&self.documents_sha256)
            || !identifier(&self.dataset_version)
            || !identifier(&self.split_seed)
            || self.split_algorithm != "sha256-group"
            || self.group_key_policy != "content-sha256"
        {
            return Err(InvalidSchema);
        }
        check(self.record_count, limits.entries, MetadataLimit)
    }

    pub(crate) fn heap_bytes(&self) -> Result<usize> {
        sum([
            &self.dataset_id,
            &self.dataset_version,
            &self.split_algorithm,
            &self.split_seed,
            &self.group_key_policy,
            &self.sources_sha256,
            &self.documents_sha256,
        ])
    }
}

fn sum<'a>(strings: impl IntoIterator<Item = &'a String>) -> Result<usize> {
    strings
        .into_iter()
        .try_fold(0, |total, s| add(total, s.capacity()))
}

pub(crate) fn version(value: u32) -> Result<()> {
    if value == 1 {
        Ok(())
    } else {
        Err(UnsupportedVersion)
    }
}

pub(crate) fn digest_text(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub(crate) fn identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 96
        && value.as_bytes()[0].is_ascii_alphanumeric()
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
        && ![
            "unknown",
            "pending",
            "unreviewed",
            "none",
            "na",
            "n/a",
            "tbd",
            "unspecified",
        ]
        .iter()
        .any(|word| value.eq_ignore_ascii_case(word))
}

fn reference(value: &str) -> bool {
    if let Some(url) = value.strip_prefix("https://") {
        let host = url.split('/').next().unwrap_or_default();
        value.len() <= 512
            && host.contains('.')
            && host
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b".-".contains(&b))
            && url
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"./_-".contains(&b))
    } else {
        identifier(value)
    }
}

fn date(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 10
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes
            .iter()
            .enumerate()
            .any(|(i, b)| i != 4 && i != 7 && !b.is_ascii_digit())
    {
        return false;
    }
    let year: u16 = value[..4].parse().unwrap_or(0);
    let month: u8 = value[5..7].parse().unwrap_or(0);
    let day: u8 = value[8..].parse().unwrap_or(0);
    let last_day = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400)) => {
            29
        }
        2 => 28,
        _ => 0,
    };
    year != 0 && (1..=last_day).contains(&day)
}

pub(crate) fn valid_path(value: &str) -> bool {
    value.len() <= 512
        && value.split('/').all(|part| {
            let stem = part.split('.').next().unwrap_or_default();
            let device = [
                "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7",
                "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8",
                "LPT9",
            ];
            !part.is_empty()
                && part.len() <= 128
                && part.as_bytes()[0].is_ascii_alphanumeric()
                && !part.ends_with('.')
                && part
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
                && !device.iter().any(|name| stem.eq_ignore_ascii_case(name))
        })
}

fn bounded_list<'de, D: serde::Deserializer<'de>, T: Deserialize<'de>, const MAX: usize>(
    deserializer: D,
) -> std::result::Result<Vec<T>, D::Error> {
    struct List<T, const MAX: usize>(std::marker::PhantomData<T>);
    impl<'de, T: Deserialize<'de>, const MAX: usize> Visitor<'de> for List<T, MAX> {
        type Value = Vec<T>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "an array of at most {MAX} items")
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<Vec<T>, A::Error> {
            use serde::de::Error as _;
            let mut values = Vec::new();
            while values.len() < MAX {
                let Some(value) = seq.next_element()? else {
                    return Ok(values);
                };
                reserve(&mut values, 1).map_err(|_| A::Error::custom("array allocation failed"))?;
                values.push(value);
            }
            if seq.next_element::<IgnoredAny>()?.is_some() {
                return Err(A::Error::custom("array limit exceeded"));
            }
            Ok(values)
        }
    }
    deserializer.deserialize_seq(List::<T, MAX>(std::marker::PhantomData))
}

/// Serialize a fixed-schema record with its required final LF.
pub fn canonical_line<T: Serialize>(record: &T, limits: DatasetLimits) -> Result<Vec<u8>> {
    limits.validate()?;
    let mut writer = CanonicalWriter {
        bytes: Vec::new(),
        max: limits.record_bytes,
        error: None,
    };
    let status = serde_json::to_writer(&mut writer, record);
    status.map_err(|_| writer.error.unwrap_or(InvalidJson))?;
    writer
        .write_all(b"\n")
        .map_err(|_| writer.error.unwrap_or(InvalidJson))?;
    Ok(writer.bytes)
}

struct CanonicalWriter {
    bytes: Vec<u8>,
    max: usize,
    error: Option<crate::DatasetError>,
}

impl Write for CanonicalWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let result = (|| {
            let needed = add(self.bytes.len(), bytes.len())?;
            check(needed, self.max, RecordLimit)?;
            if needed > self.bytes.capacity() {
                let wanted = self
                    .bytes
                    .capacity()
                    .saturating_mul(2)
                    .max(needed)
                    .min(self.max);
                let additional = wanted - self.bytes.len();
                reserve(&mut self.bytes, additional)?;
                check(self.bytes.capacity(), self.max, RecordLimit)?;
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        })();
        result.map_err(|error| {
            self.error = Some(error);
            io::Error::from(io::ErrorKind::InvalidData)
        })
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(crate) fn parse_line<T: DeserializeOwned + Serialize>(line: &[u8]) -> Result<T> {
    let body = line.strip_suffix(b"\n").ok_or(TruncatedRecord)?;
    std::str::from_utf8(body).map_err(|_| InvalidUtf8)?;
    let record = serde_json::from_slice(body).map_err(|_| InvalidJson)?;
    let mut compare = Compare(body);
    serde_json::to_writer(&mut compare, &record).map_err(|_| NonCanonical)?;
    if !compare.0.is_empty() {
        return Err(NonCanonical);
    }
    Ok(record)
}

struct Compare<'a>(&'a [u8]);

impl Write for Compare<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 = self
            .0
            .strip_prefix(bytes)
            .ok_or(io::ErrorKind::InvalidData)?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
