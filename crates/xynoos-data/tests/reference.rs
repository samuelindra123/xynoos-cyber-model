#![forbid(unsafe_code)]

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use xynoos_data::{
    BatchConfig, DatasetError, DatasetLimits, FittedTokenizer, ManifestValidator, Split,
    StreamConfig, StreamStats, Tail, canonical_line,
};
use xynoos_tokenizer::{
    Boundaries, BpeTrainer, SpecialToken, Tokenizer, TokenizerArtifact, TokenizerError,
    TokenizerLimits,
};

const RAW: &str = include_str!("fixtures/task004.json");
const TL: TokenizerLimits = TokenizerLimits::MAX;
const DL: DatasetLimits = DatasetLimits::MAX;
const SPLITS: [(Split, &str); 3] = [
    (Split::Train, "train"),
    (Split::Validation, "validation"),
    (Split::Test, "test"),
];

fn fixture() -> Value {
    let spec: Value = serde_json::from_str(RAW).unwrap();
    assert_eq!(spec["schema_version"], 1);
    spec
}

fn string(value: &Value) -> &str {
    value.as_str().unwrap()
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut output = String::new();
    for byte in bytes {
        write!(output, "{byte:02x}").unwrap();
    }
    output
}

fn unhex(text: &str) -> Vec<u8> {
    assert!(text.len().is_multiple_of(2));
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
}

fn sha(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
            "task004-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn materialize(&self, spec: &Value) -> PathBuf {
        let root = self.0.join("dataset");
        fs::create_dir_all(root.join("payload")).unwrap();
        for (path, payload) in spec["payloads"].as_object().unwrap() {
            fs::write(root.join(path), string(payload)).unwrap();
        }
        write_metadata(&root, spec);
        root
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn write_metadata(root: &Path, spec: &Value) {
    for (filename, key) in [
        ("sources.jsonl", "sources"),
        ("documents.jsonl", "documents"),
        ("manifest.json", "manifest"),
    ] {
        fs::write(root.join(filename), string(&spec[key])).unwrap();
    }
}

fn fixed_records(spec: &Value, root: &Path, sources: &str, docs: &str, count: usize) {
    // Preserve the field order of the independent sidecar, replacing exact scalar fields.
    let old_manifest = string(&spec["manifest"]);
    let old: Value = serde_json::from_str(old_manifest).unwrap();
    let mut identity = Sha256::new();
    identity.update(b"XYNOOS-DATASET-V1\0");
    for bytes in [sources.as_bytes(), docs.as_bytes()] {
        identity.update((bytes.len() as u64).to_be_bytes());
        identity.update(bytes);
    }
    let manifest = old_manifest
        .replace(string(&old["dataset_id"]), &hex(&identity.finalize()))
        .replace(string(&old["sources_sha256"]), &sha(sources.as_bytes()))
        .replace(string(&old["documents_sha256"]), &sha(docs.as_bytes()))
        .replace(
            &format!("\"record_count\":{}", old["record_count"]),
            &format!("\"record_count\":{count}"),
        );
    fs::write(root.join("sources.jsonl"), sources).unwrap();
    fs::write(root.join("documents.jsonl"), docs).unwrap();
    fs::write(root.join("manifest.json"), manifest).unwrap();
}

#[test]
fn independent_tokenizer_cases() {
    let spec = fixture();
    let baseline = Tokenizer::byte();
    assert_eq!(
        TokenizerArtifact::serialize(&baseline, TL).unwrap(),
        string(&spec["byte_artifact"]).as_bytes()
    );
    for case in spec["cases"].as_array().unwrap() {
        let mut trainer = BpeTrainer::new(8, TL).unwrap();
        for document in case["documents"].as_array().unwrap() {
            trainer.push_document(&unhex(string(document))).unwrap();
        }
        let model = trainer.finish().unwrap();
        let query = unhex(string(&case["query"]));
        let expected_artifact = string(&case["artifact"]).as_bytes();
        assert_eq!(
            TokenizerArtifact::serialize(&model, TL).unwrap(),
            expected_artifact
        );
        let loaded = TokenizerArtifact::read(expected_artifact, TL).unwrap();
        let expected = &case["bpe"];
        for selected in [&model, &loaded] {
            let tokens = selected.encode_bytes(&query, TL).unwrap();
            assert_eq!(json!(tokens), *expected);
            assert_eq!(selected.decode_bytes(&tokens, TL).unwrap(), query);
        }
        assert_eq!(
            json!(baseline.encode_bytes(&query, TL).unwrap()),
            case["baseline"]
        );
    }
}

fn counts(stats: StreamStats) -> Value {
    json!({
        "documents":stats.documents, "samples":stats.samples, "batches":stats.batches,
        "dropped_tails":stats.dropped_tails, "dropped_positions":stats.dropped_positions,
        "padded_tails":stats.padded_tails, "padded_positions":stats.padded_positions, "padded_rows":stats.padded_rows,
    })
}

fn experiment(spec: &Value, root: &Path) -> Value {
    let limits = DatasetLimits {
        reader_bytes: 7,
        ..DL
    };
    let dataset = ManifestValidator::open(root, limits).unwrap();
    assert_eq!(
        hex(&dataset.manifest_sha256()),
        string(&spec["manifest_sha256"])
    );
    assert!(dataset.retained_metadata_bytes() < DL.metadata_bytes);
    let fit = dataset.fit_bpe(Split::Train, 32, TL).unwrap();
    let fit_line = canonical_line(fit.record(), DL).unwrap();
    assert_eq!(fit_line, string(&spec["fit"]).as_bytes());
    assert_eq!(fit.artifact(), string(&spec["artifact"]).as_bytes());
    let bpe = dataset.verify_fit(&fit_line, fit.artifact(), TL).unwrap();
    assert_eq!(bpe.model().vocabulary_size(), 292);
    let baseline = Tokenizer::byte();
    let mut reports = serde_json::Map::new();
    let mut documents = Vec::new();
    let mut all_batches = serde_json::Map::new();
    for (split, label) in SPLITS {
        let mut reader = dataset.reader(split);
        let mut all_raw = Vec::new();
        let mut all_bpe = Vec::new();
        let mut count = 0;
        while let Some(document) = reader.next_document().unwrap() {
            let bytes = &document.bytes;
            let raw = baseline.encode_bytes(bytes, TL).unwrap();
            let encoded = bpe.model().encode_bytes(bytes, TL).unwrap();
            assert_eq!(baseline.decode_bytes(&raw, TL).unwrap(), *bytes);
            assert_eq!(bpe.model().decode_bytes(&encoded, TL).unwrap(), *bytes);
            let values = metrics(&raw, &encoded, bpe.model().vocabulary_size());
            documents.push(json!({"id":document.record.document_id, "split":label,
                "baseline":raw, "bpe":encoded, "metrics":values}));
            all_raw.extend(raw);
            all_bpe.extend(encoded);
            count += 1;
        }
        assert!(reader.next_document().unwrap().is_none());
        let values = metrics(&all_raw, &all_bpe, bpe.model().vocabulary_size());
        reports.insert(
            label.to_owned(),
            json!({"documents":count, "metrics":values}),
        );
        let mut runs = serde_json::Map::new();
        for (fit, family) in [(None, "baseline"), (Some(&bpe), "bpe")] {
            let drop = collect_batches(
                &dataset,
                split,
                &baseline,
                fit,
                BatchConfig::per_document(8, 3),
            );
            let pad = collect_batches(
                &dataset,
                split,
                &baseline,
                fit,
                BatchConfig {
                    sequence_length: 8,
                    stride: 3,
                    batch_size: 3,
                    tail: Tail::Pad,
                    pad_final_batch: true,
                },
            );
            runs.insert(family.to_owned(), json!({"drop":drop, "pad":pad}));
        }
        all_batches.insert(label.to_owned(), json!(runs));
    }
    json!({"per_document":documents, "splits":reports, "batches":all_batches})
}

fn metrics(raw: &[u32], bpe: &[u32], vocabulary: usize) -> Value {
    let distinct = |values: &[u32]| values.iter().copied().collect::<BTreeSet<_>>().len();
    let rate = |count| format!("{:.6}", count as f64 / raw.len() as f64);
    json!({"bytes":raw.len(), "byte_tokens":raw.len(), "bpe_tokens":bpe.len(),
        "byte_per_byte":rate(raw.len()), "bpe_per_byte":rate(bpe.len()),
        "byte_used":distinct(raw), "bpe_used":distinct(bpe), "byte_vocab":260, "bpe_vocab":vocabulary,
        "byte_round_trip":true, "bpe_round_trip":true})
}

fn collect_batches(
    dataset: &xynoos_data::Dataset,
    split: Split,
    baseline: &Tokenizer,
    fit: Option<&FittedTokenizer>,
    batch: BatchConfig,
) -> Value {
    let config = StreamConfig {
        batch,
        boundaries: Boundaries {
            bos: true,
            eos: true,
        },
        tokenizer_limits: TL,
    };
    let mut stream = match fit {
        Some(fit) => dataset.fitted_batches(split, fit, config),
        None => dataset.batches(split, baseline, config),
    }
    .unwrap();
    let mut result = Vec::new();
    while let Some(batch) = stream.next_batch().unwrap() {
        result.push(
            json!({"shape":batch.shape(), "input":batch.input(), "target":batch.target(),
            "valid":batch.valid_mask(), "loss":batch.loss_mask()}),
        );
    }
    assert!(stream.next_batch().unwrap().is_none());
    assert!(stream.stats().peak_token_buffer_bytes <= DL.token_buffer_bytes);
    json!({"batches":result, "stats":counts(stream.stats())})
}

#[test]
fn twice_matches_independent_experiment() {
    let spec = fixture();
    let first_dir = Scratch::new();
    let first = experiment(&spec, &first_dir.materialize(&spec));
    {
        let second_dir = Scratch::new();
        let second = experiment(&spec, &second_dir.materialize(&spec));
        assert_eq!(first, second);
    }
    for key in ["per_document", "splits", "batches"] {
        assert_eq!(first[key], spec[key], "{key}");
    }
    let measured = serde_json::to_vec(&first).unwrap();
    let hash = sha(&measured);
    assert_eq!(hash, string(&spec["experiment_sha256"]));
    println!("TASK004_EXPERIMENT {hash}");
}

#[test]
fn rejects_held_out_fit_and_binds_artifact() {
    let spec = fixture();
    let scratch = Scratch::new();
    let root = scratch.materialize(&spec);
    let dataset = ManifestValidator::open(&root, DL).unwrap();
    let fit = dataset.fit_bpe(Split::Train, 32, TL).unwrap();
    let cfg = StreamConfig {
        batch: BatchConfig::per_document(1, 1),
        boundaries: Boundaries::default(),
        tokenizer_limits: TL,
    };
    assert_eq!(
        dataset.batches(Split::Train, fit.model(), cfg).err(),
        Some(DatasetError::InvalidFit)
    );
    let mut reversed: Vec<_> = string(&spec["documents"]).lines().collect();
    reversed.reverse();
    fixed_records(
        &spec,
        &root,
        string(&spec["sources"]),
        &format!("{}\n", reversed.join("\n")),
        reversed.len(),
    );
    let reordered = ManifestValidator::open(&root, DL).unwrap();
    assert_eq!(
        reordered.fitted_batches(Split::Train, &fit, cfg).err(),
        Some(DatasetError::InvalidFit)
    );
    write_metadata(&root, &spec);
    for split in [Split::Validation, Split::Test] {
        assert_eq!(
            dataset.fit_bpe(split, 32, TL).err(),
            Some(DatasetError::HeldOutFit)
        );
        let selected = dataset
            .records()
            .iter()
            .find(|record| record.split == split)
            .unwrap();
        let path = root.join(&selected.payload_path);
        fs::write(path, b"synthetic held out integrity mutation").unwrap();
        assert_eq!(
            dataset.reader(split).next_document().err(),
            Some(DatasetError::IntegrityMismatch)
        );
    }
    let unchanged = dataset.fit_bpe(Split::Train, 32, TL).unwrap();
    assert_eq!(fit.artifact(), unchanged.artifact());
    for split in ["validation", "test"] {
        let false_fit = string(&spec["fit"]).replace(
            "\"training_split\":\"train\"",
            &format!("\"training_split\":\"{split}\""),
        );
        assert_eq!(
            dataset
                .verify_fit(false_fit.as_bytes(), fit.artifact(), TL)
                .err(),
            Some(DatasetError::HeldOutFit)
        );
    }
    assert_eq!(
        dataset
            .verify_fit(
                string(&spec["fit"]).as_bytes(),
                string(&spec["byte_artifact"]).as_bytes(),
                TL
            )
            .err(),
        Some(DatasetError::InvalidFit)
    );
}

#[test]
fn rejects_wrong_split_metadata_and_cross_held_out_duplicate() {
    let spec = fixture();
    let scratch = Scratch::new();
    let root = scratch.materialize(&spec);
    let rows = string(&spec["documents"]);
    let sources = string(&spec["sources"]);
    let mut lines: Vec<_> = rows.lines().map(|line| format!("{line}\n")).collect();
    let count = lines.len();
    let index = lines
        .iter()
        .position(|row| row.contains("\"document_id\":\"c\""))
        .unwrap();
    let original = lines[index].clone();
    let parsed: Value = serde_json::from_str(&original).unwrap();
    let split = string(&parsed["split"]);
    let wrong = if split == "train" { "test" } else { "train" };
    lines[index] = original.replace(
        &format!("\"split\":\"{split}\""),
        &format!("\"split\":\"{wrong}\""),
    );
    fixed_records(&spec, &root, sources, &lines.concat(), count);
    assert_eq!(
        ManifestValidator::open(&root, DL).err(),
        Some(DatasetError::InvalidSplit)
    );
    lines[index] = original;
    let test = lines
        .iter()
        .find(|line| line.contains("\"split\":\"test\""))
        .unwrap();
    let doc: Value = serde_json::from_str(test).unwrap();
    let duplicate = test
        .replace("\"split\":\"test\"", "\"split\":\"validation\"")
        .replace(
            &format!("\"document_id\":{}", doc["document_id"]),
            "\"document_id\":\"relabelled-duplicate\"",
        );
    let second_source = sources.replace(
        "\"source_id\":\"task004-synthetic\"",
        "\"source_id\":\"task004-second\"",
    );
    let registry = format!("{sources}{second_source}");
    lines.push(duplicate.replace(
        "\"source_id\":\"task004-synthetic\"",
        "\"source_id\":\"task004-second\"",
    ));
    fixed_records(&spec, &root, &registry, &lines.concat(), count + 1);
    assert_eq!(
        ManifestValidator::open(&root, DL).err(),
        Some(DatasetError::DuplicateLeakage)
    );
    let last = lines.last_mut().unwrap();
    *last = last.replace("\"split\":\"validation\"", "\"split\":\"test\"");
    fixed_records(&spec, &root, &registry, &lines.concat(), count + 1);
    assert!(ManifestValidator::open(&root, DL).is_ok());
    lines.pop();
    lines.reverse();
    fixed_records(&spec, &root, sources, &lines.concat(), count);
    let dataset = ManifestValidator::open(&root, DL).unwrap();
    assert_eq!(dataset.records()[0].document_id, "buffer-copy");
}

#[test]
fn rejects_noncanonical_metadata_rights_and_paths() {
    let spec = fixture();
    let scratch = Scratch::new();
    let root = scratch.materialize(&spec);
    let rows = string(&spec["documents"]);
    let sources = string(&spec["sources"]);
    let count = rows.lines().count();
    let duplicate_key = rows.replacen(
        "\"schema_version\":1",
        "\"schema_version\":1,\"schema_version\":1",
        1,
    );
    let wrong_order = rows.replacen(
        "\"schema_version\":1,\"document_id\":\"buffer\"",
        "\"document_id\":\"buffer\",\"schema_version\":1",
        1,
    );
    for (mutated, reason) in [
        (duplicate_key, DatasetError::InvalidJson),
        (wrong_order, DatasetError::NonCanonical),
        (
            rows.replacen("\"schema_version\":1", "\"schema_version\":2", 1),
            DatasetError::UnsupportedVersion,
        ),
        (
            rows.replacen(
                "\"source_id\":\"task004-synthetic\"",
                "\"source_id\":\"unresolved\"",
                1,
            ),
            DatasetError::InvalidProvenance,
        ),
        (
            rows.replacen("\"document_id\":\"queue\"", "\"document_id\":\"buffer\"", 1),
            DatasetError::DuplicateId,
        ),
        (
            rows.replacen(
                "\"domain_labels\":[\"english\"]",
                "\"domain_labels\":[\"unreviewed\"]",
                1,
            ),
            DatasetError::InvalidJson,
        ),
        (
            rows.replacen("payload/buffer.txt", "../synthetic-sentinel", 1),
            DatasetError::InvalidPath,
        ),
        (
            rows.replacen("payload/buffer.txt", "/synthetic-sentinel", 1),
            DatasetError::InvalidPath,
        ),
        (
            rows.replacen("payload/buffer.txt", "payload//buffer.txt", 1),
            DatasetError::InvalidPath,
        ),
        (
            rows.replacen("payload/buffer.txt", "C:/synthetic-sentinel", 1),
            DatasetError::InvalidPath,
        ),
    ] {
        fixed_records(&spec, &root, sources, &mutated, count);
        let error = ManifestValidator::open(&root, DL).err().unwrap();
        assert_eq!(error, reason);
        assert!(!error.to_string().contains("synthetic-sentinel"));
    }
    for mutated in [
        sources.replace(
            "\"training_permission\":\"approved\"",
            "\"training_permission\":\"prohibited\"",
        ),
        sources.replace(
            "\"review_status\":\"approved\"",
            "\"review_status\":\"pending\"",
        ),
        sources.replace(
            "\"terms_reference\":\"LICENSE\"",
            "\"terms_reference\":\"\"",
        ),
        sources.replace(
            "\"responsible_maintainer\":\"project-maintainers\"",
            "\"responsible_maintainer\":\"\"",
        ),
        sources.replace("2026-11-25", "2026-02-30"),
        sources.replace(
            "\"redistribution_permission\":\"approved\"",
            "\"redistribution_permission\":\"unknown\"",
        ),
    ] {
        fixed_records(&spec, &root, &mutated, rows, count);
        assert_eq!(
            ManifestValidator::open(&root, DL).err(),
            Some(DatasetError::InvalidProvenance)
        );
    }
    fixed_records(&spec, &root, &format!("{sources}{sources}"), rows, count);
    assert_eq!(
        ManifestValidator::open(&root, DL).err(),
        Some(DatasetError::DuplicateId)
    );

    fixed_records(&spec, &root, sources, rows.trim_end_matches('\n'), count);
    assert_eq!(
        ManifestValidator::open(&root, DL).err(),
        Some(DatasetError::TruncatedRecord)
    );
    write_metadata(&root, &spec);
    let manifest = string(&spec["manifest"]).replace("\"split_version\":1", "\"split_version\":2");
    fs::write(root.join("manifest.json"), manifest).unwrap();
    assert_eq!(
        ManifestValidator::open(&root, DL).err(),
        Some(DatasetError::UnsupportedVersion)
    );
    let manifest = string(&spec["manifest"]).replace("task004-synthetic-v1", "altered-seed");
    fs::write(root.join("manifest.json"), manifest).unwrap();
    assert_eq!(
        ManifestValidator::open(&root, DL).err(),
        Some(DatasetError::InvalidSplit)
    );
    write_metadata(&root, &spec);
    fs::write(
        root.join("sources.jsonl"),
        sources.replace("Apache-2.0-license-and-notice", "other-conditions"),
    )
    .unwrap();
    assert_eq!(
        ManifestValidator::open(&root, DL).err(),
        Some(DatasetError::IntegrityMismatch)
    );
}

#[test]
fn payload_integrity_utf8_and_bounds_precede_samples() {
    let spec = fixture();
    let scratch = Scratch::new();
    let root = scratch.materialize(&spec);
    let rows = string(&spec["documents"]);
    let sources = string(&spec["sources"]);
    let count = rows.lines().count();
    for (limits, expected) in [
        (
            DatasetLimits { entries: 1, ..DL },
            DatasetError::MetadataLimit,
        ),
        (
            DatasetLimits {
                metadata_bytes: 1500,
                ..DL
            },
            DatasetError::MetadataLimit,
        ),
        (
            DatasetLimits {
                record_bytes: 8,
                reader_bytes: 2,
                ..DL
            },
            DatasetError::RecordLimit,
        ),
        (
            DatasetLimits {
                document_bytes: 1,
                ..DL
            },
            DatasetError::DocumentLimit,
        ),
        (
            DatasetLimits {
                reader_bytes: DL.reader_bytes + 1,
                ..DL
            },
            DatasetError::InvalidLimits,
        ),
    ] {
        assert_eq!(ManifestValidator::open(&root, limits).err(), Some(expected));
    }
    let empty = rows.replacen("\"byte_length\":69", "\"byte_length\":0", 1);
    assert_ne!(empty, rows);
    fixed_records(&spec, &root, sources, &empty, count);
    assert_eq!(
        ManifestValidator::open(&root, DL).err(),
        Some(DatasetError::InvalidPayload)
    );
    fixed_records(&spec, &root, sources, "", 0);
    {
        let dataset = ManifestValidator::open(&root, DL).unwrap();
        assert!(
            dataset
                .reader(Split::Train)
                .next_document()
                .unwrap()
                .is_none()
        );
        assert_eq!(
            dataset
                .fit_bpe(Split::Train, 32, TL)
                .unwrap()
                .model()
                .vocabulary_size(),
            260
        );
    }
    // Rebind exact bytes so the UTF-8 and BOM checks run after successful integrity verification.
    let first = rows.lines().next().unwrap();
    let row: Value = serde_json::from_str(first).unwrap();
    for (payload, expected) in [
        (
            b"\xffprivate-synthetic-sentinel".as_slice(),
            DatasetError::InvalidUtf8,
        ),
        (
            b"\xef\xbb\xbfprivate-synthetic-sentinel".as_slice(),
            DatasetError::InvalidPayload,
        ),
    ] {
        let hash = sha(payload);
        let split = xynoos_data::assign_split("task004-synthetic-v1", &hash).unwrap();
        let label = SPLITS
            .iter()
            .find(|(candidate, _)| *candidate == split)
            .unwrap()
            .1;
        let modified = first
            .replace(string(&row["content_sha256"]), &hash)
            .replace(
                &format!("\"byte_length\":{}", row["byte_length"]),
                &format!("\"byte_length\":{}", payload.len()),
            )
            .replace(
                &format!("\"split\":{}", row["split"]),
                &format!("\"split\":\"{label}\""),
            );
        fixed_records(&spec, &root, sources, &format!("{modified}\n"), 1);
        fs::write(root.join(string(&row["payload_path"])), payload).unwrap();
        let dataset = ManifestValidator::open(&root, DL).unwrap();
        let baseline = Tokenizer::byte();
        let mut reader = dataset.reader(split);
        let mut batches = dataset
            .batches(
                split,
                &baseline,
                StreamConfig {
                    batch: BatchConfig::per_document(1, 1),
                    boundaries: Boundaries::default(),
                    tokenizer_limits: TL,
                },
            )
            .unwrap();
        for _ in 0..2 {
            assert_eq!(batches.next_batch().err(), Some(expected));
            let error = reader.next_document().err().unwrap();
            assert_eq!(error, expected);
            assert!(!error.to_string().contains("private-synthetic-sentinel"));
        }
        assert_eq!(batches.stats().samples, 0);
    }
}

#[cfg(unix)]
#[test]
fn rejects_final_intermediate_and_control_symlinks() {
    use std::os::unix::fs::symlink;

    let spec = fixture();
    let scratch = Scratch::new();
    let root = scratch.materialize(&spec);
    let outside = scratch.0.join("outside");
    fs::create_dir(&outside).unwrap();
    fs::write(
        outside.join("buffer.txt"),
        string(&spec["payloads"]["payload/buffer.txt"]),
    )
    .unwrap();
    let dataset = ManifestValidator::open(&root, DL).unwrap();
    let file = root.join("payload/buffer.txt");
    fs::remove_file(&file).unwrap();
    symlink(outside.join("buffer.txt"), &file).unwrap();
    assert_eq!(
        ManifestValidator::open(&root, DL).err(),
        Some(DatasetError::InvalidPath)
    );
    let split = dataset.records()[0].split;
    assert_eq!(
        dataset.reader(split).next_document().err(),
        Some(DatasetError::InvalidPath)
    );
    fs::remove_dir_all(root.join("payload")).unwrap();
    symlink(&outside, root.join("payload")).unwrap();
    assert_eq!(
        ManifestValidator::open(&root, DL).err(),
        Some(DatasetError::InvalidPath)
    );
    fs::remove_file(root.join("manifest.json")).unwrap();
    let safe = outside.join("manifest.json");
    fs::write(&safe, string(&spec["manifest"])).unwrap();
    symlink(safe, root.join("manifest.json")).unwrap();
    assert_eq!(
        ManifestValidator::open(&root, DL).err(),
        Some(DatasetError::InvalidPath)
    );
}

#[test]
fn stride_one_shift_padding_and_token_budget() {
    let spec = fixture();
    let scratch = Scratch::new();
    let root = scratch.materialize(&spec);
    let dataset = ManifestValidator::open(&root, DL).unwrap();
    let model = Tokenizer::byte();
    let mut cfg = StreamConfig {
        batch: BatchConfig {
            sequence_length: 4,
            stride: 1,
            batch_size: 2,
            tail: Tail::Pad,
            pad_final_batch: true,
        },
        boundaries: Boundaries::default(),
        tokenizer_limits: TL,
    };
    {
        let mut stream = dataset.batches(Split::Train, &model, cfg).unwrap();
        let batch = stream.next_batch().unwrap().unwrap();
        let mut reader = dataset.reader(Split::Train);
        let first = reader.next_document().unwrap().unwrap();
        let bytes = &first.bytes;
        let encode = |raw: &[u8]| raw.iter().map(|b| u32::from(*b)).collect::<Vec<_>>();
        assert_eq!(batch.shape(), [2, 4]);
        assert_eq!(
            batch.input(),
            [encode(&bytes[0..4]), encode(&bytes[1..5])].concat()
        );
        let target = [encode(&bytes[1..5]), encode(&bytes[2..6])].concat();
        assert_eq!(batch.target(), target);
        assert_ne!(
            batch.input(),
            target,
            "unshifted target mutation must disagree"
        );
        assert_eq!(batch.loss_mask(), Some([1; 8].as_slice()));
        while let Some(batch) = stream.next_batch().unwrap() {
            for (&token, &loss) in batch.target().iter().zip(batch.loss_mask().unwrap()) {
                assert_eq!(loss == 0, token == SpecialToken::Pad.id());
            }
            assert_eq!(batch.valid_mask(), batch.loss_mask());
        }
        assert!(stream.stats().padded_tails > 0);
    }
    for batch in [
        BatchConfig::per_document(0, 1),
        BatchConfig::per_document(1, 0),
        BatchConfig::per_document(2049, 1),
        BatchConfig::per_document(1, 65),
        BatchConfig::per_document(2048, 33),
        BatchConfig {
            stride: 5,
            ..cfg.batch
        },
        BatchConfig {
            stride: 0,
            ..cfg.batch
        },
    ] {
        assert_eq!(
            dataset
                .batches(Split::Train, &model, StreamConfig { batch, ..cfg })
                .err(),
            Some(DatasetError::InvalidConfiguration)
        );
    }
    let constrained = ManifestValidator::open(
        &root,
        DatasetLimits {
            token_buffer_bytes: 81,
            ..DL
        },
    )
    .unwrap();
    let mut stream = constrained.batches(Split::Train, &model, cfg).unwrap();
    assert_eq!(
        stream.next_batch().err(),
        Some(DatasetError::Tokenizer(TokenizerError::BufferLimitExceeded))
    );
    cfg.tokenizer_limits.input_bytes = 1;
    let mut stream = dataset.batches(Split::Train, &model, cfg).unwrap();
    assert_eq!(
        stream.next_batch().err(),
        Some(DatasetError::Tokenizer(TokenizerError::InputTooLarge))
    );
}

#[test]
fn audit_transform_and_tokenizer_compatibility() {
    let spec = fixture();
    let scratch = Scratch::new();
    let root = scratch.materialize(&spec);
    let rows = string(&spec["documents"]);
    let sources = string(&spec["sources"]);
    let first = rows.lines().next().unwrap();
    let row: Value = serde_json::from_str(first).unwrap();
    let path = string(&row["payload_path"]);
    let mut original = b"\xef\xbb\xbf".to_vec();
    original.extend_from_slice(string(&spec["payloads"][path]).as_bytes());
    let audit = format!(
        r#"[{{"id":"remove-utf8-bom","version":1,"parameters":"none","responsible_process":"fixture-review","input_sha256":"{}","output_sha256":{}}}]"#,
        sha(&original),
        row["content_sha256"],
    );
    let transformed_sources = sources.replace(
        "\"transformation_history\":[]",
        &format!("\"transformation_history\":{audit}"),
    );
    let transformed_row = first.replace(
        "\"cleaning_status\":\"original\",\"transformation_ids\":[]",
        "\"cleaning_status\":\"transformed\",\"transformation_ids\":[\"remove-utf8-bom\"]",
    );
    let transformed_rows = format!("{transformed_row}\n");
    fixed_records(&spec, &root, &transformed_sources, &transformed_rows, 1);
    let dataset = ManifestValidator::open(&root, DL).unwrap();
    let expected =
        xynoos_data::assign_split("task004-synthetic-v1", string(&row["content_sha256"])).unwrap();
    assert_eq!(
        dataset
            .reader(expected)
            .next_document()
            .unwrap()
            .unwrap()
            .bytes,
        string(&spec["payloads"][path]).as_bytes()
    );
    fixed_records(&spec, &root, sources, &transformed_rows, 1);
    assert_eq!(
        ManifestValidator::open(&root, DL).err(),
        Some(DatasetError::InvalidProvenance)
    );
    let unknown = transformed_sources.replace("fixture-review", "");
    fixed_records(&spec, &root, &unknown, &transformed_rows, 1);
    assert_eq!(
        ManifestValidator::open(&root, DL).err(),
        Some(DatasetError::InvalidProvenance)
    );

    write_metadata(&root, &spec);
    let sidecar = string(&spec["manifest"]).replace(
        "\"tokenizer_id\":\"unassigned\"",
        "\"tokenizer_id\":\"byte-v1\"",
    );
    fs::write(root.join("manifest.json"), sidecar).unwrap();
    let dataset = ManifestValidator::open(&root, DL).unwrap();
    assert_eq!(
        dataset.fit_bpe(Split::Train, 32, TL).err(),
        Some(DatasetError::IncompatibleTokenizer)
    );
}
