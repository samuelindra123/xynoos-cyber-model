#![forbid(unsafe_code)]

use std::{fs, path::PathBuf, time::SystemTime};

use serde_json::Value;
use xynoos_data::{BatchConfig, DatasetLimits, ManifestValidator, Split, StreamConfig, Tail};
use xynoos_tokenizer::{Boundaries, SpecialToken, Tokenizer, TokenizerLimits};

#[test]
fn independently_derived_padded_batch_is_shifted_and_document_local() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/task004.json")).unwrap();
    let root = temp_root();
    fs::create_dir_all(root.join("payload")).unwrap();
    for (path, encoded) in fixture["payloads"].as_object().unwrap() {
        let target = root.join(path);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(target, encoded.as_str().unwrap().as_bytes()).unwrap();
    }
    fs::write(
        root.join("manifest.json"),
        fixture["manifest"].as_str().unwrap(),
    )
    .unwrap();
    fs::write(
        root.join("sources.jsonl"),
        fixture["sources"].as_str().unwrap(),
    )
    .unwrap();
    fs::write(
        root.join("documents.jsonl"),
        fixture["documents"].as_str().unwrap(),
    )
    .unwrap();

    let expected_document =
        b"Use synthetic paths for examples.\r\nprintf '%s\\n' 'hello  buffer'\r\nls ./examples\r\n";
    let dataset = ManifestValidator::open(&root, DatasetLimits::MAX).unwrap();
    let byte_tokenizer = Tokenizer::byte();
    let mut stream = dataset
        .batches(
            Split::Train,
            &byte_tokenizer,
            StreamConfig {
                batch: BatchConfig {
                    sequence_length: 96,
                    stride: 96,
                    batch_size: 1,
                    tail: Tail::Pad,
                    pad_final_batch: true,
                },
                boundaries: Boundaries::default(),
                tokenizer_limits: TokenizerLimits::MAX,
            },
        )
        .unwrap();
    let batch = stream.next_batch().unwrap().unwrap();
    assert_eq!(batch.shape(), [1, 96]);
    let pad = SpecialToken::Pad.id();
    let valid_pairs = expected_document.len() - 1;
    let mut input: Vec<u32> = expected_document[..valid_pairs]
        .iter()
        .map(|&byte| u32::from(byte))
        .collect();
    input.resize(96, pad);
    let mut target: Vec<u32> = expected_document[1..]
        .iter()
        .map(|&byte| u32::from(byte))
        .collect();
    target.resize(96, pad);
    let mut mask = vec![1; valid_pairs];
    mask.resize(96, 0);
    assert_eq!(batch.input(), input);
    assert_eq!(batch.target(), target);
    assert_eq!(batch.valid_mask(), Some(mask.as_slice()));
    assert_eq!(batch.loss_mask(), Some(mask.as_slice()));
    assert!(batch.input()[valid_pairs..].iter().all(|&id| id == pad));
    assert!(batch.target()[valid_pairs..].iter().all(|&id| id == pad));
    while let Some(next) = stream.next_batch().unwrap() {
        assert_eq!(next.shape(), [1, 96]);
    }
    assert_eq!(stream.stats().documents, 10);
    fs::remove_dir_all(root).unwrap();
}

fn temp_root() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("xynoos-task004c-{}-{nonce}", std::process::id()))
}
