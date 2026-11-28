#![forbid(unsafe_code)]

use std::{fs, path::PathBuf};

use xynoos_data::{
    BatchConfig, DatasetLimits, ManifestValidator, Split, StreamConfig, Tail, canonical_line,
};
use xynoos_tokenizer::{Boundaries, SpecialToken, TokenizerLimits};

const RECORDS: [(Split, usize); 3] = [
    (Split::Train, 24),
    (Split::Validation, 18),
    (Split::Test, 18),
];

fn dataset_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../datasets/task006-objective-a-v3")
}

fn stream_config() -> StreamConfig {
    StreamConfig {
        batch: BatchConfig {
            sequence_length: 16,
            stride: 16,
            batch_size: 2,
            tail: Tail::Pad,
            pad_final_batch: false,
        },
        boundaries: Boundaries {
            bos: true,
            eos: true,
        },
        tokenizer_limits: TokenizerLimits::MAX,
    }
}

fn trainer_stream_config() -> StreamConfig {
    StreamConfig {
        batch: BatchConfig {
            sequence_length: 32,
            stride: 32,
            batch_size: 1,
            tail: Tail::Pad,
            pad_final_batch: false,
        },
        boundaries: Boundaries {
            bos: true,
            eos: true,
        },
        tokenizer_limits: TokenizerLimits::MAX,
    }
}

#[test]
fn manifest_train_only_fit_and_padded_split_streams_are_deterministic() {
    let root = dataset_root();
    let dataset = ManifestValidator::open(&root, DatasetLimits::MAX).unwrap();
    assert_eq!(dataset.records().len(), 60);
    for (split, expected) in RECORDS {
        assert_eq!(
            dataset
                .records()
                .iter()
                .filter(|row| row.split == split)
                .count(),
            expected
        );
    }

    let fit = dataset
        .fit_bpe(Split::Train, 32, TokenizerLimits::MAX)
        .unwrap();
    assert_eq!(fit.record().train_records, 24);
    let fit_line = canonical_line(fit.record(), DatasetLimits::MAX).unwrap();
    assert_eq!(
        fit_line,
        fs::read(root.join("tokenizer-fit.jsonl")).unwrap()
    );
    let artifact = fs::read(root.join("tokenizer.tsv")).unwrap();
    assert_eq!(fit.artifact(), artifact);
    let verified_fit = dataset
        .verify_fit(&fit_line, &artifact, TokenizerLimits::MAX)
        .unwrap();

    for (split, expected_documents) in RECORDS {
        let mut stream = dataset
            .fitted_batches(split, &verified_fit, stream_config())
            .unwrap();
        let order_identity = stream.order_identity();
        assert_eq!(stream.split(), split);
        let mut batches = 0;
        while let Some(batch) = stream.next_batch().unwrap() {
            assert_eq!(batch.shape(), [2, 16]);
            assert_eq!(batch.input().len(), 32);
            assert_eq!(batch.target().len(), 32);
            assert_eq!(batch.input()[0], SpecialToken::Bos.id());
            let valid = batch.valid_mask().unwrap();
            assert!(valid.contains(&0));
            assert!(valid.contains(&1));
            batches += 1;
        }
        let stats = stream.stats();
        assert_eq!(batches, expected_documents / 2);
        assert_eq!(stats.documents, expected_documents);
        assert_eq!(stats.samples, expected_documents);
        assert_eq!(stats.batches, expected_documents / 2);
        assert_eq!(stats.dropped_tails, 0);
        assert_eq!(stats.padded_tails, expected_documents);

        let repeated = dataset
            .fitted_batches(split, &verified_fit, stream_config())
            .unwrap();
        assert_eq!(repeated.order_identity(), order_identity);
    }
}

#[test]
fn manifest_train_documents_fit_native_trainer_shape_without_cross_document_packing() {
    let dataset = ManifestValidator::open(&dataset_root(), DatasetLimits::MAX).unwrap();
    let fitted = dataset
        .fit_bpe(Split::Train, 32, TokenizerLimits::MAX)
        .unwrap();
    let mut stream = dataset
        .fitted_batches(Split::Train, &fitted, trainer_stream_config())
        .unwrap();
    let order_identity = stream.order_identity();
    let mut documents = 0;
    let mut supervised_targets = 0;
    while let Some(batch) = stream.next_batch().unwrap() {
        assert_eq!(batch.shape(), [1, 32]);
        assert_eq!(batch.input().len(), 32);
        assert_eq!(batch.target().len(), 32);
        assert_eq!(batch.input()[0], SpecialToken::Bos.id());
        let valid = batch.valid_mask().unwrap();
        let loss = batch.loss_mask().unwrap();
        assert_eq!(valid.len(), 32);
        assert_eq!(loss.len(), 32);
        assert_eq!(valid[0], 1);
        assert!(valid.contains(&0));
        assert!(loss.iter().zip(valid).all(|(&loss, &valid)| loss <= valid));
        supervised_targets += loss.iter().map(|&value| usize::from(value)).sum::<usize>();
        documents += 1;
    }
    let stats = stream.stats();
    assert_eq!(documents, 24);
    assert_eq!(stats.documents, 24);
    assert_eq!(stats.samples, 24);
    assert_eq!(stats.batches, 24);
    assert_eq!(stats.dropped_tails, 0);
    assert_eq!(stats.padded_tails, 24);
    assert_eq!(supervised_targets, 197);

    let repeated = dataset
        .fitted_batches(Split::Train, &fitted, trainer_stream_config())
        .unwrap();
    assert_eq!(repeated.order_identity(), order_identity);
}
