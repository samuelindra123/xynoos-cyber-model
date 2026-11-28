#![forbid(unsafe_code)]

use std::{fs, path::PathBuf};

use xynoos_data::{
    BatchConfig, DatasetLimits, ManifestValidator, Split, StreamConfig, canonical_line,
};
use xynoos_tokenizer::{Boundaries, SpecialToken, TokenizerLimits};

const RECORDS: [(Split, usize); 3] = [
    (Split::Train, 24),
    (Split::Validation, 12),
    (Split::Test, 12),
];

fn dataset_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../datasets/task006-objective-a-v2")
}

fn stream_config() -> StreamConfig {
    StreamConfig {
        batch: BatchConfig::per_document(32, 1),
        boundaries: Boundaries {
            bos: true,
            eos: true,
        },
        tokenizer_limits: TokenizerLimits::MAX,
    }
}

#[test]
fn manifest_train_only_fit_and_all_split_streams_are_deterministic() {
    let root = dataset_root();
    let dataset = ManifestValidator::open(&root, DatasetLimits::MAX).unwrap();
    assert_eq!(dataset.records().len(), 48);
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
            assert_eq!(batch.shape(), [1, 32]);
            assert_eq!(batch.input().len(), 32);
            assert_eq!(batch.target().len(), 32);
            assert_eq!(batch.input()[0], SpecialToken::Bos.id());
            batches += 1;
        }
        let stats = stream.stats();
        assert_eq!(batches, expected_documents);
        assert_eq!(stats.documents, expected_documents);
        assert_eq!(stats.samples, expected_documents);
        assert_eq!(stats.batches, expected_documents);

        let repeated = dataset
            .fitted_batches(split, &verified_fit, stream_config())
            .unwrap();
        assert_eq!(repeated.order_identity(), order_identity);
    }
}
