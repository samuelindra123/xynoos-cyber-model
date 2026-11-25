#![forbid(unsafe_code)]

use sha2::{Digest, Sha256};
use xynoos_tokenizer::{
    Boundaries, BpeTrainer, SpecialDecodePolicy as Policy, SpecialToken, Tokenizer,
    TokenizerArtifact as Artifact, TokenizerError::*, TokenizerLimits,
};

const LIMITS: TokenizerLimits = TokenizerLimits::MAX;

#[test]
fn byte_mapping_text_and_special_grammar() {
    let byte = Tokenizer::byte();
    let all: Vec<_> = (0u8..=255).collect();
    for &value in &all {
        assert_eq!(
            byte.encode_bytes(&[value], LIMITS).unwrap(),
            [u32::from(value)]
        );
    }
    for payload in [
        vec![],
        all.clone(),
        all.iter().rev().copied().collect(),
        b"\0\xff\r\n\t  ".to_vec(),
        "Berkas café Δ\t<|bos|> <|eos|> <|pad|> <|doc_sep|>"
            .as_bytes()
            .to_vec(),
        (0..257).map(|i| (i * 137 + 91) as u8).collect(),
    ] {
        let tokens = byte.encode_bytes(&payload, LIMITS).unwrap();
        assert!(tokens.iter().all(|&id| id < 256));
        assert_eq!(byte.decode_bytes(&tokens, LIMITS).unwrap(), payload);
    }
    assert_eq!(
        byte.decode_text(&[255], Policy::Reject, LIMITS),
        Err(InvalidUtf8)
    );
    assert_eq!(
        byte.decode_text(
            &byte.encode_text("café", LIMITS).unwrap(),
            Policy::Reject,
            LIMITS
        )
        .unwrap(),
        "café"
    );
    let encoded = byte
        .encode_with_specials(
            b"",
            Boundaries {
                bos: true,
                eos: true,
            },
            LIMITS,
        )
        .unwrap();
    assert_eq!(encoded, [SpecialToken::Bos.id(), SpecialToken::Eos.id()]);
    assert_eq!(
        byte.decode_bytes(&encoded, LIMITS),
        Err(InvalidSpecialSequence)
    );
    assert_eq!(byte.decode(&encoded, Policy::Omit, LIMITS).unwrap(), b"");
    assert_eq!(
        byte.decode(&encoded, Policy::RenderRegisteredSpelling, LIMITS)
            .unwrap(),
        b"<|bos|><|eos|>"
    );
    for sequence in [
        [7, 256],
        [257, 7],
        [256, 256],
        [257, 257],
        [256, 259],
        [256, 258],
    ] {
        assert_eq!(
            byte.decode(&sequence, Policy::Omit, LIMITS),
            Err(InvalidSpecialSequence)
        );
    }
    for id in [260, u32::MAX] {
        assert_eq!(byte.decode_bytes(&[id], LIMITS), Err(InvalidTokenId));
    }
}

const HEADER: &str = "XYNOOS-TOKENIZER\t1\nalgorithm\tbyte-bpe-v1\nvocab_size\t264\nspecial\t256\tBOS\t<|bos|>\nspecial\t257\tEOS\t<|eos|>\nspecial\t258\tPAD\t<|pad|>\nspecial\t259\tDOC_SEP\t<|doc_sep|>\n";
const MERGES: &str = "merge\t0\t97\t98\t260\t6162\nmerge\t1\t98\t99\t261\t6263\nmerge\t2\t260\t99\t262\t616263\nmerge\t3\t97\t261\t263\t616263\n";

fn signed_tsv(body: &str) -> Vec<u8> {
    format!("{body}sha256\t{:x}\n", Sha256::digest(body.as_bytes())).into_bytes()
}

#[test]
fn valid_aliases_and_rank_conflicts() {
    let artifact = signed_tsv(&format!("{HEADER}{MERGES}"));
    let model = Artifact::parse(&artifact, LIMITS).unwrap();
    assert_eq!(model.merges()[2].expansion(), model.merges()[3].expansion());
    assert_eq!(model.decode_bytes(&[262, 263], LIMITS).unwrap(), b"abcabc");
    assert_eq!(model.encode_bytes(b"abc", LIMITS).unwrap(), [262]);
    assert_eq!(Artifact::serialize(&model, LIMITS).unwrap(), artifact);
    let swapped = "merge\t0\t98\t99\t260\t6263\nmerge\t1\t97\t98\t261\t6162\nmerge\t2\t97\t260\t262\t616263\nmerge\t3\t261\t99\t263\t616263\n";
    let reversed = Artifact::parse(&signed_tsv(&format!("{HEADER}{swapped}")), LIMITS).unwrap();
    assert_ne!(reversed.merges()[0].pair(), model.merges()[0].pair());
    assert_ne!(
        reversed.encode_bytes(b"ab", LIMITS).unwrap(),
        model.encode_bytes(b"ab", LIMITS).unwrap()
    );
}

#[test]
fn malformed_artifact_semantics_after_correct_checksum() {
    let original = format!("{HEADER}{MERGES}");
    for (old, new, error) in [
        ("special\t257", "special\t258", MalformedArtifact),
        ("0\t97\t98\t260", "0\t98\t99\t260", MalformedArtifact),
        (
            "merge\t1\t98\t99\t261\t6263",
            "merge\t1\t97\t98\t261\t6162",
            MalformedArtifact,
        ),
        (
            "merge\t1\t98\t99\t261",
            "merge\t1\t98\t99\t260",
            MalformedArtifact,
        ),
        ("merge\t1\t", "merge\t2\t", MalformedArtifact),
        ("\t97\t98\t260", "\t256\t98\t260", MalformedArtifact),
        ("\t97\t98\t260", "\t261\t98\t260", MalformedArtifact),
        ("\t97\t98\t260", "\t260\t98\t260", MalformedArtifact),
        ("\t97\t98\t260", "\t097\t98\t260", MalformedArtifact),
        ("\t6162\n", "\t6162\textra\n", MalformedArtifact),
        ("\t6162\n", "\t616F\n", MalformedArtifact),
        ("\t616263\n", "\t616264\n", MalformedArtifact),
        (
            "XYNOOS-TOKENIZER\t1",
            "XYNOOS-TOKENIZER\t2",
            UnsupportedVersion,
        ),
        (
            "algorithm\tbyte-bpe-v1",
            "algorithm\tunknown",
            UnsupportedAlgorithm,
        ),
        (
            "vocab_size\t264",
            "vocab_size\t773",
            VocabularyLimitExceeded,
        ),
        (
            "vocab_size\t264",
            "vocab_size\t65537",
            VocabularyLimitExceeded,
        ),
        ("vocab_size\t264", "vocab_size\t265", MalformedArtifact),
        ("vocab_size\t264", "unexpected\t264", MalformedArtifact),
    ] {
        let mutated = original.replacen(old, new, 1);
        assert_ne!(mutated, original);
        assert_eq!(
            Artifact::parse(&signed_tsv(&mutated), LIMITS).err(),
            Some(error),
            "{old}"
        );
    }
    let bytes = signed_tsv(&original);
    assert_eq!(
        Artifact::parse(&bytes[..bytes.len() - 1], LIMITS).err(),
        Some(MalformedArtifact)
    );
    let crlf = signed_tsv(&original.replace('\n', "\r\n"));
    assert!(Artifact::parse(&crlf, LIMITS).is_err());
    let mut corrupt = bytes.clone();
    let checksum_digit = corrupt.len() - 2;
    corrupt[checksum_digit] = if corrupt[checksum_digit] == b'0' {
        b'1'
    } else {
        b'0'
    };
    assert_eq!(
        Artifact::parse(&corrupt, LIMITS).err(),
        Some(ChecksumMismatch)
    );
    let mut appended = bytes.clone();
    appended.extend_from_slice(b"trailing\n");
    assert_eq!(
        Artifact::parse(&appended, LIMITS).err(),
        Some(MalformedArtifact)
    );
    assert_eq!(
        Artifact::parse(
            &bytes,
            TokenizerLimits {
                expansion_bytes: 2,
                ..LIMITS
            }
        )
        .err(),
        Some(ExpansionLimitExceeded)
    );
    assert_eq!(
        Artifact::read(
            bytes.as_slice(),
            TokenizerLimits {
                artifact_bytes: 8,
                ..LIMITS
            }
        )
        .err(),
        Some(ArtifactTooLarge)
    );
}

#[test]
fn configurable_limits_fail_with_no_partial_model() {
    let byte = Tokenizer::byte();
    assert_eq!(
        byte.encode_bytes(
            b"ab",
            TokenizerLimits {
                input_bytes: 1,
                ..LIMITS
            }
        ),
        Err(InputTooLarge)
    );
    assert_eq!(
        byte.encode_with_specials(
            b"a",
            Boundaries {
                bos: true,
                eos: false
            },
            TokenizerLimits {
                output_tokens: 1,
                ..LIMITS
            }
        ),
        Err(OutputTooLarge)
    );
    assert_eq!(
        byte.encode_bytes(
            b"a",
            TokenizerLimits {
                buffer_bytes: 3,
                ..LIMITS
            }
        ),
        Err(BufferLimitExceeded)
    );
    assert_eq!(
        byte.decode_bytes(
            &[97, 98],
            TokenizerLimits {
                output_bytes: 1,
                ..LIMITS
            }
        ),
        Err(OutputTooLarge)
    );
    assert_eq!(
        BpeTrainer::new(513, LIMITS).err(),
        Some(VocabularyLimitExceeded)
    );
    assert_eq!(
        BpeTrainer::new(
            1,
            TokenizerLimits {
                encoding_work: LIMITS.encoding_work + 1,
                ..LIMITS
            }
        )
        .err(),
        Some(InvalidLimits)
    );
    for (limits, reason) in [
        (
            TokenizerLimits {
                training_work: 1,
                ..LIMITS
            },
            WorkLimitExceeded,
        ),
        (
            TokenizerLimits {
                training_work: 260 * 260 + 4,
                ..LIMITS
            },
            WorkLimitExceeded,
        ),
        (
            TokenizerLimits {
                expansion_bytes: 1,
                ..LIMITS
            },
            ExpansionLimitExceeded,
        ),
        (
            TokenizerLimits {
                buffer_bytes: 256,
                ..LIMITS
            },
            BufferLimitExceeded,
        ),
    ] {
        let mut trainer = BpeTrainer::new(1, limits).unwrap();
        trainer.push_document(b"aaaa").unwrap();
        assert_eq!(trainer.finish().err(), Some(reason));
    }
    for (limits, reason) in [
        (
            TokenizerLimits {
                corpus_bytes: 1,
                ..LIMITS
            },
            CorpusTooLarge,
        ),
        (
            TokenizerLimits {
                corpus_documents: 1,
                ..LIMITS
            },
            TooManyDocuments,
        ),
    ] {
        let mut trainer = BpeTrainer::new(1, limits).unwrap();
        trainer.push_document(b"a").unwrap();
        assert_eq!(trainer.push_document(b"ab"), Err(reason));
        assert_eq!(trainer.finish().err(), Some(reason));
    }
    let model = Artifact::parse(&signed_tsv(&format!("{HEADER}{MERGES}")), LIMITS).unwrap();
    assert_eq!(
        model.encode_bytes(
            b"abc",
            TokenizerLimits {
                encoding_work: 1,
                ..LIMITS
            }
        ),
        Err(WorkLimitExceeded)
    );
    assert_eq!(
        model
            .encode_bytes(
                b"abc",
                TokenizerLimits {
                    output_tokens: 1,
                    ..LIMITS
                }
            )
            .unwrap(),
        [262]
    );
}

#[test]
fn work_budget_covers_pair_selection_replacement_and_encoding_lookup() {
    // Four document/count visits, one touched pair, two expansion bytes, then
    // five document/replacement visits are charged after table initialization.
    let cells = 260 * 260;
    for training_work in [cells + 3, cells + 4, cells + 11, cells + 12] {
        let mut trainer = BpeTrainer::new(
            1,
            TokenizerLimits {
                training_work,
                ..LIMITS
            },
        )
        .unwrap();
        trainer.push_document(b"aaaa").unwrap();
        if training_work < cells + 12 {
            assert_eq!(trainer.finish().err(), Some(WorkLimitExceeded));
        } else {
            let model = trainer.finish().unwrap();
            for encoding_work in 1..=3 {
                let limits = TokenizerLimits {
                    encoding_work,
                    ..LIMITS
                };
                assert_eq!(model.encode_bytes(b"aa", limits), Err(WorkLimitExceeded));
            }
            let limits = TokenizerLimits {
                encoding_work: 4,
                ..LIMITS
            };
            assert_eq!(model.encode_bytes(b"aa", limits).unwrap(), [260]);
        }
    }
}
