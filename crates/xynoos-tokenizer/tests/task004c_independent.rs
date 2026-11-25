#![forbid(unsafe_code)]

use sha2::{Digest, Sha256};
use xynoos_tokenizer::{BpeTrainer, TokenizerArtifact, TokenizerLimits};

const LIMITS: TokenizerLimits = TokenizerLimits::MAX;

#[test]
fn hand_calculated_ties_overlaps_rounds_and_boundaries() {
    let mut tied = BpeTrainer::new(1, LIMITS).unwrap();
    for document in [b"ab".as_slice(), b"ab", b"ba", b"ba"] {
        tied.push_document(document).unwrap();
    }
    let tied = tied.finish().unwrap();
    assert_eq!(tied.merges().len(), 1);
    assert_eq!(tied.merges()[0].pair(), (b'a' as u32, b'b' as u32));

    let mut overlap = BpeTrainer::new(1, LIMITS).unwrap();
    overlap.push_document(b"aaa").unwrap();
    let overlap = overlap.finish().unwrap();
    assert_eq!(overlap.merges()[0].pair(), (b'a' as u32, b'a' as u32));
    assert_eq!(
        overlap.encode_bytes(b"aaa", LIMITS).unwrap(),
        [260, b'a' as u32]
    );

    let mut rounds = BpeTrainer::new(2, LIMITS).unwrap();
    rounds.push_document(b"aaaaaa").unwrap();
    let rounds = rounds.finish().unwrap();
    assert_eq!(rounds.merges().len(), 2);
    assert_eq!(rounds.merges()[0].expansion(), b"aa");
    assert_eq!(rounds.merges()[1].pair(), (260, 260));
    assert_eq!(rounds.encode_bytes(b"aaaaaa", LIMITS).unwrap(), [261, 260]);

    let mut isolated = BpeTrainer::new(1, LIMITS).unwrap();
    isolated.push_document(b"a").unwrap();
    isolated.push_document(b"b").unwrap();
    assert!(isolated.finish().unwrap().merges().is_empty());

    // Rank 0 is `bc`; rank 1 is `ab`. Even though `ab` occurs earlier in
    // the query, encoding applies the lowest available rank first.
    let mut ranked = BpeTrainer::new(2, LIMITS).unwrap();
    ranked.push_document(b"bcbcbc").unwrap();
    ranked.push_document(b"ab").unwrap();
    ranked.push_document(b"ab").unwrap();
    let ranked = ranked.finish().unwrap();
    assert_eq!(ranked.merges()[0].pair(), (b'b' as u32, b'c' as u32));
    assert_eq!(ranked.merges()[1].pair(), (b'a' as u32, b'b' as u32));
    assert_eq!(
        ranked.encode_bytes(b"abc", LIMITS).unwrap(),
        [b'a' as u32, 260]
    );
}

#[test]
fn canonical_artifact_allows_flat_alias_expansions() {
    let body = concat!(
        "XYNOOS-TOKENIZER\t1\n",
        "algorithm\tbyte-bpe-v1\n",
        "vocab_size\t263\n",
        "special\t256\tBOS\t<|bos|>\n",
        "special\t257\tEOS\t<|eos|>\n",
        "special\t258\tPAD\t<|pad|>\n",
        "special\t259\tDOC_SEP\t<|doc_sep|>\n",
        "merge\t0\t97\t97\t260\t6161\n",
        "merge\t1\t260\t97\t261\t616161\n",
        "merge\t2\t97\t260\t262\t616161\n",
    );
    let digest = format!("{:x}", Sha256::digest(body.as_bytes()));
    let artifact = format!("{body}sha256\t{digest}\n");
    let tokenizer = TokenizerArtifact::parse(artifact.as_bytes(), LIMITS).unwrap();
    assert_eq!(tokenizer.merges()[1].expansion(), b"aaa");
    assert_eq!(tokenizer.merges()[2].expansion(), b"aaa");
    assert_eq!(
        tokenizer.decode_bytes(&[261, 262], LIMITS).unwrap(),
        b"aaaaaa"
    );
}
