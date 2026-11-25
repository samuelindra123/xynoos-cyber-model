"""Freeze a tiny project-authored, safe Objective A learning fixture."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "crates/xynoos-training/tests/fixtures/objective_a.json"
TOKENIZER_FIXTURE = ROOT / "crates/xynoos-data/tests/fixtures/task004.json"
SEED = 0x006B

TRAIN = [
    {"family": "conditional", "text": "if count == 0: return empty_value  # empty case\n"},
    {"family": "loop", "text": "while index < limit: index += 1  # one step\n"},
]
VALIDATION = [
    {"family": "conditional", "text": "if size == 0: return empty_value  # empty case\n"},
    {"family": "loop", "text": "while cursor < count: cursor += 1  # one step\n"},
]
TEST = [
    {"family": "iteration", "text": "for item in queue: total += item_value  # one item\n"},
    {"family": "iteration", "text": "for token in buffer: count += token_value  # one token\n"},
]


def normalized(text):
    return " ".join(text.casefold().split())


def trigrams(text):
    compact = normalized(text)
    return {compact[i : i + 3] for i in range(max(1, len(compact) - 2))}


def close_pairs(rows, threshold=0.82):
    pairs = []
    for left, right in itertools.combinations(rows, 2):
        a, b = trigrams(left["text"]), trigrams(right["text"])
        score = len(a & b) / max(1, len(a | b))
        if score >= threshold:
            pairs.append({"left": left["text"].rstrip(), "right": right["text"].rstrip(), "jaccard": round(score, 6)})
    return pairs


def build():
    tokenizer = json.loads(TOKENIZER_FIXTURE.read_text())
    artifact = tokenizer["artifact"].encode("ascii")
    rows = TRAIN + VALIDATION + TEST
    exact = [normalized(left["text"]) for left, right in itertools.combinations(rows, 2) if normalized(left["text"]) == normalized(right["text"])]
    if exact:
        raise ValueError(f"exact duplicates cross fixture rows: {exact}")
    for split in (TRAIN, VALIDATION, TEST):
        for row in split:
            if not row["text"]:
                raise ValueError("objective statement must not be empty")
    split_digests = {
        name: hashlib.sha256("".join(row["text"] for row in split).encode("ascii")).hexdigest()
        for name, split in (("train", TRAIN), ("validation", VALIDATION), ("test", TEST))
    }
    order_identity = hashlib.sha256(
        b"XYNOOS-TASK006-OBJECTIVE-A-TRAIN-ORDER-V1\0"
        + "".join(row["text"] for row in TRAIN).encode("ascii")
    ).hexdigest()
    validation_order_identity = hashlib.sha256(
        b"XYNOOS-TASK006-OBJECTIVE-A-VALIDATION-ORDER-V1\0"
        + "".join(row["text"] for row in VALIDATION).encode("ascii")
    ).hexdigest()
    fixture = {
        "schema_version": 1,
        "fixture_id": "task006-objective-a-v1",
        "provenance": "Project-authored educational patterns; no external corpus or private data.",
        "model_seed": SEED,
        "vocabulary_size": 292,
        "tokenizer_algorithm": tokenizer["artifact"].split("algorithm\t", 1)[1].splitlines()[0],
        "tokenizer_artifact_sha256": hashlib.sha256(artifact).hexdigest(),
        "tokenization": "The existing byte-bpe-v1 artifact encodes each UTF-8 statement. Append EOS, repeat that tokenized statement plus EOS until 33 IDs exist, truncate to 33, then make one B=1,T=32 next-token window. No tokenizer refit.",
        "batch": {"rows": 1, "tokens": 32, "target_shift": 1, "valid_mask": "all ones", "loss_mask": "all ones"},
        "sample_order_seed": SEED,
        "sample_order": {"train": [row["text"].rstrip("\n") for row in TRAIN], "validation": [row["text"].rstrip("\n") for row in VALIDATION], "test": [row["text"].rstrip("\n") for row in TEST]},
        "template_families": {
            "train": sorted({row["family"] for row in TRAIN}),
            "validation_combination_holdout": sorted({row["family"] for row in VALIDATION}),
            "test_family_holdout": sorted({row["family"] for row in TEST}),
        },
        "split_sha256": split_digests,
        "data_order_identity_sha256": order_identity,
        "validation_order_identity_sha256": validation_order_identity,
        "exact_duplicate_normalized_count": 0,
        "near_duplicate_threshold": 0.82,
        "near_duplicate_pairs": close_pairs(rows),
        "near_duplicate_limitations": "Case folding, whitespace collapse, and character trigram Jaccard catch surface overlap only; they do not detect semantic paraphrases or prove independent reasoning.",
        "learning_rate_candidates": [0.0001, 0.0003, 0.001],
        "fixed_integration_peak_rate": 0.0003,
        "fixed_integration_final_rate": 0.0001,
        "evaluation_procedure": "Measure deterministic mean next-token cross entropy on both held-out combination batches before training and after two updates under the fixed 0.0003-to-0.0001 schedule; do not tune on test-family rows.",
        "acceptance_metrics": {
            "must_update_nonzero_transformer_parameters": True,
            "must_update_nonzero_adamw_moments": True,
            "heldout_loss_must_be_finite": True,
            "heldout_loss_delta_is_reported_not_a_claim_of_domain_intelligence": True,
        },
        "split_rows": {"train": TRAIN, "validation": VALIDATION, "test": TEST},
    }
    return json.dumps(fixture, ensure_ascii=True, sort_keys=True, indent=2) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    data = build()
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(data)
    elif not OUTPUT.exists() or OUTPUT.read_text() != data:
        raise SystemExit(f"{OUTPUT.relative_to(ROOT)}: frozen fixture differs")
    digest = hashlib.sha256(data.encode("utf8")).hexdigest()
    print(f"{OUTPUT.relative_to(ROOT)}: SHA-256 {digest}")


if __name__ == "__main__":
    main()
