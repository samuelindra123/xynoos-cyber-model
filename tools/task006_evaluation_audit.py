"""Audit split and tokenizer-fit leakage in the frozen Objective A fixture."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBJECTIVE_A = ROOT / "crates/xynoos-training/tests/fixtures/objective_a.json"
TOKENIZER_FIXTURE = ROOT / "crates/xynoos-data/tests/fixtures/task004.json"
OUTPUT = ROOT / "tools/fixtures/task006-evaluation-audit.json"
NEAR_DUPLICATE_THRESHOLD = 0.82


def normalize(text):
    return " ".join(text.casefold().split())


def character_trigrams(text):
    value = normalize(text)
    return {value[index : index + 3] for index in range(max(1, len(value) - 2))}


def jaccard(left, right):
    a, b = character_trigrams(left), character_trigrams(right)
    return len(a & b) / max(1, len(a | b))


def build():
    fixture_bytes = OBJECTIVE_A.read_bytes()
    objective = json.loads(fixture_bytes)
    tokenizer_fixture = json.loads(TOKENIZER_FIXTURE.read_text(encoding="utf-8"))
    fit = json.loads(tokenizer_fixture["fit"])
    records = [
        (split, row["family"], row["text"])
        for split, rows in objective["split_rows"].items()
        for row in rows
    ]
    exact_duplicates = []
    pair_scores = []
    for left, right in itertools.combinations(records, 2):
        score = jaccard(left[2], right[2])
        pair_scores.append((score, left, right))
        if normalize(left[2]) == normalize(right[2]):
            exact_duplicates.append(
                {
                    "left_split": left[0],
                    "left": left[2].rstrip("\n"),
                    "right_split": right[0],
                    "right": right[2].rstrip("\n"),
                }
            )
    near_duplicates = [
        {
            "left_split": left[0],
            "left": left[2].rstrip("\n"),
            "right_split": right[0],
            "right": right[2].rstrip("\n"),
            "character_trigram_jaccard": round(score, 6),
        }
        for score, left, right in pair_scores
        if score >= NEAR_DUPLICATE_THRESHOLD
    ]
    family_overlap = {}
    for left, right in itertools.combinations(sorted(objective["split_rows"]), 2):
        left_families = {row["family"] for row in objective["split_rows"][left]}
        right_families = {row["family"] for row in objective["split_rows"][right]}
        family_overlap[f"{left}_vs_{right}"] = sorted(left_families & right_families)

    tokenizer_train = []
    for document in tokenizer_fixture["per_document"]:
        if document["split"] == "train":
            tokenizer_train.append(
                (
                    document["id"],
                    bytes(document["baseline"]).decode("utf-8"),
                )
            )
    tokenizer_overlaps = [
        (jaccard(objective_text, train_text), split, objective_text, document_id)
        for split, _, objective_text in records
        for document_id, train_text in tokenizer_train
    ]
    tokenizer_exact = [
        {"split": split, "document_id": document_id, "text": text.rstrip("\n")}
        for split, _, text in records
        for document_id, train_text in tokenizer_train
        if normalize(text) == normalize(train_text)
    ]

    return {
        "schema_version": 1,
        "objective_a_fixture_id": objective["fixture_id"],
        "objective_a_fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "rows_by_split": {
            split: len(rows) for split, rows in sorted(objective["split_rows"].items())
        },
        "window_tokens_per_row": 32,
        "supervised_tokens_by_split": {
            split: len(rows) * 32
            for split, rows in sorted(objective["split_rows"].items())
        },
        "normalized_exact_duplicate_pairs": exact_duplicates,
        "max_cross_split_character_trigram_jaccard": round(
            max(
                score
                for score, left, right in pair_scores
                if left[0] != right[0]
            ),
            6,
        ),
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "near_duplicate_pairs_at_or_above_threshold": near_duplicates,
        "template_family_overlap": family_overlap,
        "tokenizer_artifact_sha256": hashlib.sha256(
            tokenizer_fixture["artifact"].encode("ascii")
        ).hexdigest(),
        "tokenizer_fit_dataset_id": fit["dataset_id"],
        "tokenizer_fit_train_manifest_sha256": fit["training_manifest_sha256"],
        "tokenizer_fit_train_record_count": fit["train_records"],
        "tokenizer_fit_normalized_exact_overlap_pairs": tokenizer_exact,
        "max_objective_to_tokenizer_fit_train_character_trigram_jaccard": round(
            max(score for score, _, _, _ in tokenizer_overlaps), 6
        ),
        "assessment": "insufficient-for-meaningful-learning-rate-selection-or-generalization",
        "limits": [
            "The fixture has two rows per split and only 64 supervised targets per split.",
            "Training and validation reuse both template families; the conditional pair has substantial surface overlap.",
            "The held-out test has one family and two rows, so it is not a stable generalization estimate.",
            "Character-trigram Jaccard detects surface overlap only, not paraphrase or semantic contamination.",
            "Tokenizer overlap checks compare raw training texts and cannot prove that learned subword merges reveal no fragments.",
            "The Objective A fixture is not a standard manifest-backed training dataset.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(build(), ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    if args.write:
        OUTPUT.write_text(rendered, encoding="utf-8")
    elif not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != rendered:
        raise SystemExit(f"{OUTPUT.relative_to(ROOT)}: split audit fixture differs")
    print(f"{OUTPUT.relative_to(ROOT)}: SHA-256 {hashlib.sha256(rendered.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
