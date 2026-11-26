"""Independent standard-library audit for the Objective A v2 corpus bundle."""

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/task006-objective-a-v2"
OUTPUT = ROOT / "tools/fixtures/task006-objective-a-v2-audit.json"
SPLITS = ("train", "validation", "test")
FAMILIES = ("comparison", "parity", "progression")
THRESHOLD = 0.82


def sha(value):
    return hashlib.sha256(value).hexdigest()


def canonical(row):
    return (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def normalized(text):
    return " ".join(text.casefold().split())


def trigrams(text):
    value = normalized(text)
    return {value[index : index + 3] for index in range(max(1, len(value) - 2))}


def jaccard(left, right):
    a, b = trigrams(left), trigrams(right)
    return len(a & b) / max(1, len(a | b))


def assign_split(content_sha256, seed):
    digest = hashlib.sha256(f"1\0{seed}\0{content_sha256}".encode("ascii")).digest()
    prefix = int.from_bytes(digest[:8], "big")
    if prefix * 100 < 80 * 2**64:
        return "train"
    if prefix * 100 < 90 * 2**64:
        return "validation"
    return "test"


def replace_pair(tokens, pair, new_id):
    result = []
    index = 0
    while index < len(tokens):
        if tuple(tokens[index : index + 2]) == pair:
            result.append(new_id)
            index += 2
        else:
            result.append(tokens[index])
            index += 1
    return result


def train_merges(documents, requested):
    rows = [list(document) for document in documents]
    merges = []
    expansions = {index: bytes([index]) for index in range(256)}
    while len(merges) < requested:
        counts = Counter(pair for row in rows for pair in zip(row, row[1:]))
        if not counts or max(counts.values()) < 2:
            break
        pair = min(counts, key=lambda item: (-counts[item], item))
        new_id = 260 + len(merges)
        expansion = expansions[pair[0]] + expansions[pair[1]]
        expansions[new_id] = expansion
        merges.append((pair[0], pair[1], expansion.hex()))
        rows = [replace_pair(row, pair, new_id) for row in rows]
    return merges


def encode(data, merges):
    ordered = {(left, right): rank for rank, (left, right, _) in enumerate(merges)}
    tokens = list(data)
    while True:
        available = set(zip(tokens, tokens[1:])) & ordered.keys()
        if not available:
            return tokens
        pair = min(available, key=ordered.__getitem__)
        tokens = replace_pair(tokens, pair, 260 + ordered[pair])


def serialize_artifact(merges):
    specials = [(256, "BOS", "<|bos|>"), (257, "EOS", "<|eos|>"),
                (258, "PAD", "<|pad|>"), (259, "DOC_SEP", "<|doc_sep|>")]
    lines = ["XYNOOS-TOKENIZER\t1", "algorithm\tbyte-bpe-v1",
             f"vocab_size\t{260 + len(merges)}"]
    lines.extend(f"special\t{token_id}\t{name}\t{text}" for token_id, name, text in specials)
    lines.extend(
        f"merge\t{index}\t{left}\t{right}\t{260 + index}\t{expansion}"
        for index, (left, right, expansion) in enumerate(merges)
    )
    body = ("\n".join(lines) + "\n").encode("ascii")
    return body + b"sha256\t" + sha(body).encode("ascii") + b"\n"


def build():
    manifest_bytes = (DATASET / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    source_bytes = (DATASET / "sources.jsonl").read_bytes()
    document_bytes = (DATASET / "documents.jsonl").read_bytes()
    source_lines = source_bytes.splitlines(keepends=True)
    document_lines = document_bytes.splitlines(keepends=True)
    sources = [json.loads(line) for line in source_lines]
    documents = [json.loads(line) for line in document_lines]
    index_bytes = (DATASET / "task-index.json").read_bytes()
    index = json.loads(index_bytes)

    if len(source_lines) != 1 or len(sources) != 1:
        raise ValueError("expected exactly one canonical project-authored source record")
    if sha(source_bytes) != manifest["sources_sha256"]:
        raise ValueError("source registry digest mismatch")
    if sha(document_bytes) != manifest["documents_sha256"]:
        raise ValueError("document registry digest mismatch")
    identity = sha(
        b"XYNOOS-DATASET-V1\0"
        + len(source_bytes).to_bytes(8, "big")
        + source_bytes
        + len(document_bytes).to_bytes(8, "big")
        + document_bytes
    )
    if identity != manifest["dataset_id"] or identity != index["dataset_id"]:
        raise ValueError("dataset identity mismatch")
    if sha(manifest_bytes) != index["dataset_manifest_sha256"]:
        raise ValueError("manifest digest mismatch")
    if manifest["record_count"] != len(documents):
        raise ValueError("record count mismatch")
    if [row["document_id"] for row in documents] != sorted(row["document_id"] for row in documents):
        raise ValueError("document registry order is not stable")

    source = sources[0]
    payloads = {}
    normalized_seen = {}
    exact_duplicates = []
    cross_near_duplicates = []
    all_pairs = []
    families = {row["document_id"]: row for row in index["examples"]}
    if len(families) != len(documents) or set(families) != {
        row["document_id"] for row in documents
    }:
        raise ValueError("task index does not exactly cover document records")

    for row in documents:
        path = DATASET / row["payload_path"]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe or missing payload {row['payload_path']}")
        data = path.read_bytes()
        text = data.decode("utf-8")
        if len(data) != row["byte_length"] or sha(data) != row["content_sha256"]:
            raise ValueError(f"payload integrity mismatch: {row['document_id']}")
        if row["duplicate_group_id"] != row["content_sha256"]:
            raise ValueError("duplicate group is not content-hash bound")
        if assign_split(row["content_sha256"], manifest["split_seed"]) != row["split"]:
            raise ValueError(f"hash split mismatch: {row['document_id']}")
        if row["source_id"] != source["source_id"] or row["source_revision"] != source["source_revision"]:
            raise ValueError(f"source provenance mismatch: {row['document_id']}")
        key = normalized(text)
        if key in normalized_seen and normalized_seen[key][0] != row["split"]:
            exact_duplicates.append(
                {"left": normalized_seen[key][1], "right": row["document_id"]}
            )
        else:
            normalized_seen[key] = (row["split"], row["document_id"])
        payloads[row["document_id"]] = data

    if source["source_sha256"] != sha(
        b"".join(
            payloads[row["payload_path"].split("/")[-1].removesuffix(".txt")]
            for row in sorted(documents, key=lambda item: item["payload_path"])
        )
    ):
        raise ValueError("source aggregate digest mismatch")

    for left, right in itertools.combinations(documents, 2):
        left_text = payloads[left["document_id"]].decode("utf-8")
        right_text = payloads[right["document_id"]].decode("utf-8")
        score = jaccard(left_text, right_text)
        all_pairs.append((score, left, right))
        if score >= THRESHOLD and left["split"] != right["split"]:
            cross_near_duplicates.append(
                {
                    "left_document_id": left["document_id"],
                    "left_split": left["split"],
                    "right_document_id": right["document_id"],
                    "right_split": right["split"],
                    "character_trigram_jaccard": round(score, 6),
                }
            )

    fit_bytes = (DATASET / "tokenizer-fit.jsonl").read_bytes()
    fit_lines = fit_bytes.splitlines(keepends=True)
    fit = json.loads(fit_bytes)
    train_rows = [row for row in documents if row["split"] == "train"]
    train_manifest = b"".join(canonical(row) for row in train_rows)
    train_hash = sha(train_manifest)
    training_payloads = [payloads[row["document_id"]] for row in train_rows]
    merges = train_merges(training_payloads, fit["requested_merges"])
    computed_artifact = serialize_artifact(merges)
    artifact_bytes = (DATASET / "tokenizer.tsv").read_bytes()
    if fit["training_split"] != "train" or fit["train_records"] != len(train_rows):
        raise ValueError("tokenizer fit is not bound to the complete train split")
    if fit["training_manifest_sha256"] != train_hash:
        raise ValueError("tokenizer fit record has the wrong train manifest digest")
    if computed_artifact != artifact_bytes or sha(artifact_bytes) != fit["artifact_sha256"]:
        raise ValueError("independent train-only tokenizer fit does not match the artifact")
    if sha(artifact_bytes) != index["tokenizer_artifact_sha256"] or sha(fit_bytes) != index["tokenizer_fit_record_sha256"]:
        raise ValueError("tokenizer identity is not bound into task index")

    token_counts = {}
    answer_prefix_counts = {}
    answer_start_counts = {}
    for row in documents:
        task = families[row["document_id"]]
        data = payloads[row["document_id"]]
        tokens = encode(data, merges)
        target_bytes = task["target"].encode("ascii")
        target_start = task["answer_start_byte_offset"]
        target_end = task["target_end_byte_offset"]
        if (
            target_start < 0
            or target_end - target_start != len(target_bytes)
            or data[target_start:target_end] != target_bytes
            or data.count(target_bytes) != 1
        ):
            raise ValueError(f"answer must occur exactly once at the indexed offset: {row['document_id']}")
        family = task["family"]
        values = task["inputs"]
        if family == "parity":
            expected_target = "even" if values["integer"] % 2 == 0 else "odd"
            input_values = (values["integer"], values["integer"] % 2)
        elif family == "comparison":
            expected_target = (
                "below" if values["value"] < values["cutoff"]
                else "above" if values["value"] > values["cutoff"]
                else "equal"
            )
            input_values = (values["value"], values["cutoff"])
        elif family == "progression":
            expected_target = str(values["start"] + 3 * values["step"])
            input_values = (
                values["start"],
                values["start"] + values["step"],
                values["start"] + 2 * values["step"],
                values["step"],
            )
        else:
            raise ValueError(f"unknown task family: {family}")
        prompt = data[:target_start]
        if any(str(value).encode("ascii") not in prompt for value in input_values):
            raise ValueError(f"task inputs are absent from the answer prompt: {row['document_id']}")
        if task["target"] != expected_target:
            raise ValueError(f"answer does not match independently recomputed task: {row['document_id']}")
        answer_start_counts[row["document_id"]] = len(encode(prompt, merges))
        prefix = data[: task["target_end_byte_offset"]]
        answer_prefix_counts[row["document_id"]] = len(encode(prefix, merges))
        token_counts[row["document_id"]] = len(tokens)
        if len(tokens) != task["payload_bpe_tokens"]:
            raise ValueError(f"task index token count mismatch: {row['document_id']}")
        if answer_prefix_counts[row["document_id"]] != task["answer_prefix_bpe_tokens"]:
            raise ValueError(f"task index answer position mismatch: {row['document_id']}")
    if any(not 31 <= value <= 62 for value in token_counts.values()):
        raise ValueError("a document does not create exactly one complete 32-target window")
    if any(value > 31 for value in answer_prefix_counts.values()):
        raise ValueError("an answer falls outside the first supervised window")

    split_counts = Counter(row["split"] for row in documents)
    family_counts = Counter((row["split"], families[row["document_id"]]["family"]) for row in documents)
    expected_split_counts = {"train": 24, "validation": 12, "test": 12}
    expected_family_counts = {
        (split, family): (8 if split == "train" else 4)
        for split in SPLITS
        for family in FAMILIES
    }
    if dict(split_counts) != expected_split_counts or dict(family_counts) != expected_family_counts:
        raise ValueError("unexpected split or task-family balance")

    combination_overlap = {}
    input_sets = {
        (split, family): {
            json.dumps(families[row["document_id"]]["inputs"], sort_keys=True)
            for row in documents
            if row["split"] == split and families[row["document_id"]]["family"] == family
        }
        for split in SPLITS
        for family in FAMILIES
    }
    for family in FAMILIES:
        for left, right in itertools.combinations(SPLITS, 2):
            combination_overlap[f"{left}_vs_{right}/{family}"] = sorted(
                input_sets[(left, family)] & input_sets[(right, family)]
            )

    max_cross = max(score for score, left, right in all_pairs if left["split"] != right["split"])
    report = {
        "schema_version": 1,
        "dataset_id": identity,
        "manifest_sha256": sha(manifest_bytes),
        "source_registry_sha256": sha(source_bytes),
        "document_registry_sha256": sha(document_bytes),
        "task_index_sha256": sha(index_bytes),
        "tokenizer_artifact_sha256": sha(artifact_bytes),
        "tokenizer_fit_record_sha256": sha(fit_bytes),
        "record_count": len(documents),
        "rows_by_split": {split: split_counts[split] for split in SPLITS},
        "task_examples_by_split_and_family": {
            split: {family: family_counts[(split, family)] for family in FAMILIES}
            for split in SPLITS
        },
        "supervised_targets_by_split": {split: split_counts[split] * 32 for split in SPLITS},
        "normalized_exact_cross_split_duplicates": exact_duplicates,
        "near_duplicate_threshold": THRESHOLD,
        "cross_split_near_duplicate_pairs": cross_near_duplicates,
        "cross_split_near_duplicate_pair_count": len(cross_near_duplicates),
        "max_cross_split_character_trigram_jaccard": round(max_cross, 6),
        "heldout_input_combination_overlap": combination_overlap,
        "tokenizer_fit": {
            "fit_split": fit["training_split"],
            "fit_records": fit["train_records"],
            "train_manifest_sha256_recomputed": train_hash,
            "artifact_recomputed_by_separate_standard_library_audit": True,
            "heldout_rows_used_in_fit": 0,
        },
        "bpe_payload_token_count_min": min(token_counts.values()),
        "bpe_payload_token_count_max": max(token_counts.values()),
        "answer_bpe_position_max": max(answer_prefix_counts.values()),
        "answer_prompt_bpe_position_max": max(answer_start_counts.values()),
        "answers_appear_once_after_their_inputs": True,
        "all_answers_within_first_32_target_positions": True,
        "assessment": "authored-synthetic-compositional-control-only",
        "limitations": [
            "Three synthetic task families with fixed templates test operand-combination transfer, not broad reasoning or cybersecurity competence.",
            "Each split contains only 12 task examples per family-balanced set; estimates are descriptive and are not powered generalization claims.",
            "The test split is in the public repository and is not secret; runtime access logging does not prevent a human from reading it.",
            f"{len(cross_near_duplicates)} cross-split pairs exceed the fixed {THRESHOLD:.2f} character-trigram similarity threshold because templates repeat; this remains a material template-leakage risk.",
            "Character-trigram similarity misses semantic leakage; the task index is needed to detect input-combination overlap.",
            "The tokenizer is fit only on the train split, but shared task vocabulary across splits is intentional and does not imply independent lexical domains.",
            "A separate human quality review remains required before experiment admission.",
        ],
    }
    return json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    rendered = build()
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(rendered, encoding="utf-8")
    elif not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != rendered:
        raise SystemExit("Objective A v2 audit differs; regenerate without --check")
    print(f"{OUTPUT.relative_to(ROOT)}: SHA-256 {sha(rendered.encode('utf-8'))}")


if __name__ == "__main__":
    main()
