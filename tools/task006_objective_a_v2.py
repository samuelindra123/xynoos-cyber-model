"""Generate a manifest-backed, project-authored Objective A control corpus."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "datasets/task006-objective-a-v2"
SEED = "task006-objective-a-v2"
FAMILIES = ("parity", "comparison", "progression")
SPLITS = ("train", "validation", "test")
COUNTS = {"train": 8, "validation": 4, "test": 4}
BASES = {"train": 100, "validation": 400, "test": 700}

TEMPLATES = {
    "parity": (
        "Parity: n={n}, rem2={r}; class={label}. Fixed rule.\n",
        "Classify n={n}, r2={r}: {label}. Integer rule.\n",
        "n={n}, residue2={r}; parity={label}. Task.\n",
        "Parity n={n} r={r} => {label}. Rule fixed.\n",
    ),
    "comparison": (
        "Order: {a} vs {b}; result={relation}. Signed rule.\n",
        "Compare {a}/{b}; relation={relation}. Integer order.\n",
        "Pair {a},{b}; relation={relation}. Signed.\n",
        "Order {a} relative {b}: {relation}. Fixed rule.\n",
    ),
    "progression": (
        "{a},{b},{c}; step {step}; next {next}.\n",
        "Seq {a},{b},{c}; d={step}; next={next}.\n",
        "Prog {a},{b},{c}; d={step}; next={next}.\n",
        "Terms {a},{b},{c}; plus {step}; next={next}.\n",
    ),
}


def load_tokenizer_oracle():
    path = ROOT / "tools/tokenizer_data_oracle.py"
    spec = importlib.util.spec_from_file_location("task006_tokenizer_oracle", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the independent tokenizer reference")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOKENIZER_ORACLE = load_tokenizer_oracle()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def canonical(row):
    return (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def assign_split(content_sha256):
    digest = hashlib.sha256(f"1\0{SEED}\0{content_sha256}".encode("ascii")).digest()
    prefix = int.from_bytes(digest[:8], "big")
    if prefix * 100 < 80 * 2**64:
        return "train"
    if prefix * 100 < 90 * 2**64:
        return "validation"
    return "test"


def candidate(family, intended_split, index):
    base = BASES[intended_split]
    template_index = index % len(TEMPLATES[family])
    if family == "parity":
        number = base + index
        remainder = number % 2
        label = "odd" if remainder else "even"
        text = TEMPLATES[family][template_index].format(
            n=number, r=remainder, label=label
        )
        text += " End of record. Synthetic control sample.\n"
        inputs = {"integer": number}
        target = label
    elif family == "comparison":
        value = base + index
        delta = (index % 9) - 4
        cutoff = value + delta
        relation = "below" if value < cutoff else "above" if value > cutoff else "equal"
        text = TEMPLATES[family][template_index].format(
            a=value, b=cutoff, relation=relation
        )
        text += " End of record. Synthetic control sample.\n"
        inputs = {"value": value, "cutoff": cutoff}
        target = relation
    else:
        start = base + index
        step = 1 + (index % 7)
        second = start + step
        third = second + step
        next_value = third + step
        text = TEMPLATES[family][template_index].format(
            a=start, b=second, c=third, step=step, next=next_value
        )
        text += " End of record. Synthetic control sample.\n"
        inputs = {"start": start, "step": step}
        target = str(next_value)
    return text, inputs, target, template_index


def select_examples():
    selected = []
    seen_payloads = set()
    for split in SPLITS:
        for family in FAMILIES:
            pool = []
            for index in range(20_000):
                text, inputs, target, template_index = candidate(family, split, index)
                payload = text.encode("utf-8")
                checksum = sha(payload)
                if assign_split(checksum) != split or payload in seen_payloads:
                    continue
                pool.append(
                    {
                        "split": split,
                        "family": family,
                        "text": text,
                        "inputs": inputs,
                        "target": target,
                        "template_index": template_index,
                        "content_sha256": checksum,
                    }
                )
                if len(pool) >= 256:
                    break
            if len(pool) < COUNTS[split]:
                raise ValueError(f"could not populate {split}/{family}")

            rows = []
            required_templates = min(3, COUNTS[split])
            template_order = sorted({row["template_index"] for row in pool})
            if len(template_order) < required_templates:
                raise ValueError(f"insufficient target/template diversity in {split}/{family}")
            first = next(row for row in pool if row["template_index"] == template_order[0])
            rows.append(first)
            second = next(
                row
                for row in pool
                if row["template_index"] == template_order[1]
                and row["target"] != first["target"]
            )
            rows.append(second)
            for template_index in template_order[2:required_templates]:
                rows.append(next(row for row in pool if row["template_index"] == template_index))
            for row in pool:
                if len(rows) >= COUNTS[split]:
                    break
                if row not in rows:
                    rows.append(row)
            if len(rows) != COUNTS[split] or len({row["target"] for row in rows}) < 2:
                raise ValueError(f"insufficient target diversity in {split}/{family}")
            seen_payloads.update(row["text"].encode("utf-8") for row in rows)
            selected.extend(rows)
    return selected


def build():
    examples = select_examples()
    examples.sort(key=lambda row: (SPLITS.index(row["split"]), row["family"], row["content_sha256"]))
    payloads = {}
    documents = []
    task_index = []
    source_id = "task006-objective-a-v2"
    for ordinal, row in enumerate(examples, 1):
        document_id = f"oa2-{ordinal:03d}"
        payload_path = f"payload/{document_id}.txt"
        payload = row["text"].encode("utf-8")
        payloads[payload_path] = payload
        documents.append(
            {
                "schema_version": 1,
                "document_id": document_id,
                "source_id": source_id,
                "source_revision": "1",
                "payload_path": payload_path,
                "content_sha256": sha(payload),
                "byte_length": len(payload),
                "encoding": "utf-8",
                "domain_labels": ["programming"],
                "split": row["split"],
                "cleaning_status": "original",
                "transformation_ids": [],
                "tokenizer_compatibility": "unassigned",
                "duplicate_group_id": sha(payload),
            }
        )
        task_index.append(
            {
                "document_id": document_id,
                "split": row["split"],
                "family": row["family"],
                "inputs": row["inputs"],
                "target": row["target"],
                "template_index": row["template_index"],
                "answer_start_byte_offset": row["text"].find(row["target"]),
                "target_end_byte_offset": row["text"].find(row["target"])
                + len(row["target"]),
            }
        )

    sources = [
        {
            "schema_version": 1,
            "source_id": source_id,
            "source_reference": "project-authored-objective-a-v2",
            "source_revision": "1",
            "acquired_utc_date": "2026-11-26",
            "source_sha256": sha(b"".join(payloads[path] for path in sorted(payloads))),
            "license_id": "Apache-2.0",
            "license_version": "2.0",
            "terms_reference": "LICENSE",
            "training_permission": "approved",
            "redistribution_permission": "approved",
            "redistribution_conditions": "Apache-2.0-license-and-notice",
            "transformation_history": [],
            "domain": "programming",
            "responsible_maintainer": "project-maintainers",
            "review_status": "approved",
            "reviewed_utc_date": "2026-11-26",
        }
    ]
    source_bytes = canonical(sources[0])
    document_bytes = b"".join(canonical(row) for row in documents)
    identity = sha(
        b"XYNOOS-DATASET-V1\0"
        + len(source_bytes).to_bytes(8, "big")
        + source_bytes
        + len(document_bytes).to_bytes(8, "big")
        + document_bytes
    )
    manifest = {
        "schema_version": 1,
        "dataset_id": identity,
        "dataset_version": "1",
        "tokenizer_id": "unassigned",
        "split_algorithm": "sha256-group",
        "split_version": 1,
        "split_seed": SEED,
        "group_key_policy": "content-sha256",
        "record_count": len(documents),
        "sources_sha256": sha(source_bytes),
        "documents_sha256": sha(document_bytes),
    }
    manifest_bytes = canonical(manifest)

    train_rows = [row for row in documents if row["split"] == "train"]
    train_manifest = b"".join(canonical(row) for row in train_rows)
    merges = TOKENIZER_ORACLE.learn(
        [payloads[row["payload_path"]] for row in train_rows], 32
    )
    if len(merges) != 32:
        raise ValueError(f"training split produced {len(merges)} BPE merges; expected 32")
    tokenizer_artifact = TOKENIZER_ORACLE.artifact(merges)
    fit_record = {
        "schema_version": 1,
        "dataset_id": identity,
        "training_split": "train",
        "training_manifest_sha256": sha(train_manifest),
        "artifact_sha256": sha(tokenizer_artifact),
        "requested_merges": 32,
        "train_records": len(train_rows),
    }
    fit_bytes = canonical(fit_record)
    token_lengths = {}
    answer_prefix_lengths = {}
    for row, task in zip(documents, task_index):
        encoded = payloads[row["payload_path"]]
        start = task["answer_start_byte_offset"]
        end = task["target_end_byte_offset"]
        target = task["target"].encode("ascii")
        if encoded.count(target) != 1 or encoded[start:end] != target:
            raise ValueError(f"task answer must appear exactly once: {task['document_id']}")
        tokens = TOKENIZER_ORACLE.encode(encoded, merges)
        answer_prefix = len(TOKENIZER_ORACLE.encode(encoded[:end], merges))
        token_lengths[row["document_id"]] = len(tokens)
        answer_prefix_lengths[row["document_id"]] = answer_prefix
    if any(length < 31 or length > 62 for length in token_lengths.values()):
        outliers = {key: value for key, value in token_lengths.items() if not 31 <= value <= 62}
        raise ValueError(f"each task row must form exactly one 32-target window: {outliers}")
    if any(length > 31 for length in answer_prefix_lengths.values()):
        outliers = {key: value for key, value in answer_prefix_lengths.items() if value > 31}
        raise ValueError(f"the task answer must occur in the first 32 target positions: {outliers}")
    for task in task_index:
        task["payload_bpe_tokens"] = token_lengths[task["document_id"]]
        task["answer_prefix_bpe_tokens"] = answer_prefix_lengths[task["document_id"]]

    index = {
        "schema_version": 1,
        "dataset_id": identity,
        "dataset_manifest_sha256": sha(manifest_bytes),
        "tokenizer_artifact_sha256": sha(tokenizer_artifact),
        "tokenizer_fit_record_sha256": sha(fit_bytes),
        "token_count_policy": "byte-bpe-v1 payload encoding must be 31-62 tokens; the runtime adds BOS/EOS and emits exactly one complete 32-target window, and the first answer occurrence must end within those 32 supervised target positions",
        "examples": task_index,
    }
    index_bytes = canonical(index)
    result = {
        "manifest.json": manifest_bytes,
        "sources.jsonl": source_bytes,
        "documents.jsonl": document_bytes,
        "tokenizer-fit.jsonl": fit_bytes,
        "tokenizer.tsv": tokenizer_artifact,
        "task-index.json": index_bytes,
        **payloads,
    }
    return result


def write(files):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT.rglob("*"):
        if path.is_file():
            path.unlink()
    for relative, data in files.items():
        path = OUTPUT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    files = build()
    if args.write:
        write(files)
    else:
        actual = {
            path.relative_to(OUTPUT).as_posix(): path.read_bytes()
            for path in OUTPUT.rglob("*")
            if path.is_file()
        } if OUTPUT.is_dir() else {}
        if actual != files:
            raise SystemExit("Objective A v2 corpus differs; regenerate without --check")
    print(f"Objective A v2: {len(files)} files; {sha(files['manifest.json'])} manifest SHA-256")


if __name__ == "__main__":
    main()
