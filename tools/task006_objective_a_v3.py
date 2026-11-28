"""Generate the split-template-isolated Objective A v3 control corpus."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "datasets/task006-objective-a-v3"
SEED = "task006-objective-a-v3-input-group-split"
FAMILIES = ("parity", "comparison", "progression")
SPLITS = ("train", "validation", "test")
COUNTS = {"train": 8, "validation": 6, "test": 6}
TEMPLATE_SPLITS = {
    "train": (0, 1, 2, 3),
    "validation": (4, 5),
    "test": (6, 7),
}

TEMPLATES = {
    "parity": (
        "P:{n}={label}",
        "P[{n}]=>{label}",
        "par:{n}/{label}",
        "bit:{n}:{label}",
        "parity?{n}={label}",
        "class:{n}->{label}",
        "N={n};P={label}",
        "xor8({n})={label}",
    ),
    "comparison": (
        "C:{a}/{b}={relation}",
        "cmp:{a},{b}=>{relation}",
        "O({a};{b})={relation}",
        "D:{a}-{b}:{relation}",
        "c:{a},{b}:{relation}",
        "r[{a}|{b}]={relation}",
        "R:{a}vs{b}={relation}",
        "{a}?{b}:{relation}",
    ),
    "progression": (
        "N:{start}+{step}={next}",
        "next({start},{step})={next}",
        "add3:{start}|{step}|{next}",
        "A{start}x{step}:{next}",
        "term:{start}/{step}:{next}",
        "T({start};{step})={next}",
        "S{start}.{step}={next}",
        "N3:{start},{step}->{next}",
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


def input_group_id(family, inputs):
    group = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode("ascii")
    return sha(b"XYNOOS-TASK-INPUT-GROUP-V1\0" + family.encode("ascii") + b"\0" + group)


def assign_split(family, inputs):
    group_id = input_group_id(family, inputs)
    digest = hashlib.sha256(f"1\0{SEED}\0{group_id}".encode("ascii")).digest()
    prefix = int.from_bytes(digest[:8], "big")
    if prefix * 100 < 80 * 2**64:
        return "train"
    if prefix * 100 < 90 * 2**64:
        return "validation"
    return "test"


def candidate_inputs(family):
    if family == "parity":
        for number in range(256):
            yield {"integer": number}
    elif family == "comparison":
        for left in range(64):
            for right in range(64):
                yield {"left": left, "right": right}
    else:
        for start in range(64):
            for step in range(1, 5):
                yield {"start": start, "step": step}


def render_candidate(family, inputs, template_index):
    if family == "parity":
        number = inputs["integer"]
        remainder = number % 2
        label = "odd" if remainder else "even"
        text = TEMPLATES[family][template_index].format(
            n=number, r=remainder, label=label
        )
        target = label
    elif family == "comparison":
        left = inputs["left"]
        right = inputs["right"]
        relation = "below" if left < right else "above" if left > right else "equal"
        text = TEMPLATES[family][template_index].format(
            a=left, b=right, relation=relation
        )
        target = relation
    else:
        start = inputs["start"]
        step = inputs["step"]
        next_value = (start + 3 * step) % 16
        target = f"v{next_value}"
        text = TEMPLATES[family][template_index].format(
            start=start, step=step, next=target
        )
    return text, target


def select_examples():
    selected = []
    for family in FAMILIES:
        candidates = []
        for inputs in candidate_inputs(family):
            split = assign_split(family, inputs)
            group_id = input_group_id(family, inputs)
            candidates.append((group_id, split, inputs, group_id))
        candidates.sort()

        for split in SPLITS:
            required = COUNTS[split]
            template_ids = TEMPLATE_SPLITS[split]
            eligible = []
            for _, candidate_split, inputs, _ in candidates:
                if candidate_split != split:
                    continue
                _, target = render_candidate(family, inputs, template_ids[0])
                eligible.append((inputs, target))

            minimum_labels = {"parity": 2, "comparison": 3, "progression": 4}[family]
            target_order = sorted({target for _, target in eligible})
            if len(target_order) < minimum_labels:
                raise ValueError(f"insufficient {family} labels in {split}")
            selected_inputs = set()
            chosen = []
            for target in target_order[:minimum_labels]:
                inputs = next(
                    inputs
                    for inputs, candidate_target in eligible
                    if candidate_target == target and repr(inputs) not in selected_inputs
                )
                chosen.append((inputs, target))
                selected_inputs.add(repr(inputs))
            for inputs, target in eligible:
                if len(chosen) >= required:
                    break
                if repr(inputs) not in selected_inputs:
                    chosen.append((inputs, target))
                    selected_inputs.add(repr(inputs))

            rows = []
            for index, (inputs, target) in enumerate(chosen):
                template_index = template_ids[index % len(template_ids)]
                text, rendered_target = render_candidate(family, inputs, template_index)
                if rendered_target != target:
                    raise ValueError("task target changed with the selected prompt template")
                payload = text.encode("utf-8")
                rows.append(
                    {
                        "split": split,
                        "family": family,
                        "text": text,
                        "inputs": inputs,
                        "target": target,
                        "template_index": template_index,
                        "content_sha256": sha(payload),
                        "duplicate_group_id": input_group_id(family, inputs),
                    }
                )
            if len(rows) != required or len({row["target"] for row in rows}) < 2:
                raise ValueError(f"insufficient target/split diversity in {split}/{family}")
            selected.extend(rows)
    return selected


def build():
    examples = select_examples()
    examples.sort(key=lambda row: (SPLITS.index(row["split"]), row["family"], row["content_sha256"]))
    payloads = {}
    documents = []
    task_index = []
    source_id = "task006-objective-a-v3"
    for ordinal, row in enumerate(examples, 1):
        document_id = f"oa3-{ordinal:03d}"
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
                "duplicate_group_id": row["duplicate_group_id"],
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
            "source_reference": "project-authored-objective-a-v3",
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
        "group_key_policy": "task-input-sha256",
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
    if any(length < 1 or length > 15 for length in token_lengths.values()):
        outliers = {key: value for key, value in token_lengths.items() if not 1 <= value <= 15}
        raise ValueError(f"each task row must fit one padded 16-target window: {outliers}")
    if any(length > 15 for length in answer_prefix_lengths.values()):
        outliers = {key: value for key, value in answer_prefix_lengths.items() if value > 15}
        raise ValueError(f"the task answer must occur in the first 16 target positions: {outliers}")
    for task in task_index:
        task["payload_bpe_tokens"] = token_lengths[task["document_id"]]
        task["answer_prefix_bpe_tokens"] = answer_prefix_lengths[task["document_id"]]

    index = {
        "schema_version": 1,
        "dataset_id": identity,
        "dataset_manifest_sha256": sha(manifest_bytes),
        "tokenizer_artifact_sha256": sha(tokenizer_artifact),
        "tokenizer_fit_record_sha256": sha(fit_bytes),
        "token_count_policy": "byte-bpe-v1 payload encoding must be 1-15 tokens; the runtime adds BOS/EOS and emits exactly one padded 16-target window, and the first answer occurrence must end within those 16 target positions",
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
            raise SystemExit("Objective A v3 corpus differs; regenerate without --check")
    print(f"Objective A v3: {len(files)} files; {sha(files['manifest.json'])} manifest SHA-256")


if __name__ == "__main__":
    main()
