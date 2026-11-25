"""Independent standard-library reference and frozen TASK 004 synthetic fixture."""

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

FIXTURE = Path("crates/xynoos-data/tests/fixtures/task004.json")
SEED = "task004-synthetic-v1"
MERGE_LIMIT = 32
SPECIALS = [(256, "BOS", "<|bos|>"), (257, "EOS", "<|eos|>"),
            (258, "PAD", "<|pad|>"), (259, "DOC_SEP", "<|doc_sep|>")]
SPLITS = ("train", "validation", "test")
CORPUS = [
    ("buffer", "english", "A bounded buffer keeps bytes in order. A byte buffer has a capacity.\n"),
    ("queue", "indonesian", "Antrean berurutan menjaga data. Baca data, periksa ukuran, lalu simpan data.\n"),
    ("shell", "linux", "Use synthetic paths for examples.\r\nprintf '%s\\n' 'hello  buffer'\r\nls ./examples\r\n"),
    ("c", "programming", "unsigned int capacity(void) {\n\treturn 0x20u;\n}\n"),
    ("cpp", "programming", "constexpr unsigned flags() {\n    return 0b1010u;\n}\n"),
    ("rust", "programming", "fn buffer_size() -> usize {\n    let size = 0x20;\n    size\n}\n"),
    ("python", "programming", "def count_bytes(data):\n    for byte in data:\n        print(byte)\n"),
    ("address", "networking", "Documentation examples use 192.0.2.14/24 and 2001:db8::1/64.\n"),
    ("unicode", "unicode", "UTF-8 preserves café, e\u0301, Δ, 日本語, and 🔍 as exact bytes.\n"),
    ("protocol", "networking", "TCP carries a byte stream. UDP carries datagrams. DNS names records.\n"),
    ("mode", "linux", "Inspect a synthetic directory with stat ./examples and pwd.\n"),
    ("paths", "english", "The local fixture path is ./examples/buffer.txt. The file ends with LF.\n"),
    ("length", "indonesian", "Panjang data dihitung dalam byte. Ukuran buffer dicatat sebelum pembacaan.\n"),
    ("hex", "programming", "A mask of 0x0f is the same integer as 0b00001111.\n"),
    ("newline", "english", "One tab:\there. Two spaces:  here. Literal marker: <|eos|>.\n"),
    ("buffer-copy", "english", "A bounded buffer keeps bytes in order. A byte buffer has a capacity.\n"),
]


def sha(value):
    return hashlib.sha256(value).hexdigest()


def canonical(row):
    return (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def replace(tokens, pair, new_id):
    result = []
    index = 0
    while index < len(tokens):
        if tuple(tokens[index:index + 2]) == pair:
            result.append(new_id)
            index += 2
        else:
            result.append(tokens[index])
            index += 1
    return result


def learn(documents, limit):
    expansions = {i: bytes([i]) for i in range(256)}
    rows = [list(document) for document in documents]
    merges = []
    while len(merges) < limit:
        counts = Counter(pair for row in rows for pair in zip(row, row[1:]))
        if not counts or max(counts.values()) < 2:
            break
        pair = min(counts, key=lambda candidate: (-counts[candidate], candidate))
        new_id = 260 + len(merges)
        expansion = expansions[pair[0]] + expansions[pair[1]]
        expansions[new_id] = expansion
        merges.append((*pair, expansion.hex()))
        rows = [replace(row, pair, new_id) for row in rows]
    return merges


def encode(data, merges):
    ordered = {(left, right): rank for rank, (left, right, _) in enumerate(merges)}
    tokens = list(data)
    while True:
        available = set(zip(tokens, tokens[1:])) & ordered.keys()
        if not available:
            return tokens
        pair = min(available, key=ordered.__getitem__)
        tokens = replace(tokens, pair, 260 + ordered[pair])


def decode(tokens, merges):
    lookup = {i: bytes((i,)) for i in range(256)}
    lookup.update({260 + i: bytes.fromhex(m[2]) for i, m in enumerate(merges)})
    return b"".join(lookup[i] for i in tokens)


def artifact(merges, algorithm="byte-bpe-v1"):
    lines = ["XYNOOS-TOKENIZER\t1", f"algorithm\t{algorithm}",
             f"vocab_size\t{260 + len(merges)}"]
    lines += ["special\t{}\t{}\t{}".format(*entry) for entry in SPECIALS]
    lines += [f"merge\t{i}\t{left}\t{right}\t{260+i}\t{exp}"
              for i, (left, right, exp) in enumerate(merges)]
    body = ("\n".join(lines) + "\n").encode()
    return body + f"sha256\t{sha(body)}\n".encode()


def verify_checksum(value):
    if not value.endswith(b"\n"):
        raise ValueError("artifact LF")
    body, last = value[:-1].rsplit(b"\n", 1)
    if last != b"sha256\t" + sha(body + b"\n").encode():
        raise ValueError("artifact checksum")


def split(group):
    value = int.from_bytes(hashlib.sha256(f"1\0{SEED}\0{group}".encode()).digest()[:8], "big")
    return SPLITS[0 if value * 100 < 80 * 2**64 else 1 if value * 100 < 90 * 2**64 else 2]


def validate_splits(records):
    by_hash = {}
    for row in records:
        content = row["content_sha256"]
        label = row["split"]
        if content in by_hash and by_hash[content] != label:
            raise ValueError("duplicate leakage")
        by_hash[content] = label
    for row in records:
        if row["duplicate_group_id"] != row["content_sha256"] or split(row["content_sha256"]) != row["split"]:
            raise ValueError("incorrect split")


def fit_train(records, payloads):
    if any(row["split"] != "train" for row in records):
        raise ValueError("held-out fit")
    return learn([payloads[row["payload_path"]].encode() for row in records], MERGE_LIMIT)


def windows(tokens, length, stride, pad):
    pairs = list(zip(tokens, tokens[1:]))
    complete = []
    start = 0
    while start + length <= len(pairs):
        complete.append(pairs[start:start + length])
        start += stride
    tail = pairs[start:]
    if tail and pad:
        complete.append(tail)
    return complete, len(tail)


def batches(documents, length, stride, size, pad, fixed):
    rows = []
    result = []
    stats = dict(documents=len(documents), samples=0, batches=0, dropped_tails=0,
                 dropped_positions=0, padded_tails=0, padded_positions=0, padded_rows=0)
    for tokens in documents:
        samples, tail = windows([256, *tokens, 257], length, stride, pad)
        if tail:
            if pad:
                stats["padded_tails"] += 1
                stats["padded_positions"] += length - tail
            else:
                stats["dropped_tails"] += 1
                stats["dropped_positions"] += tail
        rows.extend(samples)
    stats["samples"] = len(rows)
    for offset in range(0, len(rows), size):
        portion = rows[offset:offset + size]
        if fixed:
            missing = size - len(portion)
            stats["padded_rows"] += missing
            stats["padded_positions"] += missing * length
            portion += [[] for _ in range(missing)]
        pairs = [pair for row in portion for pair in row + [(258, 258)] * (length - len(row))]
        valid = [int(left != 258) for left, _ in pairs] if pad or fixed else None
        result.append(dict(shape=[len(portion), length], input=[p[0] for p in pairs],
                           target=[p[1] for p in pairs], valid=valid, loss=valid))
    stats["batches"] = len(result)
    return dict(stats=stats, batches=result)


def metrics(byte_count, baseline, bpe, merge_count):
    return dict(bytes=byte_count, byte_tokens=len(baseline), bpe_tokens=len(bpe),
                byte_per_byte=f"{len(baseline) / byte_count:.6f}",
                bpe_per_byte=f"{len(bpe) / byte_count:.6f}",
                byte_used=len(set(baseline)), bpe_used=len(set(bpe)),
                byte_vocab=260, bpe_vocab=260 + merge_count, byte_round_trip=True, bpe_round_trip=True)


def generate():
    payloads = {f"payload/{name}.txt": text for name, _, text in CORPUS}
    source = dict(schema_version=1, source_id="task004-synthetic",
                  source_reference="task004-project-authored-v1", source_revision="1",
                  acquired_utc_date="2026-11-25",
                  source_sha256=sha(b"".join(text.encode() for text in payloads.values())),
                  license_id="Apache-2.0", license_version="2.0", terms_reference="LICENSE",
                  training_permission="approved", redistribution_permission="approved",
                  redistribution_conditions="Apache-2.0-license-and-notice",
                  transformation_history=[], domain="programming",
                  responsible_maintainer="project-maintainers", review_status="approved",
                  reviewed_utc_date="2026-11-25")
    records = []
    for name, domain, text in CORPUS:
        data = text.encode()
        checksum = sha(data)
        records.append(dict(schema_version=1, document_id=name, source_id=source["source_id"],
                            source_revision=source["source_revision"], payload_path=f"payload/{name}.txt",
                            content_sha256=checksum, byte_length=len(data), encoding="utf-8",
                            domain_labels=[domain], split=split(checksum), cleaning_status="original",
                            transformation_ids=[], tokenizer_compatibility="unassigned",
                            duplicate_group_id=checksum))
    validate_splits(records)
    registry = canonical(source)
    rows = b"".join(map(canonical, records))
    identity = sha(b"XYNOOS-DATASET-V1\0" + len(registry).to_bytes(8, "big") + registry
                   + len(rows).to_bytes(8, "big") + rows)
    manifest = canonical(dict(schema_version=1, dataset_id=identity, dataset_version="1",
                              tokenizer_id="unassigned", split_algorithm="sha256-group", split_version=1,
                              split_seed=SEED, group_key_policy="content-sha256", record_count=len(records),
                              sources_sha256=sha(registry), documents_sha256=sha(rows)))
    train = [row for row in records if row["split"] == "train"]
    merges = fit_train(train, payloads)
    tsv = artifact(merges)
    verify_checksum(tsv)
    fit = canonical(dict(schema_version=1, dataset_id=identity, training_split="train",
                         training_manifest_sha256=sha(b"".join(map(canonical, train))),
                         artifact_sha256=sha(tsv), requested_merges=MERGE_LIMIT, train_records=len(train)))

    per_document = []
    reports = {}
    all_batches = {}
    for label in SPLITS:
        docs = []
        for row in records:
            if row["split"] != label:
                continue
            data = payloads[row["payload_path"]].encode()
            baseline = list(data)
            encoded = encode(data, merges)
            assert bytes(baseline) == data and decode(encoded, merges) == data
            per_document.append(dict(id=row["document_id"], split=label, baseline=baseline, bpe=encoded,
                                     metrics=metrics(len(data), baseline, encoded, len(merges))))
            docs.append(per_document[-1])
        assert docs, f"empty fixture split {label}"
        baseline = sum((doc["baseline"] for doc in docs), [])
        encoded = sum((doc["bpe"] for doc in docs), [])
        reports[label] = dict(documents=len(docs), metrics=metrics(len(baseline), baseline, encoded, len(merges)))
        all_batches[label] = {
            family: {
                "drop": batches([doc[family] for doc in docs], 8, 8, 3, False, False),
                "pad": batches([doc[family] for doc in docs], 8, 3, 3, True, True),
            } for family in ("baseline", "bpe")
        }
    cases = []
    for name, documents, query in [
        ("all-bytes", [], bytes(range(256))),
        ("empty", [b""], b""),
        ("overlap", [b"aaa"], b"aaaaa"),
        ("tie", [b"ab", b"ab", b"ba", b"ba"], b"aba"),
        ("boundary", [b"a", b"b", b"a", b"b"], b"ab"),
        ("ranks", [b"abababcababa", b"abababcababa"], b"ababcababab"),
        ("invalid-utf8", [b"\xff\x00\xff\x00"], b"\xff\x00\xc0\t\r\n"),
    ]:
        learned = learn(documents, 8)
        encoded = encode(query, learned)
        assert decode(encoded, learned) == query
        cases.append(dict(name=name, documents=[data.hex() for data in documents],
                          query=query.hex(), baseline=list(query), bpe=encoded,
                          artifact=artifact(learned).decode()))
    measured = json.dumps(dict(per_document=per_document, splits=reports, batches=all_batches),
                          ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return dict(schema_version=1, experiment_sha256=sha(measured), payloads=payloads, sources=registry.decode(),
                documents=rows.decode(), manifest=manifest.decode(), manifest_sha256=sha(manifest),
                fit=fit.decode(), artifact=tsv.decode(), byte_artifact=artifact([], "byte-v1").decode(),
                cases=cases, per_document=per_document, splits=reports, batches=all_batches)


def rejected(action, message):
    try:
        action()
    except ValueError as error:
        assert str(error) == message
    else:
        raise AssertionError(f"undetected mutation {message}")


def compare(actual, expected, error):
    if actual != expected:
        raise ValueError(error)


def negative_controls(fixture):
    records = list(map(json.loads, fixture["documents"].splitlines()))
    wrong = fixture["cases"][0]["baseline"].copy()
    wrong[255] = 254
    rejected(lambda: compare(wrong, list(range(256)), "byte token"), "byte token")
    specials = SPECIALS.copy()
    specials[1] = (258, *specials[1][1:])
    rejected(lambda: compare(specials, SPECIALS, "special token"), "special token")
    case = next(case for case in fixture["cases"] if case["name"] == "tie")
    ordered = learn([bytes.fromhex(doc) for doc in case["documents"]], 8)
    reversed_artifact = artifact(ordered[::-1])
    verify_checksum(reversed_artifact)
    rejected(lambda: compare(reversed_artifact.decode(), case["artifact"], "merge order"), "merge order")
    tsv = fixture["artifact"].encode()
    digit = b"0" if tsv[-2:-1] != b"0" else b"1"
    rejected(lambda: verify_checksum(tsv[:-2] + digit + b"\n"), "artifact checksum")
    bad = copy.deepcopy(records)
    unique = next(row for row in bad if row["document_id"] == "c")
    unique["split"] = next(label for label in SPLITS if label != unique["split"])
    rejected(lambda: validate_splits(bad), "incorrect split")
    bad = copy.deepcopy(records)
    held_out = next(row for row in bad if row["split"] == "test")
    held_out["split"] = "validation"
    duplicate = copy.deepcopy(held_out)
    duplicate["split"] = "test"
    duplicate["document_id"] = "test-duplicate"
    rejected(lambda: validate_splits(bad + [duplicate]), "duplicate leakage")
    validate_splits(list(reversed(records)))
    train = [row for row in records if row["split"] == "train"]
    for label in ("validation", "test"):
        leaked = next(row for row in records if row["split"] == label)
        rejected(lambda: fit_train(train + [leaked], fixture["payloads"]), "held-out fit")
    row = fixture["batches"]["train"]["bpe"]["drop"]["batches"][0]
    rejected(lambda: compare(row["input"], row["target"], "target shift"), "target shift")
    print("TASK 004 Python negative controls detected: byte and special IDs, merge order, checksum, split, duplicate leakage, both held-out selections, target shift.")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    first = generate()
    encoded = canonical(first)
    assert encoded == canonical(generate()), "Python experiment is not deterministic"
    if args.write:
        FIXTURE.write_bytes(encoded)
    else:
        assert FIXTURE.read_bytes() == encoded, "TASK 004 fixture mismatch"
    negative_controls(first)
    print(f"TASK 004 fixture: {len(encoded)} bytes, SHA-256 {sha(encoded)}.")
    print("TASK 004 manifest", first["manifest_sha256"], "fit", first["fit"].strip())
    for label, report in first["splits"].items():
        print(label, report, {key: runs["drop"]["stats"] for key, runs in first["batches"][label].items()})


if __name__ == "__main__":
    main()
