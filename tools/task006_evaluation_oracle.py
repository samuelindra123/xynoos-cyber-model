#!/usr/bin/env python3
"""Independent standard-library check for token-weighted evaluation loss."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tools/fixtures/task006-evaluation-math.json"
LOSSES = (Decimal("0.125"), Decimal("2.75"))
TOKENS = (1, 3)


def fixture_bytes() -> bytes:
    weighted = sum(loss * count for loss, count in zip(LOSSES, TOKENS)) / sum(TOKENS)
    equal_batch_mean = sum(LOSSES) / len(LOSSES)
    payload = {
        "schema_version": 1,
        "batch_mean_losses": [float(loss) for loss in LOSSES],
        "supervised_tokens": list(TOKENS),
        "token_weighted_mean": float(weighted),
        "incorrect_equal_batch_mean": float(equal_batch_mean),
    }
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = fixture_bytes()
    if args.check:
        if not FIXTURE.is_file() or FIXTURE.read_bytes() != expected:
            raise SystemExit("evaluation aggregation fixture is stale; regenerate without --check")
        print("verified independent token-weighted evaluation fixture")
    else:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_bytes(expected)
        print(f"wrote {FIXTURE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
