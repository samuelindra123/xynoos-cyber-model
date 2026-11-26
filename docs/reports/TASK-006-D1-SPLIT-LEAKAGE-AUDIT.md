# TASK 006-D1 — Objective A Split and Tokenizer Audit

**Scope:** deterministic offline audit of the checked-in Objective A fixture and the tokenizer artifact used by the project. This is a software/data review aid, not independent human certification.

## Identity

| Item | Value |
|---|---|
| Objective A fixture | `task006-objective-a-v1` |
| Objective A fixture SHA-256 | `63050baf97f301c5ca6a3ad07a1a632bef77938b69f0317622738b5bbc00fa33` |
| Audit output SHA-256 | `182a4ae677a9ae0545690e48b99dab63ecc94b2ca051ce735eb2eed2abaa62a8` |
| Audit command | `python3 tools/task006_evaluation_audit.py --check` |
| Tokenizer artifact SHA-256 | `86aa40996d4a9bfe15491131bbe6fb1b04f5e325a50b23a3d7f0fa06bb56011c` |
| Tokenizer fit data | Task 004 train split, manifest `0041f6e5ff80d175265e5675da78aebe2db12007c0d4744f9b6e06f50a27e502` |

## Findings

| Measure | Result |
|---|---:|
| Rows in train / validation / test | 2 / 2 / 2 |
| Supervised targets in train / validation / test | 64 / 64 / 64 |
| Normalized exact duplicate pairs across splits | 0 |
| Character-trigram Jaccard near-duplicate threshold | 0.82 |
| Cross-split pairs at or above threshold | 0 |
| Maximum cross-split character-trigram Jaccard | 0.717391 |
| Maximum Objective A to tokenizer-fit train-text trigram Jaccard | 0.106667 |
| Exact normalized Objective A / tokenizer-fit train-text overlaps | 0 |

Training and validation share the `conditional` and `loop` template families.
The test split contains two rows from one held-out `iteration` family. This is
not enough independent test coverage for a stable generalization estimate or
learning-rate selection. Character-trigram similarity only measures surface
overlap; it does not detect semantic duplication, paraphrase, or task leakage.
Comparing raw tokenizer-training text cannot prove that BPE merges expose no
fragments relevant to Objective A.

The tokenizer fit is tied to a separate Task 004 train-only manifest rather
than Objective A. Its use avoids fitting directly on the Objective A
validation/test rows, but it does not make Objective A a standard
manifest-backed training dataset. The current fixture remains an authored
learning-control fixture only; it is not Objective B and is not evidence of
cybersecurity capability.

Regenerate and compare the exact result with:

```sh
python3 tools/task006_evaluation_audit.py --check
```
