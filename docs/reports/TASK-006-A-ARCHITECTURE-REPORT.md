# TASK 006-A — Native Training Engine Architecture Report

**Date:** 2026-11-25
**Status:** design complete; owner review required before TASK 006-B implementation.
**Verdict:** DESIGN READY — TASK 006-B IMPLEMENTATION CAN PROCEED AFTER OWNER REVIEW

## 1. Scope and baseline

This milestone defines a native training architecture. It adds no production optimizer, training loop, checkpoint codec, model weights, corpus, tokenizer change, or resource-limit increase. No training run or remote compute was started.

The accepted base is `32eca2b900c4f7a88294f7a24a5dfae0604ae798`, with successful Foundation run [45616832319, job 139214284017](https://github.com/samuelindra123/xynoos-cyber-model/actions/runs/45616832319). The branch is `research/task-006-a-native-training-architecture`; the design is submitted as draft [PR #18](https://github.com/samuelindra123/xynoos-cyber-model/pull/18), targeting `main`. It must remain unmerged until owner authorization.

The fixed target remains `xynoos-cyber-0.1`: 914,400 FP32 parameters, 74 tensors, eight blocks, hidden width 96, four heads, FFN width 256, vocabulary 292, context 128, tied embeddings, and the accepted native Transformer. The measured backward case is B=1,T=32. Existing behavior for SGD and TASK 001–005 remains a compatibility requirement.

## 2. Design decisions

- Put transactional AdamW and token-weighted gradient accumulation in `xynoos-autograd`, where ParameterSet already owns parameter identity, revisions, persistent gradients, and the resource ledger. Keep orchestration, schedule, metrics, checkpoints, run budgets, and detached runtime together in one proposed `xynoos-training` crate.
- Store parameters and AdamW moments in FP32; compute optimizer equations and reductions in deterministic FP64 with explicit FP32 publication boundaries. Use bias-corrected moments, epsilon outside the square root, and decoupled decay. Stage all values and moments, then commit atomically.
- Weight each microbatch mean gradient by its supervised-target count. Normalize once over the complete window, then apply global-norm clipping. Reject empty-supervision windows and numerical failures without publishing a partial update.
- Use a finite training state machine and linear warmup followed by linear decay. Require explicit step, token, cumulative-work, owned-memory, and runtime limits. The existing 160M candidate backward scope does not authorize repeated training.
- Checkpoint only at completed optimizer boundaries. Use a bounded versioned little-endian format, checksums, atomic replacement, staged loading, and same-platform bitwise resume equivalence.
- Separate Objective A (project-authored learning-engine proof) from Objective B (licensed, broad, held-out capability evaluation). Neither an optimizer test nor falling training loss establishes cybersecurity capability.

Full equations, operation ordering, lifecycle, resource policy, and acceptance gates are in the [training-engine spec](../design/TASK-006-TRAINING-ENGINE-SPEC.md) and [AdamW numerical spec](../design/TASK-006-ADAMW-NUMERICAL-SPEC.md). The [checkpoint spec](../design/TASK-006-CHECKPOINT-AND-RESUME-SPEC.md), [data and compute research](../design/TASK-006-DATA-AND-COMPUTE-RESEARCH.md), and [TASK 006-B handoff](../design/TASK-006-IMPLEMENTATION-HANDOFF.md) cover the supporting decisions.

## 3. Independent reference evidence

The Python-standard-library oracle in `tools/adamw_reference.py` writes no Rust-derived expected values. Its 31 frozen records have SHA-256 `dc191b568d3134a344d056619422b7fa383b0a7554b934ebcce17771de73bcc2`.

The fixture covers first and repeated updates; positive, negative, and zero gradients; decoupled decay; bias correction; epsilon placement; invalid hyperparameters; FP32 moment overflow; non-finite rejection; u64 step boundaries; unequal supervised-token counts; zero-count microbatches; clipping below, at, and above threshold; and extreme finite gradients. Negative controls distinguish coupled L2, omitted bias correction, epsilon inside the square root, and equal-weight microbatch averaging. Rejected optimizer updates preserve their input state.

Foundation CI runs this oracle in check-only mode. TASK 006-B must compare the Rust implementation against these frozen values and add transaction, owner/revision, resource, and repeated-step tests.

## 4. Data and resource findings

The existing engineering fixture has 16 project-authored documents and 1,033 bytes, split 10/3/3. Its 32-merge BPE has vocabulary 292; measured fixture ratios vary from 0.721 to 0.825 tokens per byte. The corpus is too small and repetitive to establish language or domain capability. Its exact duplicate stays in one split but is still present twice; near-duplicate leakage is not addressed. No external corpus was acquired. Candidate public sources and their source-specific rights review points are documented, not approved for training.

Exact target payloads are 3,657,600 bytes for parameters and 7,315,200 bytes for two moment arrays. One FP64 accumulation inventory is another 7,315,200 bytes; numeric checkpoint payload is 10,972,800 bytes. The accepted candidate forward/backward ledger peak is 16,037,368 bytes. A conservative modeled backward overlap with moments, accumulation, and transferred gradients totals 34,325,368 bytes. Independent dataset/tokenizer buffer maxima add 101,777,408 bytes, for a modeled 136,102,776 bytes (129.80 MiB), about 1.80 MiB over the existing 128 MiB ledger ceiling before process overhead. These are modeled owned allocations, not RSS measurements.

Keep the accepted ceilings: 64 MiB per tensor, 128 MiB aggregate reference ledger, 100M ordinary work, 160M exact-candidate backward, existing 100,000 value/operation limits, and 10,000 experiment-step ceiling. TASK 006-B should add a stricter, separately named run-owned cap at 128 MiB and reserve each subsystem within it; for Objective A, the proposed data/tokenizer allowance is at most 64 MiB pending measurements. This changes no existing cap.

## 5. Compute evidence and limitations

Accepted hosted TASK 005 measurements include 60.796 ms forward, 55.503 ms backward, 0.034 ms gradient transfer, and 0.323 ms first persistent accumulation for the candidate sample. They exclude full AdamW, repeated accumulation, data preparation, validation, checkpoint I/O, and sustained training. Forward plus backward is 116.299 ms for that isolated sample. The derived 275 target positions/second is only a component sensitivity. The illustrative 1,000/10,000-update arithmetic scenarios are not measured throughput or an approved budget.

TASK 006-B/C must measure end-to-end repeated debug/release steps, optimizer and accumulation latency, validation cadence, dataset/tokenizer memory, checkpoint write/sync latency, process high-water, and shutdown margin before selecting any experiment limits. GitHub Actions is appropriate for bounded CPU/reference validation. Colab remains optional and requires separate owner authorization. Detached execution, persistence after CLI disconnect, transfer, and cleanup are unverified.

## 6. Validation record and next gate

Local checks completed on the design branch:

- `python3 tools/adamw_reference.py --check` — 31 records matched the frozen fixture.
- `python3 tools/check_repository.py` — passed repository privacy/artifact checks.
- `cargo fmt --all -- --check` and `cargo metadata --locked --offline --format-version 1` — passed.
- Markdown hygiene and `git diff --check` — passed.

The PR's Foundation workflow must pass on its final pushed head before this milestone is accepted. Hosted CI success will validate the design fixture and protect existing behavior; it will not authorize implementation, training, external data, Colab, GPU work, or merging. TASK 006-B begins only after owner review of the specifications and unresolved decisions listed in the implementation handoff.
