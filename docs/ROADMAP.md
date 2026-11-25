# Research Roadmap

Milestones are ordered research areas, not delivery dates or completion claims. Each should have a scoped design, validation evidence, and resource plan.

| Milestone | Scope |
|---|---|
| TASK 001 — Repository and Research Foundation | Documentation, license, governance, lightweight CI, and development policies. |
| TASK 002 — Numerical Computing Foundation | Accepted and integrated at `c5250543e5a39b087b7daab3d86d69a2af65a703`; exact-head and post-merge CI passed. [Implementation report](reports/TASK-002-B-IMPLEMENTATION-REPORT.md) · [Final verification](reports/TASK-002-E-FINAL-VERIFICATION-REPORT.md) · [Acceptance report](reports/TASK-002-F-FINAL-ACCEPTANCE-REPORT.md) |
| TASK 003 — Differentiation and Optimization | Accepted implementation: bounded first-order reverse-mode autodiff, mean squared error, transactional SGD, gradient verification, and a native synthetic linear-regression proof. The TASK 003-A documents record the original proposals; TASK 003-B records implemented contracts; TASK 003-C records independent verification. No language model was trained. [Architecture plan](reports/TASK-003-A-ARCHITECTURE-REPORT.md) · [Implementation report](reports/TASK-003-B-IMPLEMENTATION-REPORT.md) · [Independent verification](reports/TASK-003-C-INDEPENDENT-VERIFICATION-REPORT.md) · [Final acceptance](reports/TASK-003-D-FINAL-ACCEPTANCE-REPORT.md) |
| TASK 004 — Tokenization and Dataset Infrastructure | Accepted and integrated at `ab675d00119c2457f140e36a1dfc02806d9e811b`; bounded byte/BPE tokenizer, provenance-aware dataset validation, train-only fitting, and local next-token batching. This reference-scale work does not establish production tokenizer efficiency or model training capability. [Final acceptance](reports/TASK-004-D-FINAL-ACCEPTANCE-REPORT.md) · [Implementation](reports/TASK-004-B-IMPLEMENTATION-REPORT.md) · [Independent verification](reports/TASK-004-C-INDEPENDENT-VERIFICATION-REPORT.md) |
| TASK 005-A — Bounded Reference Architecture Selection | Next: compare and justify one initial reference architecture against quality, compute, memory, data, and reproducibility objectives. Select a bounded design before implementing model layers. [Engineering direction](design/EFFICIENCY-FIRST-MODEL-ENGINEERING.md) |
| TASK 005-B — Native Model Architecture Implementation | Future implementation milestone for the architecture selected by TASK 005-A, with independent CPU reference verification and bounded validation. |
| TASK 006 — First Experimental Training | Randomly initialized weights, reviewed data, bounded training experiment, measured resource use, loss and quality evaluation. No checkpoint or model release is implied before its gates are met. |
| TASK 007 — Inference Runtime | Native model loading, token generation, sampling, and CPU-first low-resource inference with measured compatibility. |
| TASK 008 — Domain Knowledge Infrastructure | Linux, operating systems, cybersecurity, networking, low-level programming, and technical documentation. |
| TASK 009 — Domain-Specific Training and Evaluation | Specialized experiments, technical accuracy, reliability, and uncertainty handling. |
| TASK 010 — Performance Engineering | CPU optimization, memory efficiency, quantization, benchmarking, and reproducibility. |

Future research may consider larger architectures, retrieval, improved technical reasoning, adaptive authorized compute, and model distribution. Proposed model-family sizes and release requirements are research targets in [efficiency-first model engineering](design/EFFICIENCY-FIRST-MODEL-ENGINEERING.md), not commitments. There are no completion percentages, deadlines, or performance targets before baselines exist. No language model has been trained.
