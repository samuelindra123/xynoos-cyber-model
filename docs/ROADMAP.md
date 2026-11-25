# Research Roadmap

Milestones are ordered research areas, not delivery dates or completion claims. Each should have a scoped design, validation evidence, and resource plan.

| Milestone | Scope |
|---|---|
| TASK 001 — Repository and Research Foundation | Documentation, license, governance, lightweight CI, and development policies. |
| TASK 002 — Numerical Computing Foundation | Accepted and integrated at `c5250543e5a39b087b7daab3d86d69a2af65a703`; exact-head and post-merge CI passed. [Implementation report](reports/TASK-002-B-IMPLEMENTATION-REPORT.md) · [Final verification](reports/TASK-002-E-FINAL-VERIFICATION-REPORT.md) · [Acceptance report](reports/TASK-002-F-FINAL-ACCEPTANCE-REPORT.md) |
| TASK 003 — Differentiation and Optimization | Accepted implementation: bounded first-order reverse-mode autodiff, mean squared error, transactional SGD, gradient verification, and a native synthetic linear-regression proof. The TASK 003-A documents record the original proposals; TASK 003-B records implemented contracts; TASK 003-C records independent verification. No language model was trained. [Architecture plan](reports/TASK-003-A-ARCHITECTURE-REPORT.md) · [Implementation report](reports/TASK-003-B-IMPLEMENTATION-REPORT.md) · [Independent verification](reports/TASK-003-C-INDEPENDENT-VERIFICATION-REPORT.md) · [Final acceptance](reports/TASK-003-D-FINAL-ACCEPTANCE-REPORT.md) |
| TASK 004 — Tokenization and Dataset Infrastructure | Tokenizer research, provenance, data processing, and streaming pipelines. |
| TASK 005 — Language Model Architecture | Embeddings, causal attention, transformer implementation, and forward/backward computation. |
| TASK 006 — First Experimental Training | Randomly initialized weights, small reproducible corpus, bounded training experiment, loss measurement, and checkpoint generation. |
| TASK 007 — Inference Runtime | Model loading, token generation, sampling, and lightweight local inference. |
| TASK 008 — Domain Knowledge Infrastructure | Linux, operating systems, cybersecurity, networking, low-level programming, and technical documentation. |
| TASK 009 — Domain-Specific Training and Evaluation | Specialized experiments, technical accuracy, reliability, and uncertainty handling. |
| TASK 010 — Performance Engineering | CPU optimization, memory efficiency, quantization, benchmarking, and reproducibility. |

Future research may consider larger architectures, retrieval, improved technical reasoning, dedicated research compute, and model distribution. There are no completion percentages, deadlines, or performance targets before baselines exist.
