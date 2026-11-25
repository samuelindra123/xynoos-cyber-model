# Architecture

The accepted native workspace implements a CPU tensor library, first-order reverse-mode autodiff, mean squared error, transactional SGD, a fixed-byte and deterministic byte-level BPE tokenizer, and governed dataset streaming. The five-sample linear-regression test is a synthetic learning proof that exercises the numerical engine; it is not language-model training. See the [TASK 004 final acceptance report](reports/TASK-004-D-FINAL-ACCEPTANCE-REPORT.md).

## Data and representation

Potential stages are source registration, license and provenance review, cleaning and filtering, deduplication, train/validation/test separation, tokenization, and streaming batches. Dataset governance is specified in [DATASET_POLICY.md](DATASET_POLICY.md).

## Native compute and model

`xynoos-tensor` owns dense, contiguous, row-major `Vec<f32>` storage with dynamic rank, checked shape and index arithmetic, and fallible allocation bounded by per-tensor logical payload and retained numerical capacity. Scalar tensors contain one value and zero-sized shapes contain none. Elementwise operations use binary32, and sum, dot and matmul accumulate sequentially in binary64. A Python standard-library oracle validates the arithmetic and layout. [Tensor contracts](../crates/xynoos-tensor/README.md)

`xynoos-autograd` uses a bounded tape of immutable tensor snapshots and manually defined first-order backward kernels. A gradient batch transfers owned adjoints to the parameter set after backward. Transactional SGD stages updates before committing. The implemented operation set and ownership rules are described in [autograd contracts](../crates/xynoos-autograd/README.md); the accepted work is recorded in the [TASK 003 final acceptance report](reports/TASK-003-D-FINAL-ACCEPTANCE-REPORT.md). C kernels and external mathematical dependencies would require a separate review. The accepted xynoos-tokenizer and xynoos-data crates implement bounded tokenization and dataset streaming. No Transformer, language-model training pipeline, model checkpoint system, or inference runtime is implemented.

## Training and artifacts

A future language-model training runtime should support bounded workloads, deterministic configuration where possible, gradient accumulation if useful, checkpoint save/restore, and resource reporting. Model weights must be independently trained from random initialization; external pretrained weights and external ML frameworks do not replace the native model implementation. The synthetic regression and tokenizer experiments do not produce language-model weights; no trained language model or model checkpoint is currently included.

## Inference and evaluation

Potential inference work includes model loading, generation, sampling, quantized execution, memory-mapped files, and CPU optimization. Evaluation should measure correctness, reliability, uncertainty, responsible behavior, resource use, and reproducibility. Retrieval infrastructure is optional future research.

## Optimization hypotheses

Candidate areas include SIMD and other CPU vectorization, cache-aware kernels, memory pooling, matrix multiplication, multithreading, quantization, streaming datasets, and checkpoint efficiency. No optimization is a project guarantee until measured against a documented baseline.
