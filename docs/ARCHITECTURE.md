# Architecture

The native tensor library is the first implemented component. The remaining sections describe research directions for later milestones.

## Data and representation

Potential stages are source registration, license and provenance review, cleaning and filtering, deduplication, train/validation/test separation, tokenization, and streaming batches. Dataset governance is specified in [DATASET_POLICY.md](DATASET_POLICY.md).

## Native compute and model

The workspace contains one safe Rust crate, `xynoos-tensor`. It owns dense, contiguous, row-major `Vec<f32>` storage with dynamic rank, checked shape and index arithmetic, and bounded fallible allocation. Scalar tensors contain one value and zero-sized shapes contain none. Elementwise operations use binary32, and sum, dot and matmul accumulate sequentially in binary64. A Python standard-library oracle validates the arithmetic and layout. [Tensor contracts](../crates/xynoos-tensor/README.md)

Gradient computation, embeddings, transformer blocks, causal attention, normalization, losses, and optimizers remain future work. C kernels and external mathematical dependencies would require a separate review.

## Training and artifacts

A future training runtime should support bounded workloads, deterministic configuration where possible, gradient accumulation if useful, checkpoint save/restore, and resource reporting. Model weights are intended to be independently trained from random initialization, subject to future training milestones. No weights are currently included.

## Inference and evaluation

Potential inference work includes model loading, generation, sampling, quantized execution, memory-mapped files, and CPU optimization. Evaluation should measure correctness, reliability, uncertainty, responsible behavior, resource use, and reproducibility. Retrieval infrastructure is optional future research.

## Optimization hypotheses

Candidate areas include SIMD and other CPU vectorization, cache-aware kernels, memory pooling, matrix multiplication, multithreading, quantization, streaming datasets, and checkpoint efficiency. No optimization is a project guarantee until measured against a documented baseline.
