# TASK 005 — Bounded Decoder Transformer Specification

**Status:** TASK 005-A selected reference contract for Session B review; no model layers are implemented by this document.  This architecture is a correctness reference, not a quality claim.

## Decision

Use a decoder-only, pre-normalized Transformer with learned token embeddings, interleaved RoPE, ordinary multi-head causal self-attention (MHA), RMSNorm, SwiGLU, no linear biases, tied input/output embeddings, and explicit full attention as the CPU reference.  All weights start randomly.  The design fixes simple MHA as the reference because it avoids introducing a second head-grouping semantic into the first attention/backward oracle. GQA and MQA remain inference-memory research alternatives, not hidden configuration switches.

Architecture comparisons and citations are in the [architecture report](../reports/TASK-005-A-ARCHITECTURE-REPORT.md). Main primary references: [Transformer](https://arxiv.org/abs/1706.03762), [RoFormer](https://arxiv.org/abs/2104.09864), [RMSNorm](https://arxiv.org/abs/1910.07467), [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202), [GQA](https://arxiv.org/abs/2305.13245), [FlashAttention](https://arxiv.org/abs/2205.14135).

## Configuration and tensor conventions

One configuration contains vocabulary `V` (exact tokenizer vocabulary), width `D`, layer count `N>0`, query head count `H>0`, even head width `d=D/H`, SwiGLU width `F>0`, maximum sample length `Tmax`, and `epsilon=1e-5`. Require `D % H == 0`, `d % 2 == 0`, positive dimensions, and checked products. The reference accepts `1 <= B <= 64`, `1 <= T <= min(Tmax,2048)`, `B*T <= 65,536`, `V` equal to the loaded tokenizer's exact size, and `V >= 260`; smaller TASK 004 tokenizers are valid. The first candidates use V=260 or 292; model support does not raise the tokenizer's 772 BPE cap.

Axes are always `[batch, sequence, feature]` for activations `[B,T,D]`; parameters and flattened kernels are row-major. IDs/targets are `[B,T]` u32; validity and loss masks from TASK 004 are optional `[B,T]` byte arrays whose values must be exactly 0 or 1. Missing masks mean all true. Derived query/key masks have the same logical shape. A batch may be a partial final batch with its actual B. A sample is independent; no operation mixes batch elements. A sample's valid mask must be all true or a true prefix followed by a false suffix. PAD=258 can occur only in that suffix for input and target. Token IDs are checked against `[0,V)`. Reserved DOC_SEP=259 is rejected in valid content because the accepted per-document dataset contract does not emit it; no packed-document attention is defined.

Parameters are f32. Reductions, dot products, softmax normalization, and gradient accumulation use sequential f64 accumulators, then round once to f32 at the specified result boundary. Elementwise functions use f32 input/output. Each operation checks inputs and outputs are finite. An overflow/nonfinite value is a typed numerical error and aborts the tape; no nonfinite sentinel is permitted.

## Forward contract

Let `Xids[b,t]` be input IDs, `m[b,t]` validity, and `E[V,D]` the shared embedding/output matrix. Initialize `x[b,t,:]=E[Xids[b,t],:]` when valid and exact zero when invalid. Positions are `t=0..T-1` independently for each sample/document. No learned absolute position table is used.

For each block `l=0..N-1`, with learned RMS gains `gamma_a[l,D]`, `gamma_f[l,D]`:

1. `u = RMSNorm(x,gamma_a)` featurewise at each `[b,t]`, then set invalid rows to zero.
2. `q=u Wq`, `k=u Wk`, `v=u Wv`; each matrix has shape `[D,D]`, no bias. Reshape in exact order `[B,T,H,d]`; head `h` owns contiguous feature slice `[h*d,(h+1)*d)`.
3. For each head and pair `j=0..d/2-1`, apply RoPE to q and k at position t with `theta=10000`, `omega_j=theta^(-2j/d)`: `(a',b')=(a cos(t omega)-b sin(t omega), a sin(t omega)+b cos(t omega))` to coordinates `(2j,2j+1)`. Frequencies are computed in f64, sin/cos then rounded to f32. No rotation is applied to v.
4. For each valid query `(b,t,h)`, permitted keys are exactly `s<=t && m[b,s]`. `z_s = dot(q[b,t,h],k[b,s,h]) / sqrt(d)`; the scale is `1/sqrt(d)`. Stable softmax over only permitted s: subtract maximum, exponentiate finite shifted values, divide by f64 sum. Probability for excluded key is exact zero. A padded query has no row computation and outputs exact zero. A valid query always includes its own key. Empty allowed rows return exact zero probabilities/output and are not treated as an error. `a[b,t,h]=sum_s p_s v[b,s,h]`.
5. Concatenate heads back to `[B,T,D]` using the same contiguous head order; `c=a W_o`, `W_o[D,D]`. Mask invalid rows to zero. Attention residual: `x_a = m * (x + c)`.
6. `r=RMSNorm(x_a,gamma_f)`, invalid rows zero. `g=r W_g` and `w=r W_v`, `W_g[D,F]`, `W_v[D,F]`, no bias. `s=SiLU(g)*w`, with `SiLU(z)=z/(1+exp(-z))`; branch to stable sigmoid for positive/negative values so no overflowing exponential. `f=s W_d`, `W_d[F,D]`, no bias. `x = m * (x_a + f)`.

After N blocks, `h=m*RMSNorm(x,gamma_final)` where `gamma_final[D]`. Logits `L[b,t,v]=sum_d h[b,t,d]*E[v,d]`, shape `[B,T,V]`, no output bias. The same E parameter receives embedding lookup and output-projection gradients. No dropout, stochastic layer, learned position parameters, or other normalization scale is present.

RMSNorm for a row vector x: `r=sqrt(epsilon + (1/D) sum_i x_i^2)`; `y_i=gamma_i*x_i/r`. Epsilon is added before square root. RoPE trig and attention scale are not trainable. Layer sequence is exactly pre-norm as written.

## Backward contract

Every listed operation is differentiable with respect to floating-point inputs and parameters except integer lookup indices and masks. Masks, IDs, and targets are constants. Session B must implement and compare these contracts against the independent oracle; no approximate gradient is allowed.

For `y=ab`, local VJPs are `da=g*b`, `db=g*a`; addition sends g to both edges. For row-major dense `Y=XW`, `dX=G W^T`, `dW=X^T G`, with sums over all leading rows. Bias is absent. A reshape/permutation only reorders storage; its VJP applies inverse permutation. Mask multiplication `y=m*x` returns `m*g` for x and no mask gradient. Residual addition sends the same adjoint to both branches before masks.

Embedding gather: `Y[b,t,:]=E[id[b,t],:]`; VJP initializes `dE=0` and scatter-adds each valid position's `dY` to its ID row in deterministic b,t order. Repeated IDs sum. Invalid positions contribute zero. Output matmul also contributes to the same E gradient; shared-parameter paths must be accumulated.

RoPE per pair with `R=[[c,-s],[s,c]]`: `dx=R^T dy`; frequency/position constants have no gradient. Scaling `z=q dot k / sqrt(d)` gives `dq += dz*k/sqrt(d)`, `dk += dz*q/sqrt(d)`. For row softmax p and upstream `g`, `dz_s=p_s*(g_s-sum_j p_j*g_j)` over permitted keys; excluded positions remain exactly zero. For `a=sum_s p_s v_s`, `dp_s=g_a dot v_s`, `dv_s += p_s*g_a`. All sums are f64 then rounded once per output gradient.

RMSNorm VJP per row: write `r=sqrt(eps+mean(x^2))`, `c=sum_i (g_i*gamma_i*x_i)`; `dx_i=(g_i*gamma_i)/r - x_i*c/(D*r^3)`. `dgamma_i += g_i*x_i/r`. The epsilon derivative is fixed zero. SwiGLU with `s=SiLU(g)w`: `dg = ds*w*(sigmoid(g)+g*sigmoid(g)*(1-sigmoid(g)))`, `dw=ds*SiLU(g)`. Dense VJPs then apply. Loss VJP is specified in the validation plan and is the only root; its scalar seed is 1.

For any node with multiple outgoing paths, add all incoming adjoints in deterministic reverse-topological order using f64 accumulation and one f32 conversion per accumulated element. A nonfinite conversion fails and discards the full reverse pass. Parameter gradients are merged across repeated parameter snapshots and repeated use of a shared parameter using existing accumulation semantics or a dedicated equivalent operation.

## Initialization contract

Use a specified SplitMix64 stream initialized from one u64 seed. For each requested u64, increment state by `0x9E3779B97F4A7C15`, then apply SplitMix64 multipliers/xors `0xBF58476D1CE4E5B9`, `0x94D049BB133111EB`; all arithmetic wraps. Uniform values use the top 53 bits mapped to the open interval `(0,1)` as `(k+0.5)/2^53`. Standard normal values use Box-Muller pairs `sqrt(-2 ln u1)*cos(2 pi u2)`, `sqrt(-2 ln u1)*sin(2 pi u2)`; compute in f64 and round to f32, consuming pairs in row-major parameter order. This exact algorithm/stream order is frozen by fixtures.

Initialize E with normal(0,0.02). Every attention/FF matrix uses Xavier uniform bound `sqrt(6/(fan_in+fan_out))`; residual output matrices Wo and Wd multiply the bound by `1/sqrt(2N)`. All RMS gains are exactly 1. There are no biases. Seed, config, parameter name/order, and algorithm version are recorded. Bitwise reproducibility is required on the same reference implementation/toolchain and deterministic operation order; cross-backend parity is tolerance based, not promised bitwise. No external weights are used.

## Rejections and empty cases

Reject zero dimensions, V mismatch, out-of-range IDs/targets, incompatible masks/shapes, mask bytes outside `{0,1}`, non-prefix padding, PAD in a valid position, DOC_SEP in valid content, a true loss mask at an invalid input position, or any interior shift mismatch `target[b,t] != input[b,t+1]` for adjacent valid positions. PAD is permitted only in false-masked suffix positions. Also reject invalid epsilon/config, nonfinite parameters/input, arithmetic overflow, and resource/work exhaustion. A zero-size tensor, zero batch, zero sequence, or zero supervised loss count is rejected before recording operations. Every failure leaves parameter values, persistent gradients, revision, and caller batch unchanged; tape errors seal/drop the failed tape and release its budget charges. Allocation failure is typed; no partial batch/model state is published.

## Selected versus deferred

Selected: RoPE, MHA, RMSNorm, pre-norm, SwiGLU, tied E, no biases, explicit causal reference attention, f32 reference arithmetic with f64 reductions, mask-aware independent samples. Deferred: learned absolute position embeddings, GQA/MQA, LayerNorm/post-norm, GELU/SiLU-only FFN, untied projection, dropout, sequence packing, memory-efficient/tiled attention, FlashAttention kernel, mixed precision, quantization, checkpointing, inference and GPU dispatch. Future tiled attention must preserve this exact causal, key-mask, padded-query and normalization semantics.
