# TASK 005 — Resource and Scaling Plan

**Status:** analytic, bounded design estimates; no runtime benchmark or training result is claimed.

## Accepted ceilings

Source constants in `xynoos-tensor::TensorLimits`: 64 MiB per tensor for logical f32 payload and retained f32 Vec capacity; rank cap 32. The aggregate-autograd source constants in `xynoos-autograd::ResourceLimits`: 128 MiB charged live allocations per parameter set, 100,000 values/nodes, 100,000 operations, 100,000,000 work units per phase, 10,000 experiment steps. Work is a project accounting unit, not measured FLOPs. Ledger includes charged tensor retained capacities and tape metadata/snapshots/adjoints/staged state; excludes allocator overhead, shape/cell overhead not explicitly charged, stacks, RSS, and OS overcommit. Do not claim an RSS ceiling.

## Parameter arithmetic

For vocab V, width D, layer count N, FF width F, no biases, tied embedding/output, RoPE without parameters:

`P = V*D + D + N*(4*D^2 + 3*D*F + 2*D)`.

Terms are tied E once; final RMS gain D; per layer Q/K/V/O weights 4D²; SwiGLU gate/value/down weights 3DF; two block RMS gains 2D. RoPE, masks, attention and optimizer counters have zero parameters. No positional table, output bias, or per-head parameter is omitted.

## Bounded candidates (actual counts)

| Purpose | V | D | N | H (d) | F | Context cap | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tiny mathematical oracle | 260 | 16 | 1 | 2 (8) | 48 | 8 | 7,536 |
| Small integration reference | 292 | 32 | 4 | 4 (8) | 96 | 64 | 62,880 |
| First-training research candidate | 292 | 96 | 8 | 4 (24) | 256 | 128 | 914,400 |

Counts are exact formula outputs, not model-quality labels. Tiny is deliberately for exhaustive correctness only. Small exercises multiple blocks and real batch/sequence paths but is not a meaningful language-model claim. The 914,400-parameter candidate is a future TASK 006 hypothesis, not selected as a guaranteed quality target; representative approved data and evaluation are prerequisites. Do not shrink a quality objective to fit CI.

## Conservative memory estimates

f32 base storage for first-training candidate: parameters 3.49 MiB; one parameter-gradient inventory 3.49 MiB; transactional SGD candidate snapshot 3.49 MiB at commit. AdamW later would add two f32 moment inventories (6.99 MiB) plus candidate values and gradients, before staging/metadata. Model values and gradients are counted separately. Current parameter-set ledger cannot fit arbitrary concurrent optimizer states without measuring all Rust capacities.

For B=1,T=32,D=96,F=256,N=8, one f32 activation value at each `B*T` position is 12 KiB per D-wide tensor, 32 KiB per F-wide tensor, and 16 KiB per `[H,T,T]` attention matrix (H=4). Per layer, a conservative explicit reference list (input/residual/norm rows, q/k/v and rotated q/k, scores and probabilities, attention output, gate/value/SwiGLU intermediates, FF output) is about 0.3 MiB; eight layers about 2.4 MiB of distinct forward tensors. Reverse adjoints of same-sized outputs can approach another ~2.4 MiB; saved per-op copies/state, parameter snapshots, table capacities, temporary gradients, staging and retained Vec slack may push a plausible peak into tens of MiB. For B=2,T=64, sequence-scaled activations multiply ~4x, attention matrices ~8x, and gradient/recompute staging increases peak further. These are live tensor-payload estimates, not measured Rust allocation/Vec capacities; exact op implementation must instrument ledger peaks and independently account all retained capacities.

Single tensor limit: logits for B=64,T=1024,V=772 would be 206 MiB and invalid; with V=292,B=64,T=1024 logits are 73 MiB and exceed the tensor contract. Model config must preflight `B*T*V*4 <= 64 MiB`, in addition to all other individual tensors. At candidate B=1,T=32,V=292 logits are 36.5 KiB. Attention score/probability tensor size is `B*H*T*T*4`; preflight before allocation.

## Work and budget feasibility

Dense forward arithmetic proxy per layer is approximately `8*B*T*D^2 + 6*B*T*D*F + 4*B*T^2*D` floating multiply/add operations: four D→D attention matrices; three SwiGLU matrices; QK and probability×V. At B=1,T=32,D=96,F=256,N=8 this is about 60 million floating operations forward. Backward dense products are of comparable scale, so forward plus backward is roughly 180 million arithmetic operations before nonlinear/reduction/scatter work. This is not the accepted tape work definition. Existing 100M per-pass work ceiling may fail depending on chosen unit accounting, even when 128 MiB memory fits. Session B must give each op checked, conservative work accounting and prove the candidate fits; if not, propose a bounded model-specific ceiling/API change with new exact fixtures, leaving current `ResourceLimits` ceilings and all existing checks unchanged. Do not label 100M existing units as FLOPs. No claim that million-scale or larger full training fits CI.

The first training candidate must be tested with both inference-only and backward fixtures at an explicit batch/sequence. CI should run only tiny forward/backward fixtures. Meaningful training duration and dataset-token coverage need measured throughput at TASK 006; if CPU throughput or work ceiling blocks the objective, preserve objective and evaluate separately authorized suitable GPU resources. No runtime was provisioned for TASK 005-A.

## Context and KV cache

For MHA, f32 cache bytes for batch B, cached length C, N layers, K and V, width D: `2*N*B*C*D*4` bytes. More generally `2*N*B*C*H*d*bytes_per_element`; here `H*d=D`. Candidate at B=1,C=128,D=96,N=8 uses 786,432 bytes (768 KiB), excluding allocator/container overhead and token/history buffers. MHA KV cache is larger than GQA/MQA with same D because every query head owns a key/value head. Reference context cap is config-defined and never inferred from RoPE extrapolation.

## Family scaling implications

The approved illustrative family (Experimental 100K–50M; Nano 125M; Micro 350M; Edge 1B; Lite 3B; Core 8B; Advanced 14B) remains conditional. At f32, just 14B parameters occupy ~52.2 GiB weights; gradients plus weights ~104.3 GiB and two AdamW moments add ~104.3 GiB, before activations/temporaries. Existing tensor/autograd limits plainly do not support those scales. Larger widths/layer counts also require sharded parameters, streaming/partitioned gradients, larger checked-memory model, optimized GEMMs, tiled attention, checkpointing, mixed precision with parity, and separate checkpoint/inference design. Each requires separate architecture and evidence. Smaller editions remain first-class.
