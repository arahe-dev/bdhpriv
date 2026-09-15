# Overnight optimization contract

## Goal
Produce the fastest numerically equivalent, scalable implementation of canonical Arm-A, structured so optimized primitives can later be reused by the B/C/D arms.

## Frozen model contract
- N=16384
- D=256
- H=4
- K=4096
- L=8
- T=2048
- V=8192
- writer hidden=1040
- global batch=64
- Q=K
- strict-past same-document attention
- no softmax
- no attention scaling
- document-local pairwise RoPE, theta=2**16
- shared E/Dx/Dy/coordinator/writer across all 8 depth iterations
- scalar sigmoid(alpha) coordinator gate
- dense two-matrix writer
- positive ReLU neuron activations
- no persistent inter-chunk state
- BF16 autocast, FP32 master parameters
- fused AdamW, lr peak 1e-3, betas (.9,.95), eps 1e-8, weight decay .1
- clip norm 1.0

Do not change architecture, loss, optimizer semantics, initialization semantics, masks, RoPE, parameterization, N/D/H/L/T/V, effective global batch, or number of valid targets.

## Highest-priority exact transformations
1. Replace explicit strict-past QK score-matrix attention with the mathematically exact score-free state/scan formulation.
2. Replace coordinator mask-BMM with an exact segmented prefix-sum formulation.
3. Once memory falls, test removal/reduction of activation checkpoint recomputation.
4. Re-test B64x1, then B32x2, then B16x4.
5. Only after structural wins: fusion, layout, buffer reuse, torch.compile strategies, Triton/CUDA kernels, allocator tuning.

## Proof obligation
Every candidate must pass correctness before performance testing:
- forward equivalence
- loss equivalence
- q gradients
- v gradients
- coordinator z gradients
- packed document boundaries
- irregular segment layouts

Use float64 tiny CPU tests for the mathematical oracle, then BF16/FP32 GPU tolerances appropriate to the production path.

## Benchmarking rules
Performance claims must use complete training updates:
- H2D/input preparation where applicable
- forward
- CE loss
- backward
- grad clipping
- AdamW step

Use CUDA synchronization around timed updates, warmups, repeated samples, and report median/p10/p90.
Never call a forward-only result a training throughput result.

## Local GPU
Overnight development GPU: RTX 4060 Laptop GPU.
Known sustained test: ~98-99% utilization at 35 W for 15 minutes; 58-61 C; safety cutoff 82 C.
Query actual VRAM/PyTorch/CUDA at startup.

Do NOT extrapolate absolute 4060 tok/s to G4.
Use the 4060 to rank candidates and validate CUDA behavior.

If the full production shape does not fit locally, preserve production K/D/H where possible and reduce B/T/L for kernel work. Keep separate:
- tiny correctness tests
- representative kernel microbenchmarks
- full-step scaled integration tests

## G4 re-ranking
Tomorrow, re-run the top 2-3 distinct correctness-passing candidates on:
- NVIDIA RTX PRO 6000 Blackwell Server Edition
- sm_120
- ~95 GiB VRAM
- PyTorch 2.11.0+cu128
- CUDA 12.8

Close local winners may reorder on Blackwell, so retain distinct approaches rather than only one.

## Success ladder
- <1.3x: disappointing
- 1.5-2x: worthwhile
- 2-3x: very good
- >3x: scrutinize correctness aggressively before accepting

## B/C/D boundary
Do not invent B/C/D implementations.
The timing cell declares only adapter requirements:
- apply_moe(base_arm_a)
- apply_phase(base_arm_a)
- D = phase(moe(base)) once those canonical deltas are actually specified.
