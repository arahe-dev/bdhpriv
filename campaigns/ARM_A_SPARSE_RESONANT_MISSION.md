# Arm-A Sparse / Resonant Autoresearch — mission (verbatim)

Provenance: dictated by the project owner on 2026-09-15, following the
completed Arm-A production trainer (`training/arm_a_2p5b_trainer.py`,
commit 9857117) at the measured 80,276 tok/s. Do not weaken or paraphrase
the mission. Baseline implementation is frozen and must not be modified.

---

MISSION
=======

Start from the frozen, certified Arm-A implementation. Arm-A production is
finished and must not be modified.

Your goal is to discover a genuinely stronger BDH derivative through:

1. exact exploitation of natural Arm-A sparsity,
2. structured neuron-axis sparse/MoE computation,
3. explicit resonant/oscillatory dynamics derived from Arm-A's RoPE attention,
4. and only if independently justified, their combination.

Own the entire:
inspect -> derive -> implement -> prove -> train -> profile -> benchmark ->
kill/promote loop.

Do not merely produce proposals or reports. Run the experiments.

The desired outcome is a result important enough to affect the paper:
a large throughput/FLOP reduction at comparable learning quality, a clear
quality gain at matched active compute, or both.


NON-NEGOTIABLE BASELINE
=======================

Treat certified Arm-A as immutable oracle.

Arm-A:
N=16384
D=256
H=4
K=4096/head
L=8
T=2048
writer_hidden=1040
dense canonical coordinator
shared E/Dx/Dy/coordinator/writer across depth
strict-past same-document score-free attention
q = k
RoPE theta = 2^16
B16x4 on G4 production
BF16 autocast / FP32 masters
no activation checkpointing
opt3c_all_b1024 is the frozen execution baseline.

Do not change its source.

Fork every experimental architecture into separate modules.

Never rename or redefine Arm-A.

Every systems speed claim must be same-session against Arm-A.
Every architecture claim must state explicitly that semantics changed.


FIRST: RECONSTRUCT THE MATH
===========================

Before experimenting, write and test the actual per-level equations:

x[t,h] = ReLU(v[t] @ Dx[h])
q[t,h] = RoPE(pos[t], x[t,h])

a[t,h] =
    sum_{s<t, same_document}
      dot(q[t,h], q[s,h]) * v[s]

y[t,h] = ReLU(LN(a[t,h]) @ Dy[h])

u[t] = flatten_h(x[t,h] * y[t,h])

base[t] = LN(u[t] @ E)

then canonical coordinator/writer/residual.

Explicitly identify which operations can be skipped when:
- x coordinate == 0
- y coordinate == 0
- x*y coordinate == 0
- a neuron block/expert is inactive.

Do not assume activation sparsity from the paper transfers to our Arm-A.
Measure it.


PHASE 1 — SPARSITY CENSUS
=========================

Instrument Arm-A without changing outputs.

Measure separately for every level/head and over representative trained data:

x:
- exact zero fraction
- positive fraction
- magnitude quantiles
- top-k mass

y:
- exact zero fraction
- positive fraction
- magnitude quantiles
- top-k mass

u=x*y:
- exact zero fraction
- support intersection size
- top-k mass

Also measure:
- support Jaccard across adjacent tokens
- support Jaccard across levels
- per-document persistence
- per-RoPE-frequency-band occupancy
- block occupancy at block widths:
  16, 32, 64, 128, 256
- co-activation matrix at block granularity
- entropy/concentration by neuron
- fraction of total |u| mass captured by top
  6.25%, 12.5%, 25%, 50% of neurons.

Use a TRAINED checkpoint if available.
Random initialization is not evidence about production sparsity.

Produce machine-readable JSON.

DECISION:
If sparsity is unstructured and GPU-hostile, say so and kill exact sparse
kernel work early.
If x/y/u show block structure or strong concentration, proceed.


PHASE 2 — EXACT SPARSE ARM-A
============================

No architecture change yet.

Exploit facts that are mathematically exact:

A. y values outside support(x) never affect x*y.
B. E only needs nonzero entries of x*y.
C. q inherits zeros from x.
D. attention state updates for zero q coordinates are unnecessary.

Search:

1. partial Dy evaluation only on active x support
2. sparse x*y -> E reduction
3. sparse q state update/read
4. fused versions of the above
5. pair-preserving neuron permutation to cluster co-active neurons into blocks

Important:
RoPE operates on coordinate pairs.
Any neuron permutation must preserve 2D RoPE pairs and carry the associated
frequency with the pair. Prove permutation equivalence before benchmarking.

Try to convert unstructured sparsity into block locality through an EXACT
offline permutation of neuron pairs before considering approximate pruning.

Correctness gate:
FP64 tiny oracle
FP32 logits
BF16 logits
all gradients
packed boundaries
multiple segment patterns.

Exact branch tolerance must match the established Arm-A numerical envelope.

Promotion:
do not keep an exact systems change unless it improves full compiled
forward+backward+optimizer by >=5% same-session.
Microkernel wins that disappear in full-step benchmarking are dead.


PHASE 3 — STRUCTURED NEURON MoE
===============================

Now architecture changes are allowed.

Do NOT add conventional FFN experts.

Partition the existing BDH neuron dimension.

Within each head, divide the K=4096 coordinates into M experts while
preserving RoPE pairs.

Initial grid:

M = 8, 16
top_r = 1, 2, 4

Prioritize active fractions:
1/2
1/4
1/8

Router:
router_logits[t,h] = v[t] @ Wr[h]
select top_r experts
softmax only selected logits
negligible router dimension D -> H*M.

Route BEFORE expensive neuron computation.

For inactive experts do not execute:
- corresponding Dx columns
- attention/state work
- corresponding Dy columns
- corresponding E rows.

This is critical. A mask after dense Dx does NOT count as sparse MoE.

The routed attention semantics are:
each head/expert owns an independent strict-past state;
a token interacts through expert e only when routed to e.

Dense D-dimensional v remains the cross-expert communication channel between
BDH levels.

Test two stabilizers only if required:
- one small always-on shared expert
- standard load-balancing auxiliary loss.

Do not expand the hyperparameter grid until basic routing works.

Measure:
- actual active MAC/token
- measured wall time
- peak memory
- expert load CV
- routing entropy
- dead experts
- token-to-token route persistence
- route specialization by level/head
- loss at equal tokens
- loss at equal measured compute.


PHASE 4 — FREQUENCY-BAND / RESONANCE BRANCH
===========================================

Derive this from Arm-A rather than inventing an unrelated oscillator.

For one RoPE pair:

q_t = R(p_t * omega) x_t

therefore:

q_t^T q_s
=
x_t^T R((p_s-p_t)*omega) x_s.

Write and numerically prove this identity.

This means current Arm-A attention is already an oscillator-bank kernel;
the oscillator is currently fixed by RoPE rather than being an independent
trainable module.

Build frequency bands from contiguous RoPE pairs.

Start with M=8 bands/head.

Run these ablations independently:

O1 — learned frequency scale
omega[h,b] <- omega[h,b] * exp(beta[h,b])
beta initialized exactly 0.

O2 — stable resonant decay
give each band a trainable lambda[h,b] in (0,1],
initialized extremely close to 1.

This changes contribution with lag from:
R(delta * omega)

to approximately:
lambda^delta R(delta * omega).

O3 — learned phase/frequency correction
small bounded delta_omega or phase offset per head/band.

Do NOT start with arbitrary complex neural oscillators.
Keep the state interpretable as a stable bank of damped rotating modes.

Track:
- learned frequencies
- learned decay time constants
- frequency-band activation
- gradient contribution by band
- memory-lag sensitivity
- whether slow bands specialize for long-range/predictable structure.

Use synthetic lag/repetition/retrieval probes only as diagnostics.
Final promotion requires frozen-corpus language-training evidence.


PHASE 5 — RESONANT EXPERT BDH
=============================

Only run this if Phase 3 routing and Phase 4 frequency structure each show
independent value.

Make the experts equal to RoPE-frequency bands.

Each expert therefore owns:
- a contiguous set of RoPE pairs
- its Dx slice
- its Dy slice
- its E slice
- its own attention/synaptic state
- optional learned stable decay/frequency correction.

Router selects top-r frequency experts per token/head.

This gives structured, hardware-friendly sparsity while retaining a physical
interpretation:

the token excites only a small number of resonant neuron populations.

Search in this order:

8 bands, top-2
16 bands, top-4
16 bands, top-2
8 bands, top-1

Do not brute-force everything.

If hard routing destabilizes early training, use dense/soft warmup followed
by deterministic sparsification, but report the compute cost honestly.


TRAINING / MULTI-FIDELITY PROTOCOL
==================================

Stage A: correctness
tiny shapes, FP64 oracle.

Stage B: mechanism screen
same implementation structure, short context/small batch where required.
Kill obviously broken candidates.

Stage C: local 4060 full architecture
T2048/L8 whenever memory permits.
Benchmark complete training updates, not forward-only.

Stage D: learning screen
same seed, same corpus order, same optimizer family.
Use short fixed-token runs to compare learning curves.

For every candidate compare:
- loss after same input tokens
- loss after same measured active FLOPs
- wall-clock loss improvement
- stability
- memory.

Do not promote based on a single final loss number.
Compare the curves.

Only the top 1-2 candidates deserve a G4 confirmation.


PROMOTION BARS
==============

SYSTEMS WIN:
>= 1.20x full-step local speedup is interesting.
>= 1.35x is strong.
Anything <1.05x is dead.

ARCHITECTURE EFFICIENCY WIN:
>=2x reduction in active neuron-path FLOPs with <=1% loss regression
after the short training screen.

CAPACITY WIN:
At matched active FLOPs, materially increase stored neuron capacity N and
beat Arm-A loss.

QUALITY WIN:
>=2% relative validation-loss improvement at approximately matched compute,
reproduced across at least two seeds or clearly separated learning curves.

PAPER-LEVEL / "BOMBASTIC" WIN:
one of:
- >=2x end-to-end throughput at comparable learning quality,
- >=4x active neuron sparsity with negligible quality loss,
- materially better loss at matched wall-clock,
- larger stored model at Arm-A active compute with better learning,
- or a combined sparse-resonant model that dominates Arm-A on both
  compute and quality.


KILL RULES
==========

Kill immediately:
- masking after dense computation
- unstructured sparsity with no kernel path
- router overhead that destroys savings
- expert collapse that survives balancing
- microbench-only wins
- approximate sparse execution masquerading as exact
- random-init sparsity claims
- frequency tweaks without measurable learning effect
- custom Triton before profiling proves the operation deserves it.

Do not spend hours polishing a 2% result.


KERNEL RULE
===========

Only after a sparse/resonant architecture wins mathematically and in proxy
training, investigate Triton/CUDA.

The likely worthwhile custom kernel is a grouped expert kernel combining as
much as possible of:

routed Dx
RoPE
strict-past expert-state read/update
routed Dy
x*y
E reduction

Avoid materializing dense K=4096 intermediates for inactive experts.

Profile before writing it.


LEDGER
======

Maintain autoresearch_sparse_resonant.jsonl.

Every experiment records:

commit
branch
hypothesis
semantic_class = exact | architecture_change
shape
router config
frequency config
active fraction
params stored
active MAC/token
compile status
graph breaks
correctness maxima
tokens trained
train loss
validation loss
full-step ms
tokens/sec
peak memory
comparison baseline measured in same session
decision = kill | hold | promote
reason.

No cross-session speedup claims.


FINAL DELIVERABLE
=================

Return:

1. exact Arm-A sparsity census,
2. exact-sparse execution verdict,
3. MoE sweep Pareto frontier,
4. oscillator/resonance sweep Pareto frontier,
5. combined result if justified,
6. same-session benchmark against Arm-A,
7. short-training learning curves,
8. code for the single champion,
9. exact command for G4 confirmation,
10. list of every killed idea and why.

Do not touch the running Arm-A production trajectory.


---

## Campaign notes from the owner (verbatim)

There is a potentially important **exact** result hidden before MoE.

Because

```python
ypre = relu(a @ Dy)
prod = x * ypre
```

there is literally no mathematical reason to compute `ypre[j]` when `x[j] == 0`.

And because

```python
base = (x * ypre) @ E
```

only surviving neuron coordinates enter `E`.

So the first sparsity census might reveal an **exact sparse Arm-A** acceleration independent of any new architecture. If the support can be made block-friendly through a function-preserving neuron-pair permutation, that is another pure systems result on top of 80.3k.

Then the architectural step goes further and avoids even calculating the inactive `Dx` columns by routing **before expansion**.

That distinction matters:

> natural sparse Arm-A = exact implementation win
> routed neuron MoE = architectural compute win

And the frequency version gives us a very clean third layer. Arm-A's RoPE is not merely positional decoration here; because \(Q=K\), its dot-product kernel is explicitly a sum of rotating pairwise modes. The original paper's own frequency-bucket experiment reports the strongest memorization/repetition activity difference in slower-frequency populations.

If this campaign hits, the strongest result is not "BDH + MoE."

It is closer to:

> **BDH learns with a small dynamically selected set of resonant neuron populations, allowing stored neuronal capacity to decouple from active training compute.**
