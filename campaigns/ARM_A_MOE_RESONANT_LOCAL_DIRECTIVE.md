# Arm-A MoE / Resonant local autoresearch directive (verbatim)

Provenance: dictated by the project owner on 2026-09-15. This supersedes the
census-first framing of `campaigns/ARM_A_SPARSE_RESONANT_MISSION.md`: the
trained sparsity census is now optional later evidence, not a prerequisite.
The local RTX 4060 (through the `iclr-arm-a` container) is the engine.
Frozen Arm-A must never be modified.

---

NEW CAMPAIGN DIRECTIVE
======================

Ignore the previous idea that trained sparsity census evidence is required
before experimentation.

You are the local autoresearch engine.

Your mission is to take frozen Arm-A as the architectural starting point and
discover a MoE, resonant/oscillatory, or combined variant that trains at:

    same speed as Arm-A with materially better architecture/capacity

or preferably:

    much faster than Arm-A while preserving its useful BDH structure.

Run the entire research loop on this local RTX 4060:

derive
-> implement
-> correctness
-> torch.compile
-> full training-step benchmark
-> profile
-> mutate
-> kill/promote
-> repeat.

Do not wait for Google Drive, Colab, corpus checkpoints, or census JSON.

Those are later confirmation artifacts.

ARM-A ITSELF IS FROZEN.
Never modify the canonical/production Arm-A source.


============================================================
0. ESTABLISH THE LOCAL ARM-A ORACLE
============================================================

Use the already established optimized Arm-A as the baseline.

Benchmark every candidate in the SAME PROCESS / SAME SESSION against Arm-A.

Primary systems shape:

T=2048
L=8
B=1
full forward + CE + backward + clip + AdamW
BF16 autocast
FP32 parameters
torch.compile(default)

Use deterministic synthetic packed-document batches when real corpus data
is unavailable.

Use:
3+ warmups
10+ measured steps
randomized candidate/baseline order
median + p10/p90
peak allocated memory.

No cross-session speedup claims.

Also profile Arm-A once and keep a component ledger:

Dx projection
RoPE
local QK
carry/state read
state update
Dy
x*y
E
coordinator
writer
readout
optimizer.

The campaign optimizes the whole update, not kernels in isolation.


============================================================
1. BUILD AN EXACT EXPERTIZED ARM-A FIRST
============================================================

Partition every head's K=4096 coordinates into experts while preserving
RoPE pairs.

Start:

M=8 experts/head
Ke=512 coordinates/expert
256 RoPE pairs/expert

Transform/slice consistently:

Dx[h,:,expert_slice]
Dy[h,:,expert_slice]
E[corresponding N rows,:]
RoPE frequencies belonging to those pairs.

IMPORTANT ARM-A IDENTITY:

The attention contribution decomposes additively:

dot(q_t,q_s)
=
sum_e dot(q_t,e, q_s,e)

Therefore define:

a_e[t] =
    sum_{s<t} dot(q_t,e, q_s,e) * v_s

a_total[t] = sum_e a_e[t]

Then:

y_e[t] = ReLU(LN(a_total[t]) @ Dy_e)

and:

base[t] =
    LN(sum_e ((x_e * y_e) @ E_e))

When ALL experts are active this architecture MUST be numerically equivalent
to Arm-A.

This is the foundational correctness test.

Call it:

ExpertizedDenseArmA

Required:
FP64 tiny oracle
FP32 logits
BF16 logits
all gradients
packed documents
cross-sequence boundaries
same-document reset
full-model equivalence.

Do not proceed until this passes.


============================================================
2. HARD-ROUTED NEURON MoE
============================================================

Now route experts BEFORE Dx.

For token/group g:

router = normalized(v_group) @ Wr
selected = top_r(router)

Only selected experts execute:

Dx slice
RoPE slice
expert attention contribution
Dy slice
x*y
E slice.

Inactive experts must produce ZERO K-dependent arithmetic.

A dense computation followed by masking is NOT a MoE implementation.

Use the shared Arm-A attention output:

a_total[t] =
    sum over ACTIVE expert contributions

then active Dy slices consume the shared a_total.

This should be the FIRST sparse architecture because it stays closest to
Arm-A.


============================================================
3. GPU-FRIENDLY ROUTING GRANULARITY
============================================================

Token-wise routing may destroy the speedup through gather/scatter overhead.

Therefore autoresearch routing granularity aggressively.

Search:

route_group_tokens =
    1
    16
    32
    64
    128
    256
    512

For grouped routing, one expert set is chosen for the contiguous token group.

The group router can initially use:

mean(v over group) @ Wr

or:

LN(mean(v)) @ Wr.

A group-level route is an architecture change and that is fine.

The reason is systems performance:
contiguous groups let each active expert execute large dense GEMMs instead of
hundreds of tiny irregular operations.

Measure route overhead separately.

Prefer static top-r shapes so torch.compile sees a stable graph.


============================================================
4. FIRST MoE SEARCH GRID
============================================================

Do NOT brute-force hundreds of configs.

Run this information-efficient grid first.

A. SAME STORED WIDTH

M=8, Ke=512

top_r=4 -> Kactive=2048   (2x wide-path reduction)
top_r=2 -> Kactive=1024   (4x)
top_r=1 -> Kactive=512    (8x)

Test routing groups:

64
128
256

That's 9 serious candidates.

B. EXPANDED STORED CAPACITY

If top-2 works mechanically:

M=16
Ke=512

stored K/head = 8192
active K/head = 1024

This gives:

2x stored neuron capacity
1/4 Arm-A active neuron width.

This is the MoE result we care about most.

Also test:

M=16
Ke=256
top_r=4
stored K=4096
active K=1024

to separate "more experts" from "more stored capacity".


============================================================
5. ROUTER IMPLEMENTATION
============================================================

Keep it tiny.

Per head:

Wr: D -> M

Router overhead must be negligible compared with Dx.

Start with hard top-k plus normalized selected weights.

Do NOT use a giant routing MLP.

Record:

expert utilization
load CV
routing entropy
route persistence
dead experts
router time
gather/scatter time.

If router collapse occurs, try exactly these in order:

1. simple load-balancing auxiliary loss
2. capacity-normalized logits
3. one always-on shared expert

Do not add more machinery unless one of those fails.


============================================================
6. TWO MoE ATTENTION SEMANTICS
============================================================

Test BOTH, but in this order.

SEMANTICS A — SHARED Arm-A OUTPUT

For each active expert:

a_e = expert q/state contribution

a_total = sum active a_e

Then all active experts use:

y_e = ReLU(LN(a_total) @ Dy_e)

This is closest to Arm-A.
PRIORITIZE THIS.


SEMANTICS B — INDEPENDENT EXPERT MEMORY

Each active expert computes:

a_e
y_e = ReLU(LN(a_e) @ Dy_e)

Then combine expert products through E.

This is more like conventional MoE memory specialization.

Only continue B if it shows a clear mechanism or performance benefit.


============================================================
7. OSCILLATORY / RESONANT BRANCH
============================================================

Remember:

Arm-A ALREADY contains an oscillator bank because RoPE and Q=K imply:

q_t^T q_s
=
x_t^T R((p_s-p_t) omega) x_s.

Do not bolt on an unrelated oscillator network.

Use the existing RoPE-pair structure.

Make experts contiguous frequency bands.

For M=8 and K=4096:

each expert = 256 adjacent RoPE pairs.

Implement three architecture flags.


O1 — LEARNED FREQUENCY SCALE

For expert e:

omega'_e,j = omega_e,j * exp(beta_e)

beta initialized exactly 0.

At beta=0:
identical to baseline RoPE.

The beta tensor is tiny and cached phase generation means runtime overhead
should be nearly zero.


O2 — LEARNED BAND AMPLITUDE

q_e <- gamma_e * q_e

gamma initialized 1.

This gives the network direct control over resonance strength.

Negligible runtime overhead.


O3 — DAMPED RESONANCE

Weight a past expert contribution by:

lambda_e ^ lag

with lambda_e in (0,1], initialized close to 1.

Do this only after O1 is stable.

Implement an efficient chunkwise formulation.

Do NOT accept a version that introduces a sequential Python token loop.


IMPORTANT DEAD END:

A constant phase offset applied identically to Q and K is useless here.

Because Q=K:

R(phi)q_t dot R(phi)q_s
=
q_t dot q_s.

Do not waste experiments on a shared constant phase offset.


============================================================
8. OSCILLATORY SPEED RULE
============================================================

O1 and O2 should be essentially speed-neutral.

Reject them if they cost >3% full-step performance without a demonstrated
mechanism advantage.

O3 may cost more.

Do not keep O3 unless either:

- it is <=5% slower and clearly improves a mechanism task,

or

- it combines with routing and remains faster than Arm-A overall.


============================================================
9. RESONANT MoE — MAIN CHAMPION CANDIDATE
============================================================

Combine the best routing implementation with frequency-band experts.

Example:

M=8 frequency experts/head
Ke=512
top_r=2
route_group=128

Each selected expert executes:

Dx_e
RoPE with expert frequency band
optional learned beta_e
expert score-free state read/update
shared a_total accumulation
Dy_e
x_e*y_e
E_e.

This gives:

Kactive = 1024/head
vs Arm-A 4096/head.

The token group therefore excites only a small subset of frequency bands.

Call this family:

ResonantExpertArmA


============================================================
10. COMPILER-FIRST IMPLEMENTATION
============================================================

Do not design a mathematically elegant routing system that torch.compile hates.

Every candidate must record:

graph_count
graph_break_count
compile time
generated kernel count if available.

Prefer:

fixed M
fixed top_r
fixed route-group size
static expert capacities
contiguous expert slices
preallocated output buffers.

Avoid Python loops over individual tokens.

A Python loop over 8 or 16 static experts is acceptable initially if Dynamo
unrolls it cleanly.

Later replace it only if profiling says it matters.


============================================================
11. THREE EXECUTION STRATEGIES
============================================================

For each serious MoE configuration, implement / compare:

E1 — STATIC EXPERT LOOP
For each expert:
select assigned contiguous groups
dense GEMM
scatter result.

E2 — SORT/GROUP
Sort route groups by expert id,
execute large expert batches,
unsort.

E3 — FIXED SLOT
Preallocate fixed expert slots and pack groups into them.

Kill losers quickly.

Do not write Triton yet.


============================================================
12. LOCAL QUALITY / MECHANISM TASKS
============================================================

The local machine does not need the 5B corpus to decide which architecture
is worth G4 time.

Use generated tasks.

A. repetition / delayed-copy
B. associative retrieval
C. variable-lag retrieval
D. nested repeated motifs
E. packed independent documents
F. noisy long-range recall.

Compare:

Arm-A
best MoE
best oscillator
best ResonantMoE.

Measure:

loss vs updates
loss vs wall-clock
loss vs estimated MACs
long-lag accuracy
routing specialization
frequency specialization.

These are mechanism screens, NOT language-model claims.


============================================================
13. SPEED PROMOTION BARS
============================================================

Same-session T2048/L8 full update.

Relative to local optimized Arm-A:

<1.05x:
kill unless unique mechanism result

1.05–1.20x:
interesting but not champion

1.20–1.50x:
real systems win

1.50–2.00x:
strong G4 candidate

>2.00x:
immediate priority for G4 confirmation.


The stretch target:

ResonantExpertArmA
M=8
top2
Ke=512
route group 64–256

should attempt to achieve:

>=1.5x full-update speed

while retaining enough mechanism quality to justify real training.


============================================================
14. PARAMETER / COMPUTE LEDGER
============================================================

For every candidate record BOTH:

stored parameters
active parameters/token/group

and:

stored K/head
active K/head
stored N
active N
estimated MAC/token
measured full-update ms.

This is essential.

The strongest result may be:

2x stored neuronal capacity
with <= Arm-A active compute

rather than pure wall-clock speed.


============================================================
15. DO NOT WAIT FOR THE CENSUS
============================================================

The trained sparsity census is a later diagnostic.

It may help explain:
- why certain experts/bands work,
- which frequencies specialize,
- whether learned routing mirrors natural Arm-A sparsity.

It is NOT permission to start experimenting.

Proceed now.


============================================================
16. AUTORESEARCH LOOP
============================================================

Use this loop continuously:

1. formulate ONE concrete hypothesis
2. implement minimum candidate
3. correctness test
4. compile
5. same-session full-step benchmark
6. profile if result is ambiguous
7. mechanism screen if architecture survives
8. record JSONL
9. kill or promote
10. mutate winner.

Do not ask for permission between iterations.

Focus on >=5% effects.

A failed idea should usually die within one iteration.


============================================================
17. REQUIRED FIRST 12 EXPERIMENTS
============================================================

Run in this order:

01 ExpertizedDense M8 all-active equivalence
02 M8 top4 group128
03 M8 top2 group128
04 M8 top1 group128
05 M8 top2 group64
06 M8 top2 group256
07 M16 Ke256 top4 group128
08 M16 Ke512 top2 group128 expanded-capacity
09 best candidate + O1 frequency scaling
10 best candidate + O2 resonance amplitude
11 dense Arm-A + O1 only
12 best MoE + O1 + O2 = ResonantExpertArmA

Then choose the next experiments from the data.

Do not precommit to experiment 13.


============================================================
18. OUTPUT
============================================================

Maintain:

campaigns/autoresearch_moe_resonant_local.jsonl

Each row:

commit
timestamp
hypothesis
architecture
M
Ke
top_r
route_group
attention_semantics
oscillator_flags
stored_params
active_params
stored_K
active_K
estimated_MAC_token
correctness
graph_breaks
compile_ok
full_step_ms
tok_s
peak_mem
speedup_same_session
mechanism_task_results
decision
reason.

The goal is not a report.

The goal is executable code plus a champion.
