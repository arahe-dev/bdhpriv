# SAE / sparse population science mission — final report

Status: complete for the locally available evidence; E3/E4 track updated
(2026-09-16, second pass).

E2 finding carried forward: **stable core + conditional tail**.
The E3/E4 mission asked for frozen-corpus reproduction and a causal
static/dynamic/hybrid intervention. The frozen corpus is **not present on
this machine** (re-verified across all local roots and drives), so E3 was
not executed and no E3 claim is made. Everything that can be tested locally
was run: context-length robustness, a structured-recurrence probe, the
core/tail sensitivity ladder, a forward-only oracle ablation pilot, and the
frozen E4 protocol. See section 16 for the new results and the E3 status.

Date: 2026-09-16.
Scope: CPU + RAM only; no GPU benchmarks; no modifications to `opt/`,
`systems/`, `training/`, or the other agent's files.

---

## 1. Questions

The mission asked what kind of sparsity Arm-A training created:

- **H1** static core — a fixed global subset dominates almost every context;
- **H2** conditional populations — per-input sparse subsets that rotate;
- **H3** core + conditional tail — stable backbone plus specialists;
- **H4** frequency-structured populations — organized by RoPE band;
- **H5** artifact — concentration caused by diagnostic sampling, position,
  level effects, or pathological units.

The discriminating questions were global-vs-per-example top-N capture,
population stability over context/lag/training, core/tail decomposition,
frequency band organization under a pair-preserving null, and calibration
against established sparse representations (Gemma Scope SAEs, Anthropic
public features).

## 2. Available evidence

| artifact | what it supports |
|---|---|
| `runs/arm_a_2p5b_opt3c_all/ckpt/{step_18000,step_19000,latest}.pt`, `census_ckpts/step_0000002000.pt` | 4 trained checkpoints of the one Arm-A 2.5B trajectory (step 19,074, 2.50B tokens, final logged loss 2.998, 80.5k tok/s) |
| `runs/arm_a_2p5b_opt3c_all/logs/train.jsonl` | training telemetry (loss, LR, grad norm, throughput, memory) |
| frozen `opt/` execution code (`CensusArmA`, `opt3c_all` flags) | read-only CPU forward path used for all captures |
| `data/sae/gemma-scope-2-270m-pt/.../layer_12_width_16k_l0_{small,medium}` | static SAE structure (L0=20 and L0=60 `jump_relu`) |
| `data/sae/neuronpedia-sae-concepts/anthropic/*.parquet` | 2,149,712 public feature records with activation density (Claude 1 SAE) + 2,999 Claude 3 Sonnet concept records |
| canonical-init control (`collect.py --ckpt random_init`) | E1 calibration of what training created |

**Not available locally:** the frozen corpus (`phase_bdh_stage2_5b_v1`).
Therefore every Arm-A claim here is **E1/E2** (synthetic packed batches,
one trajectory). The explicit E3 upgrade package is prepared and documented
in `analysis/arm_a_science/E3_README.md` (not run).

## 3. Sampling and methodology

- `analysis/arm_a_science/collect.py` runs two CPU passes per checkpoint over
  four synthetic packed specs: mixed doc lengths (mean ~1429, matching the
  frozen contract), single-document windows, and all-128-token documents.
- Per batch (4 rows × 2048 tokens): 2 sampled 128-token chunks per row →
  **4,096 sampled tokens per (level, head)** for per-token statistics;
  population mass/counts use **all 32,768 tokens per level**.
- Pass 1 accumulates per-coordinate mass/positive counts, per-RoPE-pair
  masses, per-token local top-N mass ratios (N = 16…4096), top-64 identity
  sets, and per-token band statistics.
- Pass 2 recomputes, with **cross-fitting by batch parity**: masses captured
  by the frozen top-N ranking built from the *other* split, core-conditioned
  metrics for five predeclared core definitions, and in-line stability metrics
  on sampled token pairs (lag 1–127 within document, cross-document control).
- All samplers, seeds, ladders, and approximation notes are recorded in
  `results/arm_a_science/raw/meta_*.json`. RAM peak was ~6 GB per process
  (batch-level transients only); each checkpoint took ~7–8 min CPU.
- Evidence classes are carried on every claim in `campaigns/sae_science.jsonl`.

## 4. Global vs conditional sparsity (priority 1)

Per-cell means over all 32 (level, head) cells, K=4096 per head, cross-fitted:

| N | x local | x global | x Δ | u local | u global | u Δ |
|---|---|---|---|---|---|---|
| 16 | 0.069 | 0.035 | 0.034 | 0.410 | 0.181 | **0.229** |
| 64 | 0.215 | 0.123 | 0.092 | 0.784 | 0.362 | **0.421** |
| 256 | 0.593 | 0.402 | 0.191 | 0.988 | 0.617 | **0.371** |
| 1024 | 0.988 | 0.787 | 0.201 | 1.000 | 0.882 | 0.118 |

Pooled over heads (N of 16,384): x Δ = 0.008 (N=64), 0.019 (256), 0.063
(1024), 0.134 (4096); u Δ = 0.220 (64), 0.416 (256), 0.374 (1024), 0.112
(4096). Uniform reference is N/K (0.0039–0.25).

Reading:

- **u = x·y is strongly conditional.** A token's own top-64 coordinates
  capture 78% of its u mass while the frozen global top-64 captures 36%;
  every one of the 32 cells shows Δ > 0.05 (range 0.19–0.52). At N=256 the
  local set is essentially complete (0.988) while the global set still misses
  38%.
- **x has a much more global backbone.** Δ is only 0.034 at N=16 (5% relative)
  and grows to 0.20 at N=1024. The extreme top of x is nearly population-fixed;
  the mid-tail is where context changes things. Level 7 is the most
  conditional layer for x (Δ = 0.191 at N=1024).
- **Both effects coexist:** the global ranking itself is far from uniform
  (u global top-64 = 0.362 vs uniform 0.0156), so the correct description is
  a concentrated global hierarchy *plus* a rotating conditional component.

Uncertainty: document-block bootstrap on the pooled numbers (15 documents)
gives, e.g., pooled x level-4 Δ(256) = 0.0121 [0.0105, 0.0157]. Per-cell
ranges are in `results/arm_a_science/topn_global_local.json`.

Stratification (`results/arm_a_science/position_confounds.json`): the u gap
is present at document starts (Δ=0.101), mid-document, in all three packing
modes, and at every position bin; document-level ICC of the per-token gap is
0.31–0.42 for u, i.e. documents genuinely differ in how conditional their
populations are. No position boundary artifact drives the result.

## 5. Core + conditional tail

Cross-fitted core definitions (latest checkpoint; per level-head; sensitivity
over five definitions in `results/arm_a_science/core_tail.json`):

| core definition | size | x core mass share | u core mass share | u core active in |
|---|---|---|---|---|
| global mass top 1% | 41 coords | 0.083 | 0.294 | ~99.6% of tokens |
| global mass top 6.25% | 256 | 0.402 | **0.614** | ~99.6% |
| global mass top 25% | 1024 | 0.787 | 0.879 | ~99.6% |
| p_act ≥ 0.5 | 21–93% of K (level-dependent) | 0.58–0.70 | 0.19–0.40 | ~99.6% |

After removing the top-6.25% core, the remaining u mass is extremely narrow:
the residual top-64 carries 0.883 of it, and only ~34 additional coordinates
(1,024 for x) are needed for half of it. The top-1% "core" alone explains
little of x — x's mass lives in the broad top quartile, not in a handful of
super-neurons.

**Classification: H3 (core + conditional tail) with H2 dynamics in the tail;
H1 holds only for the extreme top of x; H4 is partially supported (section 7).**

**Important scope limit (added after the oracle ablation):** H3 is a
description of population statistics, not an allocation rule. The core is a
stable hierarchy and a prior; the fidelity-relevant population is the
conditional one (section 16.5). Do not phrase H3 as "therefore keep the core
always-on".

## 6. Training maturation

Checkpoint trajectory (same input distribution, one run; init control):

| metric | init | step 2k | step 18k | step 19.1k |
|---|---|---|---|---|
| x zero fraction | 0.499 | 0.651 | 0.756 | 0.752 |
| x pair-zero fraction | — | 0.452 | 0.602 | 0.597 |
| u zero fraction | 0.749 | 0.922 | 0.947 | 0.947 |
| u pair-zero fraction | — | 0.853 | 0.899 | 0.899 |
| y zero fraction | — | 0.725 | 0.690 | 0.695 |
| x Gini | 0.192 | 0.607 | 0.683 | 0.675 |
| u Gini | 0.597 | 0.756 | 0.813 | 0.805 |
| x N_eff/K | 0.927 | 0.500 | 0.408 | 0.417 |
| u N_eff/K | 0.522 | 0.299 | 0.196 | 0.207 |
| u Δ(64) | 0.178 | 0.419 | 0.405 | 0.421 |

Training did not merely increase sparsity; it **built a concentrated,
conditional population** far outside the init baseline. The conditional gap
is already large by step 2k (0.419) and then stable; the global concentration
keeps sharpening until ~18k. Notably x's top-1% mass share is nearly
training-invariant (0.074 → 0.080) while the broad inequality rises strongly —
training reshaped the mid-tail, not the extreme top.

## 7. Frequency organization (RoPE bands)

Pair-preserving permutation null (5,000 draws; pairs kept intact, band
assignment randomized):

- **x**: pooled band mass profile is far outside the null (max share 0.229
  vs 0.143, z=+13.2; entropy deficit z=−65.1); **all 32 (level,head) cells
  are significant (z = 8.6–60.9)**. Canonical init is exactly uniform
  (z=−0.2), so the organization is learned. It is already present at step 2k
  (z=+9.3) and sharpens slightly to step 19.1k (z=+13.2).
- **Head-specific directions** (this is why pooling hides structure):
  H0 and H3 are fast-band dominant (H0 L4 band shares 0.57/0.24/0.09/…),
  H1 is slow-band dominant (0.09/0.13/0.17/0.17/0.15/0.11/0.10/0.09),
  H2 peaks at mid bands. There is **no universal fast/slow direction**.
- **u**: pooled organization is weak (max share z=+0.26; entropy z=−4.6),
  but 12/32 cells are significant, including head 1 late levels (z=27.8)
  where slow bands carry the mass. Per-token, slow bands (b8–b15) carry
  37% of u mass and fast bands 62%; for x it is 19%/81%.
- Per-token band shares are themselves broad: no single band is the argmax
  for more than ~51% of tokens in any cell.

Conclusion: frequency organization is **real and highly significant for x**,
**cell-specific for u**, and should not be summarized as a single global
frequency law.

## 8. Gemma Scope SAE reference (static structure)

Analyzed both downloaded SAEs (config and weights verified against the local
hash manifest). Decoder rows are **unit-normalized by construction**
(Gini(decoder norm) = 0), so decoder norm cannot be used for inequality.

| statistic | L0=20 | L0=60 |
|---|---|---|
| encoder direction norm (median) | 2.06 | 1.83 |
| jump_relu threshold (median) | 113.2 | 64.3 |
| decoder NN cosine (median / p99) | 0.276 / 0.724 | 0.283 / 0.757 |
| encoder NN cosine (median) | 0.475 | 0.502 |
| mutual-NN rate (top-1, decoder) | 0.226 | 0.190 |
| features with NN cosine > 0.9 | 0.11% | 0.20% |
| encoder effective rank (of 640) | 182 | 164 |
| encoder norm Gini | 0.111 | 0.149 |
| Spearman(threshold, encoder norm) | −0.017 | +0.401 |

Feature geometry is low-rank and coherent with a small near-duplicate tail;
the main L0-dependent structural difference is threshold scale and the
threshold–norm relation (`results/sae/gemma_comparison.json`). These are
weight-space facts only; they say nothing about firing rates.

## 9. Anthropic public feature reference

Schema and supported-field analysis only (`results/sae/anthropic_schema.json`,
`results/sae/anthropic_analysis.json`). No Claude weights were downloaded.

- `monosemantic_2023.parquet`: 2,149,712 feature records (Claude 1 SAE via
  Neuronpedia) with `density`, `max_activation`, `concept`, `autointerp`,
  logit lists. Density is extremely heavy-tailed: median 4.3e-5, p90 1.3e-3,
  p99 2.8e-2, max 0.995; Gini 0.926; the top 1% of features account for
  53.6% of total density; ~5% of densities are exactly zero;
  Spearman(density, max_activation) = 0.11.
- `anthropic_concepts.parquet`: 2,999 curated Claude 3 Sonnet feature records
  (groups: random 2,866 / safety 83 / paper 50; three models), with
  `top_activation_text/token/value` but no density — usable only for
  categorical/token summaries.

## 10. Cross-system comparison (dimensionless only)

| system / quantity | denominator | Gini | N_eff/N | top-1% share |
|---|---|---|---|---|
| Anthropic feature density | 2.15M features | 0.926 | 0.057 | 0.536 |
| Arm-A u mass | 16,384 coords | 0.828 | 0.185 | 0.291 |
| Arm-A x mass | 16,384 coords | 0.776 | 0.261 | 0.161 |
| Arm-A x p_act | 16,384 coords | 0.529 | 0.622 | 0.040 |
| Gemma encoder norm² (small/medium) | 16,384 features | 0.23/0.30 | 0.91/0.86 | 0.04 |

Quantities differ in kind (accumulated activation mass vs activation
frequency vs weight-space energy), and the comparison is only made through
dimensionless concentration. In the one like-for-like axis (per-unit activity
frequency), Arm-A's x activation probability (Gini 0.53) is far *less*
unequal than the public Anthropic feature density (0.926). Arm-A is not more
extremely sparse-by-population than established sparse dictionaries; it is
comparable or milder. Rank–frequency log-log slopes: Arm-A u −0.70,
Anthropic density −0.59, Arm-A x −0.46, Gemma encoder norm² −0.29.

## 11. Falsified hypotheses and corrected diagnostics

- **Old "98.6% of x mass in top quarter" (E1)**: measured only 2 early tokens
  (the old accumulator sampled `min(sample_rows, rows)` tokens). On 4,096
  tokens/level the per-token top quarter captures 0.988 but the cross-fitted
  **global** top quarter captures 0.787. The number was accidentally close
  for the wrong scope. **Superseded.**
- **Old u top-25% mass ≈ 0.5**: a denominator/zero-mass artifact. Corrected
  per-token values: u top-64 = 0.784, top-256 = 0.988. **Falsified.**
- **"Slow bands dominate"**: not supported as a global statement. The pooled
  u band profile is null-consistent; x is fast-dominant on average but
  head-specific (H1 slow-dominant). The original BDH slow-frequency
  prediction was not confirmed as a universal ordering.
- **"Position/boundary artifacts"**: stratified deltas persist everywhere.
- **"The population is static"**: rejected for u (Δ up to 0.52/cell) and
  rejected for x's mid-tail; but the extreme x top is near-static.
- **"The population changes slowly over long token ranges"**: rejected in the
  synthetic random-token sample — lag-1 overlap ≈ cross-document overlap
  (1.17–1.37×). **Updated by the recurrence probe (section 16.2):** the
  rotation is content-driven, so with repeated natural content the tail
  persists across the repetition span (u top-64 overlap 0.54 at lag 256,
  0.40 at lag 1024 vs ~0.09 shuffled). "Fast" means "changes when content
  changes", not "cannot persist".

## 12. Remaining uncertainty

1. **Input distribution.** All Arm-A numbers are from random-token synthetic
   packed windows. Natural text has token-to-token correlation and semantic
   recurrence; the conditional-magnitude and lag-persistence numbers could
   change. This is the single largest uncertainty (E3).
2. **Single trajectory, single seed.** Checkpoints are not independent
   training replications.
3. **fp32 CPU census vs bf16 training.** Statistics come from exact fp32
   forwards of the stored weights; training-time bf16 activations are not
   captured.
4. **Core definitions are population-relative.** A core is stable under the
   sampled distribution; drift on other text is untested.
5. **y identity sets were not persisted**, so u's conditional component could
   not be decomposed exactly into x- and y-driven parts (evidence: u tracks y
   in the global hierarchy, 0.81–0.91 vs 0.02–0.22 for x).
6. **External systems are static references.** Gemma statistics are weight
   geometry; Anthropic density is a different model family and tokenizer.
7. **No intervention (E4)** was run: no static-vs-dynamic routing comparison.

## 13. Architecture implications

The oracle ablation (section 16.5) changed this section's conclusion; the
revised model is stated first and the original static/hybrid reasoning is
kept only where still supported.

**Revised model of the phenomenon.** A broad global hierarchy exists, but the
exact high-mass population required for faithful computation is strongly
context-dependent. The stable core is descriptive population structure and a
useful prior — it is **not** evidence that forcing the core to remain
always-on is the optimal compute allocation.

- **Dynamic selection is the architecture-relevant primitive.** Oracle
  per-token selection at 320 coords/head keeps 98.9% of u mass (4.0% logit
  perturbation); static keeps 51.1% (83.3%); hybrid is 91.2% (24.2%).
  At matched width, dynamic has 21× the fidelity headroom of static.
- **The core is a prior, not an allocation.** The 3.125–6.25% global core
  (61% of u mass at 6.25%) remains scientifically real and useful as a router
  prior, initialization bias, fallback population, capacity reservation, or
  regularization target — but it must not be forced into the active set
  unless the learned experiments justify it.
- **Static-only expertization is not viable for u**; a global top-320 budget
  keeps about half the per-token mass. Static remains relevant only for the
  x-side projection, whose extreme top is population-fixed.
- **Frequency routing should be per-head, not global.** Band profiles are
  strongly significant but head-specific (H0 fast, H1 slow, H2 mid).
  Frequency-aware routing is secondary to testing dynamic selection itself.
- **u's hierarchy is attention-driven** (rank correlation with y 0.81–0.91),
  so routing signals should be read from the recurrent state, not the
  feed-forward x. This is an argument that a learned router from v can
  recover part of the oracle gap, not that it will.

## 14. Paper-safe claims (wording)

Allowed (E2, one trajectory, synthetic contexts):

- "Across controlled packed contexts from one trained Arm-A trajectory, the
  high-mass neuronal population is strongly context-dependent while retaining
  a stable global rank hierarchy. Population identity is substantially more
  stable when content repeats, indicating that the apparent rapid turnover is
  driven by context rather than token position alone."
- "The population is consistent with a stable core plus a narrow conditional
  tail: the global top-6.25% of coordinates carries 61% of u mass and is
  active in ~99.6% of contexts; half of the remaining mass needs ~34 extra
  coordinates."
- "Attention heads develop significantly non-uniform RoPE-band profiles that
  are absent at initialization (32/32 cells significant vs a pair-preserving
  null), with head-specific directions."
- "Arm-A's learned population concentration is comparable to or milder than
  public sparse-feature density data on the one dimensionless axis available."

E4-pilot / oracle intervention evidence (must be tagged as such):

- "At matched active width, an oracle context-dependent selection preserves
  the dense computation substantially better than static or core-plus-tail
  selection (4.0% vs 83.3% vs 24.2% relative logit perturbation at 320
  coordinates per level-head)."
- "The fixed-core hybrid is a systems-cost compromise, not a fidelity
  optimum; whether a learned router recovers the oracle advantage is the
  central open question."

Not allowed:

- any frozen-corpus (E3) claim;
- any statement that Arm-A "is" or "matches" an SAE;
- "slow bands dominate";
- the old 98.6%/top-quarter phrasing without the local-vs-global qualifier;
- "static population" or "static sparsity" without the u-path exception;
- claiming a learned-router or training-quality win from the oracle pilot;
- describing the core as an always-on allocation without E4 evidence.

## 15. Experiments required for E3/E4

1. **E3**: run the single G4 cell in `campaigns/E3_G4_SCIENCE_CELL.md`
   (contract: `results/arm_a_science/e3_contract.json`), then
   `e3_finalize.py`. Success is qualitative reproduction of the predeclared
   quantities A–E; outcomes E3-A/B/C are all acceptable.
2. **E4a routing intervention (primary)**: DENSE / STATIC /
   ORACLE_DYNAMIC (reference) / LEARNED_DYNAMIC / HYBRID at matched active
   width. Predeclared ordering: DENSE > ORACLE_DYNAMIC > LEARNED_DYNAMIC >
   HYBRID > STATIC on fidelity, and LEARNED_DYNAMIC > HYBRID > STATIC on
   training loss at matched active compute. Do not reorder after seeing
   results.
3. **E4b router bridge**: oracle top-k recall, oracle mass recall, logit
   distortion, tokens-to-threshold, router entropy, utilization, and
   same-session systems throughput; predeclared gate at 0.5 oracle mass
   recall@64.
4. **E4c frequency routing**: top-r band experts per token/head vs
   frequency-agnostic experts at matched compute — secondary to E4a.
5. **E4d mechanism**: repeat the global-vs-local and stability analyses
   under training on the recurrence tasks; test whether content-driven
   persistence is causal for routing benefit.
6. **Reproduction**: one additional Arm-A seed (if compute allows) to convert
   "one trajectory" caveats into across-run statements.

---

## 16. E3/E4 track update (second pass)

### 16.1 E3 status — blocked, package ready

The frozen corpus (`phase_bdh_stage2_5b_v1`, contract in
`context/frozen_corpus_contract.json`) is not on this machine. No E3
evidence was fabricated. The prepared package is now complete:

- `analysis/arm_a_science/e3_census.py` streams the corpus with the
  production reader and runs the identical pass-1/pass-2 capture; it now
  supports `--context T` (256/512/1024) for the within-training-context
  robustness run and records the truncation design;
- `analysis/arm_a_science/e3_finalize.py` runs the standard analyzers
  against the frozen capture and writes the mission-required names without
  clobbering the E2 artifacts:
  `e3_topn_global_local.json`, `e3_population_stability.json`,
  `e3_core_tail.json`, `e3_frequency_bands.json`, `e3_frequency.json`;
- `analysis/arm_a_science/E3_README.md` contains the exact Colab cell.

Predeclared E3 primary statistics (do not redesign after seeing results):
Δ_u(64), M_core,u(256), per-head band deviation vs the pair-preserving null.

### 16.2 Context-length robustness (E2)

Same checkpoint, identically designed packed batches (exponential doc
lengths mean 0.7·T, same seeds) at T = 256/512/1024/2048:

| T | Δu(64) | Δx(16) | Δx(1024) | core u (6.25%) | band z (x / u) |
|---|---|---|---|---|---|
| 256 | 0.424 | 0.036 | 0.203 | 0.606 | 13.44 / 0.22 |
| 512 | 0.420 | 0.035 | 0.204 | 0.616 | 13.45 / 0.27 |
| 1024 | 0.419 | 0.035 | 0.200 | 0.619 | 13.45 / 0.29 |
| 2048 | 0.421 | 0.034 | 0.201 | 0.615 | 13.42 / 0.24 |

The original E2 raw run at T=2048 reproduces exactly (Δu 0.4215, core
0.6143). Top-64 stability and position strata are also flat in T. **The
structure is intrinsic to the trained weights, not a 2048-window artifact**
(`results/arm_a_science/context_length.json`, FIG 8).

### 16.3 Structured-recurrence probe (E2)

Random-token contexts maximize content change; the probe tests the opposite
regime. A 256-token block repeated 8× (top-64 Jaccard of aligned positions),
against a matched shuffled control with the identical unigram multiset:

| key | lag 256 | lag 512 | lag 1024 | shuffled control |
|---|---|---|---|---|
| x | 0.62 | 0.56 | 0.51 | ~0.13 |
| y | 0.59 | 0.51 | 0.43 | 0.29–0.34 |
| u | 0.54 | 0.47 | 0.40 | 0.09–0.10 |

The effect survives at every depth (u at L7: 0.305 repeated vs 0.118
shuffled; level-0 x = 1.0 trivially). **The conditional tail is
content-driven.** Persistence decays with repetition distance, so the
natural-text timescale remains an E3 question
(`results/arm_a_science/semantic_stability.json`, FIG 9).

### 16.4 Core/tail sensitivity ladder (E2)

Cross-fitted core fractions {1, 3.125, 6.25, 12.5, 25}% of K:

| core | u core mass | x core mass | u coords for 50/80/90% residual | u active width for 90% total |
|---|---|---|---|---|
| 1% (41) | 0.294 | 0.083 | 43 / 86 / 114 | 15.0% of K |
| 3.125% (128) | 0.480 | 0.225 | 39 / 75 / 100 | **5.6%** |
| 6.25% (256) | 0.614 | 0.402 | 34 / 64 / 84 | **8.3%** |
| 12.5% (512) | 0.753 | 0.625 | 29 / 50 / 65 | 14.1% |
| 25% (1024) | 0.879 | 0.787 | 27 / 38 / 46 | 26.1% |

The efficient operating region is a **3.125–6.25% always-on core** plus a
routed tail; the decomposition does not depend on one threshold. x has no
narrow core (90% width 15–31% of K) — the operating region is
u-specific (`results/arm_a_science/core_ladder.json`).

### 16.5 Forward-only static/dynamic/hybrid ablation pilot (E4-pilot)

Frozen cross-fitted rankings; mask u at every level; matched width; relative
L2 logit perturbation vs the dense output:

| configuration (coords/head) | u mass kept | relative perturbation |
|---|---|---|
| static 320 | 0.511 | 0.833 |
| hybrid 256+64 = 320 | 0.912 | 0.242 |
| dynamic 320 (oracle) | 0.989 | **0.040** |
| dynamic 512 (oracle) | 0.999 | 0.003 |
| zero-u (reference) | 0.000 | 0.559 |

**The forward-fidelity prediction hybrid < dynamic < static is falsified.**
At matched width, oracle per-token selection preserves the output far better
than a fixed core; hybrid is a cost compromise, and static masking can damage
the output more than deleting u entirely (0.833 > 0.559) at some layers.
Caveat: dynamic/hybrid read the true u, so they are **oracle upper bounds**;
implementability depends on a learned router (`router_recall@k`), which the
E4 protocol now requires (`results/arm_a_science/e4_forward_ablation.json`,
FIG 10).

### 16.6 E4 protocol — frozen, not run

`results/arm_a_science/e4_static_dynamic_hybrid.json` freezes the arm matrix
(STATIC, DYNAMIC-routed, DYNAMIC-oracle reference, HYBRID + controls), core
size/ranking, widths, router metrics and predeclared gate, mechanism and
language tasks, metrics (loss curves, AUC, tokens-to-threshold, wall-clock,
active width, mass coverage, routing entropy, expert knockout), statistics
(document-block bootstrap, ≥2 seeds), and the decision rule. Execution
requires the GPU training machine; the CPU agent does not own the GPU.

### 16.7 Updated architecture implication

The measured population structure predicts **dynamic conditional compute**,
not static core routing and not hybrid-first: at matched width the oracle
headroom is 21× (vs static) while hybrid gives up 6× fidelity for its
cost advantage. The fixed-core hybrid is a systems-cost compromise, not a
fidelity optimum. The revised model of the phenomenon is:

> A broad global hierarchy exists, but the exact high-mass population
> required for faithful computation is strongly context-dependent. The stable
> core is descriptive population structure and a useful prior; it is not yet
> evidence that forcing the core to remain always-on is the optimal compute
> allocation.

The predeclared E4 ordering is DENSE > ORACLE_DYNAMIC > LEARNED_DYNAMIC >
HYBRID > STATIC on fidelity at matched active width (and LEARNED_DYNAMIC >
HYBRID > STATIC on training loss at matched active compute), with a router
bridge that must recover oracle mass selection to be competitive
(`results/arm_a_science/e4_static_dynamic_hybrid.json`). This ordering is not
to be changed after seeing learned-router results.

### 16.8 CPU phase closed

The CPU descriptive phase is closed. No further SAE or population-statistics
figures will be produced; the next scientific information must come from
frozen natural text (E3) or learned/interventional GPU experiments (E4).

---

## Final claim table

| CLAIM | EVIDENCE CLASS | EFFECT | ROBUSTNESS | PAPER READY? | NEXT TEST |
|---|---|---|---|---|---|
| Training raises x/pair/u zero fractions vs init | E2 | +10.1 pts x, +14.4 pts pairs, +2.5 pts u | init control, all cells, position strata | Yes with "synthetic" qualifier | E3 census |
| Concentration grows far beyond init | E2 | Gini x 0.19→0.68, u 0.60→0.81; N_eff/K halved | init control, all cells | Yes | E3 |
| u is conditional (Δ(64) = 0.42) | E2 | local 0.78 vs global 0.36 | 32/32 cells, doc-bootstrap, modes, positions | Yes | E3 + routing intervention |
| x backbone is mostly global at extreme top | E2 | Δ(16) = 0.034, Δ(1024) = 0.20 | level-7 exception noted | Yes | E3 |
| Conditional tail is content-driven | E2 | lag1/cross = 1.17 x, 1.37 u; repeat 0.54 vs shuffle 0.09 | cross-doc + matched-multiset controls | Yes (synthetic caveat) | E3 semantic recurrence |
| Core + conditional tail (H3) | E2 | core 6.25%: 61% u mass; tail 34 coords for half residual | 5 core definitions | Yes | hybrid architecture test |
| Frequency organization is real for x | E2 | 32/32 cells z>3; pooled z=+13.2; init z=−0.2 | 5000 perms, per-cell nulls | Yes | band routing |
| u frequency organization is weak/cell-specific | WEAK | pooled z=+0.26; 12/32 cells | per-cell nulls | As a negative result | per-head band routing |
| u hierarchy tracks y, not x | E2 | rho(u,y)=0.81–0.91; rho(u,x)≤0.22 | single vector per level | Provisional | store y identity sets |
| Old 98.6% top-quarter x number | E1→superseded | local 0.988 vs global 0.787 | sampling bug documented | Cite only with qualifier | E3 |
| Arm-A ≲ Anthropic density concentration | E1/E2 | Gini 0.78/0.83 vs 0.93 | explicit semantics | Yes, carefully | like-for-like p_act census |
| Structure invariant across T=256–2048 | E2 | Δu 0.419–0.424; core 0.606–0.619 | same seeds/design, position strata | Yes | E3 at T=1024/2048 |
| Tail is content-driven (recurrence probe) | E2 | u repeat 0.54/0.47/0.40 vs shuffle 0.09 | matched multiset control, all levels | Yes (synthetic caveat) | E3 semantic recurrence |
| Core ladder has a stable operating region | E2 | 3.125–6.25% core minimizes 90% width (5.6–8.3% of K) | 5-point ladder, cross-fitted | Yes | E4 arm matrix at 3.125%/6.25% |
| Static-only expertization not viable | E4-pilot | keeps 51% u mass, 83% logit perturbation | matched width, cross-fitted | As pilot | E4 static arm |
| Hybrid-first fidelity prediction | FALSIFIED | dynamic 0.040 vs hybrid 0.242 vs static 0.833 at width 320 | oracle caveat stated | No (pilot) | E4 routed-router arms |
| Gemma SAE structure low-rank, unit decoder norms | E1 static | eff. rank 164–182/640; enc NN cos 0.48–0.50 | both L0 settings | Yes as reference | none needed |
| All Arm-A results are one-trajectory synthetic | constraint | n/a | E3 package prepared and extended | Must be stated | E3 execution + seed reproduction |
