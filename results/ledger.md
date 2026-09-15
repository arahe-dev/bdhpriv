# Optimization ledger (ARM-A AUTORESEARCH II, 2026-09-15)

Env: RTX 4060 Laptop 8GB, driver 595.79 / CUDA 13.2 WSL host. Container
`iclr-arm-a` (python:3.13-slim + torch 2.11.0+cu128, --gpus all, repo
mounted at /workspace/iclr-oc). G4 = RTX PRO 6000 Blackwell SE, 95GB.

## G4 hard evidence (anchor truth, direct measurement)

| config | ms/update | tok/s | peak alloc | note |
|---|---:|---:|---:|---|
| canonical B32x2 SAC | 2882.14 | 45,477 | 75.73 GiB | baseline |
| champion b256 B16x4 | 2056 | 63,740 | - | block tuning is HW-specific |
| champion b512 B16x4 | 2016 | 65,015 | - | laptop preferred b512 |
| **champion b1024 B16x4** | **1761.27** | **74,419** | **58.34 GiB** | **1.636x, current best** |
| checkpoint variants | ~2500 | 51-53k | - | full-checkpoint penalty |
| packed b512 boundary oracle | - | - | - | PASS ~1e-14 |

## Local results, quiet regime (same-session shuffled comparisons ONLY)

T2048_L8 B1 on 4060 (K/D/H/L production; batch reduced to fit 8GB):

| variant | eager ms | compiled ms | note |
|---|---:|---:|---|
| ref_ckpt (control) | 669.0 | 222.7 | CPU-overhead sensitive |
| champion chunkwise b512 | 333.1 | 168.0 | 2.01x eager / 1.33x compiled |
| champion chunkwise b1024 | 336.0 | 187.0 | 1.99x / 1.19x |
| parallel b512/b1024 | 353.1/366.6 | 182.7/210.4 | KILLED (twice-lost) |
| hybrid b512/b1024 | 362.3/366.9 | 188.5/204.9 | KILLED |
| static4 b512 (eager fallback) | **308.4** | 171.5 (tie) | -11% eager; kept as fallback |
| diet b512/b1024 | 379.8/385.0 | 195.3/217.0 | KILLED (+10% time) |

Packed retention, T2048 B1, champion (classify=False, branch-free), eager
(P3 session, seed 95; single measured first = warm-order row) and compiled
(P3 comp session, seed 96, b512; single row had bypass on):

| layout | eager ms | compiled ms | ref compiled | compiled x | retention |
|---|---:|---:|---:|---:|---:|
| single | 354.6* | 174.6 (bypass) | 233.3 | 1.34x | 100% |
| mixed (exp ~1429) | 331.7 | 174.4 | 234.1 | 1.34x | 100.1% |
| four docs | 360.5 | 181.6 | 230.4 | 1.27x | 96.1% |
| heavy (128-tok docs) | 353.7 | 182.8 | 230.8 | 1.26x | 95.5% |

*order-0 row; P2 measured single at 332.8 with n=2. Session-order effects,
not a mechanism difference. Mixed (corpus-realistic density) is free:
identical to single-doc in both modes.

Before the branch-free state update (P1): two/four/heavy were 363.8/363.4/
363.8 ms (retention 91.5%). The rewrite (single `state*contb + fresh`,
dead final-chunk update removed) gave **-6% on all packed modes**.

Bypass (explicit segment_start + single_doc, no [B,T,T] derivation, no
.item() graph break): compiled single 174.6 vs 174.4 for mixed which still
pays the derivation+sync — i.e., invisible in the compiled graph. Kept as
an option, NOT promoted.

Chunk-classification fast path: killed (see table above): host-side `.all()`
syncs per chunk per level cost +9..14% on packed. `classify=False` is now
the default everywhere.

## Measurement noise / regime drift (CRITICAL)

1. Within-run spread <1%; between-launch up to 12.7%. bench_matrix shuffles
   (--seed) and tags session/order; replicate only when it changes a decision.
2. **Regime drift**: ref_ckpt eager drifted 909->669ms and compiled
   336->223ms over ~2h as host CPU contention changed (ChatGPT/browsers/mpv/
   git; CPU at 2.6GHz base, Balanced plan). Champion moved <5%. ref is
   Python-overhead-sensitive (SAC policy callbacks); nockpt champion is
   GPU-bound. Consequence: the earlier 1.90x compiled / 2.63x eager claims
   are DEAD. Re-baselined same-session: eager 2.01x, compiled 1.33x (b512) /
   1.19x (b1024). G4's 1.636x is a direct same-run measurement: keep.
3. Docker incident 09:12 local: Windows Update servicing window (WMI
   service start-type flips 09:09/09:14) coincided with Docker backend exit
   status 150; container SIGKILLed (137). P2 run had completed; zero data
   lost (per-row flush to host-mounted results file). Recovered by killing
   stale backend + relaunch.

## E0 findings (profile + saved-tensor census, T2048 B1)

- b512 compiled full update 172.6ms CUDA (fwd 65.4). Spend: local QK bmm
  ~29% of fwd; dense GEMMs (project_x/encoder/Dy/readout) ~35-40%;
  RoPE/paper/other elementwise ~27%. b1024: 185.8ms.
- Census compiled b1024: 2.578 GiB saved (246 tensors) vs 5.463 GiB eager.
  Inductor already recomputes the project_x output in backward and fuses
  RoPE. Remaining big saves: paper_y [T,N], rope stack, ypre/q-side
  [4,2048,4096], encoder-side [16384,2048] (4 x 512 MB) + scores 128 MB.
  => little untouched activation memory left for the diet; E1 confirmed it.

## ARM-A III (2026-09-15): branch-free packed, zero-carry, paper/RoPE

### P1 — branch-free packed state update is now the real implementation
- `scan_chunkwise_bthk`: single branch-free recurrence
  `S = S*contb + (keep*q_b)^T v_b`, no full-bmm, no where-select, dead
  final-chunk update skipped. `scan_chunkwise_where_bthk` kept ONLY as the
  certified A/B control. Host-sync classification path deleted.
- New static production entries `forward_single_doc` / `forward_packed`
  (no `.item()`, no `.all()`, no segment derivation in the compiled graph).
  Harness compiles those entries, mirroring the certified G4 script.
- Gate `test_packed_ab_gate.py` (32 cases incl. doc-starts-inside-block,
  continuation across boundaries, all-resets, random packed): branch-free
  vs dense 7.1e-15; branch-free vs certified where **bitwise identical**
  (0.0) on fwd/gq/gv and on model logits/loss/grads. All gates pass
  (test_correctness 180, test_model_equiv).
- Same-session compiled A/B (T2048 B1, 4060):
  | layout | where (certified) | branch-free | delta |
  |---|---:|---:|---:|
  | b512 single | 174.7 | 173.4 | -0.7% |
  | b512 mixed | 197.4 | 181.0 | **-8.3%** |
  | b512 four | 196.6 | 180.6 | **-8.1%** |
  | b512 heavy | 195.9 | 181.3 | **-7.5%** |
  | b1024 single | 194.9 | 194.6 | -0.2% |
  | b1024 mixed | 207.4 | 198.9 | **-4.1%** |
  | b1024 four | 206.2 | 198.1 | **-3.9%** |
  | b1024 heavy | 206.9 | 197.1 | **-4.7%** |
  Eager mirrors it (-4.0..-6.4%). At the laptop-preferred b512 the packed
  gain clears the 5% bar; at b1024 it is 3.9-4.7% (block-dependent share
  of the removed bmm). Exact, so G4 promotion risk is zero.

### Zero-carry skip (new mechanism, exact)
- Chunk at t0==0 has S=0 exactly, so its carry bmm (`q_b @ 0`) and the
  add are provably dead; skipped for both single-doc and packed
  (`seg<0` is never true). Gate: zc vs dense 7.1e-15.
- Same-session compiled A/B vs branch-free (T2048 B1):
  | block | single | mixed | four | heavy |
  |---|---:|---:|---:|---:|
  | b1024 | **-3.9%** | **-3.8%** | **-4.8%** | **-3.5%** |
  | b512 | +2.4%* | -3.4% | -1.3% | -3.8% |
  (*b512-single anomaly is a single-row order artifact; b1024 is
  consistent across single + all packed layouts.)

### paper_y / RoPE candidates (priority 3, exact, bench in flight)
- E0 kernel census b1024 fwd (65.4 ms): local QK 19.3 (29.4%), paper mul
  9.1 + paper transpose-copy 4.7 = **13.8 (21.1%)**, RoPE 5.3 (8.2%),
  dense GEMMs 15.6 (23.9%), carry 4.5 (6.9%, half provably zero), scores@V
  2.2, state update 2.3 (dead final update already DCE'd by inductor),
  coordinator 0.6, masked_fill 0.4 (0.7% — NOT a bottleneck).
- `paper_layout="direct"`: product written straight into [B,T,H,K] so the
  N-flatten is a view (no transpose copy). Exact: 1.49e-08 vs canonical.
- `cache_rope=True`: phase tables computed once per forward instead of per
  level (pos is constant across the 8 levels). Exact: 1.49e-08.
- A/B result in the FINAL section below: direct layout pays (-2.7..-8.4%
  packed), rope cache adds nothing alone, both included in the promoted ALL.

### FINAL SAME-SESSION A/B (b1024, T2048 B1) — PROMOTION DECISION
Compiled (production decision), 24 rows, one session, shuffled order:

| variant | single | mixed | four | heavy |
|---|---:|---:|---:|---:|
| certified where (control) | 194.1 | 205.5 | 207.8 | 209.9 ms |
| branch-free | -0.3% | -3.7% | -6.2% | -6.7% |
| zero-carry | -4.0% | -8.2% | -8.2% | -8.9% |
| paper-direct | -2.7% | -6.1% | -8.4% | -8.3% |
| dir+rope | -3.0% | -5.9% | -7.8% | -8.0% |
| **ALL (bf+zc+dir+rope)** | **183.1 ms (-5.7%)** | **185.6 (-9.7%)** | **186.5 (-10.3%)** | **184.0 (-12.3%)** |

Eager same-session deltas for ALL: -4.8% single, -7.8% mixed, -8.5% four,
-9.8% heavy. Memory: 3.13 GiB vs 3.19 (compiled, B1) — no regression.

**PROMOTED_FOR_G4: `opt3c_all` = branch-free packed state update +
zero-carry skip + paper_y direct layout + cached RoPE phase.**
All four mechanisms are exact (bitwise-identical for the scan update; gate
7.1e-15 vs dense oracle; 1.49e-8 vs canonical for the model),
no host syncs, static compiled entry points, clean and packed both faster.
G4 confirmation script: `results/g4_confirm_autoresearch_iii.py`.

### Preflight tool (2026-09-15): `results/verify_opt3c_all.py`
Standalone check of the repo's actual opt3c_all_b1024 before any long run.
14 named gates, machine-readable verdict (`OPT3C_ALL_ROBUST=true|false`,
`FAILED_GATES=[...]`): fp64 scan oracle (randomized + adversarial packed
layouts); candidate-vs-certified model equivalence (fp32/bf16, eager/
compiled, packed/single, logits/loss/all grads); zero graph breaks at
production T/block; determinism (3x repeated compiled backward + two
fresh-model full updates); checkpoint save/resume; production-path smoke
train (global B64, fused AdamW, LR schedule, frozen packed corpus when
present else synthetic) with finite-loss/grad, memory-stability and OOM
checks. Local full run: 14/14 PASS, worst scan error 1.4e-14, full-update
determinism bitwise 0.0 eager / 9.3e-10 compiled, checkpoint 9.3e-10,
smoke alloc growth 0.0. The only non-bitwise signal is 4.8e-7 on
`embedding.weight` under repeated compiled backward (index_add atomic
ordering; bounded by the 1e-5 gate). G4 run must use `--require-corpus`
to certify `production_scale=true`.

### Priority 2 (big math) — measured conclusion
- G4 curve b256 63.7k / b512 65.0k / b1024 74.4k shows larger blocks win
  despite local-QK FLOPs growing linearly with W; per-chunk overhead, not
  local-QK FLOPs, sets the optimum. Every exact restructuring that removes
  local-QK work (parallel prefix with materialized G states, hybrid batched
  local, two-level/sub-block triangular) collapses to either chunkwise with
  smaller W (measured slower on G4) or the parallel form (killed twice).
- Provable dead work found and removed instead: chunk-0 carry bmm (zc) and
  dead final-chunk update (branch-free / DCE).
- Remaining lever is a fused per-chunk kernel (lower-triangular local +
  carry + update); cost model: ceiling ~8% of update at unknown Blackwell
  tile efficiency; not built locally (24-SM tuning mistransfer risk), exact
  spec left for a G4-profiled effort.

### Priority 5 (static graph)
- Production now runs through static entries with no host syncs/graph
  breaks (the previous `bool((segment_start==0).all())` graph break is
  gone from the compiled path). Explicit L=8 unroll adds nothing (dynamo
  already traces the constant loop; static4 compiled was a tie in E2).

## ARM-A III preflight tool: `results/verify_opt3c_all.py`

Standalone preflight for `opt3c_all_b1024` (uses the repo implementation,
no math changes). Local full pass on the 4060, 14/14 gates PASS,
`OPT3C_ALL_ROBUST=true`, `FAILED_GATES=[]`:

| gate | result |
|---|---|
| scan_oracle (dense fp64 vs certified/candidate, randomized + boundary layouts, 48 cases) | PASS, worst 1.42e-14 |
| model_equiv packed fp32/bf16 eager | PASS, diffs 0.0 |
| model_equiv single fp32 eager | PASS, diffs 0.0 |
| model_equiv packed fp32/bf16 compiled | PASS (<=6e-8 / bf16 tol) |
| canonical_anchor_tiny (vs canonical model) | PASS |
| graph_breaks_candidate (production T/block, forward_packed) | PASS, 0 breaks, 1 graph |
| determinism_repeat_backward (3 passes, compiled bf16) | PASS, max diff 4.77e-07 = 1 bf16 ulp, worst param embedding.weight (non-bitwise) |
| determinism_full_update_eager (2 fresh models, 1 AdamW step) | PASS, bitwise 0.0 |
| determinism_full_update_compiled | PASS, 9.31e-10 |
| checkpoint_resume_equivalence (3+2 updates, AdamW state saved/restored) | PASS, 9.31e-10 |
| smoke_train (global B64, fused AdamW, LR schedule, compiled) | PASS at microbatch 1 (local memory limit; `production_scale=false`), finite loss/grads every step, alloc growth 0.0, peak 3.44 GiB, no OOM |

Notes: the only non-bitwise behavior is the compiled bf16 backward's
1-ulp scatter-add (embedding.weight), which is expected CUDA behavior and
far below the bf16 noise floor; full-update determinism is bitwise (eager)
and 9e-10 (compiled). On G4, run with `--require-corpus` at microbatch 16
to certify production scale (`production_scale=true`, frozen packed corpus).

## Failed / rejected ideas

| ID | Hypothesis | What happened | Why rejected |
|---|---|---|---|
| diet-checkpoint-wrap | checkpoint rope/Dy+paper to cut memory for B32x2 (E1) | exact (192 cfg <=7.5e-09); +8.5-10.5% time for -18..27% memory | checkpoint boundaries blocked fusion; accum overhead only ~2% (T512_L2 B8 mb8 228 vs mb4 233) so B32x2 prize can't pay +10% |
| H1-parallel | batched local + materialized G states, b512/b1024 | eager +2-3%, compiled +6-7%; memory worse | loses twice at both blocks |
| H1-hybrid | batched local + sequential carry | eager +4-5%, compiled +4-10% | batched small local bmms less efficient than chunked |
| H2-static4-promote | explicit 4-stage static compiled graph | compiled tie (171.5 vs 172.1) | inductor already traces the loop; eager win was Python overhead. Kept as eager fallback only |
| classify-fastpath | host-side pure-chunk classification for packed | single tie; two/four/heavy +9..14% slower | `.all()` host syncs per chunk per level drain the pipeline; killed both eager and compiled |
| H4-maxautotune | max-autotune compile mode | inductor: "Not enough SMs to use max_autotune_gemm mode" | SM-gated OFF on 24-SM GPU; recorded, no run, no Blackwell extrapolation |
| Triton-fused-scan | custom fused triangular-local + carry kernel | not built: K*Dv state (1M els) cannot fit SRAM; realistic win ~37% of local-QK FLOPs (~8% of update) at unknown Blackwell tile efficiency | deferred to G4 with exact spec; 24-SM tuning would mistransfer |
| v1-outer-product-scan | [B,H,W,K*Dv] outer-product materialization | OOM at T256_L2 B2 (4 GiB ask) | superseded by chunkwise; do not retry |
| inductor-splitscan | prefix-coordinator cumsum compiled | InductorError on torch 2.11 | dense coordinator kept (<=1% FLOPs) |
| seg-from-mask-v1 | segment_start from mask w/o diagonal | round-trip failed | fixed by OR-ing eye |

## Correctness gates (all pass)

- `opt/test_correctness.py`: 180 randomized dense-vs-{chunkwise,chunked,
  cumsum,parallel,hybrid,static4} cases, fwd+gq+gv+coord, worst 1.4e-14 f64.
- `opt/test_model_equiv.py`: model-level vs canonical incl. packed layouts,
  ckpt on/off, scans {parallel,chunkwise,hybrid,static4}, dense/prefix
  coord; worst 1.5e-8 fp32; bf16 fwd 9.8e-4.
- `opt/test_diet_gate.py`: 192 configs, worst 7.5e-9 (diet exact).
- `opt/compile_smoke.py`: compiled==eager within bf16 tolerance.

## Files / harness

- `opt/scan_attn.py`: scan forms + cached masks + branch-free packed update
  + `SCAN_CLASSIFY` kill-switch.
- `opt/model_opt.py`: OptArmA (coord dense/prefix, scan forms, classify,
  optional explicit segment_start/single_doc bypass).
- `opt/model_diet.py`: checkpoint-diet variant (killed, kept for record).
- `opt/bench_matrix.py`: randomized sessions, --blocks/--packed/--gb/
  --seed/--assume-single-doc, per-row flush, unknown-name guard.
- `opt/analyze_log.py`: per-cell medians/spreads + matched speedups
  (shape/mb/compiled/packed).
- `results/local_matrix.json`: all raw rows.
- `results/e0_*.txt`, `results/profile_*.txt`: profiles and censuses.
