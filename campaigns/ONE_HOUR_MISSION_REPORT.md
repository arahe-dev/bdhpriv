# ONE-HOUR MISSION REPORT (in progress)

Mission: drive exact/declared sparse Arm-A full-step training from the
current ~3.2-3.7x local champion toward the 8.69x one-hour-equivalent local
target. All local ratios are same-process versus compiled Arm-A.

## 1. Starting point

- Optimized Arm-A (compiled, B1, T2048/L8, full update): 214-226 ms/update
  across the sessions in this mission.
- Prior champion: M8/Ke512/top1/G128 fixed cyclic window: 58.1 ms (3.67x,
  frozen round-1 session), 69.4 ms (3.16x, pass-A session), 66.5 ms (3.22x,
  autopsy session).
- Learned top2 router: 91.7 ms (2.38x), same as fixed top2 within noise.
- Target: candidate_ms / same_process_arm_a_ms <= 1/8.69 -> ~25 ms at the
  current baseline.

## 2. Measurement protocol (permanent)

One fresh process per candidate; fresh compile; Arm-A control in the same
process; static shapes; warmups before timing; graph_count/graph_break_count
recorded; no `Tensor.item()` or host syncs in the compiled hot path;
capacity and overflow recorded; absolute ms AND same-process speedup
reported; cross-session absolutes are secondary. Contaminated runs are kept
as provenance (`results/doe_round1_*contamination*.json`).

## 3. Active-width frontier (Pass A) — SATURATED at M8

| config | active K/head | candidate ms | speedup | compile | verdict |
|---|---|---|---|---|---|
| A0 M8/Ke512/top1 | 512 | 66.5-69.4 | 3.16-3.22x | 16 s | champion |
| A1 M16/Ke256/top1 | 256 | 132.5 | 1.63x | 468 s | KILL |
| A2 M32/Ke128/top1 | 128 | 254.2 | 0.87x | 1192 s | KILL |
| A3 M64/Ke64/top1 | 64 | not run | — | >20 min projected | KILLED BY EVIDENCE |

Conclusion (MEASURED): subdividing stored width into more, smaller experts
collapses throughput because per-expert GEMMs shrink and the number of
unrolled expert streams multiplies (M32 is slower than dense Arm-A).
The fitted width model T = 29.8 + 0.0575*K_active only holds when reducing
active experts at fixed M (which enlarges per-expert streams); it does not
extrapolate to width subdivision. Active-width optimization alone leaves
~2.3x to the one-hour target, so the fixed floor must fall.

## 4. Runtime-floor decomposition (Pass B autopsy, M8 top1 champion)

Candidate 66.5 ms, Arm-A 214.5 ms (3.22x), one process.

| component | estimate | share | method | confidence |
|---|---|---|---|---|
| forward + CE | 30.6 ms | 46% | compiled no-grad forward | medium |
| backward (derived) | — | unreliable | fwd+bwd minus no-grad fwd (different graphs) | low |
| optimizer (zero_grad+clip+AdamW) | 4.7 ms | 7.1% | eager repeated step | high |
| levels total (8) | 57.9 ms | 87% | minus L-level skeleton | medium |
| per level | 7.2 ms | 11% | levels/8 | low |
| non-level skeleton | 8.6 ms | 13% | identity levels full update | high |
| coordinator ablation | -1.9 ms | noise | diagnostic (returns ones) | high |
| writer ablation | 17.1 ms | 26% | diagnostic (returns zeros) | high |

Profiler (5 steps, active window): 14,404 kernels total -> **2,881 kernels
per update**, total CUDA 152.5 ms -> ~30.5 ms/update kernel time versus
66.5 ms wall. Category split per update: gemm 12.2 ms, copies 8.2 ms,
gather/scatter 3.6 ms, layernorm 3.2 ms, optimizer elementwise 2.0 ms.

Finding (MEASURED): the champion is launch/copy-bound, not FLOP-bound.
~36 ms/update of wall is not kernel time. The writer MLP is the largest
single ablational component (17.1 ms).

## 5. Causal optimization results

| experiment | change | result | decision |
|---|---|---|---|
| B1 | `torch.compile(mode="reduce-overhead")` + cudagraph boundary marking + eager mask-cache prepopulation | 76.6 ms, 2.95x, higher variance, peak 0.62 GiB, compile 355 s | KILL (negative vs 3.22x default) |

## 6. Failed ideas

- Width subdivision (M16/M32/M64) — killed above.
- CUDA-graph capture — killed above.
- Capacity padding (round 1) — killed earlier: padding is causal but avoidable
  by using exact balanced capacity.

## 7. Final stacked champion

Not yet established. Current best stack: M8 top1 fixed cyclic window, exact
capacity, default compile — 3.16-3.22x same-session.

## 8. Exactness evidence

- Expertized all-active equivalence: FP32 bitwise, FP64 4.4e-12 vs Arm-A.
- Routed executor: all-active bitwise, sparse oracle 0.0, document-boundary
  leak test.
- Learned router R0-R6: ST forward exactly hard, gradient equals soft
  surrogate to 1.1e-16, never-selected experts get exactly zero gradients,
  graph breaks 0.

## 9. Remaining bottleneck

~36 ms/update non-kernel wall + 8.2 ms copies + 17.1 ms writer ablation.
Next attacks (evidence-ordered): non-kernel gap attribution (allocator
churn, eager optimizer/clip/CE engine overhead), copy/materialization
removal, writer scheduling/fusion, microbatch geometry at constant global
batch.

## 10. G4 transfer candidates

Not yet selected. Candidates will be: (A) default-compile M8 top1 champion,
(B) whichever floor-optimized stack emerges, (C) a safer top2 fallback.
One G4 cell with environment/correctness/graph/benchmark gates is the final
deliverable; no 2.5B training until G4 throughput is confirmed.

## 11. Projected 2.5B runtime

INFERRED (local proxy only): 3.2x -> ~2.7 h; 8.69x -> ~1.0 h. No G4 claim
until measured there.

## 12. Measured vs inferred

- MEASURED: all ms/ratios above; kernel census; ablations.
- INFERRED: G4 projections; that copies/writer explain the remaining gap.
- SPECULATIVE: specific savings from future fusions (not yet demonstrated).
