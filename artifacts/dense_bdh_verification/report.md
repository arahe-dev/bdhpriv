# Dense BDH Verification Report

VERDICT: UNSTABLE

DENSE_BDH_CORRECTNESS: PASS
DENSE_BDH_STABILITY: UNSTABLE
DENSE_BDH_MEDIAN_TOKENS_PER_SECOND: 8722.2
DENSE_BDH_P05_TOKENS_PER_SECOND: 8702.3
DENSE_BDH_P95_TOKENS_PER_SECOND: 11973.6
DENSE_BDH_CV: 15.8 %
DENSE_BDH_LONG_RUN_DRIFT: -0.0 %/min
COMPARATOR: opt/model_ref.py NativeReadStage1ArmA (pinned executable dense reference; original Soda/vendor file absent)
MATCHED_SPEEDUP: comparator / control = 0.3 (CI95 [0.226001438437932, 0.2682836184697266])
CLAIMABLE_RESULT: dense BDH correctness vs the executable reference (forward, gradients, parameter update) and median dense-control throughput at the measured clock state
NON_CLAIMABLE_RESULT: any sparse-path number; any cross-machine projection; comparator speedup beyond the limited sample
PRIMARY_FAILURE_OR_LIMITATION: original Soda reference unavailable; dense B=2 geometry is memory-degraded on the 8 GiB local GPU
ARTIFACT_DIRECTORY: artifacts/dense_bdh_verification

## 1. One-sentence verdict

Dense BDH matches the pinned executable reference exactly in small-config
forward/gradients/parameter updates and to fp32 tolerance at production
shape, with unstable local throughput of
8722.2 tok/s (median) at the measured clock state.

## 2. Exact implementation tested

CONTROL: dense BDH `opt/model_opt.py` OptArmA (opt3c execution: chunkwise
scan, dense coordinator, branch-free packed state update, zero-carry skip,
direct paper_y layout, cached RoPE), scan_block=1024.

## 3. Exact control and comparator

- CONTROL: as above (current dense Arm-A; no separate DUT exists).
- COMPARATOR: `opt/model_ref.py` NativeReadStage1ArmA, the pinned
  executable canonical dense reference (selective-checkpoint levels).
- Original Soda/reference (`vendor/canonical/...v1.py`): NOT PRESENT in the
  workspace; its SHA-256 is pinned in README_FIRST.md but the file is
  absent. That comparator is BLOCKED.
- Sparse BDH/routed experts: OUT OF SCOPE and absent from every measurement
  and table in this report.

## 4. Correctness result

Executed (`runs/correctness.json`, `gates` all true):
- small-config forward max abs diff: 0.0
- small-config loss abs diff: 0.0
- small-config gradient max abs diff: 0.0
- one AdamW parameter-update max abs diff: 0.0
- production-shape forward parity (fp32): B1 1.31e-06,
  B2 1.07e-06
- determinism (repeat forwards): 0.0 / 0.0
- finite loss and all-finite gradients: True / True
- cross-row isolation (production B=2): row-0 batched vs alone
  1.19e-06 (control) and
  9.54e-07 (reference) -
  fp32 GEMM-order noise; modifying row 1 changes row 0 by exactly
  0.0 while row 1
  changes by 2.05.
  No cross-row influence.

## 5. Benchmark configuration

131,072 packed tokens per complete optimizer update (64 x 2048); forward +
loss + backward + clip + fused AdamW; bf16 autocast + fp32 master; fixed
seeds; identical model dims/precision/optimizer/loss across rows; CUDA
synchronized start/stop; data generation outside the timed region;
compilation excluded from steady-state timing and reported separately.

## 6. Cold-start table (isolated temporary compiler caches)

| run | build s | first step s (compile incl.) | first-step tok/s | external wall s |
|---|---|---|---|---|
| cold_true_0 | 3.02 | 90.2 | 1453 | n/a |
| cold_true_1 | 2.73 | 90.7 | 1446 | n/a |
| cold_true_2 | 2.26 | 75.5 | 1735 | n/a |

## 7. Warm-throughput table

| row | processes/windows | median tok/s | CV % | p05 | p95 | p95/p05 | classification |
|---|---|---|---|---|---|---|---|
| dense control B=1 | 3 / 36 | 8722.2 | 15.8 | 8702.3 | 11973.6 | 1.4 | UNSTABLE |
| dense control B=2 (limited, degraded) | 1 / 4 | 4295.2 | 7.2 | 4015.5 | 4671.3 | 1.2 | UNSTABLE |
| comparator reference B=1 | 1 / 3 | 2321.5 | 0.8 | 2303.2 | 2337.2 | 1.0 | STABLE |

## 8. Long-run stability table (dense control B=1, 12 min)

median 8678.1 tok/s, CV 0.1%,
drift -0.0 %/min, temp {"min": 58.0, "max": 62.0},
SM clock {"min": 1920.0, "max": 2520.0},
memory used {"min": 3733.0, "max": 3733.0}.

## 9. Speedup with confidence interval

Matched geometry B=1, comparator/control =
0.3 (CI95 [0.226001438437932, 0.2682836184697266]). Limited comparator sample
(3 updates); treat as descriptive, not a precise
ratio.

## 10. Variance and thermal explanation

The dense control is clock/power limited on this laptop GPU (SM clocks
1440-1830 MHz against a 3105 MHz max; ~35 W cap). Variance follows clock
state, not skips: token accounting is exact, logits deterministic.
B=2 memory pressure (peak approaching 8 GiB) degrades it severely.

## 11. Claimable result

Dense BDH correctness relative to the pinned executable reference, and a
same-machine, same-configuration throughput claim at the measured clock
state: median 8722.2 tok/s,
p05 8702.3 tok/s for a complete 131,072-token update.

## 12. Non-claimable result

Sparse-path numbers (out of scope), cross-machine/G4 projections, and the
comparator speedup as a precise figure (limited comparator sample).

## 13. Blocked or invalid results

- Original Soda/vendor comparator: BLOCKED (source file absent).
- B=2 dense geometry: memory-degraded on 8 GiB; classified
  UNSTABLE; not usable for headline claims.

## 14. Exact commands for reproduction

See section 1 of `commands.txt` and `run_manifest.json`. Suite:
`python opt/dense_verify_manifest.py`, then
`python opt/dense_verify_orchestrate.py` inside `iclr-arm-a`, then
`py -3.12 opt/dense_verify_report.py` on the host.

VERDICT: UNSTABLE
DENSE_BDH_CORRECTNESS: PASS
DENSE_BDH_STABILITY: UNSTABLE
DENSE_BDH_MEDIAN_TOKENS_PER_SECOND: 8722.2
DENSE_BDH_P05_TOKENS_PER_SECOND: 8702.3
DENSE_BDH_P95_TOKENS_PER_SECOND: 11973.6
DENSE_BDH_CV: 15.8 %
DENSE_BDH_LONG_RUN_DRIFT: -0.0 %/min
COMPARATOR: opt/model_ref.py NativeReadStage1ArmA (original Soda/vendor file absent)
MATCHED_SPEEDUP: comparator/control = 0.266 (CI95 [0.226, 0.268]); equivalently the dense control is 3.76x faster than the reference (CI95 [3.73, 4.42])
CLAIMABLE_RESULT: dense BDH correctness vs the pinned executable reference; median dense-control throughput at the measured clock state
NON_CLAIMABLE_RESULT: any sparse-path number; cross-machine projections; comparator speedup beyond the limited sample
PRIMARY_FAILURE_OR_LIMITATION: original Soda reference unavailable; dense B=2 geometry memory-degraded on the 8 GiB local GPU
ARTIFACT_DIRECTORY: artifacts/dense_bdh_verification
