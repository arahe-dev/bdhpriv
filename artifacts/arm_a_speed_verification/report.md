VERDICT: UNSTABLE
CANONICAL_COMMIT: 0dcbb878d24b99b5808359c889e97143c3cec00b (all nine candidate blobs verified MATCH the working tree; current HEAD 4c0e152 contains unrelated changes from another agent)
CANONICAL_CONFIG: M8/Ke512/top1 fixed cyclic contiguous window, head-shared, compact exact-capacity executor (opt/routed_expert.py RoutedExpertArmA); baseline dense Arm-A opt/model_opt.py OptArmA opt3c flags
DEVICE: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, driver 595.79, container iclr-arm-a (torch 2.11.0+cu128, CUDA 12.8, triton 3.6.0)
PRIMARY_METRIC: packed input tokens per full optimizer update (131072 tokens = 64x2048), CUDA-synchronized wall clock, warm steady state
MEDIAN_THROUGHPUT: 37,092 tok/s sparse mb2 (INVALID path) | 29,777 tok/s sparse mb1 (valid path) | 7,446 tok/s dense mb1
P05_THROUGHPUT: 36,914 (mb2) | 21,904 (mb1)
P95_THROUGHPUT: 37,235 (mb2) | 32,599 (mb1)
COEFFICIENT_OF_VARIATION: 0.37% (mb2) | 10.83% pooled / up to 12.84% within a process (mb1)
LONG_RUN_DRIFT: -0.04 %/min over 12 minutes (mb2, invalid path)
CORRECTNESS: FAIL for microbatch > 1 (cross-row packed-document leakage); PASS for B=1 declared-sparse semantics, all-active expertized equivalence, and learned-router gates
TARGET_STATUS: "~4x same-session" REPRODUCED on the valid B=1 matched-geometry path (3.999x, CI95 [3.895, 4.121]); the higher best-feasible 4.982x uses the invalid B>1 path and must not be claimed. The 350k packed tok/s G4 target is NOT VERIFIABLE on this machine and its local foundation (mb2) is invalidated by the correctness failure below.

# Arm-A Sparse Speed Verification Report

## 1. Exact commands executed

```
docker start iclr-arm-a
docker exec -w /workspace/iclr-oc iclr-arm-a python opt/verify_speed_orchestrate.py
docker exec iclr-arm-a sh -c 'cd /workspace/iclr-oc && for i in 0 1 2; do \
  TORCHINDUCTOR_CACHE_DIR=/tmp/cold_ind_$i TRITON_CACHE_DIR=/tmp/cold_tri_$i \
  python opt/verify_speed_run.py --kind cold --path sparse --microbatch 2 \
  --out artifacts/arm_a_speed_verification/runs/cold_true_$i.json; done'
docker exec iclr-arm-a python opt/verify_speed_run.py --kind correctness \
  --out artifacts/arm_a_speed_verification/runs/correctness.json
py -3.12 opt/verify_speed_report.py
py -3.12 opt/verify_speed_parity.py  (B=2 leakage diagnostic, see section 8)
```

All runs are one fresh process per measurement. The canonical benchmark path
is a full training update (forward + loss + backward + clip + fused AdamW),
not a forward-only or decode path. No checkpoint is loaded: the canonical
harness uses fixed-seed synthetic packed batches, the same boundary used by
every prior project benchmark. This is a benchmark/verification cell, not a
training run.

## 2. Exact source/config used

All nine source blobs match `results/350k_frozen_state.json`
(commit 0dcbb87): routed_expert.py ad974a4c, model_opt.py 6076ba04,
model_ref.py 4801c5a3, scan_attn.py 5185d032, bench_expert_moe.py b7f69f41,
doe_config.py ccdac82d, onehour_microbatch.py 549661bb,
onehour_reprofile.py 37efb50c, g4_sparse_350k_cell.py 14828c01.
Config: T=2048, L=8, D=256, N=16384, H=4; global 64 sequences; sparse
executor M=8/Ke=512/top1, cyclic contiguous windows, G=128; dense baseline
opt3c flags; bf16 autocast + fp32 master; loss and optimizer unchanged.

## 3. Cold-start results

Two distinct meanings were measured, because the container's Inductor cache
persists across processes:

- Fresh process, warm compile cache (5 runs): model build 1.9-2.3 s,
  first step 15.0-16.3 s (cache-hit compile/load + one update), first-step
  throughput 8,035-8,736 tok/s, loss identical across runs (0.2831) -
  fixed seeds are deterministic.
- Fresh process, fresh compile caches (`TORCHINDUCTOR_CACHE_DIR` and
  `TRITON_CACHE_DIR` redirected, 3 runs): first step 276.9 / 279.6 / 283.1 s
  (full Inductor compilation of the sparse graph, measured at the current
  throttled CPU/GPU state). First-step throughput 463-473 tok/s.
  True cold-start time-to-first-valid-step is therefore ~4.6 minutes, not
  ~16 seconds. Compile cost is one-time per environment/cache.

## 4. Warm steady-state results

Three independent processes per path; 5 warmups then 10 windows (dense) or
12 windows (sparse); one window = one full 131,072-token optimizer update.

| path | processes | windows | median tok/s | CV %% | p05 | p95 | p95/p05 |
|---|---|---|---|---|---|---|---|
| dense mb1 (B=1) | 3 | 30 | 7,446 | 0.14 | 7,434 | 7,462 | 1.004 |
| sparse mb1 (B=1) | 3 | 36 | 29,777 | 10.83 | 21,904 | 32,599 | 1.488 |
| sparse mb2 (B=2) | 3 | 36 | 37,092 | 0.37 | 36,914 | 37,235 | 1.009 |

Sparse mb2 is very stable but INVALID (section 8). Sparse mb1 is valid but
CONDITIONALLY stable: per-process CVs 4.72%, 10.85%, 12.84%, and within-run
first/last up to 1.091, driven by SM clock state (see section 7).

Speedups (median ratio, 95% bootstrap CI):
- matched geometry B=1 (valid): dense_mb1 / sparse_mb1 = **3.999x
  [3.895, 4.121]**.
- best-feasible mb2 (INVALID path): dense_mb1 / sparse_mb2 = 4.982x
  [4.975, 4.990] - reported for completeness only, not a supported claim.

## 5. Long-run stability results (12 minutes, mb2, INVALID path)

22 samples every ~30 s: median 37,016 tok/s, CV 0.20%, drift -0.038 %/min,
first/last 0.9914, temperature 57-58 C, SM clock 1575-1815 MHz, memory used
constant 2637 MiB, peak allocated 2.36 GiB, no skipped steps, no memory
growth. Numerically stable, but on the invalid path; no 12-minute run exists
for the valid B=1 path.

## 6. Raw statistics

`summary.json` (quantiles, MAD, bootstrap CIs), `raw_runs.csv` (129 rows,
every window), `runs/*.json` (per-process raw), `plots/throughput.png`.
Failed and superseded runs are retained (`correctness_v1_invalid.json`).

## 7. Thermal, clock and utilization observations

- Dense mb1 runs pinned at 1440-1470 MHz SM clock (near the 35 W cap,
  99% utilization), i.e. the dense baseline was clock-limited during this
  verification; absolute dense throughput (7.4k) is well below the earlier
  unthrottled sessions (11.1k), so absolute numbers are session state
  dependent.
- Sparse runs sustained 1815-1830 MHz in early repetitions; later
  repetitions saw 1575-1815 MHz with corresponding throughput variance.
- Consequence: the ratio is more robust than either absolute number, but
  sparse mb1's within-process variance exceeds the 5% CV stability bar in 2
  of 3 processes (thermal/clock dependence), which is itself a
  CONDITIONALLY STABLE condition.

## 8. Correctness and parity results - the decisive finding

Passing (B=1 and declared semantics):
- `opt/test_routed_expert.py`: all-active expertized == Arm-A (bitwise 0.0
  fp32), sparse FP64 oracle 0.0, document-boundary test, route builders.
- `opt/test_learned_router.py`: R0-R6 (ST forward exactly hard, surrogate
  gradient 1.1e-16, inactive experts exact-zero grads).
- GPU smoke (`runs/correctness.json`): finite decreasing loss, 393,216 of
  393,216 tokens, 2048/2048 route slots, zero overflow, bitwise
  determinism.

FAILING (microbatch > 1):
- All-active expertized vs dense at B=2 (`correctness.json`):
  max abs diff 1.35 - not a rounding error.
- Isolated diagnostic (`parity_failure_B2.json`, CPU, fp32, exact code):
  - routed B=2 vs per-row reference: **2.78e-3**
  - dense B=2 vs per-row reference: 7.45e-8
  - routed B=2 with row-2 segment ids offset by +1000 vs reference:
    7.45e-8
- Root cause: when B rows are flattened into expert streams, each row's
  `segment_start` restarts at 0, so tokens from different rows share the
  same document id and the scan treats them as the same document
  (cross-row attention leakage). The dense path is per-row batched and is
  unaffected. The B=1 path has one row and is exact.
- Impact: every sparse measurement with microbatch > 1 in this project -
  local mb2/mb4/mb8 results, the 4.22x "best-feasible" claim, the local
  microbatch optimum, and the planned G4 geometries B8/B16/B32 - is
  numerically invalid as declared packed-document semantics. The G4 cell
  (results/g4_sparse_350k_cell.py) would produce invalid training runs as
  written.
- Mandate compliance: no implementation changes were made. This must be
  fixed (offset packed segment ids per flattened row, or keep rows as an
  explicit batch dimension) and every B>1 benchmark re-run before any
  transfer.

## 9. Failed, invalid and outlier runs

- `correctness_v1_invalid.json`: v1 harness compared top1-sparse directly to
  dense, which is not an equivalence relation; retained as provenance.
- All B>1 runs above are evidence but not valid measurements of the
  declared architecture; they are labeled INVALID wherever cited.
- No runs were deleted. Cross-process throughput clusters (sparse mb1) are
  explained by SM clock state, not by silent skips; token accounting is
  exact in every measured window (131,072 tokens per update).

## 10. Plain-language conclusion

The "~4x same-session" historical claim does reproduce, but only on the
valid B=1 matched-geometry path: 3.999x with CI [3.895, 4.121], with the
caveat that sparse B=1 throughput carries 5-13% within-process variation
from clock state. The larger, more attractive numbers (4.98x best-feasible;
37k tok/s at mb2) come from a microbatch path where the executor leaks
attention across packed rows; they must not be promoted, and the 350k tok/s
G4 projection built on them is unsupported. The speed is real in direction
(coarse structured sparsity is ~4x at B=1), but this configuration as
written is UNSTABLE: correct at B=1, incorrect at B>1, and clock sensitive.
Remediation before any further benchmarking or transfer: fix per-row
document identity in the flattened expert streams, re-run the B>1 sweep, and
re-verify; then re-derive the G4 package.
