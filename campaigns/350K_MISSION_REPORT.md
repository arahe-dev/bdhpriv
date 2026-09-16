# 350K MISSION REPORT

Mission: fastest training-compatible sparse Arm-A configuration; G4 target
>=350k packed tok/s (preferred 380k, stretch 400k). Local RTX 4060 results are
selection evidence only; G4 is production truth.

## 1. Frozen evidence (not to be rediscovered)

- Dense production Arm-A on G4: ~80.3k packed tok/s, ~1632.8 ms/global
  update, B16x4, peak ~58.2 GiB.
- Local sparse systems: M8/Ke512/top1 fixed cyclic window, compact executor.
- Width subdivision M16/M32/M64: KILLED (fragmentation dominates; M32 is
  slower than dense).
- Microbatch geometry at constant global batch (131,072 tokens/update):
  - dense mb1 11,820 ms / 11,089 tok/s / 3.45 GiB
  - dense mb2 44,547 ms / 2,942 tok/s / 7.79 GiB -> memory-thrash collapse
    on 8 GiB; dense mb4 failed
  - sparse mb1 3,465 ms / 37,827 tok/s / 1.39 GiB
  - sparse mb2 **2,707-2,804 ms / 46.7-48.4k tok/s / 2.29-2.36 GiB**
  - sparse mb4 3,113 ms / 42,110 tok/s / 4.28 GiB
- Best-feasible local sparse/dense: 4.05-4.22x (same-session).

## 2. Measurement protocol

One process per configuration; fresh compile; same-process dense control;
warmups before timing; static shapes; graph breaks recorded; contaminated
runs kept as provenance (sparse mb1 anomaly resolved by clean repeat).

## 3. Reprofile of the mb2 champion (MEASURED)

- 2,721 ms/global update, 48,165 tok/s, peak 2.36 GiB.
- ~137,770 profiler events per global update (~4,300 per microbatch);
  launch-API CPU ~589 ms (profiler-inflated). Device-time totals are invalid
  under CUPTI; counts are valid. Execution remains fragmented.

## 4. B3 copy/materialization (MEASURED, KILLED)

- Level-invariant gathered pos/segment tensors cached once per forward
  (~128 gathers/update removed): -0.07% wall. Within noise; code reverted.
- Evidence: results/350k_b3_cache.json.

## 5. B4 writer (INVALID ATTRIBUTION, GATE NOT MET)

- Zeroing the writer produces an apparent 91.5% share, but it severs the
  autograd path through all 8 levels; that run measures "no backward through
  levels", not writer cost.
- The earlier B1 writer attribution (17.1 ms) is RETRACTED as cost evidence.
- Therefore the custom-kernel gate (writer subpath >=15% of wall with valid
  evidence) is NOT met. No custom kernel is justified.
- Evidence: results/350k_b4_writer.json.

## 6. Local STACK

STACK0 = M8/Ke512/top1 fixed cyclic window + mb2 x32 accumulation.
No compatible additional exact systems win survived B3/B4, so the stack is
the microbatch champion alone. Local same-session ratios: 4.02-4.05x (this
mission), 4.22x (earlier session).

## 7. G4 transfer package (READY)

- Cell: results/g4_sparse_350k_cell.py (single cell, fresh runtime).
- Candidate A (speed champion): M8 top1, geometries B8 -> B16 -> B32, then
  B64 if improving or B4 if improving smaller; OOM-safe.
- Candidate B (scientific champion): learned top2 (second pass; not in the
  first cell run unless the speed pass is cheap).
- Gates: environment, determinism, sparse-vs-dense diff, graph breaks,
  same-session dense B16x4 reference, machine-readable outputs
  (G4_DENSE_TOK_S_B16, G4_SPARSE_TOK_S_B*, G4_SAME_SESSION_SPEEDUP_B*,
  G4_BEST_GEOMETRY, G4_PROJECTED_2P5B_HOURS, G4_RESULT_JSON).

## 8. Projection (INFERRED ONLY)

Local-to-G4 dense ratio 80.3k/11.1k = 7.24x applied to local sparse 48.2k
gives ~349k tok/s -> ~1.99 h for 2.5B. Confidence low-medium; geometry and
hardware scaling are not transferable in general. The G4 cell is the
decision mechanism; no 2.5B run before measured G4 throughput and a short
packed training smoke.

## 9. Measured vs inferred vs speculative

- MEASURED: all local ms/tok-s/ratios, microbatch table, B3 result, B4
  invalidity, kernel event counts.
- INFERRED: G4 sparse throughput and 2.5B runtime projection.
- SPECULATIVE: any untested fusion savings; writer-cost claims (retracted).
