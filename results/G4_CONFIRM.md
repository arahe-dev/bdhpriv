# G4 confirmation pack v3 (RTX PRO 6000 Blackwell SE, torch 2.11.0+cu128)

## RUN THIS — one self-contained cell

**`results/g4_opt3c_all_final.py`** — run as the ONLY cell in a freshly
restarted G4 Colab runtime. It is fully self-contained (mounts Drive,
rebuilds the frozen packed batches, contains the certified and candidate
implementations, all gates, and the benchmark). No repo imports.

It does both jobs in order and fails closed:

1. **Robustness preflight** (gates; benchmark skipped if any fails):
   - dense fp64 scan oracle vs certified/candidate on boundary layouts
   - tiny full-model certified-vs-candidate equivalence (CPU fp32)
   - GPU equivalence fp32/bf16 × eager/compiled, packed layouts
     (single-doc, boundary-inside-block, multi-document, all-resets)
   - graph breaks on the production entry point (must be 0)
   - determinism: repeated backward + two fresh-model full updates
   - checkpoint save/resume equivalence (model + optimizer state)
   - frozen-corpus smoke train at B16x4 (finite loss/grads, memory
     stability, no OOM)
2. **Confirmation** (only if all gates pass): certified vs candidate on the
   same 13 packed global batches, 3 warmups + 10 timed full updates,
   B16x4, global B64, same init/optimizer/LR.

Final verdict lines:
```
FAILED_GATES=[...]
OPT3C_ALL_ROBUST=true|false
CANDIDATE_BEATS_CERTIFIED_ANCHOR_69183=true|false
```

Local validation of the merged cell (RTX 4060, via
`opt/validate_final_cell.py`, which runs the cell's gates with the
Blackwell-name check bypassed): 9/9 non-corpus gates PASS — model
equivalence ×4 (fp32/bf16 × eager/compiled), graph breaks (0), determinism
×3, checkpoint resume (2.3e-10). The corpus smoke gate runs only on G4.

## Legacy / local tools (not the G4 cell)

- `results/g4_confirm_autoresearch_iii.py` — previous benchmark-only cell
  (no robustness gates). Superseded by the cell above; kept as a fallback.
- `results/verify_opt3c_all.py` — repo-based preflight that imports
  `opt.*`; for local dev/CI, not for Colab.
- `opt/validate_final_cell.py` — host-side validator that executes the
  final cell's gate functions locally (stubs Colab, bypasses the GPU-name
  check only).
- `opt/validate_g4_gate.py` — older CPU validator for the tiny gates.

## ARM-A III PROMOTION (2026-09-15)

Candidate `opt3c_all` (exact, same-session compiled A/B at b1024 vs the
certified where-baseline):  single -5.7%, mixed -9.7%, four -10.3%,
heavy -12.3%; eager -4.8..-9.8%; memory flat/slightly lower. Mechanism:
branch-free packed state update + zero-carry skip + paper_y direct layout +
cached RoPE phase. Static compiled entries, no host syncs.

## Anchor evidence (already measured on G4)

| config | ms/update | tok/s | peak alloc | speedup |
|---|---:|---:|---:|---:|
| canonical B32x2 SAC | 2882.14 | 45,477 | 75.73 GiB | 1.00x |
| clean champion b1024 B16x4, no ckpt, compiled | 1761.27 | 74,419 | 58.34 GiB | 1.636x |
| **frozen packed champion (certified production)** | **1894.56** | **69,183** | **62.55 GiB** | **1.5249x** |
| packed retention clean->packed | - | 92.95% | - | - |


## Anchor evidence (already measured on G4)

| config | ms/update | tok/s | peak alloc | speedup |
|---|---:|---:|---:|---:|
| canonical B32x2 SAC | 2882.14 | 45,477 | 75.73 GiB | 1.00x |
| clean champion b1024 B16x4, no ckpt, compiled | 1761.27 | 74,419 | 58.34 GiB | 1.636x |
| **frozen packed champion (certified production)** | **1894.56** | **69,183** | **62.55 GiB** | **1.5249x** |
| packed retention clean->packed | - | 92.95% | - | - |


Block preference is HARDWARE-SPECIFIC (4060: b512 best; G4: b1024 best by
14% over b512). Never transfer block optima between GPUs.

## Session discipline (mandatory, learned the hard way)

- CPU-overhead-sensitive variants (anything with SAC/checkpoint machinery)
  drift up to 34% with host load; GPU-bound nockpt variants drift <5%.
- ALWAYS compare within one `bench_matrix` session, shuffled with `--seed`,
  ref control in the same session. Cross-session absolutes are void.
- Each row is flushed to results/local_matrix.json immediately; a container
  kill loses at most the in-flight row (learned from the 09:12 Docker
  incident: Windows servicing killed the engine mid-run, zero data lost).
- Correctness gates must pass before trusting any timing (same commands as
  below).

## Candidate 1 — champion single-doc + block retune (PRIMARY)

Mechanism: exact score-free chunkwise scan (Q=K, strict-past, same-doc, no
softmax/scale), dense canonical coordinator, no checkpoint, BF16 autocast +
FP32 master weights, torch.compile(mode=default). No semantic changes from
the current G4 champion except the harness-level session discipline and the
cached-mask/branch-free internals (bit-exact math, gate-verified).

Config: T=2048, L=8, B16x4 (global 64), blocks to test: **b1024 (current
best), b512, b2048**. Also re-test eager for the record.

Uncertainty resolved: is b1024 still the block optimum under identical
session conditions, and does the 1.636x reproducibly clear 1.60x? If b2048
single-chunk wins, promote it.

```
python opt/bench_matrix.py --compile --full --gb 16 --seed 101 \
    --shapes T2048_L8 --blocks 512,1024,2048 \
    --variants ref_ckpt,opt3c_nockpt_b512,opt3c_nockpt_b1024,opt3c_nockpt_b2048
```

## Candidate 2 — packed-production retention (SCIENTIFIC PRIORITY)

Mechanism: the champion with the **branch-free packed state update**
(`state = state*contb + fresh`, dead final-chunk update removed; exact,
gate-verified on randomized packed layouts). Local compiled retention on
realistic layouts at T2048 B1/b512: mixed (exponential doc lengths, mean
~1429 ≈ frozen corpus) 174.4 ms vs single 174.6 ms = 100%; four docs 96%;
128-token docs 95.5%. Ref control flat at 230-234 ms across layouts.
Prior version was 91.5% retention; the rewrite recovered ~6% on packed.

Uncertainty resolved: does packed-window throughput on G4 track the
frozen-window benchmark (74.4k) or degrade with boundary density? This
decides whether the final training run needs the packed-aware scheduling.

```
python opt/bench_matrix.py --compile --full --gb 16 --seed 102 \
    --shapes T2048_L8 --blocks 1024 --packed single,mixed,four,heavy \
    --variants ref_ckpt,opt3c_nockpt_b1024
```
(`--packed mixed` uses exponential doc lengths with mean 1429 tokens,
matching 5e9/3.5e6 corpus stats; `heavy` is the adversarial 128-token case.
`--packed single` is the frozen-window control in the same session.)

## Candidate 3 — selective-recompute diet enabling B32x1/B32x2

Mechanism: `opt/model_diet.py` — identical math; RoPE and (Dy+relu+paper
+reshape) wrapped in non-reentrant checkpoint (no RNG), so only x_bt and
a_ln are retained; everything else recomputed. Gate: 192 configs ≤7.5e-09
fp32. Local: -18..27% peak memory for +8.5-10.5% time at T2048 B1.

Why it may win on G4 despite losing locally: B16x4 currently uses 58.34 GiB;
the diet enables **B32x2 no-checkpoint** (~85 GiB projected) which halves
accumulation steps and doubles GEMM batch. Local accumulation overhead is
~2% for 2x steps, so the prize is GEMM efficiency at B32 — a Blackwell
question the 4060 cannot answer. KILL on G4 if B32 diet does not beat
B16 champion by ≥5%.

```
python opt/bench_matrix.py --compile --full --gb 32 --seed 103 \
    --shapes T2048_L8 --blocks 1024 \
    --variants diet_nockpt_b1024
# OOM-safe: records the exact ask/free if it does not fit; then fall back:
python opt/bench_matrix.py --compile --full --gb 16 --seed 103 \
    --shapes T2048_L8 --blocks 1024 --variants diet_nockpt_b1024
```

## Gates to run on G4 before timing (BF16 path)

```
python opt/test_correctness.py     # 180 randomized cases, fwd+gq+gv+coord, fp64 oracle
python opt/test_model_equiv.py     # model-level vs canonical, packed+single, all scan forms
python opt/test_diet_gate.py       # 192 diet configs <=7.5e-09
python opt/compile_smoke.py --shape T512_L8 --coord dense   # compiled==eager (needs gcc)
```

## Explicitly killed (do NOT re-run)

- checkpoint-everything / SAC on the scan (51-53k on G4, -30%);
- parallel-scan with materialized G states (+6-7% compiled, more memory);
- hybrid batched-local (+4-10% compiled);
- static4 compiled (tie; kept only as an eager fallback);
- chunk-classification for packed (+9-14% from per-chunk host syncs);
- prefix-coordinator cumsum under torch.compile (inductor bug; dense coord
  is ≤1% of FLOPs and already in the champion);
- max-autotune on small-SM GPUs (inductor: "Not enough SMs");
- outer-product scan forms (OOM by construction).

## What the profile says (for the G4 kernel decision, deferred)

T2048 B1 b512 compiled: local QK bmm ~29% of fwd CUDA; dense GEMMs 35-40%;
elementwise ~27%. b1024 full update 185.8 ms vs b512 172.6 ms locally, but
G4 prefers b1024 — a different GEMM/launch balance. If G4 wants a fused
Triton scan: fuse per chunk = lower-triangular local QK + local@V + carry +
state update, K/Dv tiled in SRAM (state 1M els cannot be SRAM-resident, so
state streams). Expected ceiling ~37% of local-QK FLOPs ≈ 8% of update,
at unknown Blackwell tile efficiency. Build ONLY against G4 profiles.
