# iclr-oc — Arm-A execution optimization workspace

Systems-level optimization of the frozen canonical Arm-A (BDH-style) model
for the ICLR Phase BDH campaign. The scientific model, loss, optimizer,
corpus, and mask semantics are frozen; only the execution engine changes.

## Current state (2026-09-15)

- **Production champion (certified on G4):** `opt3c_nockpt_b1024` —
  chunkwise exact score-free scan, block=1024, dense canonical coordinator,
  no checkpoint, B16x4, `torch.compile(default)`. Frozen packed corpus:
  **69,183 tok/s, 1894.56 ms/update, 62.55 GiB** (`1.5249x` packed
  canonical); clean windows 74,419 tok/s (`1.636x`).
- **Promoted candidate (same math, exact, awaiting G4 confirmation):**
  `opt3c_all_b1024` — adds four exact execution changes
  (branch-free packed state update, zero-carry skip for the t0==0 chunk,
  direct `paper_y` layout, cached RoPE phase). Local same-session compiled
  A/B vs certified: single -5.7%, mixed -9.7%, four -10.3%, heavy -12.3%.
- **Single G4 entry point:** `results/g4_opt3c_all_final.py` — one
  self-contained Colab cell: robustness gates (exactness, fp32/bf16,
  eager/compiled, determinism, checkpoint resume, frozen-corpus smoke)
  followed by the certified-vs-candidate benchmark. Prints
  `OPT3C_ALL_ROBUST` and `CANDIDATE_BEATS_CERTIFIED_ANCHOR_69183`.
- **Local preflight:** `results/verify_opt3c_all.py` (imports `opt.*`,
  for dev/CI machines); **local cell validator:**
  `opt/validate_final_cell.py`.

## Layout

| path | purpose |
|---|---|
| `reference/` | immutable handoff reference (canonical timing cell + oracle) |
| `context/` | frozen contracts, G4 baseline, optimization contract |
| `opt/` | implementation under test (`model_ref.py`, `model_opt.py`, `scan_attn.py`, `model_diet.py`), gates, benchmarks, validators |
| `results/` | ledger, G4 pack, G4 cells/scripts, raw benchmark rows, profiles |
| `scripts/` | env report + container recreation recipe |
| `PROMPT_FOR_META_MUSE.md` | overnight mandate given to the agent |
| `README_FIRST.md` | handoff instructions from the orchestrator |

## Key evidence

- `results/ledger.md` — every promoted/killed mechanism with measurements.
- `results/G4_CONFIRM.md` — what to run on G4 and why.
- `results/local_matrix.json` — all raw benchmark rows (session-tagged).
- `results/e0_*.txt`, `results/profile_*.txt` — profiler/cost evidence.

## Environment

- Dev container recipe: `scripts/container_setup.md` (torch 2.11.0+cu128,
  RTX 4060 Laptop 8 GB).
- G4: RTX PRO 6000 Blackwell SE, sm_120, ~95 GiB, torch 2.11.0+cu128.
