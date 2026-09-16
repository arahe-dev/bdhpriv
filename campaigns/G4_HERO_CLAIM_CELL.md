# G4 HERO CLAIM VALIDATION — ONE COLAB CELL

Status: **prepared, not executed.** This file contains the single cell for the
manual G4 run. Nothing here has been run on G4 by the authoring environment.

## What this validates

- **HERO 1 (sparse production speed):** sparse Arm-A top1, M8/Ke512 fixed
  cyclic window G=128, exact capacity, compiled production stack —
  **≥350k packed tok/s** target (preferred ≥380k, stretch ≥400k) at global
  batch 64×T2048, same-session against dense `opt3c_all`.
- **HERO 2 (learned conditional compute):** learned top2 hard routing
  (straight-through proxy, inactive experts skipped) — overhead vs fixed
  top2 and speedup vs dense at matched B16x4.

`ARM_B_STATUS` and `ARM_C_STATUS` are reported as
`BLOCKED_BY_MISSING_CANONICAL_SPEC` (see `campaigns/ARM_BC_SEMANTICS.md`);
no B/C implementation is invented or benchmarked.

## Preconditions

1. Fresh G4 Colab runtime: RTX PRO 6000 Blackwell (sm_120), torch
   `2.11.0+cu128`, CUDA 12.8, ≥90 GiB free VRAM. The harness fails closed on
   any other environment.
2. The repo copy on the machine must include the contract commit
   (`results/g4_hero_claim_contract.json`, harness commit
   `38ecc189ad68e9726a9fe1fc9dbc87e475591ff5`) and be fingerprint-clean.
   Sync it to `/content/iclr-oc` (or the Drive copy) first, e.g.:
   ```bash
   !git -C /content/iclr-oc fetch origin && git -C /content/iclr-oc checkout 38ecc189ad68e9726a9fe1fc9dbc87e475591ff5
   ```
   The cell verifies the harness SHA-256 before running and prints the
   expected value if the copy is stale.
3. Frozen packed corpus at
   `/content/drive/Shareddrives/ICLR PHASE BDH/phase_bdh/corpus/stage2/frozen_5b_v1`
   (manifest + artifact digests + shard sizes verified; per-shard SHA-256 is
   skipped by `fast_verify=True` and this is recorded).

## Expected runtime and output

- ~20–45 min: gates (~5–15 min, includes 4 production-shape compiles for
  graph checks) + 8–9 benchmark processes (3 warmups + 10 measured updates
  each, one fresh process and compile per config) + report.
- Output ends with the verdict table and machine-readable lines
  (`HERO_1_STATUS`, `HERO_2_STATUS`, `ARM_B_STATUS`, `ARM_C_STATUS`,
  `BEST_G4_CANDIDATE`, `BEST_PACKED_TOK_S`, `PROJECTED_2P5B_HOURS`,
  `SESSION_STABLE`, `FINAL_G4_VALIDATION_PASS`).
- Results JSON: `<drive runs root>/g4_hero_<utc>/g4_hero_claim_results.json`
  (plus `gates.json`, `plan.json`, one `bench_*.json` per config).
- Evidence labels: MEASURED (G4), INFERRED (2.5B projection), SPECULATIVE
  (training/quality consequences). No 2.5B training is launched.

```python
# ============================================================================
# ICLR — G4 HERO CLAIM VALIDATION (single cell, run top-to-bottom on G4)
# Thin cell: locates the repo, verifies the harness hash, runs
# scripts/g4_hero_claim_validation.py --mode runall, prints the verdict.
# ============================================================================
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HARNESS_REL = "scripts/g4_hero_claim_validation.py"
HARNESS_SHA256 = "99481652a882cec467fe855352abf3a22beaa2db94e9c24b6f320cbe96c5aceb"
CORPUS_ROOT = ("/content/drive/Shareddrives/ICLR PHASE BDH/"
               "phase_bdh/corpus/stage2/frozen_5b_v1")
RUNS_ROOT = ("/content/drive/Shareddrives/ICLR PHASE BDH/"
             "phase_bdh/runs/arm_a_sparse_350k")
REPO_CANDIDATES = [
    os.environ.get("ICLR_REPO", ""),
    "/content/iclr-oc",
    "/content/drive/Shareddrives/ICLR PHASE BDH/phase_bdh/iclr-oc",
    "/content/drive/MyDrive/iclr-oc",
    str(Path.cwd()),
]

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

print("python:", sys.version.split()[0], flush=True)


def locate_repo():
    seen = set()
    for candidate in REPO_CANDIDATES:
        if not candidate:
            continue
        path = Path(candidate)
        if str(path) in seen:
            continue
        seen.add(str(path))
        if (path / HARNESS_REL).is_file():
            return path
    return None


repo = locate_repo()
if repo is None:
    print("REPO_NOT_FOUND: no candidate contains", HARNESS_REL)
    for candidate in REPO_CANDIDATES:
        print("  tried:", candidate or "<empty>")
    print("Sync the repo (commit 38ecc189ad68e9726a9fe1fc9dbc87e475591ff5) "
          "to /content/iclr-oc, then rerun this cell.")
    raise SystemExit(1)

harness_bytes = (repo / HARNESS_REL).read_bytes().replace(b"\r\n", b"\n")
harness_sha = hashlib.sha256(harness_bytes).hexdigest()
print("repo:", repo, flush=True)
print("harness sha256:", harness_sha, flush=True)
print("harness hash match:", harness_sha == HARNESS_SHA256, flush=True)
if harness_sha != HARNESS_SHA256:
    print("HARNESS_HASH_MISMATCH: the repo copy is stale. Update it to the "
          "contract commit (see results/g4_hero_claim_contract.json), then "
          "rerun this cell.")
    raise SystemExit(1)

run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
outdir = Path(RUNS_ROOT) / ("g4_hero_" + run_id)
outdir.mkdir(parents=True, exist_ok=True)
print("outdir:", outdir, flush=True)

cmd = [
    sys.executable, str(repo / HARNESS_REL), "--mode", "runall",
    "--repo", str(repo), "--outdir", str(outdir),
    "--corpus-root", CORPUS_ROOT,
    "--start-sequence", "0", "--steps", "10", "--warmups", "3",
    "--seed", "4242",
]
env = dict(os.environ)
env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
print("RUN:", " ".join(cmd), flush=True)
returncode = subprocess.run(cmd, env=env).returncode
print("runall returncode:", returncode, flush=True)

results_path = outdir / "g4_hero_claim_results.json"
if results_path.is_file():
    results = json.loads(results_path.read_text(encoding="utf-8"))
    summary_keys = (
        "gates_pass", "hard_failures", "arm_b_status", "arm_c_status",
        "best_g4_candidate", "best_microbatch", "best_packed_tok_s",
        "projected_2p5b_hours", "session_stable",
        "final_g4_validation_pass",
    )
    print(json.dumps({k: results.get(k) for k in summary_keys}, indent=2))
    print("RESULTS_JSON =", results_path)
else:
    print("NO_RESULTS_JSON: gates likely failed before any benchmark; "
          "inspect", outdir)
```

## Reading the verdict (what the cell does not decide for you)

- `FINAL_G4_VALIDATION_PASS` requires all hard gates to pass, at least one
  hero claim to PASS, and the dense drift control to stay within 5%. A HOLD
  is a real outcome: it means the claim was not falsified but did not reach
  its pre-registered bar on this run.
- `PROJECTED_2P5B_HOURS` is **INFERRED** from the measured full-update
  throughput (`2.5e9 / packed_tok_s / 3600`). It is never reported as a
  measured 2.5B runtime.
- If `SESSION_STABLE` is false, rerun the cell in a clean runtime; do not use
  the numbers for a paper claim.
- `ARM_B_STATUS` / `ARM_C_STATUS` stay blocked until a canonical semantic
  definition is supplied (`campaigns/ARM_BC_SEMANTICS.md` lists the three
  acceptable ways to lift the gate).
