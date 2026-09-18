# SODA-BDH 5-MINUTE PILOT — ONE COLAB CELL

Status: **prepared, not executed by the authoring environment.** Local
execution is preflight only (`--phase selftest`, `--phase lock`); the pilot
must run on G4 because the frozen trainer's production-runtime gate requires
RTX PRO 6000 Blackwell sm_120 / torch 2.11.0+cu128 / CUDA 12.8 and the
frozen corpus.

## Naming decision in force (owner, Option A)

For this pilot, **SODA-BDH** means the hash-verified repo-local canonical
dense BDH/Arm-A path at commit `df7da680dd6420a0c4fefa35c13f107246091308`.
It does **not** claim equivalence to the absent vendor Soda file
(SHA-256 `947f8b33…`), which stays explicitly labelled unavailable.
Result labels: `MODEL=SODA-BDH`,
`MODEL_DEFINITION=repo-local canonical dense BDH/Arm-A`,
`VENDOR_SODA_EQUIVALENCE=UNVERIFIED`, `DENSE_PATH_CONFIRMED=true`,
`DATA_MODE=FROZEN_CORPUS_G4`.

## What the cell runs

`scripts/soda_bdh_5min_pilot.py --phase all` orchestrates, without modifying
the canonical trainer:

1. **lock** — git commit + SHA-256 of the 4 verified files
   (`opt/model_opt.py`, `opt/scan_attn.py`, `opt/model_ref.py`,
   `training/arm_a_2p5b_trainer.py`) against the frozen contract. Mismatch →
   `INVALID_SOURCE`, stop.
2. **preflight (Phase 2 + Stage A)** — real packed batch: forward, loss,
   backward, finite/non-zero gradients, one AdamW step (parameter-update
   magnitude measured), second step, cross-row isolation, then a fixed-batch
   overfit smoke (≤30 updates / 90 s) for the loss-trend gate.
3. **smoke** — the trainer's own checkpoint save/reload/continued-step gate:
   `python training/arm_a_2p5b_trainer.py --mode smoke --corpus-root "$CORPUS"
   --run-dir "$PILOT"`
4. **train** — fresh start (new run dir; no 2.5B continuation):
   `python training/arm_a_2p5b_trainer.py --mode train --corpus-root "$CORPUS"
   --run-dir "$PILOT" --target-tokens 24000000 --save-every 1 --log-every 1`
   bounded by the five-minute pilot budget (300 s, 20 s reserve) with
   per-update atomic checkpoints; whichever bound fires is reported.
5. **report** — required artifacts + verdict.

Artifacts under `/content/soda_bdh_5min_pilot/`:
`source_manifest.json`, `environment.json`, `correctness.json`,
`training_log.csv`, `pilot_summary.json`, `pilot_report.md`,
`soda_bdh_pilot_checkpoint.pt` (model + optimizer + progress/step/token
counts + config/seed + RNG + source hash), plus raw JSONL logs and per-phase
JSON. The cell mirrors the artifacts to Drive at the end.

Verdicts: `PASS` (all hard gates, real updates), `PASS_WITH_LIMITATIONS`
(path works, wall-clock/target not completed), `FAIL` (correctness/gradient/
optimizer/checkpoint/numerical failure), `BLOCKED` (runtime/corpus),
`INVALID_SOURCE` (hash mismatch).

## Preconditions

1. Fresh Colab G4 runtime (sm_120, torch `2.11.0+cu128`, CUDA 12.8).
2. Repo copy containing this commit (or a descendant of
   `df7da680dd6420a0c4fefa35c13f107246091308`) with the four verified files
   and `scripts/soda_bdh_5min_pilot.py`. The cell verifies the orchestrator
   SHA-256 before running:
   `31729694ab494c74b16efa5dc2a1c9485b0a346e79065562063ea16aed7f264d`.
3. Frozen corpus at
   `/content/drive/Shareddrives/ICLR PHASE BDH/phase_bdh/corpus/stage2/frozen_5b_v1`.
   Fast verification (default) skips per-shard SHA-256; sizes, artifact index
   and manifest digests are still checked. Pass `--full-verify` for full
   per-shard SHA-256 (slower).

## Local preflight (authoring machine, no pilot evidence)

```bash
py -3.12 scripts/soda_bdh_5min_pilot.py --phase selftest   # HARNESS_SELFTEST_PASS=true
py -3.12 scripts/soda_bdh_5min_pilot.py --phase lock --pilot-dir /tmp/soda_lock
```

The local 4060 cannot run smoke/train: the trainer fails closed on non-G4.

## The cell

```python
# ============================================================================
# SODA-BDH 5-minute training pilot (run on G4, fresh runtime)
# Canonical dense BDH/Arm-A path (naming decision Option A); the absent
# vendor Soda file is labelled unavailable. No sparse/routed/top-k paths.
# ============================================================================
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PILOT_SCRIPT_REL = "scripts/soda_bdh_5min_pilot.py"
PILOT_SCRIPT_SHA256 = (
    "31729694ab494c74b16efa5dc2a1c9485b0a346e79065562063ea16aed7f264d")
CORPUS_ROOT = ("/content/drive/Shareddrives/ICLR PHASE BDH/"
               "phase_bdh/corpus/stage2/frozen_5b_v1")
PILOT_DIR = "/content/soda_bdh_5min_pilot"
DRIVE_MIRROR_ROOT = ("/content/drive/Shareddrives/ICLR PHASE BDH/"
                     "phase_bdh/runs/soda_bdh_5min_pilot")
TARGET_TOKENS = 24_000_000
PILOT_BUDGET_S = 300
RESERVE_S = 20
MIRROR_CHECKPOINT = True
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
        if (path / PILOT_SCRIPT_REL).is_file():
            return path
    return None


repo = locate_repo()
if repo is None:
    print("REPO_NOT_FOUND: no candidate contains", PILOT_SCRIPT_REL)
    for candidate in REPO_CANDIDATES:
        print("  tried:", candidate or "<empty>")
    raise SystemExit(1)

script = repo / PILOT_SCRIPT_REL
sha = hashlib.sha256(script.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
print("repo:", repo, flush=True)
print("pilot script sha256:", sha, flush=True)
print("pilot script hash match:", sha == PILOT_SCRIPT_SHA256, flush=True)
if sha != PILOT_SCRIPT_SHA256:
    print("PILOT_SCRIPT_HASH_MISMATCH: sync the repo to the commit that "
          "contains this pilot script, then rerun.")
    raise SystemExit(1)

shutil.rmtree(PILOT_DIR, ignore_errors=True)
Path(PILOT_DIR).mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, str(script), "--phase", "all",
    "--corpus-root", CORPUS_ROOT, "--pilot-dir", PILOT_DIR,
    "--target-tokens", str(TARGET_TOKENS),
    "--pilot-budget-s", str(PILOT_BUDGET_S),
    "--reserve-s", str(RESERVE_S),
]
env = dict(os.environ)
env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
print("RUN:", " ".join(cmd), flush=True)
returncode = subprocess.run(cmd, env=env).returncode
print("pilot returncode:", returncode, flush=True)

summary_path = Path(PILOT_DIR) / "pilot_summary.json"
if summary_path.is_file():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps({
        "verdict": summary.get("verdict"),
        "labels": summary.get("labels"),
        "completed_updates": summary.get("train", {}).get(
            "completed_updates"),
        "tokens_consumed": summary.get("train", {}).get("tokens_consumed"),
        "throughput_input_tokens_per_s": summary.get("train", {}).get(
            "throughput_input_tokens_per_s"),
        "hard_failures": summary.get("hard_failures"),
        "limitations": summary.get("limitations"),
    }, indent=2), flush=True)

run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
mirror = Path(DRIVE_MIRROR_ROOT) / ("soda_pilot_" + run_id)
mirror.mkdir(parents=True, exist_ok=True)
required = [
    "source_manifest.json", "environment.json", "correctness.json",
    "training_log.csv", "pilot_summary.json", "pilot_report.md",
    "soda_bdh_pilot_checkpoint.pt",
    "smoke_outcome.json", "train_outcome.json",
    "correctness_preflight.json", "logs/smoke.jsonl",
    "logs/train.jsonl", "logs/startup.jsonl",
]
for rel in required:
    src = Path(PILOT_DIR) / rel
    if not src.is_file():
        continue
    if rel == "soda_bdh_pilot_checkpoint.pt" and not MIRROR_CHECKPOINT:
        continue
    dst = mirror / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
print("DRIVE_MIRROR =", mirror, flush=True)
print("ARTIFACT_DIRECTORY =", PILOT_DIR, flush=True)
```

## Scope boundary (final claim)

> The repo-local canonical dense BDH/Arm-A training path completed a
> five-minute frozen-corpus pilot. The original vendor Soda implementation
> was unavailable, so vendor-level Soda equivalence was not tested.

Five minutes of training establishes execution integrity only — not
convergence, not final model quality, and no sparse-path result.
