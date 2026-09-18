# SODA-BDH pilot — local preflight (authoring machine, NOT pilot evidence)

Date: 2026-09-16 (authoring host)
Pilot target: Colab G4, run by the human with
`campaigns/SODA_BDH_5MIN_PILOT_CELL.md`.

Do not mistake this file for the pilot report. The pilot report (with
checkpoint and training log) is produced on G4 by
`scripts/soda_bdh_5min_pilot.py --phase all`.

## Naming decision (owner, Option A)

SODA-BDH := hash-verified repo-local canonical dense BDH/Arm-A path at commit
`df7da680dd6420a0c4fefa35c13f107246091308`.
Vendor Soda file (pinned SHA-256 `947f8b33…`) remains UNAVAILABLE and
equivalence is UNVERIFIED.

## Commands executed locally

```
py -3.12 scripts/soda_bdh_5min_pilot.py --phase selftest
py -3.12 scripts/soda_bdh_5min_pilot.py --phase lock --pilot-dir <temp>
```

## Results

### 1. Harness selftest (tiny CPU model + synthetic log records)

`HARNESS_SELFTEST_PASS=true`

- model construction: true
- loss finite (step 1 / step 2): true / true
- gradients finite: true; non-zero: true
- AdamW changed parameters: true
- param update max-abs / L2: 2.517e-04 / 2.140e-02
- report builder verdict on synthetic records: PASS
- CSV rows: 5; pilot_report.md written; hard failures: none

This validates the orchestrator's model-step code path (using the frozen
trainer's own `ArmAConfig`, `_build_model`, `make_optimizer`,
`one_full_update`), the artifact builders, and the verdict logic. It is a
harness test only; no corpus, no G4, no pilot evidence.

### 2. Phase-1 source lock (real repository)

`SOURCE_HASH_STATUS=VERIFIED`, `SOURCE_COMMIT=df7da680dd6420a0c4fefa35c13f107246091308`,
lock exit code 0.

| file | SHA-256 match vs frozen contract | git worktree clean |
|---|---|---|
| `opt/model_opt.py` | true | true |
| `opt/scan_attn.py` | true | true |
| `opt/model_ref.py` | true | true |
| `training/arm_a_2p5b_trainer.py` | true | true |

- git commit matches the pinned commit: true
- vendor Soda file: UNAVAILABLE (explicitly labelled, not required)
- corpus verification mode recorded: fast (per-shard SHA-256 skipped; sizes,
  artifact index, manifest digests verified)

### 3. Not run locally (by design)

- `--phase preflight` / `smoke` / `train`: the frozen trainer's
  `enforce_production_runtime` requires RTX PRO 6000 Blackwell sm_120,
  torch 2.11.0+cu128, CUDA 12.8, and the frozen corpus. The local RTX 4060
  host cannot satisfy this; per instruction the local CLI is not the pilot.

## G4 rerun (human action)

Paste the cell from `campaigns/SODA_BDH_5MIN_PILOT_CELL.md` into a fresh G4
Colab runtime. Artifacts land in `/content/soda_bdh_5min_pilot/` and are
mirrored to Drive under `runs/soda_bdh_5min_pilot/soda_pilot_<utc>/`.
