# SODA-BDH 5-Minute Training Pilot — BLOCKED Report

Status: **BLOCKED at Phase 1 (source lock). No pilot cell was produced and no
training was executed.**

This report is the honest outcome of the mission's own gate:

> If the source does not contain a dense SODA-BDH training path, stop with
> BLOCKED. Do not invent one.

## 1. What was required vs what exists

| Required input (source authority order) | Status | Evidence |
|---|---|---|
| 1. The supplied original dense Arm-A training cell | **ABSENT** | No cell bytes, attachment, or path were supplied with the request. No cell content exists in the conversation. |
| 2. Canonical dense BDH source | **VENDOR FILE ABSENT** | `vendor/canonical/icrl_reintegrated_bdh_baseline_v1.py` is not in git and not on disk; only its SHA-256 is pinned (`947f8b33…`) in `context/canonical_source_provenance.md`, `README_FIRST.md`, and `artifacts/dense_bdh_verification/source_manifest.json`. |
| 3. Known-good dense verification at `df7da68` | **PRESENT** | Commit `df7da68` verifies the hash-matched repo-local dense path: correctness PASS (small-config fwd/grads/param update exact; production forward 1.3e-6; cross-row isolation); its own message and report state **"Soda reference absent"**. |
| 4. Repository files matching the frozen contract | **PRESENT (dense BDH/Arm-A only)** | `opt/model_opt.py` `722fbdc1…`, `opt/scan_attn.py` `855f868e…`, `opt/model_ref.py` `2ff16aaa…`, `training/arm_a_2p5b_trainer.py` `1985fa42…` all match the frozen project contract. |

## 2. Why this is BLOCKED and not PASS_WITH_LIMITATIONS

- No source anywhere defines a **SODA-BDH** model, configuration, checkpoint,
  test, or training path. The string "Soda" appears only as the name of the
  **missing** original vendor reference (`opt/dense_verify_*.py`,
  `artifacts/dense_bdh_verification/*`, commit `df7da68`).
- Searches performed: `rg -ri soda` over the working tree (excluding `.git`),
  `git log --all -i --grep=soda` (single hit: `df7da68`, "Soda reference
  absent"), archive listing of `campaigns.zip`, and targeted greps of
  `training/`, `reference/`, `results/`. No definition exists.
- The hash-verified dense BDH path (`opt/model_opt.py` OptArmA opt3c,
  `opt/model_ref.py` NativeReadStage1ArmA, `training/arm_a_2p5b_trainer.py`)
  is a **different, repo-local implementation** from the canonical Soda
  vendor source. Building a "SODA-BDH pilot" on it and presenting it as such
  would silently replace the required source — explicitly forbidden by the
  mission ("Do not reconstruct the model from memory", "Do not silently
  replace the source with another implementation").
- Therefore no dense SODA-BDH training path was confirmed, and the correct
  classification is **BLOCKED** (required dense source/reference
  unavailable), not FAIL (the path never ran) and not PASS_WITH_LIMITATIONS
  (a working training path was never established for the required model).

## 3. What was deliberately not done

- No replacement cell was written. Writing one would have required inventing
  or renaming model semantics.
- No forward/backward/optimizer step was executed, because Phase 2 requires
  instantiating **the actual dense SODA-BDH model**, which does not exist in
  any available source. Running the dense Arm-A path instead would be a
  substitution.
- No artifacts were fabricated: `correctness.json`, `training_log.csv`,
  `pilot_summary.json`, `soda_bdh_pilot_checkpoint.pt` are **not present**.
  Only `source_manifest.json` and this report were produced.

## 4. Existing parity evidence (context only, not a SODA-BDH claim)

For the hash-verified repo-local dense path, `artifacts/dense_bdh_verification/`
(commit `df7da68`) already contains: small-config forward/loss/gradient/param
update diffs `0.0`; production forward parity `1.31e-06` (B1) / `1.07e-06`
(B2); determinism `0.0`; finite loss/grads true; cross-row isolation exact
(modify row 1 → row 0 changes by `0.0`). This is evidence that the dense
carrier is sound **if** the owner confirms it is the intended semantic source.

`training/arm_a_2p5b_trainer.py --mode smoke` already performs real packed
updates, atomic checkpoint save, fresh-model reload, and one identical update
on both paths with loss/parameter/cursor equivalence. It is the natural
carrier for a five-minute pilot once source identity is resolved.

## 5. Exact unblock conditions (any one)

1. Supply the original dense Arm-A training cell (the top-priority source in
   the mission's precedence order).
2. State explicitly that the hash-verified dense BDH/Arm-A path
   (`opt/model_opt.py` opt3c control + `opt/model_ref.py` reference +
   `training/arm_a_2p5b_trainer.py`) **is** the intended semantic source for
   the SODA-BDH pilot. This is an owner naming decision; it cannot be
   inferred from the repository.
3. Supply the canonical vendor file bytes matching the pinned SHA-256
   `947f8b33e740adede3e13382f8cb9e8e374d845de75e282bda5feb05c582f624`.

On any of these, the pilot can be derived and executed immediately: source
lock → live preflight (forward/loss/backward/AdamW) → 300 s two-stage pilot
(fixed-batch overfit smoke + real data path) → checkpoint save/reload/step →
gates → artifacts.

## 6. Environment (authoring)

- Python 3.12.10; torch 2.10.0+cu128; CUDA available.
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, capability (8, 9).
- Project-pinned container: torch 2.11.0+cu128 (`iclr-arm-a`); pilot target:
  Colab G4 / RTX PRO 6000 Blackwell sm_120 with the frozen packed corpus.
- `/content/soda_bdh_5min_pilot/` is not available in the authoring
  environment (no Colab runtime); artifacts were written to the local mirror
  `artifacts/soda_bdh_5min_pilot/`.

## 7. Required final report block

VERDICT: BLOCKED
SOURCE_COMMIT: df7da680dd6420a0c4fefa35c13f107246091308
SOURCE_HASH_STATUS: DENSE_BDH_PATH_VERIFIED_BY_HASH / CANONICAL_SODA_SOURCE_ABSENT / SUPPLIED_CELL_ABSENT
MODEL: NOT_IDENTIFIED (no SODA-BDH definition exists in any available source)
DENSE_PATH_CONFIRMED: false (SODA-BDH); true only for the hash-verified dense BDH/Arm-A path
DATA_MODE: NOT_RUN
COMPLETED_UPDATES: 0
TOTAL_TOKENS: 0
INITIAL_LOSS: N/A
FINAL_LOSS: N/A
LOSS_TREND: N/A
GRADIENT_STATUS: NOT_RUN
PARAMETER_UPDATE_STATUS: NOT_RUN
CHECKPOINT_RELOAD_STATUS: NOT_RUN
PEAK_GPU_MEMORY_MB: N/A
NAN_INF_EVENTS: 0
OOM_EVENTS: 0
THROUGHPUT_DIAGNOSTIC: NOT_RUN
PRIMARY_FAILURE: Required dense SODA-BDH source and the supplied dense Arm-A training cell are unavailable; the canonical Soda vendor file is absent and no SODA-BDH path exists to execute without inventing or substituting semantics.
LIMITATIONS: No replacement cell was produced (would require invented semantics); no training artifacts were generated; only source_manifest.json and this report exist; /content path unavailable in the authoring environment.
ARTIFACT_DIRECTORY: artifacts/soda_bdh_5min_pilot (local mirror; uncommitted)
