# ICLR Arm-A overnight optimization handoff

## Yes: this is the code to give Meta Muse

`reference/arm_a_timing_cell_colab.py` is the exact Arm-A timing cell supplied for the verified G4 run. It contains the executable canonical Arm-A model definitions used by the benchmark.

Do **not** run that file unchanged on Windows:
- it imports `google.colab`;
- it expects the frozen corpus under Google Drive;
- it performs the expensive corpus census before benchmarking.

It is included as the source-of-truth reference from which the local optimization harness should be extracted.

## What is authoritative in this handoff

- Model semantics: `reference/arm_a_timing_cell_colab.py`
- Mathematical exactness oracle: `reference/scan_coordinator_oracle.py`
- G4 measured baseline: `context/g4_baseline.json`
- Frozen corpus contract: `context/frozen_corpus_contract.json`
- Optimization constraints: `context/optimization_contract.md`
- Agent instructions: `PROMPT_FOR_META_MUSE.md`

## Important provenance limitation

The original repository file
`vendor/canonical/icrl_reintegrated_bdh_baseline_v1.py`
is not physically present in this zip.

Its pinned SHA-256 is:
`947f8b33e740adede3e13382f8cb9e8e374d845de75e282bda5feb05c582f624`

The timing cell states that its model/init/RoPE/selective-checkpoint definitions are copied verbatim from that pinned source. Until the original vendor file is transferred, the timing cell is the executable reference. Do not falsely claim the extracted local file itself has the vendor hash.

## Install location

Extract this zip into:

`C:\iclr-oc`

Then tell Meta Muse 1.3:

> Read `PROMPT_FOR_META_MUSE.md` and execute the mandate autonomously. Treat `reference/` as immutable.

## Overnight hardware

Use the local RTX 4060 Laptop GPU for CUDA development and relative candidate ranking. Do not treat its absolute throughput as a G4 projection. Tomorrow, re-rank the top 2-3 candidates on the RTX PRO 6000 Blackwell G4.

## Files

- `reference/arm_a_timing_cell_colab.py` — exact supplied timing/reference cell
- `reference/scan_coordinator_oracle.py` — exact tiny CPU forward+gradient equivalence oracle
- `context/g4_baseline.json` — measured canonical Arm-A G4 baseline
- `context/transformer_control_context.json` — separate Transformer control context
- `context/frozen_corpus_contract.json` — frozen corpus contract
- `context/canonical_source_provenance.md` — source path/hash caveat
- `context/optimization_contract.md` — hard constraints and priorities
- `context/known_findings.md` — already-established findings
- `scripts/report_env.py` — local CUDA/PyTorch/GPU inventory
- `workspace_template/results/ledger.md` — candidate/failed-idea ledger template
- `PROMPT_FOR_META_MUSE.md` — paste-free agent handoff
