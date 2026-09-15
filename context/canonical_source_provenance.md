# Canonical Arm-A provenance

Canonical project source path:
`vendor/canonical/icrl_reintegrated_bdh_baseline_v1.py`

Pinned SHA-256:
`947f8b33e740adede3e13382f8cb9e8e374d845de75e282bda5feb05c582f624`

IMPORTANT:
- This handoff zip does NOT contain the original vendor file bytes.
- `reference/arm_a_timing_cell_colab.py` contains the canonical model/init/RoPE/selective-checkpoint definitions copied into the verified timing cell.
- Treat that timing cell as the executable source of truth until the original vendor file is separately supplied.
- Do not claim that a re-extracted local file matches the pinned SHA unless the actual vendor file is obtained and hashed.
