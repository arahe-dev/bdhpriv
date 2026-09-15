# Arm-A 2.5B run — launch (G4 Colab)

The trainer is a single self-contained file:
`training/arm_a_2p5b_trainer.py` (`arm_a_2p5b_trainer_v1_opt3c_all_b1024`).
It uses the certified `opt3c_all_b1024` implementation exactly (branch-free
packed update, zero-carry skip, direct paper_y layout, cached RoPE, dense
coordinator, no activation checkpointing, B16x4, BF16 autocast + FP32 master).

Copy the file next to the run output on Drive, e.g.
`/content/drive/Shareddrives/ICLR PHASE BDH/phase_bdh/runs/arm_a_2p5b_opt3c_all/arm_a_2p5b_trainer.py`,
then run these two cells in order in a fresh G4 runtime.

## 1. Smoke gate (must pass before training)

```python
from google.colab import drive
drive.mount("/content/drive")

TRAINER = ("/content/drive/Shareddrives/ICLR PHASE BDH/phase_bdh/"
           "runs/arm_a_2p5b_opt3c_all/arm_a_2p5b_trainer.py")

!python {TRAINER} --mode smoke
```

Required output line: `ARM_A_2P5B_TRAINER_READY=true`.
The smoke does 2 real packed updates, an atomic checkpoint, a fresh-model
reload, one identical update on both paths, then proves loss / parameter /
corpus-cursor equality. It also gates on graph breaks (`graph_break_count==0`).

## 2. Training (2.5B budget, auto-resume)

```python
!python {TRAINER} --mode train
```

- Target defaults to `2_500_000_000` input tokens (19,074 updates at
  131,072 tokens/update → `TOKENS_CONSUMED=2500067328`; the last update
  crosses the budget by 67,328).
- Later 5B continuation (no optimizer/LR/token-schedule restart):

```python
!python {TRAINER} --mode train --target-tokens full
```

- Resume is automatic and fail-closed: newest valid of `ckpt/latest.pt` or
  `ckpt/step_XXXXXXXXXX.pt`; a checkpoint whose code fingerprint, config,
  corpus hashes, or token accounting does not match is rejected.
- Rerunning after a Colab disconnect restarts exactly where it stopped.
- `--fast-verify` skips the ~11 GB Drive SHA-256 pass (sizes + index digest
  still checked) when restarting quickly.

Expected final lines:

```
TRAIN_STATUS=COMPLETE
UPDATES_DONE=19074
TOKENS_CONSUMED=2500067328
```

Outputs: `runs/arm_a_2p5b_opt3c_all/logs/train.jsonl` (loss, LR, tok/s,
valid pairs, memory per 10 updates), `ckpt/latest.pt` every 200 updates, two
rotating `ckpt/step_*.pt` archives (every 1000 updates).

Runtime estimate: ~1.9 s/update certified, `opt3c_all` measurably faster
(local A/B −5.7%…−12.3% vs certified on the same session), so ~9–10 h.
