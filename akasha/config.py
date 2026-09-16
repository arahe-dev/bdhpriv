"""Frozen provenance and artifact-status records for the Akasha V0 mission.

The implementation source of truth is ``training/arm_a_2p5b_trainer.py`` at
commit ``0dcbb878d24b99b5808359c889e97143c3cec00b`` (frozen systems state).
Where documentation and the trainer disagree, the trainer wins.
"""

from __future__ import annotations

REPO_SOURCE_COMMIT = "0dcbb87"
REPO_SOURCE_COMMIT_FULL = "0dcbb878d24b99b5808359c889e97143c3cec00b"
SOURCE_HEAD_AT_IMPLEMENTATION = "c8d751b5320f3cd95234687484362c707cd975fd"
TRAINER_PATH = "training/arm_a_2p5b_trainer.py"
TRAINER_SHA256 = (
    "1985fa42042033c842c7ed0faea2c34ead6516bb1fe56753426b6774ee2d0b49"
)
TRAINER_IMPLEMENTATION_VERSION = "arm_a_2p5b_trainer_v1_opt3c_all_b1024"
TRAINER_CKPT_FORMAT = "arm_a_2p5b_ckpt_v1"

TOKENIZER_IDENTITY = "bytelevel-bpe-8192-9a05bca4c065d995"
TOKENIZER_EXPECTED_SHA256 = (
    "9a05bca4c065d9952a01ee1e3f4b6e23e39200e1f33da65e5dad1819419f71c3"
)

TOKENIZER_ARTIFACT_ENV = "AKASHA_TOKENIZER_PATH"
TRAINED_CHECKPOINT_ENV = "AKASHA_ARM_A_CHECKPOINT"

AKASHA_V0_REFERENCE_PASS_ENV = "AKASHA_V0_REFERENCE_PASS"

REQUIRED_V0_LENGTHS = (1, 2, 7, 31, 32, 127, 128, 511, 512, 1024, 2048)

FP32_ATOL = 1e-5
FP32_RTOL = 1e-4
FP64_ATOL = 1e-11
