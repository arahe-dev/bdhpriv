"""Validation cell for the real trained Arm-A checkpoint.

Runs the complete conversion + load + token-ID inference smoke test against a
real trainer checkpoint. When the Drive artifact is not present, records
``TRAINED_CHECKPOINT_LOCAL_STATUS = BLOCKED`` and exits zero (the rest of V0
does not depend on it).

Exact usage when the artifact is available:

    py -3.12 -m akasha.checkpoint.validate_real_checkpoint ^
        --checkpoint "G:/My Drive/.../runs/arm_a_2p5b_opt3c_all/ckpt/latest.pt" ^
        --out-dir results/akasha/real_checkpoint_package ^
        --json-out results/akasha/v0_real_checkpoint.json

Or set ``AKASHA_ARM_A_CHECKPOINT`` and omit ``--checkpoint``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

from akasha.config import TRAINED_CHECKPOINT_ENV
from akasha.checkpoint.convert_arma import convert_trainer_checkpoint
from akasha.checkpoint.loader import load_package
from akasha.models.arma.reference_recurrent import (
    create_state,
    logits_from_state,
    prefill_tokens,
    step,
)


def find_checkpoint(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    env = os.environ.get(TRAINED_CHECKPOINT_ENV)
    if env:
        return Path(env)
    return None


def validate(checkpoint: Path, out_dir: Path, allow_code_change: bool = False):
    summary = convert_trainer_checkpoint(
        checkpoint, out_dir, allow_code_change=allow_code_change
    )
    loaded = load_package(out_dir)
    weights = loaded.weights
    cfg = loaded.cfg

    generator = torch.Generator(device="cpu").manual_seed(20260916)
    prompt = torch.randint(0, cfg.V, (8,), generator=generator).tolist()
    state = create_state(weights, cfg)
    logits = prefill_tokens(weights, cfg, state, prompt)
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("non-finite logits during real-checkpoint smoke")
    next_id = int(torch.argmax(logits).item())
    logits2 = step(weights, cfg, state, next_id)
    if not bool(torch.isfinite(logits2).all()):
        raise RuntimeError("non-finite logits during decode step")
    restored_logits = logits_from_state(weights, state)
    if not bool(torch.allclose(restored_logits, logits2, atol=1e-5, rtol=1e-4)):
        raise RuntimeError("last_hidden logits do not match the decode-step logits")

    return {
        "ok": True,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "package_dir": str(out_dir),
        "weights_fingerprint": summary["weights_fingerprint"],
        "progress": summary["progress"],
        "smoke_prompt": prompt,
        "smoke_next_token": next_id,
        "smoke_logits_finite": True,
        "last_hidden_roundtrip": True,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--allow-code-change", action="store_true")
    args = parser.parse_args(argv)

    checkpoint = find_checkpoint(args.checkpoint)
    if checkpoint is None or not checkpoint.is_file():
        report = {
            "TRAINED_CHECKPOINT_LOCAL_STATUS": "BLOCKED",
            "reason": (
                "no local trainer checkpoint supplied; pass --checkpoint or set "
                f"{TRAINED_CHECKPOINT_ENV}"
            ),
            "provided": None if checkpoint is None else str(checkpoint),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.json_out:
            Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json_out).write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
            )
        return 0

    out_dir = Path(args.out_dir) if args.out_dir else Path(
        tempfile.mkdtemp(prefix="akasha_real_ckpt_")
    )
    try:
        report = validate(checkpoint, out_dir, allow_code_change=args.allow_code_change)
    except Exception as exc:  # noqa: BLE001
        report = {
            "ok": False,
            "TRAINED_CHECKPOINT_LOCAL_STATUS": "VALIDATION_FAILED",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    report["TRAINED_CHECKPOINT_LOCAL_STATUS"] = "VALIDATED"
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
