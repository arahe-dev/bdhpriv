"""Convert a frozen Arm-A trainer checkpoint into a canonical package.

Source format (produced by ``build_checkpoint`` in
``training/arm_a_2p5b_trainer.py`` @ 0dcbb87): a ``torch.save`` payload with
``format == "arm_a_2p5b_ckpt_v1"``, ``implementation`` and a ``model``
``state_dict``. The converter validates provenance, maps tensor names, checks
shapes and writes ``model.safetensors`` + ``manifest.json``. It never mutates
the source checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from akasha.config import (
    TRAINER_CKPT_FORMAT,
    TRAINER_IMPLEMENTATION_VERSION,
)
from akasha.models.arma.config import ArmAConfig, production_config
from akasha.checkpoint.loader import map_trainer_state_dict, save_package

TRAINER_CONFIG_KEYS = (
    "T", "V", "D", "N", "H", "K", "L", "HIDDEN", "SEED", "INIT_STD", "THETA",
    "READ_BLOCK", "PEAK_LR", "BETAS", "EPS", "WEIGHT_DECAY", "CLIP_NORM",
    "GLOBAL_BATCH", "MICROBATCH", "SCAN_BLOCK", "WARMUP_TOKENS",
)


def sha256_file(path: Path, block: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(block)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_trainer_config(payload_config: Dict[str, Any], cfg: ArmAConfig) -> None:
    frozen = {
        "T": cfg.T, "V": cfg.V, "D": cfg.D, "N": cfg.N, "H": cfg.H,
        "K": cfg.K, "L": cfg.L, "HIDDEN": cfg.HIDDEN, "SEED": cfg.SEED,
        "INIT_STD": cfg.INIT_STD, "THETA": float(cfg.THETA),
        "READ_BLOCK": cfg.READ_BLOCK,
    }
    mismatches = {}
    for key in TRAINER_CONFIG_KEYS:
        if key not in payload_config:
            continue
        expected = frozen.get(key)
        if expected is None:
            continue
        actual = payload_config[key]
        if isinstance(expected, float):
            actual = float(actual)
        if isinstance(expected, int) and not isinstance(expected, bool):
            actual = int(actual)
        if actual != expected:
            mismatches[key] = {"expected": expected, "found": actual}
    if mismatches:
        raise ValueError(
            "checkpoint configuration does not match the frozen Arm-A config: "
            + json.dumps(mismatches, sort_keys=True)
        )


def convert_trainer_checkpoint(
    ckpt_path: str | Path,
    out_dir: str | Path,
    cfg: Optional[ArmAConfig] = None,
    allow_code_change: bool = False,
) -> Dict[str, Any]:
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"trainer checkpoint not found: {path}")
    cfg = cfg or production_config()
    cfg.validate()
    digest = sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"not a trainer checkpoint payload: {path}")
    if payload.get("format") != TRAINER_CKPT_FORMAT:
        raise ValueError(
            f"unexpected checkpoint format {payload.get('format')!r}; "
            f"expected {TRAINER_CKPT_FORMAT!r}"
        )
    implementation = payload.get("implementation")
    if implementation != TRAINER_IMPLEMENTATION_VERSION and not allow_code_change:
        raise ValueError(
            f"checkpoint implementation {implementation!r} != "
            f"{TRAINER_IMPLEMENTATION_VERSION!r}"
        )
    payload_config = payload.get("config", {})
    if payload_config:
        validate_trainer_config(payload_config, cfg)
    model_sd = payload.get("model")
    if not isinstance(model_sd, dict):
        raise ValueError("checkpoint payload has no model state_dict")
    weights = map_trainer_state_dict(model_sd, cfg, strict=True)
    provenance = {
        "source_checkpoint": str(path),
        "source_checkpoint_sha256": digest,
        "trainer_format": payload.get("format"),
        "trainer_implementation": implementation,
        "trainer_code_sha256": payload.get("code_sha256"),
        "progress": payload.get("progress"),
        "corpus": payload.get("corpus"),
        "saved_at": payload.get("saved_at"),
        "torch": payload.get("torch"),
        "cuda": payload.get("cuda"),
    }
    manifest = save_package(out_dir, weights, cfg, extra_manifest=provenance)
    return {
        "ok": True,
        "checkpoint": str(path),
        "checkpoint_sha256": digest,
        "package_dir": str(out_dir),
        "weights_fingerprint": manifest["weights_fingerprint"],
        "progress": payload.get("progress"),
        "config": payload_config,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--allow-code-change", action="store_true")
    args = parser.parse_args(argv)
    summary = convert_trainer_checkpoint(
        args.checkpoint, args.out_dir, allow_code_change=args.allow_code_change
    )
    print(json.dumps(summary, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
