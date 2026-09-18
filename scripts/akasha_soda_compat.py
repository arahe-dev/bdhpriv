"""Explicitly convert a SODA-BDH checkpoint for Akasha inference.

The SODA trainer and the canonical Akasha trainer use different checkpoint
formats and provenance contracts. This adapter accepts a SODA checkpoint only
when its implementation prefix, frozen dense Arm-A architecture, and complete
tensor set match the reference runtime. It writes a normal Akasha package with
provenance that says the package is compatibility-only.

The adapter does not certify the SODA trainer as the canonical Akasha trainer.
It also runs only on CPU RAM. Use the canonical converter for canonical
``arm_a_2p5b_ckpt_v1`` checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

# This must happen before importing torch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from akasha.checkpoint.loader import map_trainer_state_dict, save_package
from akasha.models.arma.config import ArmAConfig, production_config


if torch.version.cuda is not None:
    raise RuntimeError(
        "akasha_soda_compat requires a CPU-only torch build; CUDA is forbidden"
    )


SODA_FORMAT = "soda_bdh_ckpt_v1"
SODA_IMPLEMENTATION_PREFIX = "soda_bdh_trainer_v1_dense_arm_a_opt3c"
ARCHITECTURE_KEYS = (
    "T",
    "V",
    "D",
    "N",
    "H",
    "L",
    "HIDDEN",
    "SEED",
    "INIT_STD",
    "THETA",
    "READ_BLOCK",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _architecture_config(payload: dict[str, Any]) -> ArmAConfig:
    source = payload.get("config")
    if not isinstance(source, dict):
        raise ValueError("SODA checkpoint has no config dictionary")

    expected = production_config()
    mismatches = {}
    for key in ARCHITECTURE_KEYS:
        if key not in source:
            raise ValueError(f"SODA checkpoint config is missing {key}")
        actual = source[key]
        reference = getattr(expected, key)
        if key in {"THETA", "INIT_STD"}:
            actual, reference = float(actual), float(reference)
        elif key == "READ_BLOCK":
            actual, reference = int(actual), int(reference)
        if actual != reference:
            mismatches[key] = {"expected": reference, "found": actual}

    if mismatches:
        raise ValueError(
            "SODA checkpoint is not the frozen dense Arm-A architecture: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return expected


def convert(checkpoint: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Convert one validated SODA checkpoint into an Akasha package."""
    source_path = Path(checkpoint)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a dictionary")
    if payload.get("format") != SODA_FORMAT:
        raise ValueError(
            f"expected {SODA_FORMAT!r}; found {payload.get('format')!r}"
        )

    implementation = str(payload.get("implementation", ""))
    if not implementation.startswith(SODA_IMPLEMENTATION_PREFIX):
        raise ValueError(f"unexpected SODA implementation: {implementation!r}")

    cfg = _architecture_config(payload)
    model_sd = payload.get("model")
    if not isinstance(model_sd, dict):
        raise ValueError("SODA checkpoint has no model state_dict")

    # Strict mapping is the compatibility boundary. Optimizer/runtime tensors,
    # missing weights, extra model tensors, and shape drift all fail here.
    weights = map_trainer_state_dict(model_sd, cfg, strict=True)
    source_sha = sha256_file(source_path)
    provenance = {
        "adapter": "scripts.akasha_soda_compat",
        "adapter_status": "EXPLICIT_COMPATIBILITY_ONLY",
        "canonical_akasha_trainer_status": "NOT_CANONICAL_AKASHA_TRAINER",
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_sha,
        "soda_format": payload.get("format"),
        "soda_implementation": implementation,
        "soda_code_sha256": payload.get("code_sha256"),
        "progress": payload.get("progress"),
        "corpus": payload.get("corpus"),
        "saved_at": payload.get("saved_at"),
        "cpu_only": True,
    }
    manifest = save_package(out_dir, weights, cfg, extra_manifest=provenance)
    return {
        "ok": True,
        "adapter_status": provenance["adapter_status"],
        "canonical_akasha_trainer_status": provenance[
            "canonical_akasha_trainer_status"
        ],
        "checkpoint": str(source_path),
        "checkpoint_sha256": source_sha,
        "package_dir": str(out_dir),
        "weights_fingerprint": manifest["weights_fingerprint"],
        "progress": payload.get("progress"),
        "device": "cpu",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(convert(args.checkpoint, args.out_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
