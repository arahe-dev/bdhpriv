"""Canonical Arm-A checkpoint package and session-state serialization.

Canonical package layout (no custom tensor format):

    model.safetensors   ten canonical tensors, FP32
    manifest.json       architecture manifest + weights fingerprint

The loader validates tensor names, shapes, dtype, manifest format and the
weights fingerprint. Session snapshots serialize the complete state
(``S``, ``C``, ``position``, ``segment_count``, ``last_hidden``,
``context_policy``, fingerprint) with the sampler RNG in a separate sidecar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from akasha.config import (
    REPO_SOURCE_COMMIT,
    REPO_SOURCE_COMMIT_FULL,
    TRAINER_PATH,
    TRAINER_SHA256,
)
from akasha.models.arma.config import ArmAConfig
from akasha.models.arma.manifest import FORMAT as MANIFEST_FORMAT
from akasha.models.arma.manifest import manifest_dict
from akasha.models.arma.ops import (
    ArmAWeights,
    expected_tensor_shapes,
    validate_weights,
    weights_fingerprint,
)
from akasha.models.arma.state import STATE_FORMAT, AkashaState
from akasha.sampling.sampler import Sampler

PACKAGE_FORMAT = "akasha_arma_package_v1"
RNG_FORMAT = "akasha_sampler_rng_v1"

CANONICAL_NAMES = (
    "embedding",
    "encoder",
    "decoder_x",
    "decoder_y",
    "readout",
    "coord_Wc",
    "coord_bc",
    "coord_alpha",
    "writer_W1",
    "writer_W2",
)

TRAINER_TO_CANONICAL = {
    "embedding.weight": "embedding",
    "encoder": "encoder",
    "decoder_x": "decoder_x",
    "decoder_y": "decoder_y",
    "readout": "readout",
    "coordinator.Wc": "coord_Wc",
    "coordinator.bc": "coord_bc",
    "coordinator.alpha": "coord_alpha",
    "writer.W1": "writer_W1",
    "writer.W2": "writer_W2",
}
CANONICAL_TO_TRAINER = {v: k for k, v in TRAINER_TO_CANONICAL.items()}


@dataclass
class LoadedPackage:
    weights: ArmAWeights
    cfg: ArmAConfig
    manifest: Dict[str, Any]
    weights_fingerprint: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def map_trainer_state_dict(
    state_dict: Dict[str, torch.Tensor], cfg: ArmAConfig, strict: bool = True
) -> ArmAWeights:
    """Map a frozen-trainer ``state_dict`` onto canonical Arm-A tensors."""
    expected = expected_tensor_shapes(cfg)
    mapped: Dict[str, torch.Tensor] = {}
    unmatched = set(state_dict)
    for key, tensor in state_dict.items():
        canonical = TRAINER_TO_CANONICAL.get(key)
        if canonical is None:
            continue
        if canonical in mapped:
            raise ValueError(f"duplicate source key for tensor {canonical!r}")
        mapped[canonical] = tensor
        unmatched.discard(key)
    if strict and unmatched:
        raise ValueError(f"unexpected trainer state_dict keys: {sorted(unmatched)}")
    missing = [name for name in expected if name not in mapped]
    if missing:
        raise ValueError(f"missing canonical tensors: {missing}")
    for name, shape in expected.items():
        tensor = mapped[name]
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(
                f"tensor {name}: expected {tuple(shape)}, found {tuple(tensor.shape)}"
            )
    return ArmAWeights(**{name: mapped[name] for name in CANONICAL_NAMES})


def canonical_state_dict(weights: ArmAWeights, cfg: ArmAConfig) -> Dict[str, torch.Tensor]:
    validate_weights(weights, cfg)
    return {
        name: tensor.detach().to("cpu", torch.float32).contiguous()
        for name, tensor in weights.tensors().items()
    }


def save_package(
    out_dir: str | Path,
    weights: ArmAWeights,
    cfg: ArmAConfig,
    extra_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensors = canonical_state_dict(weights, cfg)
    fingerprint = weights_fingerprint(weights)
    manifest = manifest_dict(cfg)
    manifest["format"] = PACKAGE_FORMAT
    manifest["architecture_manifest_format"] = MANIFEST_FORMAT
    manifest["weights_fingerprint"] = fingerprint
    manifest["created_at"] = _utc_now()
    manifest["source"] = {
        "repo_commit": REPO_SOURCE_COMMIT,
        "repo_commit_full": REPO_SOURCE_COMMIT_FULL,
        "trainer_path": TRAINER_PATH,
        "trainer_sha256": TRAINER_SHA256,
    }
    if extra_manifest:
        manifest["provenance"] = dict(extra_manifest)
    save_file(
        tensors,
        str(out / "model.safetensors"),
        metadata={"format": PACKAGE_FORMAT, "weights_fingerprint": fingerprint},
    )
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def load_package(
    package_dir: str | Path,
    device=None,
    dtype=None,
    validate: bool = True,
) -> LoadedPackage:
    from safetensors.torch import load_file

    root = Path(package_dir)
    manifest_path = root / "manifest.json"
    weights_path = root / "model.safetensors"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"missing weights: {weights_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if validate and manifest.get("format") != PACKAGE_FORMAT:
        raise ValueError(
            f"unexpected package format {manifest.get('format')!r}; "
            f"expected {PACKAGE_FORMAT!r}"
        )
    cfg = ArmAConfig(**{k: v for k, v in manifest.get("config", {}).items() if k != "K"})
    cfg.validate()
    tensors = load_file(str(weights_path))
    expected = expected_tensor_shapes(cfg)
    if validate:
        missing = [n for n in CANONICAL_NAMES if n not in tensors]
        extra = [n for n in tensors if n not in CANONICAL_NAMES]
        if missing:
            raise ValueError(f"missing tensors in package: {missing}")
        if extra:
            raise ValueError(f"unexpected tensors in package: {extra}")
        for name, shape in expected.items():
            if tuple(tensors[name].shape) != tuple(shape):
                raise ValueError(
                    f"tensor {name}: expected {tuple(shape)}, "
                    f"found {tuple(tensors[name].shape)}"
                )
            if tensors[name].dtype != torch.float32:
                raise ValueError(
                    f"tensor {name}: expected float32, found {tensors[name].dtype}"
                )
    weights = ArmAWeights(**{name: tensors[name] for name in CANONICAL_NAMES})
    fingerprint = weights_fingerprint(weights)
    if validate and manifest.get("weights_fingerprint") != fingerprint:
        raise ValueError(
            "weights fingerprint mismatch: package manifest says "
            f"{manifest.get('weights_fingerprint')!r}, computed {fingerprint!r}"
        )
    if device is not None or dtype is not None:
        weights = weights.to(device=device, dtype=dtype)
    return LoadedPackage(
        weights=weights, cfg=cfg, manifest=manifest, weights_fingerprint=fingerprint
    )


def _state_path(path: str | Path) -> Path:
    p = Path(path)
    if p.suffix == ".safetensors":
        return p
    return p.with_suffix(".safetensors")


def save_state(
    path: str | Path,
    state: AkashaState,
    sampler: Optional[Sampler] = None,
    include_rng: bool = True,
) -> Dict[str, Any]:
    from safetensors.torch import save_file

    out = _state_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tensors, meta = state.to_serializable()
    save_file(tensors, str(out), metadata={"state_json": json.dumps(meta)})
    summary: Dict[str, Any] = {"state_path": str(out), "meta": meta}
    if include_rng and sampler is not None:
        rng = sampler.get_rng_state()
        rng_path = out.with_suffix(".rng.safetensors")
        rng_tensors = {}
        if rng is not None:
            rng_tensors["rng_state"] = rng.to(torch.uint8).contiguous()
        rng_meta = {
            "format": RNG_FORMAT,
            "sampler": json.dumps(sampler.to_metadata()),
            "has_rng": str(rng is not None),
        }
        save_file(rng_tensors, str(rng_path), metadata=rng_meta)
        summary["rng_path"] = str(rng_path)
    return summary


def load_state(
    path: str | Path,
    sampler: Optional[Sampler] = None,
    restore_rng: bool = True,
    validate_fingerprint: bool = True,
    expected_fingerprint: str = "",
) -> tuple[AkashaState, Optional[Sampler]]:
    from safetensors.torch import load_file
    from safetensors import safe_open

    out = _state_path(path)
    if not out.is_file():
        raise FileNotFoundError(f"missing state snapshot: {out}")
    tensors = load_file(str(out))
    with safe_open(str(out), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    meta = json.loads(metadata.get("state_json", "{}"))
    if meta.get("format") != STATE_FORMAT:
        raise ValueError(f"unexpected state format: {meta.get('format')!r}")
    if (
        validate_fingerprint
        and expected_fingerprint
        and meta.get("model_fingerprint")
        and meta["model_fingerprint"] != expected_fingerprint
    ):
        raise ValueError(
            "state model fingerprint mismatch: "
            f"{meta.get('model_fingerprint')!r} != {expected_fingerprint!r}"
        )
    state = AkashaState.from_serializable(tensors, meta)

    session_sampler = sampler
    if session_sampler is None:
        session_sampler = Sampler()
    rng_path = out.with_suffix(".rng.safetensors")
    if restore_rng and rng_path.is_file():
        rng_tensors = load_file(str(rng_path))
        with safe_open(str(rng_path), framework="pt", device="cpu") as handle:
            rng_meta = handle.metadata() or {}
        sampler_meta = json.loads(rng_meta.get("sampler", "{}"))
        if sampler_meta.get("method"):
            session_sampler.method = session_sampler.method.__class__(
                sampler_meta["method"]
            )
        if "temperature" in sampler_meta:
            session_sampler.temperature = float(sampler_meta["temperature"])
        if "top_p" in sampler_meta:
            session_sampler.top_p = float(sampler_meta["top_p"])
        if "seed" in sampler_meta and sampler_meta["seed"] is not None:
            session_sampler.seed = int(sampler_meta["seed"])
        if "rng_state" in rng_tensors:
            session_sampler.set_rng_state(rng_tensors["rng_state"])
    return state, session_sampler
