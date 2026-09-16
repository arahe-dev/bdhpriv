"""Frozen architecture manifest for Arm-A exact dense inference.

Records the source provenance, tensor shapes, LayerNorm epsilon and RoPE
convention used by the reference runtime. Nothing in this module may be
inferred: every field is transcribed or derived from
``training/arm_a_2p5b_trainer.py`` @ 0dcbb87 (SHA-256 in
:mod:`akasha.config`).

LayerNorm epsilon: ``nn.LayerNorm(cfg.D, elementwise_affine=False,
bias=False)`` is constructed in the trainer without an explicit ``eps``.
``torch.nn.LayerNorm.__init__`` declares ``eps: float = 1e-5`` (verified
against the installed PyTorch source contract). The default is therefore
frozen to ``1e-5`` with no affine parameters.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict

from akasha.config import (
    REPO_SOURCE_COMMIT,
    REPO_SOURCE_COMMIT_FULL,
    TRAINER_IMPLEMENTATION_VERSION,
    TRAINER_PATH,
    TRAINER_SHA256,
)
from akasha.models.arma.config import ArmAConfig

FORMAT = "akasha_arma_manifest_v1"
MODEL_ID = "arm-a"

LAYERNORM_EPS = 1e-5
LAYERNORM_AFFINE = False

ROPE_CONVENTION = {
    "style": "interleaved_pairs",
    "pairing": "adjacent (x[2i], x[2i+1]) -> (q[2i], q[2i+1])",
    "theta": 2**16,
    "freq": "freq_i = THETA ** (-2i/K) / (2*pi), i in [0, K/2)",
    "phase": "phase_i = remainder(pos * freq_i, 1) * (2*pi)",
    "rotate": "q[2i] = x[2i]*cos(phase_i) - x[2i+1]*sin(phase_i); "
              "q[2i+1] = x[2i+1]*cos(phase_i) + x[2i]*sin(phase_i)",
    "cache": "phase cached per position tensor; replayed for all L levels",
}

TENSOR_SPEC: Dict[str, Dict[str, Any]] = {
    "embedding": {"source_name": "embedding.weight", "shape": ["V", "D"],
                  "role": "token embedding"},
    "encoder": {"source_name": "encoder", "shape": ["N", "D"],
                "role": "neuronal collapse N -> D"},
    "decoder_x": {"source_name": "decoder_x", "shape": ["H", "D", "K"],
                  "role": "wide positive projection D -> H*K"},
    "decoder_y": {"source_name": "decoder_y", "shape": ["H", "D", "K"],
                  "role": "second positive projection D -> H*K"},
    "readout": {"source_name": "readout", "shape": ["D", "V"],
                "role": "logits"},
    "coord_Wc": {"source_name": "coordinator.Wc", "shape": ["D", "D"],
                 "role": "coordinator projection"},
    "coord_bc": {"source_name": "coordinator.bc", "shape": ["D"],
                 "role": "coordinator bias"},
    "coord_alpha": {"source_name": "coordinator.alpha", "shape": [],
                    "role": "coordinator gate logit; rho = sigmoid(alpha)"},
    "writer_W1": {"source_name": "writer.W1", "shape": ["D", "HIDDEN"],
                  "role": "writer up-projection, ReLU"},
    "writer_W2": {"source_name": "writer.W2", "shape": ["HIDDEN", "D"],
                  "role": "writer down-projection"},
}

TENSOR_ORDER = tuple(TENSOR_SPEC)


def manifest_dict(cfg: ArmAConfig | None = None) -> Dict[str, Any]:
    cfg = cfg or ArmAConfig()
    cfg.validate()
    tensors = {}
    for name in TENSOR_ORDER:
        spec = dict(TENSOR_SPEC[name])
        spec["shape"] = [_resolve(dim, cfg) for dim in spec["shape"]]
        tensors[name] = spec
    return {
        "format": FORMAT,
        "model": MODEL_ID,
        "source_commit": REPO_SOURCE_COMMIT,
        "source_commit_full": REPO_SOURCE_COMMIT_FULL,
        "trainer_path": TRAINER_PATH,
        "trainer_sha256": TRAINER_SHA256,
        "trainer_implementation_version": TRAINER_IMPLEMENTATION_VERSION,
        "config": cfg.to_dict(),
        "layernorm": {"eps": LAYERNORM_EPS, "affine": LAYERNORM_AFFINE,
                      "normalized_dims": "last"},
        "rope": dict(ROPE_CONVENTION),
        "tensors": tensors,
    }


def _resolve(dim: str, cfg: ArmAConfig) -> int:
    if dim == "K":
        return cfg.K
    return int(getattr(cfg, dim))


def manifest_fingerprint(cfg: ArmAConfig | None = None) -> str:
    import json

    payload = json.dumps(manifest_dict(cfg), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
