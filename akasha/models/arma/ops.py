"""Shared Arm-A operators transcribed from the frozen trainer.

All functions are dtype-generic (``float32`` for the production contract,
``float64`` for the algebra gates) and device-generic. No operation is
approximated: LayerNorm has no affine parameters, RoPE is the trainer's cached
phase construction, and the wide projection reproduces the trainer's
``permute(1, 0, 2).reshape(D, N)`` weight layout exactly.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from akasha.models.arma.config import ArmAConfig

LAYERNORM_EPS = 1e-5


@dataclass(frozen=True)
class ArmAWeights:
    embedding: torch.Tensor
    encoder: torch.Tensor
    decoder_x: torch.Tensor
    decoder_y: torch.Tensor
    readout: torch.Tensor
    coord_Wc: torch.Tensor
    coord_bc: torch.Tensor
    coord_alpha: torch.Tensor
    writer_W1: torch.Tensor
    writer_W2: torch.Tensor

    def tensors(self) -> Dict[str, torch.Tensor]:
        return {
            "embedding": self.embedding,
            "encoder": self.encoder,
            "decoder_x": self.decoder_x,
            "decoder_y": self.decoder_y,
            "readout": self.readout,
            "coord_Wc": self.coord_Wc,
            "coord_bc": self.coord_bc,
            "coord_alpha": self.coord_alpha,
            "writer_W1": self.writer_W1,
            "writer_W2": self.writer_W2,
        }

    def to(self, device=None, dtype=None) -> "ArmAWeights":
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        return ArmAWeights(**{k: v.to(**kwargs) for k, v in self.tensors().items()})

    @property
    def dtype(self) -> torch.dtype:
        return self.embedding.dtype

    @property
    def device(self) -> torch.device:
        return self.embedding.device


def expected_tensor_shapes(cfg: ArmAConfig) -> Dict[str, Tuple[int, ...]]:
    cfg.validate()
    return {
        "embedding": (cfg.V, cfg.D),
        "encoder": (cfg.N, cfg.D),
        "decoder_x": (cfg.H, cfg.D, cfg.K),
        "decoder_y": (cfg.H, cfg.D, cfg.K),
        "readout": (cfg.D, cfg.V),
        "coord_Wc": (cfg.D, cfg.D),
        "coord_bc": (cfg.D,),
        "coord_alpha": (),
        "writer_W1": (cfg.D, cfg.HIDDEN),
        "writer_W2": (cfg.HIDDEN, cfg.D),
    }


def validate_weights(weights: ArmAWeights, cfg: ArmAConfig) -> None:
    expected = expected_tensor_shapes(cfg)
    for name, shape in expected.items():
        tensor = weights.tensors()[name]
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(
                f"tensor {name}: expected shape {tuple(shape)}, "
                f"found {tuple(tensor.shape)}"
            )


def weights_fingerprint(weights: ArmAWeights) -> str:
    h = hashlib.sha256()
    for name, tensor in weights.tensors().items():
        t = tensor.detach().to("cpu", torch.float32).contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def layer_norm(x: torch.Tensor, eps: float = LAYERNORM_EPS) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), eps=eps)


def rope_pair_freq(cfg: ArmAConfig, device) -> torch.Tensor:
    return (
        1.0
        / (cfg.THETA ** ((2.0 * torch.arange(cfg.K // 2, dtype=torch.float32,
                                              device=device)) / cfg.K))
        / (2.0 * math.pi)
    )


_FREQ_CACHE: Dict[Tuple[int, int, str], torch.Tensor] = {}


def cached_pair_freq(cfg: ArmAConfig, device) -> torch.Tensor:
    key = (cfg.K, int(cfg.THETA), str(device))
    freq = _FREQ_CACHE.get(key)
    if freq is None or freq.device != torch.device(device):
        freq = rope_pair_freq(cfg, device)
        _FREQ_CACHE[key] = freq
    return freq


def rope_phase(positions: torch.Tensor, freq: torch.Tensor):
    phase = positions.to(torch.float32).unsqueeze(-1).unsqueeze(-1) * freq.view(
        1, 1, 1, -1
    )
    phase = torch.remainder(phase, 1.0) * (2.0 * math.pi)
    return torch.cos(phase), torch.sin(phase)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate interleaved pairs of the last dimension.

    ``x`` has shape ``[..., H, K]``; ``cos``/``sin`` broadcast to
    ``[..., 1, K // 2]``. Pairs are ``(x[..., 2i], x[..., 2i+1])``.
    """
    *lead, h, k = x.shape
    qp = x.reshape(*lead, h, k // 2, 2)
    even = qp[..., 0]
    odd = qp[..., 1]
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return torch.stack(
        (even * cos - odd * sin, odd * cos + even * sin), dim=-1
    ).reshape(*lead, h, k)


_WIDE_MATRIX_CACHE: Dict[int, Tuple[torch.Tensor, int, torch.Tensor]] = {}


def wide_projection_matrix(weights: ArmAWeights, cfg: ArmAConfig) -> torch.Tensor:
    """Cached ``decoder_x.permute(1, 0, 2).reshape(D, N)`` derived constant."""
    key = id(weights.decoder_x)
    entry = _WIDE_MATRIX_CACHE.get(key)
    if entry is not None and entry[0] is weights.decoder_x and entry[1] == cfg.N:
        return entry[2]
    matrix = weights.decoder_x.permute(1, 0, 2).reshape(cfg.D, cfg.N)
    _WIDE_MATRIX_CACHE[key] = (weights.decoder_x, cfg.N, matrix)
    return matrix


def wide_project_x(
    v: torch.Tensor,
    weights: ArmAWeights,
    cfg: ArmAConfig,
    w_wide: torch.Tensor | None = None,
):
    """``relu(v @ decoder_x.permute(1, 0, 2).reshape(D, N))``.

    Returns ``[..., H, K]`` for any leading shape of ``v`` ending in ``D``.
    """
    if w_wide is None:
        w_wide = wide_projection_matrix(weights, cfg)
    out = torch.matmul(v, w_wide)
    return torch.relu(out).reshape(*v.shape[:-1], cfg.H, cfg.K)
