"""Akasha recurrent session state.

The mathematical recurrence state is ``S`` (neuronal, ``[L, H, K, D]``) and
``C`` (coordinator, ``[L, D]``) plus integer metadata ``position`` and
``segment_count``. A resumable generation session additionally stores
``last_hidden`` so that the logits of the last processed prompt token can be
recovered without reprocessing it: ``logits = last_hidden @ readout``.

``position`` (RoPE document position) and ``segment_count`` (tokens already
processed in the current memory segment, also the coordinator mean
denominator) are strictly independent. A memory reset zeroes ``S``, ``C`` and
``segment_count`` but never rewrites ``position``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Tuple

import torch

from akasha.models.arma.config import ArmAConfig

STATE_FORMAT = "akasha_arma_state_v1"


class ContextPolicy(str, Enum):
    """How recurrent memory behaves at the trained 2048-token boundary."""

    TRAINING_WINDOW = "TRAINING_WINDOW"
    CONTINUOUS_EXPERIMENTAL = "CONTINUOUS_EXPERIMENTAL"


@dataclass
class AkashaState:
    S: torch.Tensor
    C: torch.Tensor
    position: int
    segment_count: int
    last_hidden: torch.Tensor
    context_policy: ContextPolicy = ContextPolicy.TRAINING_WINDOW
    has_last_hidden: bool = False
    segment_id: Optional[int] = None
    model_fingerprint: str = ""
    config_fingerprint: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "AkashaState":
        return AkashaState(
            S=self.S.clone(),
            C=self.C.clone(),
            position=int(self.position),
            segment_count=int(self.segment_count),
            last_hidden=self.last_hidden.clone(),
            context_policy=ContextPolicy(self.context_policy),
            has_last_hidden=bool(self.has_last_hidden),
            segment_id=self.segment_id,
            model_fingerprint=self.model_fingerprint,
            config_fingerprint=self.config_fingerprint,
            metadata=dict(self.metadata),
        )

    def reset_memory(self) -> None:
        self.S.zero_()
        self.C.zero_()
        self.segment_count = 0

    def begin_segment(self, position: int | None = None) -> None:
        self.reset_memory()
        if position is not None:
            self.position = int(position)

    def device(self) -> torch.device:
        return self.S.device

    def dtype(self) -> torch.dtype:
        return self.S.dtype

    def nbytes(self) -> int:
        return int(self.S.numel() * self.S.element_size()) + int(
            self.C.numel() * self.C.element_size()
        )

    def to_serializable(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        tensors = {
            "S": self.S.detach().to("cpu", torch.float32).contiguous(),
            "C": self.C.detach().to("cpu", torch.float32).contiguous(),
            "last_hidden": self.last_hidden.detach().to("cpu", torch.float32).contiguous(),
        }
        meta = {
            "format": STATE_FORMAT,
            "position": int(self.position),
            "segment_count": int(self.segment_count),
            "context_policy": ContextPolicy(self.context_policy).value,
            "has_last_hidden": bool(self.has_last_hidden),
            "segment_id": self.segment_id,
            "model_fingerprint": self.model_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "metadata": dict(self.metadata),
        }
        return tensors, meta

    @classmethod
    def from_serializable(
        cls,
        tensors: Dict[str, torch.Tensor],
        meta: Dict[str, Any],
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> "AkashaState":
        if meta.get("format") != STATE_FORMAT:
            raise ValueError(f"unexpected state format: {meta.get('format')!r}")
        for required in ("S", "C", "last_hidden"):
            if required not in tensors:
                raise ValueError(f"serialized state missing tensor {required!r}")
        move = {}
        if device is not None:
            move["device"] = device
        if dtype is not None:
            move["dtype"] = dtype
        return cls(
            S=tensors["S"].to(**move),
            C=tensors["C"].to(**move),
            last_hidden=tensors["last_hidden"].to(**move),
            position=int(meta["position"]),
            segment_count=int(meta["segment_count"]),
            context_policy=ContextPolicy(meta["context_policy"]),
            has_last_hidden=bool(meta.get("has_last_hidden", False)),
            segment_id=(
                None if meta.get("segment_id") is None else int(meta["segment_id"])
            ),
            model_fingerprint=str(meta.get("model_fingerprint", "")),
            config_fingerprint=str(meta.get("config_fingerprint", "")),
            metadata=dict(meta.get("metadata", {})),
        )


def state_element_counts(cfg: ArmAConfig) -> Dict[str, int]:
    cfg.validate()
    return {
        "S": cfg.L * cfg.H * cfg.K * cfg.D,
        "C": cfg.L * cfg.D,
        "last_hidden": cfg.D,
        "total_mathematical": cfg.L * cfg.H * cfg.K * cfg.D + cfg.L * cfg.D,
    }


def production_state_bytes(dtype: torch.dtype = torch.float32) -> int:
    cfg = ArmAConfig()
    es = torch.empty((), dtype=dtype).element_size()
    return (cfg.L * cfg.H * cfg.K * cfg.D + cfg.L * cfg.D) * es
