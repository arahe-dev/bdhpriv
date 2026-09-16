"""Minimal Arm-A inference runtime.

``AkashaModel`` binds weights, configuration and manifest to the token-major
reference recurrence. Prefill/decode semantics are explicit:

    logits = model.prefill_tokens(state, [x0, ..., xn])

returns the logits produced *after* processing ``xn``; the returned state
already includes ``xn``. The next operation must consume a newly sampled
token, never ``xn`` again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch

from akasha.models.arma.config import ArmAConfig
from akasha.models.arma.ops import ArmAWeights
from akasha.models.arma.reference_recurrent import (
    create_state,
    logits_from_state,
    prefill_all_logits,
    prefill_tokens as _prefill_tokens,
    step,
)
from akasha.models.arma.state import AkashaState, ContextPolicy


@dataclass
class AkashaModel:
    weights: ArmAWeights
    cfg: ArmAConfig
    manifest: Dict[str, Any] = field(default_factory=dict)
    model_fingerprint: str = ""

    @classmethod
    def load(cls, package_dir, device=None, dtype=None, validate=True) -> "AkashaModel":
        from akasha.checkpoint.loader import load_package

        loaded = load_package(
            package_dir, device=device, dtype=dtype, validate=validate
        )
        return cls(
            weights=loaded.weights,
            cfg=loaded.cfg,
            manifest=loaded.manifest,
            model_fingerprint=loaded.weights_fingerprint,
        )

    def create_state(
        self,
        context_policy: ContextPolicy | str = ContextPolicy.TRAINING_WINDOW,
        device=None,
        dtype=None,
    ) -> AkashaState:
        return create_state(
            self.weights,
            self.cfg,
            context_policy=context_policy,
            device=device,
            dtype=dtype,
            model_fingerprint=self.model_fingerprint,
        )

    def prefill_tokens(
        self,
        state: AkashaState,
        token_ids: Sequence[int] | torch.Tensor,
        positions: Optional[Sequence[int] | torch.Tensor] = None,
        segment_ids: Optional[Sequence[int] | torch.Tensor] = None,
    ) -> torch.Tensor:
        return _prefill_tokens(
            self.weights,
            self.cfg,
            state,
            token_ids,
            positions=positions,
            segment_ids=segment_ids,
        )

    def decode_one(
        self, state: AkashaState, token_id: int | torch.Tensor
    ) -> torch.Tensor:
        return step(self.weights, self.cfg, state, token_id)

    def logits_for_next_token(self, state: AkashaState) -> torch.Tensor:
        return logits_from_state(self.weights, state)

    def prefill_all_logits(
        self,
        state: AkashaState,
        token_ids: Sequence[int] | torch.Tensor,
        positions: Optional[Sequence[int] | torch.Tensor] = None,
        segment_ids: Optional[Sequence[int] | torch.Tensor] = None,
        return_hidden: bool = False,
    ):
        return prefill_all_logits(
            self.weights,
            self.cfg,
            state,
            token_ids,
            positions=positions,
            segment_ids=segment_ids,
            return_hidden=return_hidden,
        )

    def save_package(self, out_dir, extra_manifest: Optional[Dict[str, Any]] = None):
        from akasha.checkpoint.loader import save_package

        return save_package(
            out_dir, self.weights, self.cfg, extra_manifest=extra_manifest
        )
