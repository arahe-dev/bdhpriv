"""Generation session: model + recurrent state + sampler.

A session snapshot contains the complete mathematical state (``S``, ``C``,
``position``, ``segment_count``), the resumable ``last_hidden``, the context
policy and the model/config fingerprints. Sampler RNG state is stored in a
separate sidecar and is optional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from akasha.models.arma.state import AkashaState, ContextPolicy
from akasha.runtime.model import AkashaModel
from akasha.sampling.sampler import Sampler


@dataclass
class AkashaSession:
    model: AkashaModel
    state: AkashaState
    sampler: Sampler = field(default_factory=Sampler)
    history: List[int] = field(default_factory=list)
    pending_logits: Optional[torch.Tensor] = None

    @classmethod
    def create(
        cls,
        model: AkashaModel,
        context_policy: ContextPolicy | str = ContextPolicy.TRAINING_WINDOW,
        sampler: Optional[Sampler] = None,
    ) -> "AkashaSession":
        return cls(
            model=model,
            state=model.create_state(context_policy=context_policy),
            sampler=sampler or Sampler(),
        )

    def prefill(
        self,
        token_ids: Sequence[int] | torch.Tensor,
        positions: Optional[Sequence[int]] = None,
        segment_ids: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        ids = [int(x) for x in torch.as_tensor(token_ids).reshape(-1).tolist()]
        logits = self.model.prefill_tokens(
            self.state, ids, positions=positions, segment_ids=segment_ids
        )
        self.history.extend(ids)
        self.pending_logits = logits
        return logits

    def sample(self, logits: Optional[torch.Tensor] = None) -> int:
        logits = self.pending_logits if logits is None else logits
        if logits is None:
            raise ValueError("no logits available to sample from")
        return self.sampler.sample(logits)

    def decode(self, token_id: Optional[int] = None) -> int:
        if token_id is None:
            token_id = self.sample()
        logits = self.model.decode_one(self.state, int(token_id))
        self.history.append(int(token_id))
        self.pending_logits = logits
        return token_id

    def generate(self, max_new_tokens: int) -> List[int]:
        produced = []
        for _ in range(int(max_new_tokens)):
            produced.append(self.decode())
        return produced

    def clone(self) -> "AkashaSession":
        return AkashaSession(
            model=self.model,
            state=self.state.clone(),
            sampler=self.sampler.clone(),
            history=list(self.history),
            pending_logits=None
            if self.pending_logits is None
            else self.pending_logits.detach().clone(),
        )

    def save(self, path, include_rng: bool = True) -> Dict[str, Any]:
        from akasha.checkpoint.loader import save_state

        return save_state(
            path, self.state, sampler=self.sampler, include_rng=include_rng
        )

    @classmethod
    def load(
        cls,
        model: AkashaModel,
        path,
        restore_rng: bool = True,
        validate_fingerprint: bool = True,
    ) -> "AkashaSession":
        from akasha.checkpoint.loader import load_state

        state, sampler = load_state(
            path,
            sampler=Sampler(),
            restore_rng=restore_rng,
            validate_fingerprint=validate_fingerprint,
            expected_fingerprint=model.model_fingerprint,
        )
        return cls(model=model, state=state, sampler=sampler)
