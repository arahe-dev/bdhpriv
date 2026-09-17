"""Deterministic and stochastic samplers with cloneable RNG state.

The sampler RNG is serialized separately from :class:`AkashaState` so that a
session snapshot can be restored with or without its stochastic branch
reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import torch


class SamplerMethod(str, Enum):
    GREEDY = "greedy"
    MULTINOMIAL = "multinomial"


@dataclass
class Sampler:
    method: SamplerMethod | str = SamplerMethod.GREEDY
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    seed: Optional[int] = None
    generator: Optional[torch.Generator] = None

    def __post_init__(self) -> None:
        self.method = SamplerMethod(self.method)
        if self.method == SamplerMethod.MULTINOMIAL and self.generator is None:
            self.generator = torch.Generator(device="cpu")
            if self.seed is not None:
                self.generator.manual_seed(int(self.seed))

    def sample(self, logits: torch.Tensor) -> int:
        logits = logits.reshape(-1)
        if self.method == SamplerMethod.GREEDY:
            return int(torch.argmax(logits).item())
        if self.temperature <= 0:
            raise ValueError("temperature must be positive for multinomial")
        scaled = logits.to(torch.float32) / float(self.temperature)
        if self.top_k and self.top_k > 0:
            k = min(int(self.top_k), scaled.numel())
            top_values, _ = torch.topk(scaled, k)
            threshold = top_values[..., -1].unsqueeze(-1)
            scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
        if self.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            cutoff = cumulative - probs > float(self.top_p)
            sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
            scaled = torch.full_like(scaled, float("-inf")).scatter(
                0, sorted_idx, sorted_logits
            )
        probs = torch.softmax(scaled, dim=-1)
        return int(
            torch.multinomial(probs, num_samples=1, generator=self.generator).item()
        )

    def clone(self) -> "Sampler":
        twin = Sampler(
            method=self.method,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            seed=self.seed,
        )
        if self.generator is not None:
            state = self.generator.get_state().clone()
            twin.generator = torch.Generator(device=self.generator.device)
            twin.generator.set_state(state)
        return twin

    def get_rng_state(self) -> Optional[torch.Tensor]:
        if self.generator is None:
            return None
        return self.generator.get_state().detach().clone()

    def set_rng_state(self, state: Optional[torch.Tensor]) -> None:
        if state is None:
            return
        if self.generator is None:
            self.generator = torch.Generator(device="cpu")
        self.generator.set_state(state.detach().to("cpu"))

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "top_k": int(self.top_k),
            "seed": None if self.seed is None else int(self.seed),
        }
