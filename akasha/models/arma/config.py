"""Arm-A inference configuration.

Frozen production values are transcribed from ``ArmAConfig`` in
``training/arm_a_2p5b_trainer.py`` @ 0dcbb87. ``K = N // H`` exactly as in the
trainer; ``T`` is simultaneously the trained window length and the maximum
source-aligned memory segment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict


@dataclass(frozen=True)
class ArmAConfig:
    T: int = 2048
    V: int = 8192
    D: int = 256
    N: int = 16384
    H: int = 4
    L: int = 8
    HIDDEN: int = 1040
    SEED: int = 1337
    INIT_STD: float = 0.02
    THETA: float = 2**16
    READ_BLOCK: int = 256

    @property
    def K(self) -> int:
        return self.N // self.H

    def validate(self) -> None:
        if self.N % self.H != 0:
            raise ValueError(f"N={self.N} must be divisible by H={self.H}")
        if self.K % 2 != 0:
            raise ValueError(f"K={self.K} must be even for interleaved RoPE")
        if self.T <= 0 or self.V <= 0 or self.D <= 0 or self.L <= 0:
            raise ValueError("T, V, D and L must be positive")
        if self.HIDDEN <= 0:
            raise ValueError("HIDDEN must be positive")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["K"] = self.K
        return data

    def with_overrides(self, **kwargs: Any) -> "ArmAConfig":
        return replace(self, **kwargs)


def production_config() -> ArmAConfig:
    cfg = ArmAConfig()
    cfg.validate()
    return cfg


def tiny_config() -> ArmAConfig:
    cfg = ArmAConfig(T=16, V=32, D=8, N=32, H=2, L=2, HIDDEN=12)
    cfg.validate()
    return cfg
