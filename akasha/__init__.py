"""AkashaSDK: exact dense Arm-A reference inference runtime.

V0 scope is the dense, token-major recurrent reference established against the
frozen trainer at commit ``0dcbb87``. No optimized kernels, no serving, no
sparse routing. Correctness first.
"""

from __future__ import annotations

__version__ = "0.1.0"

from akasha.config import (  # noqa: F401
    AKASHA_V0_REFERENCE_PASS_ENV,
    REPO_SOURCE_COMMIT,
    REPO_SOURCE_COMMIT_FULL,
    SOURCE_HEAD_AT_IMPLEMENTATION,
    TOKENIZER_EXPECTED_SHA256,
    TOKENIZER_IDENTITY,
    TRAINER_PATH,
    TRAINER_SHA256,
)
from akasha.models.arma.config import (  # noqa: F401
    ArmAConfig,
    production_config,
    tiny_config,
)
from akasha.models.arma.state import AkashaState, ContextPolicy  # noqa: F401


def load_model(path, **kwargs):
    """Load a canonical Arm-A package (``model.safetensors`` + manifest)."""
    from akasha.runtime.model import AkashaModel

    return AkashaModel.load(path, **kwargs)


__all__ = [
    "AkashaState",
    "ArmAConfig",
    "ContextPolicy",
    "REPO_SOURCE_COMMIT",
    "REPO_SOURCE_COMMIT_FULL",
    "TOKENIZER_EXPECTED_SHA256",
    "TOKENIZER_IDENTITY",
    "TRAINER_PATH",
    "TRAINER_SHA256",
    "load_model",
    "production_config",
    "tiny_config",
]
