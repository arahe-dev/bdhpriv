"""Exact training tokenizer adapter.

Known identity: ``bytelevel-bpe-8192-9a05bca4c065d995`` with expected
SHA-256 ``9a05bca4c065d9952a01ee1e3f4b6e23e39200e1f33da65e5dad1819419f71c3``.

The serialized artifact is not part of this repository. This module never
fabricates an equivalent tokenizer: it loads an artifact only when one is
supplied, verifies its SHA-256, and reports ``BLOCKED_ARTIFACT_NOT_LOCAL``
otherwise. Token-ID inference does not depend on the tokenizer.
"""

from __future__ import annotations

import hashlib
import os
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from akasha.config import (
    TOKENIZER_ARTIFACT_ENV,
    TOKENIZER_EXPECTED_SHA256,
    TOKENIZER_IDENTITY,
)


class TokenizerStatus(str, Enum):
    READY = "READY"
    BLOCKED_ARTIFACT_NOT_LOCAL = "BLOCKED_ARTIFACT_NOT_LOCAL"
    BLOCKED_DEPENDENCY = "BLOCKED_DEPENDENCY"
    HASH_MISMATCH = "HASH_MISMATCH"


class TokenizerUnavailable(RuntimeError):
    pass


def sha256_file(path: Path, block: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(block)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def discover_local(extra_paths: Iterable[str] = ()) -> Optional[Path]:
    candidates: List[Path] = []
    env_path = os.environ.get(TOKENIZER_ARTIFACT_ENV)
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(Path(p) for p in extra_paths)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


class ArmATokenizerAdapter:
    identity = TOKENIZER_IDENTITY
    expected_sha256 = TOKENIZER_EXPECTED_SHA256
    vocab_size = 8192

    def __init__(self, artifact_path: Optional[str | Path] = None):
        self.artifact_path = Path(artifact_path) if artifact_path else None
        self._impl = None
        self._status = TokenizerStatus.BLOCKED_ARTIFACT_NOT_LOCAL
        self._reason = "no tokenizer artifact supplied or found locally"
        if self.artifact_path is not None:
            self._try_load(self.artifact_path)

    def _try_load(self, path: Path) -> None:
        if not path.is_file():
            self._status = TokenizerStatus.BLOCKED_ARTIFACT_NOT_LOCAL
            self._reason = f"artifact not found: {path}"
            return
        digest = sha256_file(path)
        if self.expected_sha256 and digest != self.expected_sha256:
            self._status = TokenizerStatus.HASH_MISMATCH
            self._reason = (
                f"artifact SHA-256 {digest} does not match expected "
                f"{self.expected_sha256}; refusing to load"
            )
            return
        try:
            from tokenizers import Tokenizer  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self._status = TokenizerStatus.BLOCKED_DEPENDENCY
            self._reason = f"tokenizers package unavailable: {exc}"
            return
        try:
            self._impl = Tokenizer.from_file(str(path))
        except Exception as exc:  # noqa: BLE001
            self._status = TokenizerStatus.BLOCKED_DEPENDENCY
            self._reason = f"tokenizers could not load artifact: {exc}"
            return
        self._status = TokenizerStatus.READY
        self._reason = f"loaded verified artifact {path} (sha256={digest})"

    def status(self) -> TokenizerStatus:
        return self._status

    def status_detail(self) -> dict:
        return {
            "identity": self.identity,
            "expected_sha256": self.expected_sha256,
            "status": self._status.value,
            "reason": self._reason,
            "artifact_path": None if self.artifact_path is None else str(self.artifact_path),
        }

    def _require_ready(self) -> None:
        if self._status != TokenizerStatus.READY:
            raise TokenizerUnavailable(
                f"tokenizer is {self._status.value}: {self._reason}; "
                "supply the real artifact and do not fabricate a substitute"
            )

    def encode(self, text: str) -> List[int]:
        self._require_ready()
        return list(self._impl.encode(text).ids)

    def decode(self, token_ids: Sequence[int]) -> str:
        self._require_ready()
        return self._impl.decode([int(t) for t in token_ids])

    def golden_fixtures(self) -> Tuple[Tuple[str, List[int]], ...]:
        """Placeholder until the real artifact is hashed and loaded."""
        return ()
