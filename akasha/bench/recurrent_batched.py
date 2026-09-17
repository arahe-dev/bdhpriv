"""Batched bench-side Arm-A recurrence, verified against the V0 oracle.

This is a benchmark artifact, not a replacement for the verified reference
recurrence. For identical sessions it reproduces the reference recurrence to
machine precision (batched BLAS reduction order can differ in the last ulp;
float64 observed <= ~1e-16, float32 <= ~6e-8). The equivalence is asserted by
``tests/akasha/test_batched_recurrent.py`` and by the profiling harness before
any timing is recorded.

The batched engine exists because aggregate multi-session throughput cannot be
measured honestly by serializing Python per session.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch

from akasha.models.arma.config import ArmAConfig
from akasha.models.arma.ops import (
    LAYERNORM_EPS,
    ArmAWeights,
    apply_rope,
    cached_pair_freq,
    layer_norm,
    rope_phase,
    wide_projection_matrix,
)
from akasha.models.arma.state import ContextPolicy


@dataclass
class BatchedArmAState:
    S: torch.Tensor
    C: torch.Tensor
    last_hidden: torch.Tensor
    position: torch.Tensor
    segment_count: torch.Tensor
    context_policy: ContextPolicy = ContextPolicy.TRAINING_WINDOW

    @property
    def batch(self) -> int:
        return int(self.S.shape[0])


def create_batched_state(
    weights: ArmAWeights,
    cfg: ArmAConfig,
    batch: int,
    context_policy: ContextPolicy | str = ContextPolicy.TRAINING_WINDOW,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> BatchedArmAState:
    dev = device if device is not None else weights.device
    return BatchedArmAState(
        S=torch.zeros((batch, cfg.L, cfg.H, cfg.K, cfg.D), dtype=dtype, device=dev),
        C=torch.zeros((batch, cfg.L, cfg.D), dtype=dtype, device=dev),
        last_hidden=torch.zeros((batch, cfg.D), dtype=dtype, device=dev),
        position=torch.zeros((batch,), dtype=torch.long, device=dev),
        segment_count=torch.zeros((batch,), dtype=torch.long, device=dev),
        context_policy=ContextPolicy(context_policy),
    )


def _apply_window_reset(cfg: ArmAConfig, state: BatchedArmAState) -> torch.Tensor:
    counts = state.segment_count
    if state.context_policy != ContextPolicy.TRAINING_WINDOW:
        return counts
    reset = counts >= cfg.T
    if bool(reset.any()):
        state.S[reset] = 0.0
        state.C[reset] = 0.0
        counts = torch.where(reset, torch.zeros_like(counts), counts)
        state.segment_count = counts
    return counts


def _profile_range(name: str, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.profiler.record_function(name)


def batched_step(
    weights: ArmAWeights,
    cfg: ArmAConfig,
    state: BatchedArmAState,
    token_ids: torch.Tensor,
    eps: float = LAYERNORM_EPS,
    ranges: bool = False,
) -> torch.Tensor:
    """One decode step for ``B`` synchronized sessions. Mutates ``state``.

    ``ranges=True`` adds ``torch.profiler`` ranges for component attribution;
    it does not change the mathematics (used by the V0.5 profile harness).
    """
    cfg.validate()
    token_ids = torch.as_tensor(
        token_ids, dtype=torch.long, device=state.S.device
    ).reshape(-1)
    batch = int(token_ids.shape[0])
    dtype = state.S.dtype
    dev = state.S.device
    counts = _apply_window_reset(cfg, state)

    w_wide = wide_projection_matrix(weights, cfg)
    freq = cached_pair_freq(cfg, dev)
    cos, sin = rope_phase(state.position.unsqueeze(1), freq)
    cos = cos.reshape(batch, 1, -1)
    sin = sin.reshape(batch, 1, -1)

    with _profile_range("phase:embed_ln", ranges):
        v = layer_norm(weights.embedding[token_ids], eps)
    den = counts.clamp_min(1).to(dtype).unsqueeze(-1)
    rho = torch.sigmoid(weights.coord_alpha).to(dtype)
    S = state.S
    C = state.C

    for level in range(cfg.L):
        with _profile_range("phase:wide_x", ranges):
            x = torch.relu(torch.matmul(v, w_wide)).reshape(batch, cfg.H, cfg.K)
        with _profile_range("phase:rope", ranges):
            q = apply_rope(x, cos, sin)
        with _profile_range("phase:state_read", ranges):
            a = torch.matmul(q.unsqueeze(-2), S[:, level]).squeeze(-2)
        with _profile_range("phase:state_write", ranges):
            for head in range(cfg.H):
                S[:, level, head].add_(q[:, head].unsqueeze(-1) * v.unsqueeze(1))
        with _profile_range("phase:attn_ln_y", ranges):
            a = layer_norm(a, eps)
            y = torch.relu(torch.einsum("bhd,hdk->bhk", a, weights.decoder_y))
        with _profile_range("phase:collapse", ranges):
            u = x * y
            base = layer_norm(u.reshape(batch, cfg.N) @ weights.encoder, eps)
        with _profile_range("phase:coordinator", ranges):
            z = v @ weights.coord_Wc + weights.coord_bc
            c = C[:, level] / den - z
            g = 1.0 + rho * torch.tanh(c)
            C[:, level].add_(z)
        with _profile_range("phase:writer", ranges):
            delta = torch.relu((g * base) @ weights.writer_W1) @ weights.writer_W2
        with _profile_range("phase:ln_residual", ranges):
            v = layer_norm(v + delta, eps)

    state.last_hidden = v.detach().clone()
    state.position = state.position + 1
    state.segment_count = counts + 1
    with _profile_range("phase:readout", ranges):
        return v @ weights.readout


class BatchedDecodeModule(torch.nn.Module):
    """Compiler-friendly wrapper: weights and state are module buffers.

    The TRAINING_WINDOW boundary reset is intentionally handled outside the
    compiled graph (it is a data-dependent control decision); decode benchmarks
    run below the trained window. The verified reference path owns the reset
    semantics.
    """

    def __init__(
        self,
        weights: ArmAWeights,
        cfg: ArmAConfig,
        batch: int,
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.cfg = cfg
        for name, tensor in weights.tensors().items():
            self.register_buffer(
                name, tensor.detach().to(device=device, dtype=dtype).clone()
            )
        state = create_batched_state(
            weights, cfg, batch, device=device, dtype=dtype
        )
        self.register_buffer("S", state.S)
        self.register_buffer("C", state.C)
        self.register_buffer("last_hidden", state.last_hidden)
        self.register_buffer("position", state.position)
        self.register_buffer("segment_count", state.segment_count)

    @property
    def arm_weights(self) -> ArmAWeights:
        return ArmAWeights(
            embedding=self.embedding,
            encoder=self.encoder,
            decoder_x=self.decoder_x,
            decoder_y=self.decoder_y,
            readout=self.readout,
            coord_Wc=self.coord_Wc,
            coord_bc=self.coord_bc,
            coord_alpha=self.coord_alpha,
            writer_W1=self.writer_W1,
            writer_W2=self.writer_W2,
        )

    def reset(self) -> None:
        self.S.zero_()
        self.C.zero_()
        self.last_hidden.zero_()
        self.position.zero_()
        self.segment_count.zero_()

    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        batch = int(token_ids.shape[0])
        dtype = self.S.dtype
        w_wide = self.decoder_x.permute(1, 0, 2).reshape(cfg.D, cfg.N)
        freq = cached_pair_freq(cfg, self.S.device)
        cos, sin = rope_phase(self.position.unsqueeze(1), freq)
        cos = cos.reshape(batch, 1, -1)
        sin = sin.reshape(batch, 1, -1)
        v = layer_norm(self.embedding[token_ids])
        den = self.segment_count.clamp_min(1).to(dtype).unsqueeze(-1)
        rho = torch.sigmoid(self.coord_alpha).to(dtype)
        S = self.S
        C = self.C
        for level in range(cfg.L):
            x = torch.relu(torch.matmul(v, w_wide)).reshape(batch, cfg.H, cfg.K)
            q = apply_rope(x, cos, sin)
            a = torch.matmul(q.unsqueeze(-2), S[:, level]).squeeze(-2)
            for head in range(cfg.H):
                S[:, level, head].add_(q[:, head].unsqueeze(-1) * v.unsqueeze(1))
            a = layer_norm(a)
            y = torch.relu(torch.einsum("bhd,hdk->bhk", a, self.decoder_y))
            u = x * y
            base = layer_norm(u.reshape(batch, cfg.N) @ self.encoder)
            z = v @ self.coord_Wc + self.coord_bc
            c = C[:, level] / den - z
            g = 1.0 + rho * torch.tanh(c)
            C[:, level].add_(z)
            delta = torch.relu((g * base) @ self.writer_W1) @ self.writer_W2
            v = layer_norm(v + delta)
        self.last_hidden = v.detach().clone()
        self.position.add_(1)
        self.segment_count.add_(1)
        return v @ self.readout


class CompiledDecodeModule(torch.nn.Module):
    """Compiler-friendly decode module with selectable state layout.

    The V0.5 profile measured that compiling strided slice updates on a single
    stacked state tensor makes inductor emit full-buffer ``select`` kernels
    (measured ~53 ms/step at B=1). Keeping one contiguous ``[B, K, D]`` tensor
    per (level, head) lets inductor generate plain pointwise update kernels
    (~3.9 ms/step). All layouts implement identical mathematics and are
    verified against the V0 oracle.
    """

    def __init__(
        self,
        weights: ArmAWeights,
        cfg: ArmAConfig,
        batch: int,
        layout: str = "buffers",
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if layout not in ("buffers", "grouped", "stacked"):
            raise ValueError(f"unknown layout {layout!r}")
        self.cfg = cfg
        self.layout = layout
        for name, tensor in weights.tensors().items():
            self.register_buffer(
                name, tensor.detach().to(device=device, dtype=dtype).clone()
            )
        if layout == "buffers":
            self._buffers_state = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.zeros(batch, cfg.K, cfg.D, dtype=dtype, device=device),
                        requires_grad=False,
                    )
                    for _ in range(cfg.L * cfg.H)
                ]
            )
        elif layout == "grouped":
            self.register_buffer(
                "_grouped_state",
                torch.zeros(batch, cfg.L * cfg.H, cfg.K, cfg.D, dtype=dtype,
                            device=device),
            )
        else:
            self.register_buffer(
                "_stacked_state",
                torch.zeros(batch, cfg.L, cfg.H, cfg.K, cfg.D, dtype=dtype,
                            device=device),
            )
        self.register_buffer(
            "C", torch.zeros(batch, cfg.L, cfg.D, dtype=dtype, device=device)
        )
        self.register_buffer(
            "position", torch.zeros(batch, dtype=torch.long, device=device)
        )
        self.register_buffer(
            "segment_count", torch.zeros(batch, dtype=torch.long, device=device)
        )

    @property
    def arm_weights(self) -> ArmAWeights:
        return ArmAWeights(
            embedding=self.embedding,
            encoder=self.encoder,
            decoder_x=self.decoder_x,
            decoder_y=self.decoder_y,
            readout=self.readout,
            coord_Wc=self.coord_Wc,
            coord_bc=self.coord_bc,
            coord_alpha=self.coord_alpha,
            writer_W1=self.writer_W1,
            writer_W2=self.writer_W2,
        )

    @property
    def S(self) -> torch.Tensor:
        if self.layout == "buffers":
            batch = self.C.shape[0]
            stacked = torch.stack([tensor for tensor in self._buffers_state], dim=1)
            return stacked.reshape(batch, self.cfg.L, self.cfg.H, self.cfg.K, self.cfg.D)
        if self.layout == "grouped":
            batch = self.C.shape[0]
            return self._grouped_state.reshape(
                batch, self.cfg.L, self.cfg.H, self.cfg.K, self.cfg.D
            )
        return self._stacked_state

    def reset(self) -> None:
        if self.layout == "buffers":
            for tensor in self._buffers_state:
                tensor.zero_()
        elif self.layout == "grouped":
            self._grouped_state.zero_()
        else:
            self._stacked_state.zero_()
        self.C.zero_()
        self.position.zero_()
        self.segment_count.zero_()

    def _level_state(self, level: int):
        if self.layout == "buffers":
            return [self._buffers_state[level * self.cfg.H + head]
                    for head in range(self.cfg.H)]
        if self.layout == "grouped":
            return self._grouped_state[
                :, level * self.cfg.H:(level + 1) * self.cfg.H
            ]
        return self._stacked_state[:, level]

    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        batch = int(token_ids.shape[0])
        dtype = self.C.dtype
        w_wide = self.decoder_x.permute(1, 0, 2).reshape(cfg.D, cfg.N)
        freq = cached_pair_freq(cfg, self.C.device)
        cos, sin = rope_phase(self.position.unsqueeze(1), freq)
        cos = cos.reshape(batch, 1, -1)
        sin = sin.reshape(batch, 1, -1)
        v = layer_norm(self.embedding[token_ids])
        den = self.segment_count.clamp_min(1).to(dtype).unsqueeze(-1)
        rho = torch.sigmoid(self.coord_alpha).to(dtype)
        C = self.C
        for level in range(cfg.L):
            level_state = self._level_state(level)
            x = torch.relu(torch.matmul(v, w_wide)).reshape(batch, cfg.H, cfg.K)
            q = apply_rope(x, cos, sin)
            if self.layout == "buffers":
                reads = [
                    torch.matmul(q[:, head].unsqueeze(-2), level_state[head]).squeeze(-2)
                    for head in range(cfg.H)
                ]
            else:
                reads = [
                    torch.matmul(
                        q[:, head].unsqueeze(-2), level_state[:, head]
                    ).squeeze(-2)
                    for head in range(cfg.H)
                ]
            a = torch.stack(reads, dim=1)
            if self.layout == "stacked":
                for head in range(cfg.H):
                    level_state[:, head].add_(
                        q[:, head].unsqueeze(-1) * v.unsqueeze(1)
                    )
            elif self.layout == "grouped":
                for head in range(cfg.H):
                    level_state[:, head].add_(
                        q[:, head].unsqueeze(-1) * v.unsqueeze(1)
                    )
            else:
                for head in range(cfg.H):
                    level_state[head].add_(
                        q[:, head].unsqueeze(-1) * v.unsqueeze(1)
                    )
            a = layer_norm(a)
            y = torch.relu(torch.einsum("bhd,hdk->bhk", a, self.decoder_y))
            u = x * y
            base = layer_norm(u.reshape(batch, cfg.N) @ self.encoder)
            z = v @ self.coord_Wc + self.coord_bc
            c = C[:, level] / den - z
            g = 1.0 + rho * torch.tanh(c)
            C[:, level].add_(z)
            delta = torch.relu((g * base) @ self.writer_W1) @ self.writer_W2
            v = layer_norm(v + delta)
        self.position.add_(1)
        self.segment_count.add_(1)
        return v @ self.readout
