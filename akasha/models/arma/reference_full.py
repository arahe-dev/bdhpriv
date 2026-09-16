"""Source-faithful full-context Arm-A oracle.

This module is the reference oracle: it executes every token at every level
with the same operators as ``OptArmA`` in the frozen trainer, including the
certified chunkwise scan (``scan_chunkwise_candidate``, transcribed here
including the zero-carry t0==0 skip and cross-chunk segment carry logic) and
the dense coordinator prefix sum. It may be instantiated with tiny
configurable dimensions for CPU gates.

Level-major execution: all tokens at level 0, then all tokens at level 1, ...
This is the schedule the trainer uses and the schedule the token-major
recurrence must be proven equivalent to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from akasha.models.arma.config import ArmAConfig
from akasha.models.arma.ops import (
    LAYERNORM_EPS,
    ArmAWeights,
    apply_rope,
    cached_pair_freq,
    layer_norm,
    rope_phase,
    wide_project_x,
)


@dataclass
class FullReferenceOutput:
    logits: torch.Tensor
    debug: Optional[Dict[int, Dict[str, torch.Tensor]]] = None


def segment_starts_from_ids(segment_ids: torch.Tensor) -> torch.Tensor:
    """Convert arbitrary contiguous segment labels to start indices.

    The certified scan (like the trainer's packed corpus) identifies a segment
    by the token index at which it starts, so ``segment_ids`` labels must be
    translated before chunked execution.
    """
    b, t = segment_ids.shape
    starts = torch.zeros_like(segment_ids)
    for i in range(1, t):
        same = segment_ids[:, i] == segment_ids[:, i - 1]
        starts[:, i] = torch.where(
            same, starts[:, i - 1], torch.full_like(starts[:, i], i)
        )
    return starts


def scan_attention(
    qh: torch.Tensor,
    vh: torch.Tensor,
    segment_start: torch.Tensor,
    block: int,
    zero_carry: bool = True,
) -> torch.Tensor:
    """Transcription of the trainer's certified chunkwise state scan.

    ``qh``/``vh`` are ``[B, H, T, K]`` / ``[B, H, T, Dv]``; ``segment_start``
    is ``[B, T]`` holding the token index where each segment begins. The
    returned tensor follows the trainer's ``[B, H, T, Dv]`` orientation.
    """
    b, h, t, k = qh.shape
    dv = vh.shape[-1]
    dev = qh.device
    state = torch.zeros((b, h, k, dv), dtype=qh.dtype, device=dev)
    outs = []
    for t0 in range(0, t, block):
        t1 = min(t0 + block, t)
        w = t1 - t0
        qb = qh[:, :, t0:t1]
        vb = vh[:, :, t0:t1]
        seg = segment_start[:, t0:t1]
        scores = qb @ qb.transpose(-1, -2)
        causal = torch.ones((w, w), dtype=torch.bool, device=dev).tril(diagonal=-1)
        samedoc = seg[:, :, None] == seg[:, None, :]
        local = scores.masked_fill(~(samedoc.unsqueeze(1) & causal), 0.0) @ vb
        if zero_carry and t0 == 0:
            outs.append(local)
        else:
            cont = (seg < t0).to(qb.dtype).view(b, 1, w, 1)
            carry = torch.einsum("bhwk,bhkd->bhwd", qb, state) * cont
            outs.append(local + carry)
        if t1 < t:
            segb = segment_start[:, t1]
            contb = (segb < t0).to(qb.dtype).view(b, 1, 1, 1)
            j = (segb - t0).clamp_min(0)
            keep = (
                (torch.arange(w, device=dev).unsqueeze(0) >= j.unsqueeze(1))
                .to(qb.dtype)
                .view(b, 1, w, 1)
            )
            fresh = torch.einsum("bhwk,bhwd->bhkd", qb * keep, vb)
            state = state * contb + fresh
    return torch.cat(outs, dim=2)


def _as_batched(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    return tensor


def full_forward(
    weights: ArmAWeights,
    cfg: ArmAConfig,
    token_ids: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
    segment_ids: Optional[torch.Tensor] = None,
    input_valid: Optional[torch.Tensor] = None,
    scan_block: Optional[int] = None,
    collect_debug: bool = False,
    eps: float = LAYERNORM_EPS,
) -> FullReferenceOutput:
    """Run the complete level-major forward pass.

    ``token_ids``/``positions``/``segment_ids`` are ``[B, T]`` (a 1-D input is
    promoted to ``B=1``). ``segment_ids`` must be equal for every pair of
    tokens in the same document segment; ``positions`` are the RoPE document
    positions. ``scan_block=None`` uses a single dense chunk (the oracle
    semantics); ``scan_block=1024`` reproduces the trainer's certified
    chunkwise execution.
    """
    cfg.validate()
    token_ids = _as_batched(token_ids)
    b, t = token_ids.shape
    dev = token_ids.device
    dtype = weights.embedding.dtype
    if positions is None:
        positions = torch.arange(t, device=dev).unsqueeze(0).expand(b, t)
    positions = _as_batched(positions)
    if segment_ids is None:
        segment_ids = torch.zeros((b, t), dtype=torch.long, device=dev)
    segment_ids = _as_batched(segment_ids)
    if input_valid is None:
        input_valid = torch.ones((b, t), dtype=torch.bool, device=dev)
    input_valid = _as_batched(input_valid)
    block = t if scan_block is None else min(int(scan_block), t)
    if block <= 0:
        raise ValueError("scan_block must be positive")

    freq = cached_pair_freq(cfg, dev)
    cos, sin = rope_phase(positions, freq)

    causal = torch.ones((t, t), dtype=torch.bool, device=dev).tril(diagonal=-1)
    coord_mask = (
        (segment_ids[:, :, None] == segment_ids[:, None, :])
        & causal.unsqueeze(0)
        & input_valid[:, :, None]
        & input_valid[:, None, :]
    )
    scan_segments = segment_starts_from_ids(segment_ids)

    v = layer_norm(weights.embedding[token_ids], eps)
    debug: Optional[Dict[int, Dict[str, torch.Tensor]]] = {} if collect_debug else None
    logits = None
    for level in range(cfg.L):
        x = wide_project_x(v, weights, cfg)
        q = apply_rope(x, cos, sin).permute(0, 2, 1, 3)
        vh = v.unsqueeze(1).expand(-1, cfg.H, -1, -1)
        attn_pre = scan_attention(q, vh, scan_segments, block)
        attn_post = layer_norm(attn_pre, eps)
        y = torch.relu(torch.matmul(attn_post, weights.decoder_y))
        u = x * y.permute(0, 2, 1, 3)
        base = layer_norm(u.reshape(b, t, cfg.N) @ weights.encoder, eps)
        z = v @ weights.coord_Wc + weights.coord_bc
        prev_sum = torch.einsum("bts,bsd->btd", coord_mask.to(dtype), z)
        counts = coord_mask.to(dtype).sum(dim=-1).clamp_min(1.0).unsqueeze(-1)
        c = prev_sum / counts - z
        rho = torch.sigmoid(weights.coord_alpha).to(dtype)
        g = 1.0 + rho * torch.tanh(c)
        delta = torch.relu((g * base) @ weights.writer_W1) @ weights.writer_W2
        v_new = layer_norm(v + delta, eps)
        if debug is not None:
            debug[level] = {
                "x": x.detach(),
                "q": q.detach(),
                "attention_pre_ln": attn_pre.detach(),
                "attention_post_ln": attn_post.detach(),
                "y": y.detach(),
                "u": u.detach(),
                "base": base.detach(),
                "z": z.detach(),
                "c": c.detach(),
                "g": g.detach(),
                "delta": delta.detach(),
                "v": v_new.detach(),
            }
        v = v_new
    logits = v @ weights.readout
    return FullReferenceOutput(logits=logits, debug=debug)
