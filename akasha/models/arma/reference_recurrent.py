"""Token-major Arm-A recurrence.

For each token, process levels ``0..L-1`` sequentially. Each level reads its
own strict-past neuronal state *before* writing the current token's outer
product, and reads/writes the coordinator accumulator with the segment-count
mean. The schedule is token-major: ``token 0 levels 0..L-1``, then
``token 1 levels 0..L-1``, ...

This module is deliberately readable: no ``torch.compile``, no Triton, no
custom CUDA. The state it maintains is the complete mathematical recurrence
state, and ``last_hidden`` completes the resumable session state.
"""

from __future__ import annotations

from typing import Optional, Sequence

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
    wide_projection_matrix,
)
from akasha.models.arma.state import AkashaState, ContextPolicy


def create_state(
    weights: ArmAWeights,
    cfg: ArmAConfig,
    context_policy: ContextPolicy | str = ContextPolicy.TRAINING_WINDOW,
    device=None,
    dtype: Optional[torch.dtype] = None,
    model_fingerprint: str = "",
    config_fingerprint: str = "",
) -> AkashaState:
    cfg.validate()
    dev = device if device is not None else weights.device
    dt = dtype if dtype is not None else weights.dtype
    return AkashaState(
        S=torch.zeros((cfg.L, cfg.H, cfg.K, cfg.D), dtype=dt, device=dev),
        C=torch.zeros((cfg.L, cfg.D), dtype=dt, device=dev),
        position=0,
        segment_count=0,
        last_hidden=torch.zeros((cfg.D,), dtype=dt, device=dev),
        context_policy=ContextPolicy(context_policy),
        has_last_hidden=False,
        model_fingerprint=model_fingerprint,
        config_fingerprint=config_fingerprint,
    )


def _window_reset_if_needed(cfg: ArmAConfig, state: AkashaState) -> None:
    if (
        state.context_policy == ContextPolicy.TRAINING_WINDOW
        and state.segment_count >= cfg.T
    ):
        state.reset_memory()


def step(
    weights: ArmAWeights,
    cfg: ArmAConfig,
    state: AkashaState,
    token_id: int | torch.Tensor,
    position: Optional[int] = None,
    eps: float = LAYERNORM_EPS,
) -> torch.Tensor:
    """Process one token and return its logits. Mutates ``state`` in place."""
    cfg.validate()
    dev = state.S.device
    dtype = state.S.dtype
    if position is None:
        position = int(state.position)
    _window_reset_if_needed(cfg, state)

    freq = cached_pair_freq(cfg, dev)
    pos_t = torch.tensor([int(position)], dtype=torch.long, device=dev)
    cos, sin = rope_phase(pos_t, freq)
    cos = cos.reshape(-1)
    sin = sin.reshape(-1)

    tok = torch.as_tensor(token_id, dtype=torch.long, device=dev).reshape(())
    v = layer_norm(weights.embedding[tok], eps)

    n = int(state.segment_count)
    den = float(max(n, 1))
    rho = torch.sigmoid(weights.coord_alpha).to(dtype)
    w_wide = wide_projection_matrix(weights, cfg)

    for level in range(cfg.L):
        x = wide_project_x(v, weights, cfg, w_wide)
        q = apply_rope(x, cos, sin)
        state_level = state.S[level]
        a = torch.empty((cfg.H, cfg.D), dtype=dtype, device=dev)
        for head in range(cfg.H):
            a[head] = torch.matmul(q[head], state_level[head])
        for head in range(cfg.H):
            state_level[head].addr_(q[head], v)
        a = layer_norm(a, eps)
        y = torch.relu(torch.einsum("hd,hdk->hk", a, weights.decoder_y))
        u = x * y
        base = layer_norm(u.reshape(cfg.N) @ weights.encoder, eps)
        z = v @ weights.coord_Wc + weights.coord_bc
        c = state.C[level] / den - z
        g = 1.0 + rho * torch.tanh(c)
        state.C[level] += z
        delta = torch.relu((g * base) @ weights.writer_W1) @ weights.writer_W2
        v = layer_norm(v + delta, eps)

    state.last_hidden = v.detach().clone()
    state.has_last_hidden = True
    logits = v @ weights.readout
    state.position = int(position) + 1
    state.segment_count = n + 1
    return logits


def prefill_all_logits(
    weights: ArmAWeights,
    cfg: ArmAConfig,
    state: AkashaState,
    token_ids: Sequence[int] | torch.Tensor,
    positions: Optional[Sequence[int] | torch.Tensor] = None,
    segment_ids: Optional[Sequence[int] | torch.Tensor] = None,
    return_hidden: bool = False,
    eps: float = LAYERNORM_EPS,
):
    """Process a token sequence and return the logits of every token.

    With ``return_hidden=True`` returns ``(logits, hidden)`` where ``hidden``
    is the post-level hidden state of every token (``[T, D]``).

    ``segment_ids`` marks document segments; a change resets recurrent memory
    (``S``, ``C``, ``segment_count``) before the first token of the new
    segment, exactly as the trainer resets at a packed-row document boundary.
    An explicit ``positions`` entry sets the RoPE position for that token
    (document positions may continue across memory resets).
    """
    ids = torch.as_tensor(token_ids, dtype=torch.long, device=state.S.device).reshape(-1)
    n = int(ids.numel())
    if positions is None:
        pos = None
    else:
        pos = torch.as_tensor(positions, dtype=torch.long, device=state.S.device).reshape(-1)
        if int(pos.numel()) != n:
            raise ValueError("positions length must match token_ids length")
    seg = None
    if segment_ids is not None:
        seg = torch.as_tensor(
            segment_ids, dtype=torch.long, device=state.S.device
        ).reshape(-1)
        if int(seg.numel()) != n:
            raise ValueError("segment_ids length must match token_ids length")

    out = []
    hidden = []
    prev_seg = state.segment_id
    for i in range(n):
        if seg is not None:
            current = int(seg[i].item())
            if prev_seg is not None and current != prev_seg:
                state.reset_memory()
            prev_seg = current
            state.segment_id = current
        p = int(pos[i].item()) if pos is not None else None
        out.append(step(weights, cfg, state, int(ids[i].item()), position=p, eps=eps))
        if return_hidden:
            hidden.append(state.last_hidden.clone())
    if not out:
        raise ValueError("prefill_all_logits requires at least one token")
    logits = torch.stack(out, dim=0)
    if return_hidden:
        return logits, torch.stack(hidden, dim=0)
    return logits


def prefill_tokens(
    weights: ArmAWeights,
    cfg: ArmAConfig,
    state: AkashaState,
    token_ids: Sequence[int] | torch.Tensor,
    positions: Optional[Sequence[int] | torch.Tensor] = None,
    segment_ids: Optional[Sequence[int] | torch.Tensor] = None,
    eps: float = LAYERNORM_EPS,
) -> torch.Tensor:
    """Process a prompt and return the logits of its final token.

    The returned state already includes the final prompt token. The next
    operation must consume a *newly sampled* token; never re-feed the last
    prompt token.
    """
    ids = torch.as_tensor(token_ids, dtype=torch.long, device=state.S.device).reshape(-1)
    if int(ids.numel()) == 0:
        raise ValueError("prefill_tokens requires at least one token")
    all_logits = prefill_all_logits(
        weights, cfg, state, ids, positions=positions, segment_ids=segment_ids, eps=eps
    )
    return all_logits[-1]


def logits_from_state(weights: ArmAWeights, state: AkashaState) -> torch.Tensor:
    """Logits of the last token already processed, without reprocessing it."""
    if not state.has_last_hidden:
        raise ValueError("state has no processed token; nothing to decode")
    return state.last_hidden @ weights.readout
