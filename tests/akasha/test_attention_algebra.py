from __future__ import annotations

import torch

from akasha.bench.correctness import attention_algebra
from akasha.config import FP64_ATOL


def test_recurrent_attention_matches_strict_past_dense(tiny_cfg):
    result = attention_algebra(tiny_cfg)
    assert result["pass"], result
    assert result["max_abs_error"] <= FP64_ATOL


def test_strict_past_ordering_is_read_before_write(tiny_cfg):
    """Updating the state before the read must change the result."""
    cfg = tiny_cfg
    torch.manual_seed(7)
    q = torch.randn(cfg.T, cfg.H, cfg.K, dtype=torch.float64)
    v = torch.randn(cfg.T, cfg.D, dtype=torch.float64)

    correct = []
    state = torch.zeros(cfg.H, cfg.K, cfg.D, dtype=torch.float64)
    for i in range(cfg.T):
        read = torch.einsum("hk,hkd->hd", q[i], state)
        correct.append(read)
        state = state + torch.einsum("hk,d->hkd", q[i], v[i])

    inclusive = []
    state = torch.zeros(cfg.H, cfg.K, cfg.D, dtype=torch.float64)
    for i in range(cfg.T):
        state = state + torch.einsum("hk,d->hkd", q[i], v[i])
        inclusive.append(torch.einsum("hk,hkd->hd", q[i], state))

    gap = (torch.stack(correct) - torch.stack(inclusive)).abs().max().item()
    assert gap > 1e-3


def test_segment_reset_zeroes_recurrent_sum(tiny_cfg):
    cfg = tiny_cfg
    torch.manual_seed(11)
    q = torch.randn(cfg.T, cfg.H, cfg.K, dtype=torch.float64)
    v = torch.randn(cfg.T, cfg.D, dtype=torch.float64)
    cut = cfg.T // 2

    state = torch.zeros(cfg.H, cfg.K, cfg.D, dtype=torch.float64)
    outputs = []
    for i in range(cfg.T):
        if i == cut:
            state = torch.zeros_like(state)
        outputs.append(torch.einsum("hk,hkd->hd", q[i], state))
        state = state + torch.einsum("hk,d->hkd", q[i], v[i])

    expected = torch.zeros(cfg.T, cfg.H, cfg.D, dtype=torch.float64)
    for i in range(cfg.T):
        for j in range(i):
            if (j < cut) == (i < cut):
                score = torch.einsum("hk,hk->h", q[i], q[j])
                expected[i] = expected[i] + score.unsqueeze(-1) * v[j]
    assert (torch.stack(outputs) - expected).abs().max().item() <= FP64_ATOL
