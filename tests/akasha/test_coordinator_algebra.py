from __future__ import annotations

import torch

from akasha.bench.correctness import coordinator_algebra
from akasha.config import FP64_ATOL


def test_recurrent_coordinator_matches_dense_masked_mean(tiny_cfg):
    result = coordinator_algebra(tiny_cfg)
    assert result["pass"], result
    assert result["max_abs_error"] <= FP64_ATOL


def test_coordinator_accumulator_updates_after_read(tiny_cfg):
    cfg = tiny_cfg
    torch.manual_seed(13)
    z = torch.randn(cfg.T, cfg.D, dtype=torch.float64)
    cut = cfg.T // 3

    acc = torch.zeros(cfg.D, dtype=torch.float64)
    count = 0
    outputs = []
    for i in range(cfg.T):
        if i == cut:
            acc = torch.zeros_like(acc)
            count = 0
        outputs.append(acc / max(count, 1) - z[i])
        acc = acc + z[i]
        count += 1

    assert count == cfg.T - cut
    assert (acc - z[cut:].sum(dim=0)).abs().max().item() <= FP64_ATOL
    assert torch.isfinite(torch.stack(outputs)).all()
