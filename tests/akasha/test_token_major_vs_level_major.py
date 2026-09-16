from __future__ import annotations

import pytest
import torch

from akasha.bench.correctness import canonical_init, token_major_gates
from akasha.config import FP64_ATOL
from akasha.models.arma.config import ArmAConfig
from akasha.models.arma.reference_full import full_forward
from akasha.models.arma.reference_recurrent import create_state, prefill_all_logits


def _segments(t: int, cuts):
    seg = torch.zeros(t, dtype=torch.long)
    for index, cut in enumerate(cuts):
        seg[cut:] = index + 1
    return seg


def test_token_major_equals_level_major_tiny(tiny_cfg):
    result = token_major_gates(tiny_cfg)
    assert result["pass"], result
    assert result["logits_max_abs_error"] <= FP64_ATOL
    assert result["hidden_max_abs_error"] <= FP64_ATOL
    assert result["chunked_scan_max_abs_error"] <= FP64_ATOL


@pytest.mark.parametrize("cuts", [(7,), (5, 11), (3, 8, 13)])
def test_schedule_equivalence_with_segment_resets(cuts):
    cfg = ArmAConfig(T=16, V=32, D=8, N=32, H=2, L=2, HIDDEN=12)
    weights = canonical_init(cfg).to(dtype=torch.float64)
    gen = torch.Generator().manual_seed(17)
    ids = torch.randint(0, cfg.V, (cfg.T,), generator=gen)
    seg = _segments(cfg.T, list(cuts))
    positions = torch.arange(cfg.T) + 100

    full = full_forward(weights, cfg, ids, positions=positions, segment_ids=seg)
    state = create_state(weights, cfg)
    rec = prefill_all_logits(
        weights, cfg, state, ids, positions=positions, segment_ids=seg
    )
    assert (full.logits[0] - rec).abs().max().item() <= FP64_ATOL
    assert state.segment_count == cfg.T - cuts[-1]


def test_chunked_scan_matches_dense_for_mid_chunk_segment_starts():
    cfg = ArmAConfig(T=16, V=32, D=8, N=32, H=2, L=2, HIDDEN=12)
    weights = canonical_init(cfg).to(dtype=torch.float64)
    gen = torch.Generator().manual_seed(19)
    ids = torch.randint(0, cfg.V, (cfg.T,), generator=gen)
    seg = _segments(cfg.T, [3])
    positions = torch.arange(cfg.T)

    dense = full_forward(weights, cfg, ids, positions=positions, segment_ids=seg)
    for block in (2, 4, 5, 8):
        chunked = full_forward(
            weights, cfg, ids, positions=positions, segment_ids=seg, scan_block=block
        )
        err = (dense.logits[0] - chunked.logits[0]).abs().max().item()
        assert err <= FP64_ATOL, f"block={block} err={err}"
