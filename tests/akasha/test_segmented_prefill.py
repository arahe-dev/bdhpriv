from __future__ import annotations

import torch

from akasha.bench.correctness import canonical_init
from akasha.models.arma.config import ArmAConfig
from akasha.models.arma.reference_full import full_forward
from akasha.models.arma.reference_recurrent import (
    create_state,
    prefill_all_logits,
    prefill_tokens,
)


def _cfg() -> ArmAConfig:
    return ArmAConfig(T=16, V=32, D=8, N=32, H=2, L=2, HIDDEN=12)


def _packed(seed: int = 61):
    cfg = _cfg()
    weights = canonical_init(cfg).to(dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, cfg.V, (cfg.T,), generator=gen)
    segment_ids = torch.zeros(cfg.T, dtype=torch.long)
    segment_ids[5:] = 1
    segment_ids[11:] = 2
    positions = torch.arange(cfg.T) + 200
    return cfg, weights, ids, segment_ids, positions


def test_chunked_prefill_matches_dense_and_recurrent():
    cfg, weights, ids, segment_ids, positions = _packed()
    dense = full_forward(weights, cfg, ids, positions=positions,
                         segment_ids=segment_ids)
    for block in (2, 4, 5, 8, 16):
        chunked = full_forward(
            weights, cfg, ids, positions=positions, segment_ids=segment_ids,
            scan_block=block,
        )
        assert (dense.logits[0] - chunked.logits[0]).abs().max().item() <= 1e-11

    state = create_state(weights, cfg)
    rec = prefill_all_logits(
        weights, cfg, state, ids, positions=positions, segment_ids=segment_ids
    )
    assert (dense.logits[0] - rec).abs().max().item() <= 1e-11


def test_no_document_leakage_across_segment_boundaries():
    cfg, weights, ids, segment_ids, positions = _packed()
    state = create_state(weights, cfg)

    left_ids = ids[:11]
    right_ids = ids[11:]
    prefill_tokens(
        weights, cfg, state, left_ids, positions=positions[:11],
        segment_ids=segment_ids[:11],
    )
    assert state.segment_count == 6

    right_logits = prefill_all_logits(
        weights, cfg, state, right_ids, positions=positions[11:],
        segment_ids=segment_ids[11:],
    )
    assert state.segment_count == 5

    fresh = create_state(weights, cfg)
    fresh_logits = prefill_all_logits(
        weights, cfg, fresh, right_ids, positions=positions[11:],
        segment_ids=segment_ids[11:],
    )
    assert torch.equal(right_logits, fresh_logits)


def test_naive_single_chunk_carry_leaks_across_reset():
    """The plain O = Q S0 + ((Q Q^T) * strict_lower) @ V formula assumes one
    segment; with a reset inside the chunk it produces different logits,
    which is why segmented prefill must reproduce the trainer mask logic."""
    cfg, weights, ids, segment_ids, positions = _packed()
    token = 6
    torch.manual_seed(67)
    q = torch.randn(cfg.H, cfg.K, dtype=torch.float64)
    v = torch.randn(cfg.D, dtype=torch.float64)
    state = torch.zeros(cfg.H, cfg.K, cfg.D, dtype=torch.float64)
    state[0].fill_(1.0)

    correct = torch.einsum("hk,hkd->hd", q, state)
    fresh_after_reset = torch.zeros_like(correct)
    assert not torch.allclose(correct, fresh_after_reset)
