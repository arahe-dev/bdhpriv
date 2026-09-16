from __future__ import annotations

import torch

from akasha.config import FP64_ATOL
from akasha.models.arma.reference_full import full_forward
from akasha.models.arma.reference_recurrent import (
    create_state,
    prefill_all_logits,
    prefill_tokens,
    step,
)
from akasha.models.arma.state import ContextPolicy


def test_begin_segment_zeroes_memory_but_not_position(tiny_weights, tiny_cfg):
    weights, cfg = tiny_weights, tiny_cfg
    state = create_state(weights, cfg)
    rng = torch.Generator().manual_seed(37)
    prompt = torch.randint(0, cfg.V, (5,), generator=rng)
    prefill_tokens(weights, cfg, state, prompt)
    state.S.fill_(1.0)
    state.C.fill_(1.0)

    state.begin_segment(position=91)
    assert state.segment_count == 0
    assert state.position == 91
    assert torch.count_nonzero(state.S) == 0
    assert torch.count_nonzero(state.C) == 0

    token = 3
    logits = step(weights, cfg, state, token)
    fresh = create_state(weights, cfg)
    fresh.position = 91
    expected = step(weights, cfg, fresh, token)
    assert torch.equal(logits, expected)
    assert state.position == 92
    assert state.segment_count == 1


def test_training_window_resets_memory_at_boundary():
    from akasha.models.arma.config import ArmAConfig

    cfg = ArmAConfig(T=4, V=32, D=8, N=32, H=2, L=2, HIDDEN=12)
    from akasha.bench.correctness import canonical_init

    weights = canonical_init(cfg)
    state = create_state(weights, cfg)
    rng = torch.Generator().manual_seed(41)
    prompt = torch.randint(0, cfg.V, (4,), generator=rng)
    positions = list(range(100, 104))
    prefill_tokens(weights, cfg, state, prompt, positions=positions)
    assert state.segment_count == 4
    assert int(torch.count_nonzero(state.S)) > 0

    extra = int(torch.randint(0, cfg.V, (1,), generator=rng).item())
    step(weights, cfg, state, extra)
    assert state.position == 105
    assert state.segment_count == 1

    fresh = create_state(weights, cfg)
    fresh.position = 104
    step(weights, cfg, fresh, extra)
    assert torch.equal(state.S, fresh.S)
    assert torch.equal(state.C, fresh.C)
    assert torch.equal(state.last_hidden, fresh.last_hidden)


def test_continuous_experimental_carries_past_window():
    from akasha.bench.correctness import canonical_init
    from akasha.models.arma.config import ArmAConfig

    cfg = ArmAConfig(T=4, V=32, D=8, N=32, H=2, L=2, HIDDEN=12)
    weights = canonical_init(cfg)
    state = create_state(
        weights, cfg, context_policy=ContextPolicy.CONTINUOUS_EXPERIMENTAL
    )
    rng = torch.Generator().manual_seed(43)
    prompt = torch.randint(0, cfg.V, (4,), generator=rng)
    prefill_tokens(weights, cfg, state, prompt)
    extra = int(torch.randint(0, cfg.V, (1,), generator=rng).item())
    step(weights, cfg, state, extra)
    assert state.segment_count == 5
    assert int(torch.count_nonzero(state.S)) > 0


def test_segment_reset_matches_separate_prefill(tiny_weights, tiny_cfg):
    weights, cfg = tiny_weights, tiny_cfg
    left = torch.tensor([1, 2, 3, 4])
    right = torch.tensor([5, 6, 7])
    state = create_state(weights, cfg)
    prefill_tokens(weights, cfg, state, left, positions=range(4),
                   segment_ids=[0, 0, 0, 0])
    logits = prefill_tokens(weights, cfg, state, right, positions=range(10, 13),
                            segment_ids=[1, 1, 1])

    fresh = create_state(weights, cfg)
    expected = prefill_all_logits(
        weights, cfg, fresh, right, positions=torch.arange(10, 13),
        segment_ids=torch.zeros(3, dtype=torch.long),
    )
    assert (logits - expected[-1]).abs().max().item() <= 1e-6
