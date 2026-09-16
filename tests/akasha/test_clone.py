from __future__ import annotations

import torch

from akasha.models.arma.reference_recurrent import (
    create_state,
    logits_from_state,
    prefill_tokens,
    step,
)


def test_clone_is_independent(tiny_weights, tiny_cfg):
    weights, cfg = tiny_weights, tiny_cfg
    state = create_state(weights, cfg)
    rng = torch.Generator().manual_seed(53)
    prompt = torch.randint(0, cfg.V, (6,), generator=rng).tolist()
    prefill_tokens(weights, cfg, state, prompt)

    clone = state.clone()
    assert clone.S.data_ptr() != state.S.data_ptr()
    assert clone.C.data_ptr() != state.C.data_ptr()
    assert clone.last_hidden.data_ptr() != state.last_hidden.data_ptr()
    assert torch.equal(clone.S, state.S)
    assert torch.equal(clone.C, state.C)
    assert torch.equal(clone.last_hidden, state.last_hidden)
    assert clone.position == state.position
    assert clone.segment_count == state.segment_count

    step(weights, cfg, state, 7)
    assert not torch.equal(clone.S, state.S)
    assert clone.position == state.position - 1

    expected = create_state(weights, cfg)
    prefill_tokens(weights, cfg, expected, prompt)
    step(weights, cfg, expected, 7)
    clone_logits = step(weights, cfg, clone, 7)
    assert torch.equal(clone.S, expected.S)
    assert torch.equal(clone.C, expected.C)
    assert torch.equal(clone.last_hidden, expected.last_hidden)
    assert clone.position == expected.position
    assert clone.segment_count == expected.segment_count


def test_clone_preserves_sampling_semantics(tiny_weights, tiny_cfg):
    weights, cfg = tiny_weights, tiny_cfg
    state = create_state(weights, cfg)
    prefill_tokens(weights, cfg, state, [1, 2, 3, 4])

    clone = state.clone()
    token = int(torch.argmax(logits_from_state(weights, state)).item())
    left = step(weights, cfg, state, token)
    right = step(weights, cfg, clone, token)
    assert torch.equal(left, right)


def test_clone_preserves_context_policy(tiny_weights, tiny_cfg):
    from akasha.models.arma.state import ContextPolicy

    state = create_state(
        tiny_weights, tiny_cfg, context_policy=ContextPolicy.CONTINUOUS_EXPERIMENTAL
    )
    clone = state.clone()
    assert clone.context_policy == ContextPolicy.CONTINUOUS_EXPERIMENTAL
