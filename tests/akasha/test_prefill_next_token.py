from __future__ import annotations

import torch

from akasha.models.arma.reference_recurrent import (
    create_state,
    prefill_all_logits,
    prefill_tokens,
    step,
)
from akasha.runtime.model import AkashaModel
from akasha.runtime.session import AkashaSession
from akasha.sampling.sampler import Sampler


def _model(weights, cfg) -> AkashaModel:
    return AkashaModel(weights=weights, cfg=cfg, model_fingerprint="test")


def test_prefill_returns_logits_after_the_last_prompt_token(tiny_weights, tiny_cfg):
    weights, cfg = tiny_weights, tiny_cfg
    prompt = [3, 1, 4, 1, 5, 9, 2, 6]
    state = create_state(weights, cfg)
    logits = prefill_tokens(weights, cfg, state, prompt)

    replay = prefill_all_logits(weights, cfg, create_state(weights, cfg), prompt)
    assert torch.equal(logits, replay[-1])
    assert state.segment_count == len(prompt)
    assert state.position == len(prompt)


def test_next_decode_must_consume_a_new_token(tiny_weights, tiny_cfg):
    weights, cfg = tiny_weights, tiny_cfg
    prompt = [3, 1, 4, 1, 5]
    state = create_state(weights, cfg)
    logits = prefill_tokens(weights, cfg, state, prompt)

    next_id = int(torch.argmax(logits).item())
    after = step(weights, cfg, state, next_id)

    expected = prefill_all_logits(
        weights, cfg, create_state(weights, cfg), prompt + [next_id]
    )
    assert torch.equal(after, expected[-1])

    refeed = create_state(weights, cfg)
    prefill_tokens(weights, cfg, refeed, prompt)
    wrong = step(weights, cfg, refeed, prompt[-1])
    assert not torch.allclose(wrong, after)


def test_model_api_prefill_and_decode_one(tiny_weights, tiny_cfg):
    model = _model(tiny_weights, tiny_cfg)
    state = model.create_state()
    logits = model.prefill_tokens(state, [1, 2, 3])
    restored = model.logits_for_next_token(state)
    assert torch.equal(logits, restored)

    next_id = int(torch.argmax(logits).item())
    decoded = model.decode_one(state, next_id)
    assert restored.shape == decoded.shape
    assert state.position == 4


def test_session_greedy_generation_and_clone(tiny_weights, tiny_cfg):
    model = _model(tiny_weights, tiny_cfg)
    session = AkashaSession.create(model, sampler=Sampler())
    session.prefill([1, 2, 3, 4])
    produced = session.generate(5)
    assert len(produced) == 5

    twin = session.clone()
    twin.prefill([9, 9])
    only_original = session.generate(2)
    both = twin.generate(2)
    assert len(only_original) == 2
    assert len(both) == 2
    assert session.state.S.data_ptr() != twin.state.S.data_ptr()
