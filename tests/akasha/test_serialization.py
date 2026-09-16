from __future__ import annotations

import pytest
import torch

from akasha.checkpoint.loader import load_state, save_state
from akasha.models.arma.reference_recurrent import (
    create_state,
    logits_from_state,
    prefill_tokens,
    step,
)
from akasha.models.arma.state import ContextPolicy
from akasha.sampling.sampler import Sampler, SamplerMethod


def test_state_serialization_round_trip(tmp_path, tiny_weights, tiny_cfg):
    weights, cfg = tiny_weights, tiny_cfg
    state = create_state(
        weights, cfg, context_policy=ContextPolicy.CONTINUOUS_EXPERIMENTAL
    )
    prompts = [3, 1, 4, 1, 5, 9, 2, 6]
    prefill_tokens(weights, cfg, state, prompts, positions=range(100, 108))
    state.segment_id = 4

    path = tmp_path / "session"
    save_state(path, state)
    loaded, sampler = load_state(path)

    assert torch.equal(loaded.S, state.S)
    assert torch.equal(loaded.C, state.C)
    assert torch.equal(loaded.last_hidden, state.last_hidden)
    assert loaded.position == state.position
    assert loaded.segment_count == state.segment_count
    assert loaded.context_policy == ContextPolicy.CONTINUOUS_EXPERIMENTAL
    assert loaded.has_last_hidden is True
    assert loaded.segment_id == 4

    torch.testing.assert_close(
        logits_from_state(weights, loaded), logits_from_state(weights, state)
    )
    token = 11
    torch.testing.assert_close(
        step(weights, cfg, loaded, token), step(weights, cfg, state, token)
    )


def test_last_hidden_is_mandatory_in_snapshot(tmp_path, tiny_weights, tiny_cfg):
    state = create_state(tiny_weights, tiny_cfg)
    prefill_tokens(tiny_weights, tiny_cfg, state, [1, 2, 3])
    save_state(tmp_path / "s", state)
    from safetensors.torch import load_file

    tensors = load_file(str(tmp_path / "s.safetensors"))
    assert "last_hidden" in tensors
    assert "S" in tensors and "C" in tensors


def test_fingerprint_mismatch_is_rejected(tmp_path, tiny_weights, tiny_cfg):
    state = create_state(tiny_weights, tiny_cfg)
    state.model_fingerprint = "abc"
    prefill_tokens(tiny_weights, tiny_cfg, state, [1, 2])
    save_state(tmp_path / "s", state)
    with pytest.raises(ValueError, match="fingerprint"):
        load_state(tmp_path / "s", expected_fingerprint="different")


def test_sampler_rng_serialized_separately(tmp_path, tiny_weights, tiny_cfg):
    sampler = Sampler(method=SamplerMethod.MULTINOMIAL, temperature=0.8, seed=99)
    state = create_state(tiny_weights, tiny_cfg)
    logits = prefill_tokens(tiny_weights, tiny_cfg, state, [1, 2, 3])
    for _ in range(3):
        sampler.sample(logits)

    save_state(tmp_path / "s", state, sampler=sampler)
    assert (tmp_path / "s.rng.safetensors").exists()

    _, restored = load_state(
        tmp_path / "s", sampler=Sampler(method=SamplerMethod.MULTINOMIAL)
    )
    assert restored.method == SamplerMethod.MULTINOMIAL
    assert restored.temperature == pytest.approx(0.8)
    assert restored.seed == 99
    a = sampler.sample(logits)
    b = restored.sample(logits)
    assert a == b

    clone = sampler.clone()
    assert clone.sample(logits) == sampler.sample(logits)


def test_clone_of_sampler_does_not_share_rng(tiny_weights, tiny_cfg):
    sampler = Sampler(method=SamplerMethod.MULTINOMIAL, temperature=1.0, seed=7)
    logits = torch.zeros(tiny_cfg.V)
    sampler.sample(logits)
    clone = sampler.clone()
    a = sampler.sample(logits)
    b = clone.sample(logits)
    assert a == b
