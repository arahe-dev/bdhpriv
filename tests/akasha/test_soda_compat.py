from __future__ import annotations

import importlib

import pytest
import torch

from akasha.runtime.model import AkashaModel
from akasha.runtime.session import AkashaSession


def _trainer_state_dict(weights):
    return {
        "embedding.weight": weights.embedding,
        "encoder": weights.encoder,
        "decoder_x": weights.decoder_x,
        "decoder_y": weights.decoder_y,
        "readout": weights.readout,
        "coordinator.Wc": weights.coord_Wc,
        "coordinator.bc": weights.coord_bc,
        "coordinator.alpha": weights.coord_alpha,
        "writer.W1": weights.writer_W1,
        "writer.W2": weights.writer_W2,
    }


def _payload(weights, cfg):
    return {
        "format": "soda_bdh_ckpt_v1",
        "implementation": "soda_bdh_trainer_v1_dense_arm_a_opt3c_test",
        "code_sha256": "test-code-sha",
        "config": cfg.to_dict(),
        "model": _trainer_state_dict(weights),
        "progress": {"updates_done": 3, "tokens_consumed": 96},
        "corpus": {"corpus_id": "test-soda"},
        "saved_at": "2026-09-18T00:00:00Z",
    }


def _adapter(monkeypatch, tiny_cfg):
    adapter = importlib.import_module("scripts.akasha_soda_compat")
    monkeypatch.setattr(adapter, "production_config", lambda: tiny_cfg)
    return adapter


def test_soda_checkpoint_converts_with_explicit_noncanonical_provenance(
    tmp_path, monkeypatch, tiny_weights, tiny_cfg
):
    adapter = _adapter(monkeypatch, tiny_cfg)
    checkpoint = tmp_path / "soda.pt"
    torch.save(_payload(tiny_weights, tiny_cfg), checkpoint)

    result = adapter.convert(checkpoint, tmp_path / "package")

    assert result["ok"] is True
    assert result["adapter_status"] == "EXPLICIT_COMPATIBILITY_ONLY"
    assert result["canonical_akasha_trainer_status"] == "NOT_CANONICAL_AKASHA_TRAINER"
    model = AkashaModel.load(tmp_path / "package", device="cpu")
    provenance = model.manifest["provenance"]
    assert provenance["soda_format"] == "soda_bdh_ckpt_v1"
    assert provenance["cpu_only"] is True

    session = AkashaSession.create(model)
    prompt_logits = session.prefill([1, 2, 3, 4])
    generated = session.generate(3)
    assert bool(torch.isfinite(prompt_logits).all())
    assert len(generated) == 3
    assert all(0 <= token < tiny_cfg.V for token in generated)


def test_soda_checkpoint_rejects_wrong_format(
    tmp_path, monkeypatch, tiny_weights, tiny_cfg
):
    adapter = _adapter(monkeypatch, tiny_cfg)
    payload = _payload(tiny_weights, tiny_cfg)
    payload["format"] = "arm_a_2p5b_ckpt_v1"
    checkpoint = tmp_path / "wrong.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="expected 'soda_bdh_ckpt_v1'"):
        adapter.convert(checkpoint, tmp_path / "package")


def test_soda_checkpoint_rejects_architecture_drift(
    tmp_path, monkeypatch, tiny_weights, tiny_cfg
):
    adapter = _adapter(monkeypatch, tiny_cfg)
    payload = _payload(tiny_weights, tiny_cfg)
    payload["config"]["READ_BLOCK"] += 1
    checkpoint = tmp_path / "drift.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="frozen dense Arm-A architecture"):
        adapter.convert(checkpoint, tmp_path / "package")


def test_soda_checkpoint_rejects_extra_model_tensor(
    tmp_path, monkeypatch, tiny_weights, tiny_cfg
):
    adapter = _adapter(monkeypatch, tiny_cfg)
    payload = _payload(tiny_weights, tiny_cfg)
    payload["model"]["unexpected"] = torch.zeros(1)
    checkpoint = tmp_path / "extra.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="unexpected trainer state_dict keys"):
        adapter.convert(checkpoint, tmp_path / "package")
