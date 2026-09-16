from __future__ import annotations

import json

import pytest
import torch

from akasha.checkpoint.convert_arma import convert_trainer_checkpoint
from akasha.checkpoint.loader import (
    TRAINER_TO_CANONICAL,
    load_package,
    save_package,
)
from akasha.config import (
    TRAINER_CKPT_FORMAT,
    TRAINER_IMPLEMENTATION_VERSION,
)
from akasha.models.arma.manifest import manifest_dict
from akasha.models.arma.reference_recurrent import (
    create_state,
    prefill_all_logits,
)
from akasha.runtime.model import AkashaModel
from akasha.tokenizer.adapter import ArmATokenizerAdapter, TokenizerStatus
from akasha.bench.correctness import canonical_init


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


def _payload(weights, cfg, **overrides):
    payload = {
        "format": TRAINER_CKPT_FORMAT,
        "implementation": TRAINER_IMPLEMENTATION_VERSION,
        "code_sha256": "0" * 64,
        "config": {**manifest_dict(cfg)["config"], "GLOBAL_BATCH": 64},
        "model": _trainer_state_dict(weights),
        "progress": {"updates_done": 3, "tokens_consumed": 3, "next_sequence": 3},
        "saved_at": "2026-09-16T00:00:00Z",
    }
    payload.update(overrides)
    return payload


def test_trainer_checkpoint_conversion_round_trip(tmp_path, tiny_weights, tiny_cfg):
    src = tmp_path / "latest.pt"
    torch.save(_payload(tiny_weights, tiny_cfg), src)
    summary = convert_trainer_checkpoint(src, tmp_path / "package", cfg=tiny_cfg)
    assert summary["ok"] is True
    assert summary["weights_fingerprint"]

    loaded = load_package(tmp_path / "package")
    for name, tensor in tiny_weights.tensors().items():
        assert torch.equal(loaded.weights.tensors()[name], tensor)

    model = AkashaModel.load(tmp_path / "package")
    state = model.create_state()
    logits = prefill_all_logits(
        model.weights, model.cfg, state, [1, 2, 3, 4]
    )
    assert logits.shape == (4, tiny_cfg.V)
    assert bool(torch.isfinite(logits).all())


def test_conversion_rejects_config_mismatch(tmp_path, tiny_weights, tiny_cfg):
    payload = _payload(tiny_weights, tiny_cfg)
    payload["config"]["T"] = 4096
    src = tmp_path / "bad_config.pt"
    torch.save(payload, src)
    with pytest.raises(ValueError, match="configuration"):
        convert_trainer_checkpoint(src, tmp_path / "package")


def test_conversion_rejects_bad_format(tmp_path, tiny_weights, tiny_cfg):
    payload = _payload(tiny_weights, tiny_cfg, format="not_arm_a")
    src = tmp_path / "bad_format.pt"
    torch.save(payload, src)
    with pytest.raises(ValueError, match="format"):
        convert_trainer_checkpoint(src, tmp_path / "package")


def test_conversion_rejects_missing_tensor(tmp_path, tiny_weights, tiny_cfg):
    payload = _payload(tiny_weights, tiny_cfg)
    del payload["model"]["writer.W1"]
    src = tmp_path / "missing.pt"
    torch.save(payload, src)
    with pytest.raises(ValueError, match="missing"):
        convert_trainer_checkpoint(src, tmp_path / "package", cfg=tiny_cfg)


def test_package_loader_rejects_tampered_fingerprint(tmp_path, tiny_weights, tiny_cfg):
    save_package(tmp_path / "pkg", tiny_weights, tiny_cfg)
    manifest_path = tmp_path / "pkg" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["weights_fingerprint"] = "deadbeef"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_package(tmp_path / "pkg")


def test_package_loader_rejects_wrong_shape(tmp_path, tiny_weights, tiny_cfg):
    from safetensors.torch import load_file, save_file

    save_package(tmp_path / "pkg", tiny_weights, tiny_cfg)
    tensors = {name: tensor.clone() for name, tensor in load_file(
        str(tmp_path / "pkg" / "model.safetensors")
    ).items()}
    tensors["encoder"] = tensors["encoder"][:-1]
    import shutil

    shutil.copytree(tmp_path / "pkg", tmp_path / "pkg2")
    save_file(tensors, str(tmp_path / "pkg2" / "model.safetensors"), metadata={})
    with pytest.raises(ValueError, match="encoder"):
        load_package(tmp_path / "pkg2")


def test_tokenizer_is_blocked_without_artifact():
    adapter = ArmATokenizerAdapter()
    assert adapter.status() == TokenizerStatus.BLOCKED_ARTIFACT_NOT_LOCAL
    with pytest.raises(Exception):
        adapter.encode("hello")


def test_tokenizer_rejects_wrong_hash(tmp_path):
    bogus = tmp_path / "tokenizer.json"
    bogus.write_text("{}", encoding="utf-8")
    adapter = ArmATokenizerAdapter(bogus)
    assert adapter.status() == TokenizerStatus.HASH_MISMATCH
