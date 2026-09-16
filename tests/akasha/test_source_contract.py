from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from akasha.config import (
    REPO_SOURCE_COMMIT,
    REPO_SOURCE_COMMIT_FULL,
    TRAINER_PATH,
    TRAINER_SHA256,
)
from akasha.models.arma.config import production_config
from akasha.models.arma.manifest import (
    LAYERNORM_EPS,
    ROPE_CONVENTION,
    manifest_dict,
    manifest_fingerprint,
)

ROOT = Path(__file__).resolve().parents[2]


def test_trainer_source_hash_matches_manifest():
    digest = hashlib.sha256((ROOT / TRAINER_PATH).read_bytes()).hexdigest()
    assert digest == TRAINER_SHA256
    manifest = manifest_dict(production_config())
    assert manifest["trainer_sha256"] == TRAINER_SHA256
    assert manifest["source_commit"] == REPO_SOURCE_COMMIT
    assert manifest["source_commit_full"] == REPO_SOURCE_COMMIT_FULL


def test_manifest_config_matches_trainer_prod_config(trainer_module):
    manifest = manifest_dict(production_config())
    trainer_cfg = trainer_module.PROD_CFG
    for key in ("T", "V", "D", "N", "H", "L", "HIDDEN", "THETA",
                "READ_BLOCK", "SEED", "INIT_STD"):
        assert manifest["config"][key] == getattr(trainer_cfg, key), key
    assert manifest["config"]["K"] == trainer_cfg.K


def test_layernorm_epsilon_matches_torch_default():
    import inspect

    from torch.nn.modules.normalization import LayerNorm

    source = inspect.getsource(LayerNorm.__init__)
    assert "eps: float = 1e-5" in source
    assert LAYERNORM_EPS == 1e-5


def test_rope_convention_records_trainer_formula(trainer_module):
    assert ROPE_CONVENTION["theta"] == 2**16
    cfg = production_config()
    ours = manifest_dict(cfg)["rope"]
    assert ours["style"] == "interleaved_pairs"
    assert "remainder" in ours["phase"]


def test_manifest_fingerprint_is_stable():
    assert manifest_fingerprint() == manifest_fingerprint(production_config())


def test_manifest_tensor_shapes_are_frozen():
    manifest = manifest_dict(production_config())
    shapes = {name: spec["shape"] for name, spec in manifest["tensors"].items()}
    assert shapes["embedding"] == [8192, 256]
    assert shapes["encoder"] == [16384, 256]
    assert shapes["decoder_x"] == [4, 256, 4096]
    assert shapes["decoder_y"] == [4, 256, 4096]
    assert shapes["readout"] == [256, 8192]
    assert shapes["coord_Wc"] == [256, 256]
    assert shapes["coord_bc"] == [256]
    assert shapes["coord_alpha"] == []
    assert shapes["writer_W1"] == [256, 1040]
    assert shapes["writer_W2"] == [1040, 256]
