from __future__ import annotations

import torch

from akasha.models.arma.config import production_config
from akasha.models.arma.reference_recurrent import create_state
from akasha.models.arma.state import (
    production_state_bytes,
    state_element_counts,
)


def test_production_state_element_counts():
    cfg = production_config()
    counts = state_element_counts(cfg)
    assert counts["S"] == 8 * 4 * 4096 * 256 == 33_554_432
    assert counts["C"] == 8 * 256 == 2_048
    assert counts["total_mathematical"] == 33_556_480


def test_production_state_bytes_fp32():
    assert production_state_bytes(torch.float32) == 134_225_920
    mib = production_state_bytes(torch.float32) / 2**20
    assert abs(mib - 128.0078125) < 1e-9


def test_create_state_allocates_exactly_the_contract(prod_weights, prod_cfg):
    state = create_state(prod_weights, prod_cfg)
    assert state.S.numel() == 33_554_432
    assert state.C.numel() == 2_048
    assert state.S.dtype == torch.float32
    assert state.C.dtype == torch.float32
    assert state.nbytes() == production_state_bytes(torch.float32)
    assert state.last_hidden.numel() == prod_cfg.D
    assert state.position == 0
    assert state.segment_count == 0
    assert state.has_last_hidden is False
