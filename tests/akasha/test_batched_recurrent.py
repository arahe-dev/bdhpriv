"""Batched bench engine must reproduce the verified oracle to machine precision.

The engine is not bit-identical by construction: batched BLAS reductions can
differ in the last ulp from the reference's per-head ops. The contract is
numerical equivalence at machine precision (float64 <= 1e-14, float32 <= 1e-6).
"""

from __future__ import annotations

import pytest
import torch

from akasha.bench.correctness import canonical_init
from akasha.models.arma.config import ArmAConfig
from akasha.models.arma.reference_recurrent import create_state, step
from akasha.bench.recurrent_batched import (
    BatchedDecodeModule,
    CompiledDecodeModule,
    batched_step,
    create_batched_state,
)


def _cfg() -> ArmAConfig:
    return ArmAConfig(T=16, V=32, D=8, N=32, H=2, L=2, HIDDEN=12)


def test_batched_step_matches_reference_per_session():
    cfg = _cfg()
    weights = canonical_init(cfg).to(dtype=torch.float64)
    batch = 3
    sessions = [create_state(weights, cfg) for _ in range(batch)]
    batched = create_batched_state(weights, cfg, batch, dtype=torch.float64)

    gen = torch.Generator().manual_seed(71)
    for _ in range(6):
        tokens = torch.randint(0, cfg.V, (batch,), generator=gen)
        out = batched_step(weights, cfg, batched, tokens)
        for row in range(batch):
            expected = step(weights, cfg, sessions[row], int(tokens[row].item()))
            torch.testing.assert_close(out[row], expected, atol=1e-14, rtol=1e-14)
            torch.testing.assert_close(
                batched.S[row], sessions[row].S, atol=1e-14, rtol=1e-14
            )
            torch.testing.assert_close(
                batched.C[row], sessions[row].C, atol=1e-14, rtol=1e-14
            )
            torch.testing.assert_close(
                batched.last_hidden[row],
                sessions[row].last_hidden,
                atol=1e-14,
                rtol=1e-14,
            )
            assert int(batched.position[row]) == sessions[row].position
            assert int(batched.segment_count[row]) == sessions[row].segment_count


def test_batched_decode_module_matches_reference():
    cfg = _cfg()
    weights = canonical_init(cfg).to(dtype=torch.float32)
    batch = 2
    module = BatchedDecodeModule(weights, cfg, batch)
    reference = [create_state(weights, cfg) for _ in range(batch)]

    gen = torch.Generator().manual_seed(73)
    for _ in range(5):
        tokens = torch.randint(0, cfg.V, (batch,), generator=gen)
        out = module.decode(tokens)
        for row in range(batch):
            expected = step(weights, cfg, reference[row], int(tokens[row].item()))
            torch.testing.assert_close(out[row], expected, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(
                module.S[row], reference[row].S, atol=1e-6, rtol=1e-6
            )
            torch.testing.assert_close(
                module.C[row], reference[row].C, atol=1e-6, rtol=1e-6
            )


@pytest.mark.parametrize("layout", ["buffers", "grouped", "stacked"])
def test_compiled_layouts_match_reference(layout):
    cfg = _cfg()
    weights = canonical_init(cfg).to(dtype=torch.float32)
    batch = 2
    module = CompiledDecodeModule(weights, cfg, batch, layout=layout)
    reference = [create_state(weights, cfg) for _ in range(batch)]

    gen = torch.Generator().manual_seed(83)
    for _ in range(5):
        tokens = torch.randint(0, cfg.V, (batch,), generator=gen)
        out = module.decode(tokens)
        for row in range(batch):
            expected = step(weights, cfg, reference[row], int(tokens[row].item()))
            torch.testing.assert_close(out[row], expected, atol=1e-6, rtol=1e-6)
    for row in range(batch):
        torch.testing.assert_close(
            module.S[row], reference[row].S, atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            module.C[row], reference[row].C, atol=1e-6, rtol=1e-6
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batched_step_gpu_matches_reference():
    cfg = _cfg()
    weights = canonical_init(cfg).to(device="cuda")
    batch = 2
    sessions = [create_state(weights, cfg) for _ in range(batch)]
    batched = create_batched_state(weights, cfg, batch, device="cuda")

    gen = torch.Generator().manual_seed(79)
    for _ in range(4):
        tokens = torch.randint(0, cfg.V, (batch,), generator=gen)
        out = batched_step(weights, cfg, batched, tokens)
        for row in range(batch):
            expected = step(weights, cfg, sessions[row], int(tokens[row].item()))
            torch.testing.assert_close(
                out[row].cpu(), expected.cpu(), atol=1e-5, rtol=1e-5
            )
