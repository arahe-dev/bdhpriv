"""Correctness gates for opt/expert_moe.ExpertizedArmA (campaign exp. 01).

All-active expertization must be numerically equivalent to frozen Arm-A:
FP32/FP64 logits, gradients, packed documents, cross-sequence boundaries,
and RoPE pair-band layout. Run: py -3.12 opt/test_expert_equivalence.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opt.expert_moe import ExpertizedArmA
from opt.model_opt import OptArmA
from opt.model_ref import ArmAConfig, canonical_init, load_init, \
    rope_pair_freq, synthetic_packed_batch
from opt.test_resonant_math import oracle_logits

TINY = dict(T=16, V=32, D=8, N=16, H=2, L=2, HIDDEN=8)
MODES = ("single", "two", "four", "heavy", "mixed")


def build_pair(cfg, experts: int, dtype=torch.float32):
    device = torch.device("cpu")
    ref = OptArmA(
        cfg, device, scan_block=cfg.K, use_checkpoint=False, coord="dense",
        single_scan="chunkwise", packed_update="branchfree", zero_carry=True,
        paper_layout="direct", cache_rope=True,
    )
    exp = ExpertizedArmA(cfg, device, experts=experts, scan_block=cfg.K)
    load_init(ref, canonical_init(cfg), device)
    exp.load_canonical(ref.state_dict())
    ref = ref.to(dtype).eval()
    exp = exp.to(dtype).eval()
    return ref, exp


def logits_pair(ref, exp, batch):
    with torch.no_grad():
        a = ref.forward_packed(batch["x"], batch["pos"], batch["segpos"],
                               batch["full_mask"], batch["segment_start"])
        b = exp.forward_packed(batch["x"], batch["pos"], batch["segpos"],
                               batch["full_mask"], batch["segment_start"])
    return a, b


def test_modes_fp32():
    cfg = ArmAConfig(**TINY)
    worst = 0.0
    per_mode = {}
    for experts in (2, 4):
        ref, exp = build_pair(cfg, experts)
        for mode in MODES:
            batch = synthetic_packed_batch(cfg, 2, torch.device("cpu"),
                                           seed=7, mode=mode)
            a, b = logits_pair(ref, exp, batch)
            err = float((a - b).abs().max())
            per_mode[f"M{experts}_{mode}"] = err
            worst = max(worst, err)
    assert worst < 1e-5, per_mode
    return {"worst_abs_logit_error_fp32": worst, "per_mode": per_mode}


def test_fp64_oracle():
    cfg = ArmAConfig(**TINY)
    ref, exp = build_pair(cfg, 4, dtype=torch.float64)
    batch = synthetic_packed_batch(cfg, 2, torch.device("cpu"), seed=3,
                                   mode="mixed")
    with torch.no_grad():
        got_ref, got_exp = logits_pair(ref, exp, batch)
        state = {k: v.double() for k, v in exp.state_dict().items()
                 if k != "rope_freq_bands"}
        got_oracle = oracle_logits(cfg, state, batch)
    err_ref = float((got_ref - got_exp).abs().max())
    err_oracle = float((got_oracle - got_exp).abs().max())
    assert err_ref < 1e-8, err_ref
    assert err_oracle < 1e-7, err_oracle
    return {"expertized_vs_arma_fp64": err_ref,
            "expertized_vs_oracle_fp64": err_oracle}


def test_gradients():
    cfg = ArmAConfig(**TINY)
    ref, exp = build_pair(cfg, 4)
    batch = synthetic_packed_batch(cfg, 2, torch.device("cpu"), seed=11,
                                   mode="mixed")
    targets = torch.randint(0, cfg.V, batch["x"].shape, generator=None)
    valid = torch.ones(batch["x"].shape, dtype=torch.bool)
    worst = 0.0
    for model in (ref, exp):
        model.zero_grad(set_to_none=True)
        logits = model.forward_packed(batch["x"], batch["pos"], batch["segpos"],
                                      batch["full_mask"],
                                      batch["segment_start"])
        loss = F.cross_entropy(logits.reshape(-1, cfg.V),
                               targets.reshape(-1),
                               reduction="none")[valid.reshape(-1)].sum()
        loss.backward()
    for (name, pr), (_, pe) in zip(ref.named_parameters(),
                                   exp.named_parameters()):
        assert pr.grad is not None and pe.grad is not None, name
        diff = float((pr.grad - pe.grad).abs().max())
        worst = max(worst, diff)
        assert diff < 1e-5, (name, diff)
    return {"worst_grad_abs_diff": worst}


def test_oscillator_flags():
    cfg = ArmAConfig(**TINY)
    plain = ExpertizedArmA(cfg, torch.device("cpu"), experts=4, scan_block=cfg.K)
    scaled = ExpertizedArmA(cfg, torch.device("cpu"), experts=4, scan_block=cfg.K,
                            learn_freq_scale=True, learn_band_amp=True)
    load_init(plain, canonical_init(cfg), torch.device("cpu"))
    scaled.load_canonical(plain.state_dict())
    batch = synthetic_packed_batch(cfg, 2, torch.device("cpu"), seed=5,
                                   mode="mixed")
    with torch.no_grad():
        a = plain.forward_packed(batch["x"], batch["pos"], batch["segpos"],
                                 batch["full_mask"], batch["segment_start"])
        b = scaled.forward_packed(batch["x"], batch["pos"], batch["segpos"],
                                  batch["full_mask"], batch["segment_start"])
    identity = float((a - b).abs().max())
    assert identity == 0.0, identity
    with torch.no_grad():
        scaled.beta.fill_(0.1)
        c = scaled.forward_packed(batch["x"], batch["pos"], batch["segpos"],
                                  batch["full_mask"], batch["segment_start"])
    changed = float((a - c).abs().max())
    assert changed > 1e-6, changed
    assert set(scaled.state_dict()) >= {"beta", "gamma"}
    return {"beta0_gamma1_identity": identity, "beta_effect": changed}


def test_band_layout():
    cfg = ArmAConfig(**TINY)
    exp = ExpertizedArmA(cfg, torch.device("cpu"), experts=4, scan_block=cfg.K)
    full = rope_pair_freq(cfg, torch.device("cpu"))
    for expert in range(4):
        want = full[expert * (cfg.K // 8):(expert + 1) * (cfg.K // 8)]
        got = exp.rope_freq_bands[expert]
        assert torch.allclose(got, want), expert
    ledger = exp.parameter_ledger(top_r=1)
    assert ledger["stored_K_per_head"] == cfg.K
    assert ledger["active_K_per_head"] == cfg.K // 4
    assert ledger["active_fraction"] == 0.25
    return {"band_slices": 4, "ledger": ledger}


def main():
    checks = {
        "modes_fp32": test_modes_fp32(),
        "fp64_oracle": test_fp64_oracle(),
        "gradients": test_gradients(),
        "oscillator_flags": test_oscillator_flags(),
        "band_layout": test_band_layout(),
    }
    for name, result in checks.items():
        print(f"PASS {name}: {json.dumps(result, default=str)[:400]}")
    print("EXPERT_EQUIVALENCE_ALL_PASS=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
