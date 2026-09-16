"""Learned-router correctness gates R0-R6 (CPU tiny).

R0 constant logits == fixed deterministic routing
R1 each of the M cyclic windows selectable
R2/R3 mixed per-group windows == equivalent fixed packed route
R4 forward/backward determinism
R5 router grads nonzero; never-selected experts get zero grads
R6 straight-through forward is exactly hard; gradient equals soft surrogate
Run: py -3.12 opt/test_learned_router.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opt.learned_router import (
    LearnedRoutedExpertArmA,
    build_learned_route,
    straight_through,
)
from opt.model_opt import OptArmA
from opt.model_ref import ArmAConfig, canonical_init, load_init
from opt.routed_expert import build_route_tensors, group_route_table
from opt.test_routed_expert import make_batch

TINY = dict(T=16, V=32, D=8, N=16, H=2, L=2, HIDDEN=8)
GROUP = 4
M = 4
KE = 2


def build(cfg, dtype=torch.float32):
    device = torch.device("cpu")
    dense = OptArmA(cfg, device, scan_block=cfg.K, use_checkpoint=False,
                    coord="dense", single_scan="chunkwise",
                    packed_update="branchfree", zero_carry=True,
                    paper_layout="direct", cache_rope=True)
    load_init(dense, canonical_init(cfg), device)
    model = LearnedRoutedExpertArmA(cfg, device, experts=M, expert_width=KE,
                                    scan_block=cfg.K, top_r=2,
                                    route_group=GROUP)
    model.load_canonical(dense.state_dict())
    model.capacity_override = cfg.T
    return dense.to(dtype).eval(), model.to(dtype).eval()


def route_summary(model, v):
    capacity = model.capacity_tokens(v.shape[0], v.shape[1])
    route, overflow = build_learned_route(model, v, GROUP, capacity)
    return route, overflow


def test_r0_constant():
    cfg = ArmAConfig(**TINY)
    dense, model = build(cfg, dtype=torch.float64)
    batch = make_batch(cfg, torch.device("cpu"))
    with torch.no_grad():
        model.router.Wr.zero_()
        model.router.br.zero_()
        model.router.br[0] = 5.0
        v = model.ln(model.embedding(batch["x"]))
        route, overflow = route_summary(model, v)
        got = model.forward_learned(batch["x"], batch["pos"], batch["segpos"],
                                    batch["full_mask"],
                                    batch["segment_start"])
        table = group_route_table([(0, 1)], cfg.T // GROUP)
        fixed = build_route_tensors(table, GROUP, 1, cfg.T, M,
                                    torch.device("cpu"))
        import opt.learned_router as lr
        fwd = model.forward_route(batch["x"], batch["pos"], batch["segpos"],
                                  batch["full_mask"], batch["segment_start"],
                                  fixed)
    err = float((got - fwd).abs().max())
    selected_per_token = route.sel_mask.sum(0)
    assert overflow == 0, overflow
    assert err < 1e-8, err
    assert float(selected_per_token.min()) == 2.0, selected_per_token
    return {"max_diff_vs_fixed": err, "overflow": overflow,
            "selected_per_token": float(selected_per_token.float().mean())}


def test_r1_all_windows():
    cfg = ArmAConfig(**TINY)
    _, model = build(cfg)
    batch = make_batch(cfg, torch.device("cpu"))
    with torch.no_grad():
        model.router.Wr.zero_()
        v = model.ln(model.embedding(batch["x"]))
        chosen = []
        for window in range(M):
            model.router.br.zero_()
            model.router.br[window] = 5.0
            _, _, idx = straight_through(model.router.logits_for(v, GROUP,
                                                                cfg.T // GROUP))
            assert bool((idx == window).all()), (window, idx)
            chosen.append(int(idx[0, 0]))
    return {"windows": chosen}


def test_r23_mixed_groups():
    cfg = ArmAConfig(**TINY)
    _, model = build(cfg, dtype=torch.float64)
    batch = make_batch(cfg, torch.device("cpu"))
    groups = cfg.T // GROUP
    target = torch.arange(groups) % M
    explicit = torch.full((1, groups, M), -5.0, dtype=torch.float64)
    explicit[0, torch.arange(groups), target] = 5.0
    with torch.no_grad():
        v = model.ln(model.embedding(batch["x"]))
        route, overflow = build_learned_route(
            model, v, GROUP, model.capacity_override, logits=explicit)
        got = model.forward_route(batch["x"], batch["pos"], batch["segpos"],
                                  batch["full_mask"],
                                  batch["segment_start"], route)
        table = [(int(target[ga]), (int(target[ga]) + 1) % M)
                 for ga in range(groups)]
        fixed = build_route_tensors(table, GROUP, 1, cfg.T, M,
                                    torch.device("cpu"))
        want = model.forward_route(batch["x"], batch["pos"], batch["segpos"],
                                   batch["full_mask"],
                                   batch["segment_start"], fixed)
    err = float((got - want).abs().max())
    assert int(torch.unique(target).numel()) == M
    assert err < 1e-8, err
    return {"distinct_windows": int(torch.unique(target).numel()),
            "max_diff": err, "overflow": overflow,
            "capacity": int(route.sel_idx.shape[1])}


def test_r4_determinism():
    cfg = ArmAConfig(**TINY)
    _, model = build(cfg)
    model.train()
    batch = make_batch(cfg, torch.device("cpu"))
    losses = []
    grads = []
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        logits = model.forward_learned(batch["x"], batch["pos"],
                                       batch["segpos"], batch["full_mask"],
                                       batch["segment_start"])
        loss = F.cross_entropy(logits.reshape(-1, cfg.V),
                               batch["y"].reshape(-1))
        loss.backward()
        losses.append(float(loss.detach()))
        grads.append(model.router.Wr.grad.detach().clone())
    assert losses[0] == losses[1], losses
    assert torch.equal(grads[0], grads[1])
    assert float(grads[0].abs().max()) > 0.0
    return {"loss": losses[0], "router_grad_norm": float(grads[0].norm())}


def test_r5_grad_routing():
    cfg = ArmAConfig(**TINY)
    _, model = build(cfg)
    model.train()
    batch = make_batch(cfg, torch.device("cpu"))
    with torch.no_grad():
        model.router.Wr.zero_()
        model.router.br.zero_()
        model.router.br[0] = 5.0
    model.zero_grad(set_to_none=True)
    logits = model.forward_learned(batch["x"], batch["pos"], batch["segpos"],
                                   batch["full_mask"], batch["segment_start"])
    loss = F.cross_entropy(logits.reshape(-1, cfg.V),
                           batch["y"].reshape(-1))
    loss.backward()
    router_norm = float(model.router.Wr.grad.norm())
    grad_norms = {}
    for expert in range(M):
        grad = model.DxE.grad[expert]
        grad_norms[expert] = float(grad.norm())
    assert router_norm > 0.0, router_norm
    assert grad_norms[0] > 0 and grad_norms[1] > 0, grad_norms
    assert grad_norms[2] == 0.0 and grad_norms[3] == 0.0, grad_norms
    return {"router_grad_norm": router_norm, "expert_grad_norms": grad_norms}


def test_r6_straight_through():
    torch.manual_seed(3)
    logits = torch.randn(5, M, dtype=torch.float64, requires_grad=True)
    w = torch.randn(M, dtype=torch.float64)
    p = torch.softmax(logits, dim=-1)
    h, g_st, idx = straight_through(logits)
    hard = (h * w).sum()
    proxy = (g_st * w).sum()
    assert torch.equal(hard, proxy), (hard.item(), proxy.item())
    proxy.backward()
    expected = ((torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2))
                @ w.unsqueeze(-1)).squeeze(-1)
    err = float((logits.grad - expected).abs().max())
    assert err < 1e-12, err
    return {"forward_exact_hard": True,
            "grad_vs_soft_surrogate_max_err": err}


def main():
    checks = {
        "R0_constant_route": test_r0_constant(),
        "R1_all_windows": test_r1_all_windows(),
        "R2R3_mixed_groups": test_r23_mixed_groups(),
        "R4_determinism": test_r4_determinism(),
        "R5_grad_routing": test_r5_grad_routing(),
        "R6_straight_through": test_r6_straight_through(),
    }
    for name, result in checks.items():
        print(f"PASS {name}: {json.dumps(result)}")
    print("LEARNED_ROUTER_ALL_PASS=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
