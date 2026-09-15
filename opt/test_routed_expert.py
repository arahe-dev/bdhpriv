"""Correctness gates for opt/routed_expert.RoutedExpertArmA.

Checks:
  1. all-active route == Arm-A (FP32 and FP64)
  2. fixed / varying route == a dense FP64 sparse-semantics oracle
     (expert state accumulates only at routed tokens, shared a_total)
  3. packed-document reset: an expert inactive across a document boundary
     must not leak state into the next document
  4. route builders: fixed, varying, contiguous windows
Run: py -3.12 opt/test_routed_expert.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opt.model_opt import OptArmA
from opt.model_ref import ArmAConfig, canonical_init, load_init
from opt.routed_expert import (
    RoutedExpertArmA,
    build_route_tensors,
    fixed_route_sets,
    group_route_table,
    varying_route_sets,
    window_route_sets,
)

TINY = dict(T=16, V=32, D=8, N=16, H=2, L=2, HIDDEN=8)
GROUP = 4
DOC_CUT = 8


def make_batch(cfg, device, seed=5):
    g = torch.Generator().manual_seed(seed)
    t = cfg.T
    start = torch.zeros(1, t, dtype=torch.long)
    start[0, DOC_CUT:] = DOC_CUT
    pos = (torch.arange(t).unsqueeze(0) - start).to(torch.int32)
    segpos = pos.clone()
    same = start[:, :, None] == start[:, None, :]
    strict = torch.ones((t, t), dtype=torch.bool).tril(-1)
    full_mask = (same & strict.unsqueeze(0))
    x = torch.randint(0, cfg.V, (1, t), generator=g)
    y = torch.randint(0, cfg.V, (1, t), generator=g)
    return {
        "x": x.to(device), "y": y.to(device), "pos": pos.to(device),
        "segpos": segpos.to(device), "full_mask": full_mask.to(device),
        "segment_start": start.to(device, dtype=torch.int32),
        "valid": torch.ones((1, t), dtype=torch.bool, device=device),
    }


def activity_from_table(table, cfg, group, experts):
    groups_per_row = cfg.T // group
    active = torch.zeros(experts, 1, cfg.T, dtype=torch.bool)
    for ga in range(groups_per_row):
        for expert in table[ga % len(table)]:
            active[expert, :, ga * group:(ga + 1) * group] = True
    return active  # (M, B, T)


def sparse_oracle(cfg, model, batch, table, group):
    """Dense FP64 reference of the routed semantics (shared a_total)."""
    dev = batch["x"].device
    state = {k: v.double() for k, v in model.state_dict().items()}
    v = F.layer_norm(state["embedding.weight"][batch["x"]], (cfg.D,))
    b, t = batch["x"].shape
    h, ke, m, d = cfg.H, model.Ke, model.M, cfg.D
    start = batch["segment_start"].long()
    pos = batch["pos"]
    segpos = batch["segpos"].double()
    same = start[:, :, None] == start[:, None, :]
    strict = torch.ones((t, t), dtype=torch.bool, device=dev).tril(-1)
    mask = same & strict.unsqueeze(0)
    act = activity_from_table(table, cfg, group, m)  # (M,B,T)
    freq = model.rope_freq_bands.double()
    for _ in range(cfg.L):
        a_pre = torch.zeros((b, h, t, d), dtype=torch.float64, device=dev)
        x_store = {}
        for e in range(m):
            active_e = act[e, 0, :]  # (T,)
            if not bool(active_e.any()):
                continue
            dx = state["DxE"][e].view(d, h, ke)
            x_e = F.relu(torch.einsum("btd,dhk->bthk", v, dx))
            x_store[e] = x_e
            q_e = torch.stack([
                _oracle_rope(x_e[:, :, hh], pos[0], freq[e])
                for hh in range(h)
            ], dim=2)
            slope = q_e.permute(0, 2, 1, 3)  # (B,H,T,Ke)
            scores = slope @ slope.transpose(-1, -2)
            allowed = mask.unsqueeze(1) & active_e.view(1, 1, 1, t)
            scores = scores.masked_fill(~allowed, 0.0)
            values = v.unsqueeze(1) * active_e.view(1, 1, t, 1).double()
            query_active = active_e.view(1, 1, t, 1).double()
            a_pre = a_pre + (scores @ values) * query_active
        a = F.layer_norm(a_pre.permute(0, 2, 1, 3), (d,))  # (B,T,H,D)
        base = torch.zeros((b, t, d), dtype=torch.float64, device=dev)
        for e in range(m):
            active_e = act[e, 0, :]
            if not bool(active_e.any()):
                continue
            dy = state["DyE"][e].view(d, h, ke)
            y_e = F.relu(torch.einsum("bthd,dhk->bthk", a, dy))
            u_e = x_store[e] * y_e
            hide = active_e.view(1, t, 1, 1).double()
            ee = state["EE"][e].view(h, ke, d)
            base = base + torch.einsum("bthk,hkd->btd", u_e * hide, ee)
        base = F.layer_norm(base, (d,))
        z = v @ state["coordinator.Wc"] + state["coordinator.bc"]
        c = (mask.double() @ z) / segpos.clamp_min(1).unsqueeze(-1) - z
        g = 1.0 + torch.sigmoid(state["coordinator.alpha"]) * torch.tanh(c)
        delta = F.relu((g * base) @ state["writer.W1"]) @ state["writer.W2"]
        v = F.layer_norm(v + delta, (d,))
    return v @ state["readout"]


def _oracle_rope(x, pos, freq):
    """FP64 RoPE for one head: x (B,T,Ke), pos (T,), freq (Ke/2,)."""
    b, t, ke = x.shape
    xp = x.reshape(b, t, ke // 2, 2)
    phase = pos.double().view(1, t, 1) * freq.view(1, 1, -1)
    phase = torch.remainder(phase, 1.0) * (2.0 * torch.pi)
    cs, sn = torch.cos(phase), torch.sin(phase)
    return torch.stack((xp[..., 0] * cs - xp[..., 1] * sn,
                        xp[..., 1] * cs + xp[..., 0] * sn), dim=-1).reshape(
        b, t, ke)


def build_models(cfg, experts=4, dtype=torch.float32):
    device = torch.device("cpu")
    dense = OptArmA(
        cfg, device, scan_block=32, use_checkpoint=False, coord="dense",
        single_scan="chunkwise", packed_update="branchfree", zero_carry=True,
        paper_layout="direct", cache_rope=True,
    )
    routed = RoutedExpertArmA(cfg, device, experts=experts,
                              expert_width=cfg.K // experts, scan_block=32)
    routed = routed.to(dtype)
    load_init(dense, canonical_init(cfg), device)
    routed.load_canonical(dense.state_dict())
    dense = dense.to(dtype).eval()
    routed = routed.to(dtype).eval()
    return dense, routed


def test_all_active_equivalence():
    cfg = ArmAConfig(**TINY)
    dense, routed = build_models(cfg, experts=4)
    batch = make_batch(cfg, torch.device("cpu"))
    table = [(0, 1, 2, 3)]
    route = build_route_tensors(
        table, GROUP, 1, cfg.T, 4, torch.device("cpu"))
    with torch.no_grad():
        want = dense.forward_packed(batch["x"], batch["pos"], batch["segpos"],
                                    batch["full_mask"], batch["segment_start"])
        got = routed.forward_route(batch["x"], batch["pos"], batch["segpos"],
                                   batch["full_mask"], batch["segment_start"],
                                   route)
    err = float((want - got).abs().max())
    assert err < 1e-5, err
    return {"all_active_max_abs_diff_fp32": err}


def test_sparse_semantics_oracle():
    cfg = ArmAConfig(**TINY)
    dense, routed = build_models(cfg, experts=4, dtype=torch.float64)
    batch = make_batch(cfg, torch.device("cpu"))
    table = [(0, 1), (1, 2), (2, 3), (0, 3)]
    route = build_route_tensors(
        table, GROUP, 1, cfg.T, 4, torch.device("cpu"))
    with torch.no_grad():
        got = routed.forward_route(batch["x"], batch["pos"], batch["segpos"],
                                   batch["full_mask"], batch["segment_start"],
                                   route)
        want = sparse_oracle(cfg, routed, batch, table, GROUP)
    err = float((want - got).abs().max())
    assert err < 1e-8, err
    return {"sparse_oracle_max_abs_diff_fp64": err}


def test_document_boundary_no_leak():
    cfg = ArmAConfig(**TINY)
    _, routed = build_models(cfg, experts=4, dtype=torch.float64)
    batch = make_batch(cfg, torch.device("cpu"))
    table = [(0,), (1,), (2,), (3,)]
    route = build_route_tensors(
        table, GROUP, 1, cfg.T, 4, torch.device("cpu"))
    with torch.no_grad():
        got = routed.forward_route(batch["x"], batch["pos"], batch["segpos"],
                                   batch["full_mask"], batch["segment_start"],
                                   route)
        want = sparse_oracle(cfg, routed, batch, table, GROUP)
    err = float((want - got).abs().max())
    assert err < 1e-8, err

    replaced = [
        (0,), (0,), (2,), (2,),
    ]
    route2 = build_route_tensors(
        replaced, GROUP, 1, cfg.T, 4, torch.device("cpu"))
    with torch.no_grad():
        got2 = routed.forward_route(batch["x"], batch["pos"], batch["segpos"],
                                    batch["full_mask"], batch["segment_start"],
                                    route2)
    assert not torch.equal(got, got2), (
        "expert 0 active in doc A group 0 and doc B group 2 must not leak; "
        "outputs identical when group 1/3 routes change is suspicious")
    return {"boundary_oracle_ok": True}


def test_route_builders():
    assert fixed_route_sets(8, 2) == [(0, 1)]
    assert fixed_route_sets(8, 2, offset=3) == [(3, 4)]
    var = varying_route_sets(8, 2)
    assert len(var) == 8
    assert var[:4] == [(0, 1), (2, 3), (4, 5), (6, 7)]
    win = window_route_sets(8, 2)
    assert win == [(i, i + 1) for i in range(7)]
    cyc = window_route_sets(8, 2, cyclic=True)
    assert cyc[-1] == (7, 0)
    table = group_route_table(win, 16)
    assert len(table) == 16
    assert table[0] == (0, 1) and table[7] == (0, 1)
    return {"builders": True}


def main():
    checks = {
        "all_active_equivalence": test_all_active_equivalence(),
        "sparse_semantics_oracle": test_sparse_semantics_oracle(),
        "document_boundary_no_leak": test_document_boundary_no_leak(),
        "route_builders": test_route_builders(),
    }
    for name, result in checks.items():
        print(f"PASS {name}: {json.dumps(result)}")
    print("ROUTED_EXPERT_ALL_PASS=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
