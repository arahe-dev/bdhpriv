"""Campaign 'FIRST: reconstruct the math' checks + census zero-change test.

Validates against the frozen micro-semantics of Arm-A:
  1. an independent FP64 dense oracle reproduces the repo model numerically,
  2. the RoPE oscillator identity q_t^T q_s = x_t^T R((p_s-p_t)w) x_s,
  3. q support = pair-completion of x support (RoPE mixes within pairs),
  4. exact skips: y outside supp(x) is irrelevant; E only sees supp(x*y);
     state updates/reads for zero q coordinates are unnecessary,
  5. CensusArmA + SparsityAccumulator change no outputs.

Run: py -3.12 opt/test_resonant_math.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opt.census_model import CensusArmA, SparsityAccumulator
from opt.model_opt import OptArmA
from opt.model_ref import ArmAConfig, canonical_init, load_init, \
    synthetic_packed_batch

OUT = Path(__file__).resolve().parents[1] / "results" / "resonant_math_checks.json"
TINY = dict(T=16, V=32, D=8, N=16, H=2, L=2, HIDDEN=8)


def build_tiny():
    cfg = ArmAConfig(**TINY)
    batch = synthetic_packed_batch(cfg, 2, torch.device("cpu"), seed=3,
                                   mode="mixed")
    return cfg, batch


def rope_double(x, pos, freq):
    b, t, h, k = x.shape
    xp = x.reshape(b, t, h, k // 2, 2)
    phase = pos.double().unsqueeze(-1).unsqueeze(-1) * freq.view(1, 1, 1, -1)
    phase = torch.remainder(phase, 1.0) * (2.0 * math.pi)
    cs, sn = torch.cos(phase), torch.sin(phase)
    return torch.stack((xp[..., 0] * cs - xp[..., 1] * sn,
                        xp[..., 1] * cs + xp[..., 0] * sn),
                       dim=-1).reshape_as(x)


def oracle_logits(cfg, state, batch):
    def ln(t):
        return F.layer_norm(t, (t.shape[-1],))

    emb = state["embedding.weight"][batch["x"]]
    v = ln(emb)
    b, t = batch["x"].shape
    start = batch["segment_start"].long()
    pos = batch["pos"]
    segpos = batch["segpos"].double()
    mask = (start[:, :, None] == start[:, None, :]) & torch.ones(
        (t, t), dtype=torch.bool
    ).tril(-1)
    freq = (
        1.0 / (cfg.THETA ** ((2.0 * torch.arange(cfg.K // 2,
                                                 dtype=torch.float64)) / cfg.K))
        / (2.0 * math.pi)
    )
    wdx = state["decoder_x"].permute(1, 0, 2).reshape(cfg.D, cfg.N)
    for _ in range(cfg.L):
        x = F.relu(v.reshape(b * t, cfg.D) @ wdx).reshape(
            b, t, cfg.H, cfg.K)
        q = rope_double(x, pos, freq)
        scores = q.permute(0, 2, 1, 3) @ q.permute(0, 2, 1, 3).transpose(-1, -2)
        scores = scores.masked_fill(~mask.unsqueeze(1), 0.0)
        a = ln(scores @ v.unsqueeze(1).expand(-1, cfg.H, -1, -1))
        y = F.relu(a @ state["decoder_y"])
        u = (x.permute(0, 2, 1, 3) * y).transpose(1, 2).reshape(b, t, cfg.N)
        base = ln(u @ state["encoder"])
        z = v @ state["coordinator.Wc"] + state["coordinator.bc"]
        c = (mask.double() @ z) / segpos.clamp_min(1).unsqueeze(-1) - z
        g = 1.0 + torch.sigmoid(state["coordinator.alpha"]) * torch.tanh(c)
        delta = F.relu((g * base) @ state["writer.W1"]) @ state["writer.W2"]
        v = ln(v + delta)
    return v @ state["readout"]


def check_oracle():
    cfg, batch = build_tiny()
    model = OptArmA(
        cfg, torch.device("cpu"), scan_block=cfg.K, use_checkpoint=False,
        coord="dense", single_scan="chunkwise", packed_update="branchfree",
        zero_carry=True, paper_layout="direct", cache_rope=True,
    )
    load_init(model, canonical_init(cfg), torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        got = model.forward_packed(
            batch["x"], batch["pos"], batch["segpos"], batch["full_mask"],
            batch["segment_start"],
        )
        state = {k: v.double() for k, v in model.state_dict().items()}
        want = oracle_logits(cfg, state, batch)
    err = float((got.double() - want).abs().max())
    return {"max_abs_logit_error_fp64": err, "pass": err < 1e-5}


def check_rope_identity():
    cfg, _ = build_tiny()
    g = torch.Generator().manual_seed(11)
    pairs = cfg.K // 2
    x = torch.randn(12, cfg.K, generator=g, dtype=torch.float64)
    pos = torch.arange(12, dtype=torch.float64).unsqueeze(1)
    freq = (
        1.0 / (cfg.THETA ** ((2.0 * torch.arange(pairs,
                                                 dtype=torch.float64)) / cfg.K))
        / (2.0 * math.pi)
    )
    q = rope_double(x.unsqueeze(1).unsqueeze(1), pos,
                    freq).squeeze(1).squeeze(1)
    err = 0.0
    for i in range(0, 12, 3):
        for j in range(i + 1, 12, 4):
            delta = float(pos[j, 0] - pos[i, 0])
            w = 2.0 * math.pi * freq
            cs, sn = torch.cos(delta * w), torch.sin(delta * w)
            lhs = float(q[i] @ q[j])
            xp = x.reshape(12, pairs, 2)
            xt, xs = xp[i], xp[j]
            rot = torch.stack(
                (xs[:, 0] * cs - xs[:, 1] * sn,
                 xs[:, 0] * sn + xs[:, 1] * cs), dim=-1
            )
            rhs = float((xt * rot).sum())
            err = max(err, abs(lhs - rhs))
    return {"max_abs_identity_error_fp64": err, "pass": err < 1e-10}


def check_pair_support():
    cfg, _ = build_tiny()
    g = torch.Generator().manual_seed(12)
    x = (torch.randn(8, cfg.K, generator=g) > 0).float()
    x[torch.rand_like(x) < 0.6] = 0.0
    pos = torch.arange(8).unsqueeze(1).float()
    freq = (
        1.0 / (cfg.THETA ** ((2.0 * torch.arange(cfg.K // 2,
                                                 dtype=torch.float32)) / cfg.K))
        / (2.0 * math.pi)
    )
    q = rope_double(x.unsqueeze(1).unsqueeze(1), pos,
                    freq.double()).squeeze(1).squeeze(1)
    xp = x.reshape(8, cfg.K // 2, 2)
    qp = q.reshape(8, cfg.K // 2, 2)
    x_pair_zero = (xp == 0).all(-1)
    q_pair_zero = (qp == 0).all(-1)
    mismatch = int((x_pair_zero != q_pair_zero).sum())
    coordinate_inflation = int((qp != 0).sum()) - int((xp != 0).sum())
    return {
        "pair_zero_mismatches": mismatch,
        "q_coordinate_support_inflation": coordinate_inflation,
        "pass": mismatch == 0,
    }


def check_exact_skips():
    cfg, _ = build_tiny()
    g = torch.Generator().manual_seed(13)
    b, t, h, k, d = 2, cfg.T, cfg.H, cfg.K, cfg.D
    x = torch.relu(torch.randn(b, t, h, k, generator=g))
    x[torch.rand_like(x) < 0.7] = 0.0
    a = torch.randn(b, t, h, d, generator=g)
    dy = torch.randn(h, d, k, generator=g)
    e = torch.randn(cfg.N, d, generator=g)
    y = F.relu(torch.einsum("bthd,hdk->bthk", a, dy))
    u = x * y
    y_masked = y * (x > 0)
    u_masked = x * y_masked
    u_diff = float((u - u_masked).abs().max())
    base = u.reshape(b, t, cfg.N) @ e
    base_masked = (u * (x > 0)).reshape(b, t, cfg.N) @ e
    e_diff = float((base - base_masked).abs().max())

    q = torch.randn(t, k, generator=g)
    q[torch.rand_like(q) < 0.8] = 0.0
    v = torch.randn(t, d, generator=g)
    state = torch.einsum("tk,td->kd", q, v)
    dense = q @ state
    sparse = torch.zeros_like(dense)
    active = (q != 0).any(0)
    state_sparse = torch.zeros_like(state)
    state_sparse[active] = torch.einsum("tk,td->kd", q[:, active], v)
    sparse = q @ state_sparse
    state_diff = float((dense - sparse).abs().max())
    return {
        "u_irrelevant_outside_supp_x_max_diff": u_diff,
        "e_support_only_max_diff": e_diff,
        "q_zero_state_skip_max_diff": state_diff,
        "pass": max(u_diff, e_diff, state_diff) == 0.0,
    }


def check_census_zero_change_and_metrics():
    cfg, batch = build_tiny()
    census = CensusArmA(cfg, torch.device("cpu"), scan_block=cfg.K)
    plain = OptArmA(
        cfg, torch.device("cpu"), scan_block=cfg.K, use_checkpoint=False,
        coord="dense", single_scan="chunkwise", packed_update="branchfree",
        zero_carry=True, paper_layout="direct", cache_rope=True,
    )
    init = canonical_init(cfg)
    load_init(census, init, torch.device("cpu"))
    load_init(plain, init, torch.device("cpu"))
    census.eval()
    plain.eval()
    acc = SparsityAccumulator(cfg, sample_rows=8, block_widths=(2, 4),
                              bands=4, quantile_stride=2)
    with torch.no_grad():
        gen = torch.Generator().manual_seed(0)
        acc.new_batch(batch["x"].shape[0], gen)
        census.begin_forward(
            lambda level, xx, yy, uu, ss: acc.add(level, xx, yy, uu, ss)
        )
        got = census.forward_packed(
            batch["x"], batch["pos"], batch["segpos"], batch["full_mask"],
            batch["segment_start"],
        )
        want = plain.forward_packed(
            batch["x"], batch["pos"], batch["segpos"], batch["full_mask"],
            batch["segment_start"],
        )
    exact = torch.equal(got, want)
    census_out = acc.finalize()
    return {
        "output_bitwise_equal": bool(exact),
        "levels_captured": len(census_out["levels"]),
        "sample_rows": census_out["sample_rows"],
        "pass": bool(exact) and len(census_out["levels"]) == cfg.L,
    }


def main():
    checks = {
        "fp64_dense_oracle": check_oracle(),
        "rope_oscillator_identity": check_rope_identity(),
        "q_pair_support": check_pair_support(),
        "exact_skips": check_exact_skips(),
        "census_zero_change": check_census_zero_change_and_metrics(),
    }
    passed = all(v["pass"] for v in checks.values())
    report = {"checks": checks, "all_pass": passed}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, result in checks.items():
        print(f"{'PASS' if result['pass'] else 'FAIL'} {name}: "
              f"{json.dumps({k: v for k, v in result.items() if k != 'pass'})}")
    print("RESONANT_MATH_ALL_PASS=" + str(passed).lower())
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
