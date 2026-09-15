"""Targeted gate for the branch-free packed state update (ARM-A III, P1).

Proves on randomized documented-boundary layouts:
  1. scan_chunkwise_bthk (branch-free)  == dense oracle (fwd + q/v grads)
  2. scan_chunkwise_where_bthk (certified control) == dense oracle
  3. branch-free == certified control (direct A/B, all outputs and grads)
  4. model-level: OptArmA(branchfree) == OptArmA(where) == canonical
     for logits/loss/all parameter grads, packed + single.

Boundary cases are constructed explicitly: document starts inside a block,
document continues across a block boundary, reset+continue inside one block,
single-token docs, and random layouts.
"""

import random
import sys

import torch

sys.path.insert(0, ".")

from opt.model_opt import OptArmA
from opt.model_ref import (
    ArmAConfig,
    NativeReadStage1ArmA,
    canonical_init,
    load_init,
)
from opt.scan_attn import (
    dense_segstart_attention,
    scan_chunkwise_bthk,
    scan_chunkwise_where_bthk,
)

RTOL = ATOL = 1e-10


def random_packed_layouts(rng, b, t):
    layouts = []
    for _ in range(6):
        row = torch.zeros((b, t), dtype=torch.long)
        for i in range(b):
            cuts = sorted(rng.sample(range(1, t), k=rng.randrange(max(1, t // 2))))
            s, col = 0, []
            for c in cuts + [t]:
                col += [s] * (c - s)
                s = c
            row[i] = torch.tensor(col)
        layouts.append(row)
    return layouts


def boundary_layouts(t, block):
    """Explicit worst-case boundary patterns inside one block and across."""
    out = []
    # doc starts exactly mid-block and continues across the next boundary
    seg = torch.zeros((1, t), dtype=torch.long)
    seg[0, block // 2:] = block // 2
    out.append(seg)
    # doc reset one token after a block boundary, then continues
    seg = torch.zeros((1, t), dtype=torch.long)
    seg[0, block + 1:] = block + 1
    out.append(seg)
    # reset, single-token doc, reset again (strict-past sparse)
    seg = torch.zeros((1, t), dtype=torch.long)
    seg[0, block - 2:block] = block - 2
    seg[0, block:] = block
    seg[0, block + 1:] = block + 1
    out.append(seg)
    # every token its own doc (all resets)
    out.append(torch.arange(t, dtype=torch.long).unsqueeze(0))
    return out


def scan_ab(b, h, t, k, dv, seg, seed):
    g = torch.Generator().manual_seed(seed)
    q0 = torch.randn((b, h, t, k), generator=g, dtype=torch.float64)
    v0 = torch.randn((b, t, dv), generator=g, dtype=torch.float64)
    probe = torch.randn((b, h, t, dv), generator=g, dtype=torch.float64)

    def run(fn, blk, **kw):
        q = q0.clone().requires_grad_(True)
        v = v0.clone().requires_grad_(True)
        vh = v.unsqueeze(1).expand(-1, h, -1, -1)
        out = fn(q, vh, seg, block=blk, **kw)
        gq, gv = torch.autograd.grad((out * probe).sum(), (q, v))
        return out.detach(), gq, gv

    def run_dense():
        q = q0.clone().requires_grad_(True)
        v = v0.clone().requires_grad_(True)
        out = dense_segstart_attention(q, v, seg)
        gq, gv = torch.autograd.grad((out * probe).sum(), (q, v))
        return out.detach(), gq, gv

    ref = run_dense()
    block = t // 2 if t >= 2 else 1
    branch = run(scan_chunkwise_bthk, block)
    where = run(scan_chunkwise_where_bthk, block)
    zc = run(scan_chunkwise_bthk, block, skip_zero_carry=True)
    errs = {}
    for tag, a, c in zip(("fwd", "gq", "gv"), branch, ref):
        errs[f"bf_vs_dense_{tag}"] = float((a - c).abs().max())
        torch.testing.assert_close(a, c, rtol=RTOL, atol=ATOL)
    for tag, a, c in zip(("fwd", "gq", "gv"), where, ref):
        errs[f"where_vs_dense_{tag}"] = float((a - c).abs().max())
        torch.testing.assert_close(a, c, rtol=RTOL, atol=ATOL)
    for tag, a, c in zip(("fwd", "gq", "gv"), branch, where):
        errs[f"bf_vs_where_{tag}"] = float((a - c).abs().max())
        torch.testing.assert_close(a, c, rtol=RTOL, atol=ATOL)
    for tag, a, c in zip(("fwd", "gq", "gv"), zc, ref):
        errs[f"zc_vs_dense_{tag}"] = float((a - c).abs().max())
        torch.testing.assert_close(a, c, rtol=RTOL, atol=ATOL)
    return errs


def model_ab(cfg, seed):
    from opt.test_model_equiv import packed_batch, run_grads
    batch = packed_batch(cfg, 2, seed, single=False)
    ref = NativeReadStage1ArmA(cfg, torch.device("cpu"))
    init = canonical_init(cfg)
    load_init(ref, init, torch.device("cpu"))
    ref.train()
    r_logits, r_loss, r_grads = run_grads(ref, batch, cfg)
    errs = {}
    outs = {}
    for name, kw in (("branchfree", {}), ("where", {"packed_update": "where"}),
                     ("zero_carry", {"zero_carry": True}),
                     ("paper_direct", {"paper_layout": "direct"}),
                     ("rope_cache", {"cache_rope": True}),
                     ("dirrope", {"paper_layout": "direct", "cache_rope": True})):
        m = OptArmA(cfg, torch.device("cpu"), scan_block=cfg.T // 2,
                    use_checkpoint=False, coord="dense",
                    single_scan="chunkwise", **kw)
        load_init(m, init, torch.device("cpu"))
        m.train()
        o_logits, o_loss, o_grads = run_grads(m, batch, cfg)
        outs[name] = (o_logits, o_loss, o_grads)
        d = float((o_logits - r_logits).abs().max())
        dg = max(float((o_grads[k] - r_grads[k]).abs().max()) for k in r_grads)
        errs[f"{name}_vs_canonical"] = max(d, dg)
        torch.testing.assert_close(o_logits, r_logits, rtol=1e-4, atol=1e-4)
        for k in r_grads:
            torch.testing.assert_close(o_grads[k], r_grads[k], rtol=1e-3, atol=1e-3)
    for tag, a, c in zip(("logits", "loss", "grads"),
                         (outs["branchfree"][0], outs["branchfree"][1], None),
                         (outs["where"][0], outs["where"][1], None)):
        if a is not None:
            errs[f"model_bf_vs_where_{tag}"] = float((a - c).abs().max())
            torch.testing.assert_close(a, c, rtol=1e-10, atol=1e-10)
    gmax = max(float((outs["branchfree"][2][k] - outs["where"][2][k]).abs().max())
               for k in r_grads)
    errs["model_bf_vs_where_grads"] = gmax
    torch.testing.assert_close(gmax, 0.0, rtol=0, atol=1e-11)
    return errs


def main():
    rng = random.Random(0)
    cfg = ArmAConfig(T=32, V=64, D=16, N=64, H=2, L=2, HIDDEN=32)
    worst = {}
    n = 0

    # scan-level: explicit boundaries at tiny scale (block=T/2 gives many)
    for t in (8, 16, 32):
        b, h, k, dv = 2, 2, 4, 3
        cases = boundary_layouts(t, block=max(1, t // 2))
        cases += random_packed_layouts(rng, b, t)
        for i, seg in enumerate(cases):
            if seg.shape[0] != b:
                seg = seg.expand(b, -1).contiguous()
            e = scan_ab(b, h, t, k, dv, seg.clone(), seed=100 + i)
            for kk, vv in e.items():
                worst[kk] = max(worst.get(kk, 0.0), vv)
            n += 1

    # model-level: packed, block=T//2
    for seed in (0, 1):
        e = model_ab(cfg, seed)
        for kk, vv in e.items():
            worst[kk] = max(worst.get(kk, 0.0), vv)
        n += 1

    print({"status": "PACKED_AB_GATE_PASS", "cases": n})
    for kk in sorted(worst):
        print(f"  worst {kk}: {worst[kk]:.3e}")


if __name__ == "__main__":
    main()
