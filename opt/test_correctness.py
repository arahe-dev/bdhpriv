"""Systematic randomized exactness tests: scan/coordinator vs dense oracles.

Covers forward + gradients (q, v, z), packed documents, irregular segments,
and edge layouts (single doc, all singletons, T=1). float64 CPU math oracle.
Also checks dense_fullmask == dense_segstart given consistent inputs, which
bridges the scan candidates to the canonical harness (full_mask-based).
"""

import itertools
import sys

import torch

sys.path.insert(0, ".")

from opt.scan_attn import (
    dense_coordinator_fullmask,
    dense_fullmask_attention,
    dense_segstart_attention,
    scan_chunked_attention,
    scan_chunkwise_bthk,
    scan_cumsum_attention,
    scan_hybrid_bthk,
    scan_parallel_bthk,
    scan_static4_bthk,
    segmented_prefix_coordinator,
)

RTOL = ATOL = 1e-10


def random_segment_start(rng, b, t):
    """Random packed-doc layout; returns segment_start [B,T] (doc-start cols)."""
    seg = torch.zeros((b, t), dtype=torch.long)
    for i in range(b):
        cuts = sorted(rng.sample(range(1, t), k=rng.randrange(t))) if t > 1 else []
        starts, s = [], 0
        for c in cuts + [t]:
            starts += [s] * (c - s)
            s = c
        seg[i] = torch.tensor(starts, dtype=torch.long)
    return seg


def full_mask_from_seg(seg):
    b, t = seg.shape
    same = seg[:, :, None] == seg[:, None, :]
    strict = torch.ones((t, t), dtype=torch.bool).tril(diagonal=-1)
    return same & strict.unsqueeze(0)


def check_attention(b, h, t, k, dv, seg, seed):
    g = torch.Generator().manual_seed(seed)
    q0 = torch.randn((b, h, t, k), generator=g, dtype=torch.float64)
    v0 = torch.randn((b, t, dv), generator=g, dtype=torch.float64)
    probe = torch.randn((b, h, t, dv), generator=g, dtype=torch.float64)
    fm = full_mask_from_seg(seg)

    def run(fn, *args):
        q = q0.clone().requires_grad_(True)
        v = v0.clone().requires_grad_(True)
        out = fn(q, v, *args)
        gq, gv = torch.autograd.grad((out * probe).sum(), (q, v))
        return out.detach(), gq, gv

    ref = run(dense_segstart_attention, seg)
    m = run(dense_fullmask_attention, fm)
    s1 = run(scan_cumsum_attention, seg)

    def run_chunkwise(q, v, seg, blk):
        qh = q
        vh = v.unsqueeze(1).expand(-1, q.shape[1], -1, -1)
        sd = bool((seg == 0).all())
        return scan_chunkwise_bthk(qh, vh, seg, block=blk, single_doc=sd)

    errs = {}
    for name, got in (("fullmask", m), ("cumsum", s1)):
        for tag, a, c in zip(("fwd", "gq", "gv"), got, ref):
            errs[f"{name}_{tag}"] = float((a - c).abs().max())
            torch.testing.assert_close(a, c, rtol=RTOL, atol=ATOL)
    # chunked form at several block sizes (incl. block > T, block = 1)
    for blk in (1, 2, 3, 5, 7, 64):
        if blk > t and blk != 64:
            continue
        q = q0.clone().requires_grad_(True)
        v = v0.clone().requires_grad_(True)
        out = scan_chunked_attention(q, v, seg, block=blk)
        gq, gv = torch.autograd.grad((out * probe).sum(), (q, v))
        for tag, a, c in zip(("fwd", "gq", "gv"), (out.detach(), gq, gv), ref):
            errs[f"chunk{blk}_{tag}"] = float((a - c).abs().max())
            torch.testing.assert_close(a, c, rtol=RTOL, atol=ATOL)
    # chunkwise (production execution form) at several block sizes
    for blk in (1, 2, 3, 5, 8, 64):
        if blk > t and blk != 64:
            continue
        q = q0.clone().requires_grad_(True)
        v = v0.clone().requires_grad_(True)
        out = run_chunkwise(q, v, seg, blk)
        gq, gv = torch.autograd.grad((out * probe).sum(), (q, v))
        for tag, a, c in zip(("fwd", "gq", "gv"), (out.detach(), gq, gv), ref):
            errs[f"wise{blk}_{tag}"] = float((a - c).abs().max())
            torch.testing.assert_close(a, c, rtol=RTOL, atol=ATOL)
    # parallel scan (single-doc only) incl. padding path
    # hybrid scan (single-doc only) incl. padding path
    # static4 scan (single-doc only, T == 4W exactly)
    if bool((seg == 0).all()):
        dev = q0.device
        if t % 4 == 0 and t >= 4:
            blk = t // 4
            q = q0.clone().requires_grad_(True)
            v = v0.clone().requires_grad_(True)
            qh = q
            vh = v.unsqueeze(1).expand(-1, q.shape[1], -1, -1)
            mask = torch.ones((blk, blk), dtype=torch.bool, device=dev).tril(diagonal=-1)
            out = scan_static4_bthk(qh, vh, mask)
            gq, gv = torch.autograd.grad((out * probe).sum(), (q, v))
            for tag, a, c in zip(("fwd", "gq", "gv"), (out.detach(), gq, gv), ref):
                errs[f"static4_{tag}"] = float((a - c).abs().max())
                torch.testing.assert_close(a, c, rtol=RTOL, atol=ATOL)
    if bool((seg == 0).all()):
        for blk in (1, 3, 7, 64):
            for fname, fn in (("par", scan_parallel_bthk), ("hyb", scan_hybrid_bthk)):
                q = q0.clone().requires_grad_(True)
                v = v0.clone().requires_grad_(True)
                qh = q
                vh = v.unsqueeze(1).expand(-1, q.shape[1], -1, -1)
                out = fn(qh, vh, block=blk)
                gq, gv = torch.autograd.grad((out * probe).sum(), (q, v))
                for tag, a, c in zip(("fwd", "gq", "gv"), (out.detach(), gq, gv), ref):
                    errs[f"{fname}{blk}_{tag}"] = float((a - c).abs().max())
                    torch.testing.assert_close(a, c, rtol=RTOL, atol=ATOL)
    return errs


def check_coordinator(b, t, w, seg, seed):
    g = torch.Generator().manual_seed(seed + 10**6)
    z0 = torch.randn((b, t, w), generator=g, dtype=torch.float64)
    probe = torch.randn_like(z0)
    pos = torch.arange(t).expand(b, -1)
    segpos = pos - seg
    fm = full_mask_from_seg(seg)

    def run(fn, *args):
        z = z0.clone().requires_grad_(True)
        out = fn(z, *args)
        gz = torch.autograd.grad((out * probe).sum(), z)[0]
        return out.detach(), gz

    ref = run(dense_coordinator_fullmask, segpos, fm)
    got = run(segmented_prefix_coordinator, segpos, seg)
    errs = {}
    for tag, a, c in zip(("val", "gz"), got, ref):
        errs[f"coord_{tag}"] = float((a - c).abs().max())
        torch.testing.assert_close(a, c, rtol=RTOL, atol=ATOL)
    # single-doc fast path where applicable
    if bool((seg == 0).all()):
        z = z0.clone().requires_grad_(True)
        out = segmented_prefix_coordinator(z, segpos, seg, single_doc=True)
        gz = torch.autograd.grad((out * probe).sum(), z)[0]
        torch.testing.assert_close(out.detach(), ref[0], rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(gz, ref[1], rtol=RTOL, atol=ATOL)
        errs["coord_sd_val"] = float((out.detach() - ref[0]).abs().max())
    return errs


def main():
    rng = __import__("random").Random(0)
    worst = {}
    n = 0
    shapes = [
        (b, h, t, k, dv)
        for b, h, t, k, dv in itertools.product(
            (1, 2), (1, 2, 4), (1, 2, 5, 11, 17, 33), (2, 7), (1, 5)
        )
        if b * h * t * k * dv <= 6000
    ]
    # Fixed edge layouts: single doc, all singletons, two halves.
    edges = []
    for b, h, t, k, dv in shapes[:: max(1, len(shapes) // 12)]:
        edges.append((b, h, t, k, dv, torch.zeros((b, t), dtype=torch.long)))
        if t > 1:
            edges.append(
                (b, h, t, k, dv, torch.arange(t).unsqueeze(0).expand(b, -1))
            )
            mid = t // 2
            s = torch.zeros((b, t), dtype=torch.long)
            s[:, mid:] = mid
            edges.append((b, h, t, k, dv, s))
    cases = list(edges)
    for b, h, t, k, dv in shapes:
        cases.append((b, h, t, k, dv, random_segment_start(rng, b, t)))
    # Explicit T-multiple-of-4 single-doc cases for the static4 gate.
    for (b, h, t, k, dv) in ((1, 1, 8, 4, 3), (2, 2, 12, 5, 4), (1, 2, 16, 3, 5),
                             (2, 1, 20, 4, 2)):
        cases.append((b, h, t, k, dv, torch.zeros((b, t), dtype=torch.long)))
    for i, (b, h, t, k, dv, seg) in enumerate(cases):
        e = check_attention(b, h, t, k, dv, seg, seed=1000 + i)
        e.update(check_coordinator(b, t, dv, seg, seed=1000 + i))
        for kk, vv in e.items():
            worst[kk] = max(worst.get(kk, 0.0), vv)
        n += 1
    print({"status": "SYSTEMATIC_CHECKS_PASSED", "cases": n})
    for kk in sorted(worst):
        print(f"  worst {kk}: {worst[kk]:.3e}")


if __name__ == "__main__":
    main()
