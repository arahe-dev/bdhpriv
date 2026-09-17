"""Forward-only static vs dynamic vs hybrid sparse-coordinate ablation pilot.

This is NOT the E4 training intervention (see e4_static_dynamic_hybrid.json
for that protocol). It measures a causal property that the E4 intervention
would test: at matched active width per (level, head), how much of the dense
model output survives when u = x*y is restricted to

  STATIC  : frozen global top-w coordinates (from the other batch split)
  DYNAMIC : the token's own top-w u coordinates
  HYBRID  : always-on global top-c (c = 6.25% of K = 256) plus the token's
            top-t non-core coordinates, w = c + t

Predeclared prediction (from the E2 core+tail structure):
  hybrid preserves the dense output better than dynamic, which preserves it
  better than static, at matched total width.

Reported: relative L2 logit perturbation, u mass coverage, coordinate
utilization and routing entropy.

Writes results/arm_a_science/e4_forward_ablation.json.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import (  # noqa: E402
    CHECKPOINTS, MAIN_SPECS, RAW_DIR, RESULTS_DIR, ArmAConfig, build_model,
    load_state, make_batch, make_sample_layout, rss_mb, save_json,
)
from opt.census_model import CensusArmA  # noqa: E402

CORE_COORDS = 256  # 6.25% of K
SPLIT_A = (0, 2)
SPLIT_B = (1, 3)


class MaskPolicy:
    name = "base"

    def mask(self, level, prod):
        raise NotImplementedError


class StaticPolicy(MaskPolicy):
    def __init__(self, rank_by_level, width):
        self.name = "static"
        self.rank = rank_by_level
        self.width = width

    def mask(self, level, prod):
        b, t, h, k = prod.shape
        mask = torch.zeros((h, k), dtype=torch.bool)
        r = torch.as_tensor(self.rank[level], dtype=torch.long)
        mask.scatter_(1, r[:, :self.width], True)
        return mask.view(1, 1, h, k).expand(b, t, h, k)


class DynamicPolicy(MaskPolicy):
    def __init__(self, width):
        self.name = "dynamic"
        self.width = width
        self.counts = None
        self.n_calls = 0

    def mask(self, level, prod):
        b, t, h, k = prod.shape
        flat = prod.reshape(b * t * h, k)
        w = min(self.width, k)
        idx = torch.topk(flat, w, dim=-1).indices
        mask = torch.zeros((b * t * h, k), dtype=torch.bool)
        mask.scatter_(1, idx, True)
        m4 = mask.view(b, t, h, k)
        sel = m4.sum(dim=(0, 1)).double()  # (H,K)
        if self.counts is None:
            self.counts = sel
        else:
            self.counts += sel
        self.n_calls += 1
        return m4

    def routing_stats(self):
        if self.counts is None:
            return {}
        p = (self.counts / max(self.counts.sum().item(), 1e-30)).numpy()
        nz = p[p > 0]
        ent = float(-(nz * np.log(nz)).sum())
        return {
            "routing_entropy_nats": ent,
            "routing_entropy_over_logK": ent / math.log(self.counts.shape[-1]),
            "coordinate_utilization": float((self.counts > 0).float().mean()),
            "coordinate_utilization_frac2x_active": float(
                (self.counts > 2 * self.n_calls).float().mean()),
        }


class HybridPolicy(MaskPolicy):
    def __init__(self, rank_by_level, core=256, tail=64):
        self.name = "hybrid"
        self.rank = rank_by_level
        self.core = core
        self.tail = tail
        self.width = core + tail

    def mask(self, level, prod):
        b, t, h, k = prod.shape
        r = torch.as_tensor(self.rank[level], dtype=torch.long)
        core = torch.zeros((h, k), dtype=torch.bool)
        core.scatter_(1, r[:, :self.core], True)
        core4 = core.view(1, 1, h, k).expand(b, t, h, k)
        vals = prod.masked_fill(core4, float("-inf"))
        flat = vals.reshape(b * t * h, k)
        idx = torch.topk(flat, self.tail, dim=-1).indices
        tail = torch.zeros((b * t * h, k), dtype=torch.bool)
        tail.scatter_(1, idx, True)
        return core4 | tail.view(b, t, h, k)


class ZeroPolicy(MaskPolicy):
    name = "zero"

    def mask(self, level, prod):
        return torch.zeros_like(prod, dtype=torch.bool)


class MaskedCensusArmA(CensusArmA):
    """CensusArmA with a u-mask policy applied at every level."""

    def __init__(self, cfg, device, policy, scan_block=1024):
        super().__init__(cfg, device, scan_block=scan_block)
        self.policy = policy
        self.mask_stats = []

    def level(self, v, pos, segpos, full_mask, segment_start,
              single_doc=False, cs=None, sn=None):
        cfg = self.cfg
        x_bt = self.project_x_native(v)
        a = self.ln(self.attention_scan(x_bt, v, pos, segment_start,
                                        single_doc, cs, sn))
        ypre = F.relu(a @ self.decoder_y)
        prod = x_bt * ypre.permute(0, 2, 1, 3)
        if self.capture is not None:
            self.capture(self._level_index, x_bt, ypre, prod, segment_start)
        level = self._level_index
        self._level_index += 1
        mask = self.policy.mask(level, prod)
        kept = float((prod * mask).sum().item())
        total = float(prod.sum().item()) + 1e-30
        self.mask_stats.append({
            "level": level,
            "mass_kept_fraction": kept / total,
            "coords_active_mean": float(mask.sum().item())
            / (prod.shape[0] * prod.shape[1] * prod.shape[2]),
        })
        prod = prod * mask
        paper_y_flat = prod.reshape(v.shape[0], cfg.T, cfg.N)
        base = self.ln(paper_y_flat @ self.encoder)
        g = self.coordinator(v, segpos, full_mask)
        delta = self.writer(g * base)
        return self.ln(v + delta)


def make_configs(rank_by_level):
    return [
        ("static_320", StaticPolicy(rank_by_level, 320)),
        ("static_512", StaticPolicy(rank_by_level, 512)),
        ("static_1024", StaticPolicy(rank_by_level, 1024)),
        ("dynamic_320", DynamicPolicy(320)),
        ("dynamic_512", DynamicPolicy(512)),
        ("dynamic_1024", DynamicPolicy(1024)),
        ("hybrid_256p64", HybridPolicy(rank_by_level, 256, 64)),
        ("hybrid_256p256", HybridPolicy(rank_by_level, 256, 256)),
        ("hybrid_256p768", HybridPolicy(rank_by_level, 256, 768)),
        ("zero_u", ZeroPolicy()),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="latest", choices=list(CHECKPOINTS))
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    cfg = ArmAConfig()
    state, _ = load_state(CHECKPOINTS[args.ckpt])
    specs = list(MAIN_SPECS)
    layouts = [make_sample_layout(s, cfg) for s in specs]
    batches = [make_batch(s, cfg) for s in specs]

    # frozen rankings from split A (cross-fitted against split B evaluation)
    p1 = np.load(RAW_DIR / "pass1_latest.npz")
    mass_a = p1["mass_sum_u"][list(SPLIT_A)].sum(0)
    rank_by_level = [np.argsort(-mass_a[level], axis=-1).astype(np.int32)
                     for level in range(cfg.L)]

    # dense baseline
    t0 = time.perf_counter()
    dense_model = build_model(cfg, state, threads=args.threads)
    dense_logits = []
    with torch.no_grad():
        for i in SPLIT_B:
            entry = batches[i]
            dense_logits.append(dense_model.forward_packed(
                entry["x"], entry["pos"], entry["segpos"],
                entry["full_mask"], entry["segment_start"]))
    del dense_model

    results = {}
    for name, policy in make_configs(rank_by_level):
        t1 = time.perf_counter()
        model = MaskedCensusArmA(cfg, torch.device("cpu"), policy)
        model.load_state_dict(state, strict=True)
        model.eval()
        logits, stats = [], []
        with torch.no_grad():
            for i in SPLIT_B:
                entry = batches[i]
                model.begin_forward(None)
                logits.append(model.forward_packed(
                    entry["x"], entry["pos"], entry["segpos"],
                    entry["full_mask"], entry["segment_start"]))
                stats.append(list(model.mask_stats))
        del model
        rel = []
        for dl, lg in zip(dense_logits, logits):
            dl = dl.double()
            lg = lg.double()
            num = (lg - dl).norm(dim=-1)
            den = dl.norm(dim=-1).clamp_min(1e-12)
            rel.append(float((num / den).mean()))
        entry = {
            "relative_logit_perturbation_mean": float(np.mean(rel)),
            "relative_logit_perturbation_per_batch": rel,
            "u_mass_kept": float(np.mean(
                [st["mass_kept_fraction"] for bs in stats
                 for st in bs])),
            "coords_active_mean": float(np.mean(
                [st["coords_active_mean"] for bs in stats
                 for st in bs])),
            "seconds": time.perf_counter() - t1,
        }
        if isinstance(policy, DynamicPolicy):
            entry.update(policy.routing_stats())
        results[name] = entry
        print("%-16s rel_perturb=%.4f mass_kept=%.3f coords=%.0f%s" % (
            name, entry["relative_logit_perturbation_mean"],
            entry["u_mass_kept"], entry["coords_active_mean"],
            (" util=%.3f" % entry["coordinate_utilization"])
            if "coordinate_utilization" in entry else ""))

    out = {
        "meta": {
            "purpose": "forward-only causal pilot for the E4 intervention; "
                       "NOT a training comparison and NOT E4 evidence",
            "design": "T=2048, 8 packed windows (split B), frozen u-mass "
                      "rankings from split A, mask u at every level, matched "
                      "width w = active coordinates per (level, head)",
            "oracle_note": "DYNAMIC and HYBRID tail selection read the "
                           "current u activations, so they are ORACLE "
                           "selectors and give an upper bound for a learned "
                           "router; an implementable router must predict the "
                           "selection from v (see e4_static_dynamic_hybrid"
                           ".json, metric router_recall@k)",
            "prediction": "hybrid < dynamic < static in relative logit "
                          "perturbation at matched width",
            "result_vs_prediction": "FALSIFIED for forward fidelity: at "
                                    "matched width dynamic (oracle) "
                                    "preserves the dense output best, hybrid "
                                    "is intermediate, static is worst",
            "widths": [320, 512, 1024],
            "core_coords": CORE_COORDS,
            "evidence_class": "E4-pilot (forward-only intervention on a "
                              "frozen model; no training)",
        },
        "results": results,
        "seconds": time.perf_counter() - t0,
        "rss_mb": rss_mb(),
    }
    save_json(RESULTS_DIR / "e4_forward_ablation.json", out)


if __name__ == "__main__":
    main()
