"""P2 analysis: population stability + core/tail decomposition.

Reads raw NPZ from collect.py and writes
  results/arm_a_science/population_stability.json
  results/arm_a_science/core_tail.json

Stability categories: lag_<L> token pairs inside a document at distance L,
and cross_row pairs (different packed rows, i.e. different documents).
Metrics: top-N Jaccard identity overlap, full-support Jaccard, Spearman rank
correlation over all K coordinates, RBO (p=0.9, depth 1024).

Cross-checkpoint stability: global ranking overlap and mass-vector Spearman
between checkpoints of the same trajectory (not independent runs).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import KEYS, LADDER, RAW_DIR, RESULTS_DIR, save_json  # noqa: E402

CORE_DEFS = ("mass_top_1pct", "mass_top_6p25", "mass_top_25",
             "pact_ge_0p5", "pact_ge_0p9")


def _q(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {"mean": float(v.mean()), "p10": float(np.quantile(v, 0.1)),
            "p25": float(np.quantile(v, 0.25)),
            "p50": float(np.quantile(v, 0.5)),
            "p75": float(np.quantile(v, 0.75)),
            "p90": float(np.quantile(v, 0.9)), "n": int(v.size)}


def analyze_stability(ckpt: str, raw: Path) -> dict:
    p2 = np.load(raw / f"pass2_{ckpt}.npz")
    jac = p2["stab_jaccard"]          # (entries, max_pairs, 9)
    sup = p2["stab_support"][..., 0]  # (entries, max_pairs)
    sp = p2["stab_spearman"][..., 0]
    rbo = p2["stab_rbo"][..., 0]
    meta = p2["stab_meta"]            # (entries, 4) level, head, key, cat
    cats = [str(c) for c in p2["stab_categories"]]
    out = {"categories": cats, "ladder": list(LADDER), "cells": []}
    for e in range(meta.shape[0]):
        level, head, kidx, cidx = [int(x) for x in meta[e]]
        category = cats[cidx]
        key = ("x", "u")[kidx]
        n_valid = np.isfinite(sup[e]).sum()
        if n_valid == 0:
            continue
        entry = {
            "level": level, "head": head, "key": key,
            "category": category, "n_pairs": int(n_valid),
            "jaccard": {str(n): _q(jac[e][:, i])
                        for i, n in enumerate(LADDER)},
            "support_jaccard": _q(sup[e]),
            "spearman": _q(sp[e]),
            "rbo": _q(rbo[e]),
        }
        out["cells"].append(entry)
    return out


def summarize_stability(stab: dict) -> dict:
    """Aggregate cells into lag curves (mean across levels/heads)."""
    curves = {}
    for key in ("x", "u"):
        entry = {}
        for cat in stab["categories"]:
            sel = [c for c in stab["cells"]
                   if c["key"] == key and c["category"] == cat]
            if not sel:
                continue
            entry[cat] = {
                "n_cells": len(sel),
                "jaccard_mean_per_N": {
                    n: float(np.mean([c["jaccard"][n]["mean"]
                                      for c in sel if c["jaccard"][n]["n"]]))
                    for n in map(str, LADDER)},
                "support_jaccard_mean": float(np.mean(
                    [c["support_jaccard"]["mean"] for c in sel])),
                "spearman_mean": float(np.mean(
                    [c["spearman"]["mean"] for c in sel])),
                "rbo_mean": float(np.mean([c["rbo"]["mean"] for c in sel])),
            }
        curves[key] = entry
    return curves


def analyze_core(ckpt: str, raw: Path) -> dict:
    p2 = np.load(raw / f"pass2_{ckpt}.npz")
    p1 = np.load(raw / f"pass1_{ckpt}.npz")
    ranks = np.load(raw / f"ranks_{ckpt}.npz")
    L = p1["mass_sum_x"].shape[1]
    H = p1["mass_sum_x"].shape[2]
    K = p1["mass_sum_x"].shape[3]
    out = {"checkpoint": ckpt, "ladder": list(LADDER), "defs": {}}
    for name in CORE_DEFS:
        d = {"levels": []}
        for key in ("x", "u"):
            level_list = []
            for level in range(L):
                frac = p2[f"core_{name}_frac_{key}"][:, level, :, :]
                cpos = p2[f"core_{name}_cpos_{key}"][:, level, :, :]
                res = p2[f"core_{name}_res_ratio_{key}"][:, level, :, :, :]
                need = p2[f"core_{name}_res_need_{key}"][:, level, :, :, :]
                frac_flat = frac.reshape(-1)
                active = (cpos.reshape(-1) > 0)
                n_q = need.shape[-1]
                q_map = ({0: "p50", 1: "p80", 2: "p90"} if n_q >= 3
                         else {0: "p50", 1: "p90"})
                entry = {
                    "level": level,
                    "mass_fraction": _q(frac_flat),
                    "tokens_with_core_active": float(active.mean()),
                    "residual_topN": {
                        str(n): _q(res[..., i].reshape(-1))
                        for i, n in enumerate(LADDER)},
                    "residual_need_coords": {
                        label: _q(need[..., qi].reshape(-1))
                        for qi, label in q_map.items()},
                }
                level_list.append(entry)
            d["levels"].append({"key": key, "levels": level_list})
        # core sizes per def (fraction of the pooled H*K dictionary)
        sizes = []
        for level in range(L):
            for split in ("A", "B"):
                m = ranks[f"core_{name}_x_L{level}_{split}"]
                sizes.append(float(m.sum()) / (K * m.shape[0]))
        d["core_size_fraction_K"] = {"mean": float(np.mean(sizes)),
                                     "min": float(np.min(sizes)),
                                     "max": float(np.max(sizes))}
        out["defs"][name] = d
    return out


def cross_checkpoint(ckpts, raw: Path) -> dict:
    """Global ranking overlap and mass Spearman across checkpoints."""
    mass = {}
    ranks = {}
    for c in ckpts:
        p1 = np.load(raw / f"pass1_{c}.npz")
        mass[c] = {k: p1[f"mass_sum_{k}"].sum(0) for k in KEYS}
        ranks[c] = np.load(raw / f"ranks_{c}.npz")
    pairs = {}
    for i, a in enumerate(ckpts):
        for b in ckpts[i + 1:]:
            entry = {}
            for key in KEYS:
                per_level = []
                for level in range(mass[a][key].shape[0]):
                    ma = mass[a][key][level].reshape(-1)
                    mb = mass[b][key][level].reshape(-1)
                    ra = np.argsort(np.argsort(ma))
                    rb = np.argsort(np.argsort(mb))
                    ra = ra - ra.mean()
                    rb = rb - rb.mean()
                    rho = float((ra * rb).sum() /
                                np.sqrt((ra * ra).sum() * (rb * rb).sum()))
                    top = {}
                    for n in (16, 64, 256, 1024):
                        sa = set(np.argsort(-ma)[:n].tolist())
                        sb = set(np.argsort(-mb)[:n].tolist())
                        top[str(n)] = len(sa & sb) / len(sa | sb)
                    per_level.append({"level": level, "mass_spearman": rho,
                                      "topn_jaccard": top})
                entry[key] = per_level
            pairs[f"{a}->{b}"] = entry
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", nargs="+", default=["latest"])
    ap.add_argument("--raw", default=str(RAW_DIR))
    ap.add_argument("--out-stability",
                    default=str(RESULTS_DIR / "population_stability.json"))
    ap.add_argument("--out-core",
                    default=str(RESULTS_DIR / "core_tail.json"))
    args = ap.parse_args()
    raw = Path(args.raw)
    stability = {"meta": {
        "evidence_class": "E1/E2 synthetic packed batches, one trajectory",
        "categories": "lag_L = same-document token pairs L apart; cross_row = "
                      "different packed rows (different documents)",
    }, "checkpoints": {}}
    core = {"meta": {
        "evidence_class": "E1/E2 synthetic packed batches, one trajectory",
        "core_defs": list(CORE_DEFS),
        "core_def_note": "mass_top_* = top fraction of K by cross-fitted "
                         "global mass; pact_ge_* = activation probability "
                         "threshold",
    }, "checkpoints": {}}
    for ckpt in args.ckpts:
        try:
            s = analyze_stability(ckpt, raw)
            stability["checkpoints"][ckpt] = {
                "summary": summarize_stability(s),
                "cells": s["cells"],
            }
            core["checkpoints"][ckpt] = analyze_core(ckpt, raw)
        except FileNotFoundError as e:
            print(f"skip {ckpt}: {e}")
    if len([c for c in args.ckpts if (raw / f"pass1_{c}.npz").exists()]) > 1:
        core["cross_checkpoint_rank_overlap"] = cross_checkpoint(
            [c for c in args.ckpts if (raw / f"pass1_{c}.npz").exists()], raw)
    save_json(Path(args.out_stability), stability)
    save_json(Path(args.out_core), core)
    print(json.dumps({"ckpts": list(stability["checkpoints"]),
                      "core_ckpts": list(core["checkpoints"])}, indent=2))


if __name__ == "__main__":
    main()
