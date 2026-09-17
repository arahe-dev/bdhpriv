"""Confounder / stratification analysis for the P1 global-vs-local result.

Writes results/arm_a_science/position_confounds.json:
  1. M_local / M_global / delta stratified by document-relative position,
     sequence position, document length, first-token status and data mode.
  2. Between- vs within-document variance of M_global (document-level
     conditional structure).
  3. Token-level correlation of the conditional gap across keys (x, y, u)
     to identify which pathway (pre-activation x or attention output y)
     supplies the conditional component of u = x * y.
  4. Coordinate-level rank correlation between global mass vectors of the
     three keys.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import KEYS, LADDER, RAW_DIR, RESULTS_DIR, save_json  # noqa: E402


def mean_ci(v, mask=None, seed=3, n_boot=500):
    v = np.asarray(v, dtype=np.float64)
    if mask is not None:
        v = v[mask]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    rng = np.random.default_rng(seed)
    boots = np.array([v[rng.integers(0, v.size, v.size)].mean()
                      for _ in range(n_boot)])
    return {"mean": float(v.mean()), "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975)), "n": int(v.size)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="latest")
    ap.add_argument("--raw", default=str(RAW_DIR))
    ap.add_argument("--out", default=str(
        RESULTS_DIR / "position_confounds.json"))
    args = ap.parse_args()
    raw = Path(args.raw)
    p1 = np.load(raw / f"pass1_{args.ckpt}.npz")
    p2 = np.load(raw / f"pass2_{args.ckpt}.npz")
    meta = json.loads((raw / f"meta_{args.ckpt}.json").read_text(
        encoding="utf-8"))
    labels = [s["label"] for s in meta["specs"]]
    B, S, _ = p1["tok_meta"].shape
    doc_rel = p1["tok_meta"][:, :, 4]
    doc_len = p1["tok_meta"][:, :, 3]
    batch_of = np.repeat(np.arange(B), 1)[:, None] * np.ones((1, S),
                                                             dtype=int)
    out = {
        "checkpoint": args.ckpt,
        "ladder": list(LADDER),
        "strata_defs": {
            "doc_rel_pos": ["==0", "1-4", "5-16", "17-64", "65-256", ">256"],
            "seq_pos": ["0-255", "256-1023", "1024-2047"],
            "doc_len": ["<=8", "9-64", "65-512", ">512"],
            "mode": labels,
        },
        "stratified": {},
    }
    levels = p1["mass_sum_x"].shape[1]
    H = p1["mass_sum_x"].shape[2]

    def gap(key, key2, N):
        i = LADDER.index(N)
        loc = p1[f"tA_topn_{key}"][..., i]      # (B,L,S,H)
        glo = p2[f"g_topn_ratio_{key}"][..., i]
        tot = p1[f"tA_total_{key}"]
        valid = tot > 0
        # average over levels and heads -> (B,S)
        d = (loc - glo).mean(axis=(1, 3))
        g = glo.mean(axis=(1, 3))
        l = loc.mean(axis=(1, 3))
        return l, g, d, valid.mean(axis=(1, 3)) > 0.5

    for key in ("x", "u"):
        keyout = {}
        for N in (64, 256):
            l, g, d, valid = gap(key, key, N)
            strat = {}
            masks = {
                "doc_rel_pos:==0": doc_rel == 0,
                "doc_rel_pos:1-4": (doc_rel >= 1) & (doc_rel <= 4),
                "doc_rel_pos:5-16": (doc_rel >= 5) & (doc_rel <= 16),
                "doc_rel_pos:17-64": (doc_rel >= 17) & (doc_rel <= 64),
                "doc_rel_pos:65-256": (doc_rel >= 65) & (doc_rel <= 256),
                "doc_rel_pos:>256": doc_rel > 256,
                "seq_pos:0-255": p1["tok_meta"][:, :, 1] <= 255,
                "seq_pos:256-1023": (p1["tok_meta"][:, :, 1] >= 256)
                                    & (p1["tok_meta"][:, :, 1] <= 1023),
                "seq_pos:1024-2047": p1["tok_meta"][:, :, 1] >= 1024,
                "doc_len:<=8": doc_len <= 8,
                "doc_len:9-64": (doc_len >= 9) & (doc_len <= 64),
                "doc_len:65-512": (doc_len >= 65) & (doc_len <= 512),
                "doc_len:>512": doc_len > 512,
            }
            for label in labels:
                bi = labels.index(label)
                m = np.zeros_like(valid)
                m[bi, :] = True
                masks[f"mode:{label}"] = m
            for name, m in masks.items():
                mm = m & valid
                strat[name] = {
                    "n": int(mm.sum()),
                    "M_local": mean_ci(l, mm),
                    "M_global": mean_ci(g, mm),
                    "delta": mean_ci(d, mm),
                }
            keyout[str(N)] = strat
        out["stratified"][key] = keyout

    # document-level decomposition of M_global and delta at N=64/256
    doc_id = p1["tok_meta"][:, :, 2]
    doc_flat = (np.arange(B)[:, None] * 100000 + doc_id).reshape(-1)
    dec = {}
    for key in ("x", "u"):
        for N in (64, 256):
            i = LADDER.index(N)
            loc = p1[f"tA_topn_{key}"][..., i]
            glo = p2[f"g_topn_ratio_{key}"][..., i]
            tot = p1[f"tA_total_{key}"]
            d = ((loc - glo).mean(axis=(1, 3))).reshape(-1)
            g = (glo.mean(axis=(1, 3))).reshape(-1)
            valid = (tot.mean(axis=(1, 3)).reshape(-1) > 0)
            df, gf = d[valid], g[valid]
            bf = doc_flat[valid]
            doc_means = {}
            for dv in np.unique(bf):
                doc_means[dv] = df[bf == dv]
            if len(doc_means) > 1:
                between = np.var([v.mean() for v in doc_means.values()])
                within = np.mean([v.var() for v in doc_means.values()])
            else:
                between = within = float("nan")
            dec[f"{key}_{N}"] = {
                "n_docs": int(len(doc_means)),
                "delta_between_doc_var": float(between),
                "delta_within_doc_var": float(within),
                "delta_icc_approx": float(between / (between + within))
                if (between + within) > 0 else None,
                "M_global_between_doc_sd": float(np.std(
                    [v.mean() for v in doc_means.values()])),
            }
    out["document_decomposition"] = dec

    # token-level correlation of conditional gaps across keys
    corr = {}
    for N in (64, 256):
        i = LADDER.index(N)
        gaps = {}
        for key in KEYS:
            loc = p1[f"tA_topn_{key}"][..., i]
            glo = p2[f"g_topn_ratio_{key}"][..., i]
            gaps[key] = (loc - glo).mean(axis=-1).reshape(-1)  # (B*L*S)
        valid = np.isfinite(gaps["u"]) & (p1["tA_total_u"].mean(
            axis=-1).reshape(-1) > 0)
        entry = {}
        for a, b in (("x", "u"), ("y", "u"), ("x", "y")):
            va, vb = gaps[a][valid], gaps[b][valid]
            if va.std() > 0 and vb.std() > 0:
                entry[f"{a}_{b}"] = {
                    "pearson": float(np.corrcoef(va, vb)[0, 1]),
                    "spearman": float(_spearman(va, vb)),
                }
        corr[str(N)] = entry
    out["gap_correlations"] = corr

    # coordinate-level global mass rank correlations
    rankcorr = {}
    M = {k: p1[f"mass_sum_{k}"].sum(0) for k in KEYS}  # (L,H,K)
    for level in range(levels):
        entry = {}
        for a, b in (("x", "u"), ("y", "u"), ("x", "y")):
            va = M[a][level].reshape(-1)
            vb = M[b][level].reshape(-1)
            entry[f"{a}_{b}"] = {"spearman": float(_spearman(va, vb))}
        rankcorr[str(level)] = entry
    out["global_mass_rank_correlation"] = rankcorr

    save_json(Path(args.out), out)
    print(json.dumps({"out": args.out}, indent=2))


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() /
                 np.sqrt((ra * ra).sum() * (rb * rb).sum()))


if __name__ == "__main__":
    main()
