"""Context-length robustness of the core+conditional-tail structure.

Runs the same trained checkpoint at T = 256/512/1024/2048 with identically
designed packed batches (doc lengths scaled to 0.7*T) and asks whether the
predeclared statistics are intrinsic to the population or an artifact of the
2048-token window regime:

  A. Delta_u(64) = M_local,u(64) - M_global,u(64)   (primary)
  B. M_core,u(256) = per-token u mass fraction in the cross-fitted global
     top-6.25% of coordinates                        (primary)
  C. per-head RoPE-band deviation vs pair-preserving permutation null

Secondary: Delta_x(16), Delta_x(1024), local/global ladder, Gini, N_eff,
stability, position strata.

Writes results/arm_a_science/context_length.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import LADDER, RAW_DIR, RESULTS_DIR, save_json, band_slices  # noqa: E402
from analyze_topn import cell_stats  # noqa: E402
from analyze_maturation import gini, neff, pratio  # noqa: E402

N_PERM = 2000
SEED = 20260916


def load_ctx(T: int):
    raw = RAW_DIR / "raw_ctx" / f"ctx{T}"
    p1 = np.load(raw / "pass1_latest.npz")
    p2 = np.load(raw / "pass2_latest.npz")
    meta = json.loads((raw / "meta_latest.json").read_text(encoding="utf-8"))
    return raw, p1, p2, meta


def band_result(p1, n_perm=N_PERM):
    K2 = p1["pair_mass_sum_x"].shape[-1]
    rng = np.random.default_rng(SEED)
    out = {}
    for key in ("x", "u"):
        pm = p1[f"pair_mass_sum_{key}"].sum((0, 1, 2))
        edges = band_slices(pm.size * 2, 8)
        obs = np.array([pm[lo // 2:hi // 2].sum() for lo, hi in edges])
        obs_share = obs / obs.sum()
        nulls = np.empty(n_perm)
        for i in range(n_perm):
            idx = rng.permutation(K2)
            v = np.empty_like(pm)
            v[idx] = pm
            bs = np.array([v[lo // 2:hi // 2].sum() for lo, hi in edges])
            nulls[i] = (bs / bs.sum()).max()
        sd = nulls.std()
        out[key] = {
            "pooled_max_band_share": float(obs_share.max()),
            "null_mean": float(nulls.mean()),
            "z": float((obs_share.max() - nulls.mean()) / sd) if sd > 0
            else None,
            "pooled_band_share": obs_share.tolist(),
        }
    return out


def stability_result(p2):
    jac = p2["stab_jaccard"]
    sup = p2["stab_support"][..., 0]
    sp = p2["stab_spearman"][..., 0]
    rbo = p2["stab_rbo"][..., 0]
    meta = p2["stab_meta"]
    cats = [str(c) for c in p2["stab_categories"]]
    out = {}
    for key, kidx in (("x", 0), ("u", 1)):
        per_cat = {}
        for ci, cat in enumerate(cats):
            m = meta[:, 3] == ci
            if not m.any():
                continue
            if "lag_1" != cat and cat not in ("cross_row",):
                continue
            v = jac[m]
            if key == "x":
                pass
            # filter by key
            mk = m & (meta[:, 2] == kidx)
            if not mk.any():
                continue
            v = jac[mk]
            s = sp[mk]
            per_cat[cat] = {
                "jaccard64_mean": float(np.nanmean(v[:, :, 2])),
                "jaccard1024_mean": float(np.nanmean(v[:, :, 6])),
                "support_mean": float(np.nanmean(sup[mk])),
                "spearman_mean": float(np.nanmean(s)),
                "rbo_mean": float(np.nanmean(rbo[mk])),
                "n_cells": int(mk.sum()),
            }
        out[key] = per_cat
    return out


def main():
    out = {
        "meta": {
            "purpose": "is the core+conditional-tail structure intrinsic or a "
                       "2048-window artifact?",
            "design": "same checkpoint, T in {256,512,1024,2048}, packed "
                      "batches with exponential doc lengths mean 0.7*T, same "
                      "seeds; 4 batches x 4 rows; sampled Tier-A tokens per "
                      "batch = 2 chunks x 128 tokens (all tokens when T<=256)",
            "evidence_class": "E2 (synthetic contexts; E3 corpus version "
                              "available via e3_census.py --context)",
            "predeclared_primary": ["Delta_u(64)", "M_core_u(256)",
                                    "band pooled max-share z"],
            "band_null": f"{N_PERM} pair-preserving permutations, seed {SEED}",
        },
        "by_T": {},
    }
    for T in (256, 512, 1024, 2048):
        raw, p1, p2, meta = load_ctx(T)
        B, S = p1["tok_meta"].shape[:2]
        doc_id = p1["tok_meta"][:, :, 2]
        blocks = (np.arange(B)[:, None] * 100000 + doc_id).reshape(-1)
        L = p1["mass_sum_x"].shape[1]
        H = p1["mass_sum_x"].shape[2]
        K = p1["mass_sum_x"].shape[3]
        n_batches = p1["mass_sum_x"].shape[0]
        rows = meta["specs"][0]["rows"]
        n_tokens = n_batches * rows * T
        entry = {"T": T, "n_tokens_per_level": int(n_tokens)}

        def cellmat(key, field, N):
            vals = []
            for level in range(L):
                for h in range(H):
                    lv = p1[f"tA_topn_{key}"][:, level, :, h, :].reshape(
                        -1, len(LADDER))
                    gl = p2[f"g_topn_ratio_{key}"][:, level, :, h, :].reshape(
                        -1, len(LADDER))
                    to = p1[f"tA_total_{key}"][:, level, :, h].reshape(-1)
                    i = LADDER.index(N)
                    valid = to > 0
                    vals.append(float((lv[valid, i] - gl[valid, i]).mean()))
            return np.asarray(vals)

        for key, Ns in (("u", (64, 256, 1024)), ("x", (16, 64, 1024))):
            for N in Ns:
                d = cellmat(key, "delta", N)
                entry[f"delta_{key}_{N}"] = {
                    "mean_over_cells": float(d.mean()),
                    "min_cell": float(d.min()),
                    "max_cell": float(d.max()),
                    "cells_gt_0.05": int((d > 0.05).sum()),
                    "n_cells": int(d.size),
                }
        # core mass fraction (top 6.25% cross-fitted) for u and x
        for key in ("x", "u"):
            frac = p2[f"core_mass_top_6p25_frac_{key}"]
            cpos = p2[f"core_mass_top_6p25_cpos_{key}"]
            entry[f"core_6p25_mass_frac_{key}"] = {
                "mean": float(frac.mean()), "p10": float(np.quantile(frac, 0.1)),
                "p90": float(np.quantile(frac, 0.9)),
                "active_fraction": float((cpos > 0).mean()),
            }
            res = p2[f"core_mass_top_6p25_res_ratio_{key}"]
            need = p2[f"core_mass_top_6p25_res_need_{key}"]
            entry[f"residual_6p25_{key}"] = {
                "res_top64": float(np.nanmean(res[..., 2])),
                "res_top1024": float(np.nanmean(res[..., 6])),
                "coords_for_50pct_residual": float(need[..., 0].mean()),
                "coords_for_90pct_residual": float(need[..., -1].mean()),
            }
        # concentration
        conc = {}
        for key in ("x", "u"):
            m = p1[f"mass_sum_{key}"].sum(0)
            neffs, ginis, prs, top5 = [], [], [], []
            for level in range(L):
                for h in range(H):
                    p = m[level, h] / max(m[level, h].sum(), 1e-30)
                    neffs.append(neff(p) / K)
                    ginis.append(gini(m[level, h]))
                    prs.append(pratio(p) / K)
                    top5.append(float(np.sort(p)[::-1][:K // 20].sum()))
            conc[key] = {"neff_over_K": float(np.mean(neffs)),
                         "gini": float(np.mean(ginis)),
                         "pr_over_K": float(np.mean(prs)),
                         "top5pct_mass_share": float(np.mean(top5))}
        entry["concentration"] = conc
        entry["stability"] = stability_result(p2)
        entry["bands"] = band_result(p1)
        # position strata: delta_u(64) by sequence-position quartile
        pos = p1["tok_meta"][:, :, 1]
        i64 = LADDER.index(64)
        dmap = (p1["tA_topn_u"][..., i64] -
                p2["g_topn_ratio_u"][..., i64]).mean(axis=3).mean(axis=1)
        to = p1["tA_total_u"].mean(axis=(1, 3))
        valid = to > 0
        edges = np.quantile(pos.reshape(-1), [0, 0.25, 0.5, 0.75, 1.0])
        strata = {}
        for qi in range(4):
            lo, hi = edges[qi], edges[qi + 1]
            mask = valid & (pos >= lo) & (pos <= hi if qi == 3 else pos < hi)
            if mask.sum():
                strata[f"q{qi+1}_pos{int(lo)}-{int(hi)}"] = {
                    "n": int(mask.sum()),
                    "delta_u64_mean": float(dmap[mask].mean()),
                }
        entry["position_strata_delta_u64"] = strata
        out["by_T"][str(T)] = entry

    # E2 baseline cross-check (raw/, not scaled specs)
    try:
        p1 = np.load(RAW_DIR / "pass1_latest.npz")
        p2 = np.load(RAW_DIR / "pass2_latest.npz")
        B, S = p1["tok_meta"].shape[:2]
        doc_id = p1["tok_meta"][:, :, 2]
        blocks = (np.arange(B)[:, None] * 100000 + doc_id).reshape(-1)
        i64 = LADDER.index(64)
        d = []
        for level in range(p1["mass_sum_x"].shape[1]):
            for h in range(p1["mass_sum_x"].shape[2]):
                lv = p1["tA_topn_u"][:, level, :, h, :].reshape(
                    -1, len(LADDER))
                gl = p2["g_topn_ratio_u"][:, level, :, h, :].reshape(
                    -1, len(LADDER))
                to = p1["tA_total_u"][:, level, :, h].reshape(-1)
                valid = to > 0
                d.append(float((lv[valid, i64] - gl[valid, i64]).mean()))
        frac = p2["core_mass_top_6p25_frac_u"]
        out["e2_baseline_raw_T2048"] = {
            "delta_u_64_mean_over_cells": float(np.mean(d)),
            "core_6p25_mass_frac_u_mean": float(frac.mean()),
            "note": "original E2 run (mixed mode mean doc len 1429) for "
                    "cross-checking the scaled T=2048 row",
        }
    except FileNotFoundError:
        pass
    save_json(RESULTS_DIR / "context_length.json", out)
    print(json.dumps({"ok": True}, indent=2))


if __name__ == "__main__":
    main()
