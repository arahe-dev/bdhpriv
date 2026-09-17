"""Headline statistics used in the claim ledger and the final report.

Aggregates the per-cell results from the analysis JSONs into a small,
explicitly defined set of numbers so the report does not restate raw arrays.
Writes results/arm_a_science/headline_stats.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import KEYS, LADDER, RESULTS_DIR, save_json  # noqa: E402


def load(name):
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


def cell_matrix(topn, ckpt, key, field, N):
    levels = topn[ckpt]["per_cell"][key]["levels"]
    vals = []
    for lvl in levels:
        for head in lvl["heads"]:
            vals.append(head[field][str(N)]["mean"])
    return np.asarray(vals)


def main():
    topn = load("topn_global_local.json")["checkpoints"]
    stab = load("population_stability.json")["checkpoints"]
    core = load("core_tail.json")["checkpoints"]
    freq = load("frequency_null.json")["checkpoints"]
    mat = load("sparsity_maturation.json")["checkpoints"]
    ctrl = load("controls.json")["checkpoints"]
    sae = {
        "small": json.loads((RESULTS_DIR.parent / "sae" / "gemma_small.json")
                            .read_text(encoding="utf-8")),
        "medium": json.loads((RESULTS_DIR.parent / "sae" / "gemma_medium.json")
                             .read_text(encoding="utf-8")),
    }
    cross = json.loads((RESULTS_DIR.parent / "sae" /
                        "cross_system_comparison.json").read_text(
        encoding="utf-8"))
    anthro = json.loads((RESULTS_DIR.parent / "sae" /
                         "anthropic_analysis.json").read_text(
        encoding="utf-8"))

    out = {"meta": {
        "evidence_class": "E1/E2 synthetic packed batches, one trajectory; "
                          "random-init control; external files are static "
                          "reference data",
        "aggregation": "equal weight per (level, head) cell, then mean; "
                       "32 cells for per-head keys",
    }}

    # P1 headline
    p1 = {}
    for key in ("x", "u"):
        entry = {}
        for N in (16, 64, 256, 1024):
            loc = cell_matrix(topn, "latest", key, "local", N)
            glo = cell_matrix(topn, "latest", key, "global", N)
            d = loc - glo
            entry[str(N)] = {
                "local_mean_over_cells": float(loc.mean()),
                "global_mean_over_cells": float(glo.mean()),
                "delta_mean_over_cells": float(d.mean()),
                "delta_min_cell": float(d.min()),
                "delta_max_cell": float(d.max()),
                "cells_with_delta_gt_0.05": int((d > 0.05).sum()),
                "n_cells": int(d.size),
            }
        p1[key] = entry
    pooled = {}
    for key in ("x", "u"):
        lv = topn["latest"]["pooled"][key]["levels"]
        pooled[key] = {}
        for N in (64, 256, 1024, 4096):
            loc = np.array([l["local"][str(N)]["mean"] for l in lv])
            glo = np.array([l["global"][str(N)]["mean"] for l in lv])
            pooled[key][str(N)] = {
                "local_mean_over_levels": float(loc.mean()),
                "global_mean_over_levels": float(glo.mean()),
                "delta_mean_over_levels": float((loc - glo).mean()),
            }
    out["p1_global_vs_local"] = {"per_head": p1, "pooled": pooled}

    # maturation: u delta across checkpoints at N=64
    matr = {}
    for ck in ("random_init", "step2000", "step18000", "step19000", "latest"):
        if ck not in topn:
            continue
        loc = cell_matrix(topn, ck, "u", "local", 64)
        glo = cell_matrix(topn, ck, "u", "global", 64)
        matr[ck] = {
            "u_delta64_mean": float((loc - glo).mean()),
            "u_local64_mean": float(loc.mean()),
            "u_global64_mean": float(glo.mean()),
            "u_zero_fraction": mat[ck]["zero_fraction"]["u"]["mean"],
            "x_zero_fraction": mat[ck]["zero_fraction"]["x"]["mean"],
            "u_gini": mat[ck]["concentration"]["u"]["gini"]["mean"],
            "u_neff_over_K": mat[ck]["concentration"]["u"]["neff_over_K"][
                "mean"],
        }
    out["maturation"] = matr

    # stability headline
    st = stab["latest"]["summary"]
    out["stability"] = {}
    for key in ("x", "u"):
        out["stability"][key] = {
            "jaccard64_lag1": st[key]["lag_1"]["jaccard_mean_per_N"]["64"],
            "jaccard64_cross": st[key]["cross_row"]["jaccard_mean_per_N"]["64"],
            "jaccard1024_lag1": st[key]["lag_1"]["jaccard_mean_per_N"]["1024"],
            "jaccard1024_cross":
                st[key]["cross_row"]["jaccard_mean_per_N"]["1024"],
            "spearman_lag1": st[key]["lag_1"]["spearman_mean"],
            "spearman_cross": st[key]["cross_row"]["spearman_mean"],
            "support_lag1": st[key]["lag_1"]["support_jaccard_mean"],
            "support_cross": st[key]["cross_row"]["support_jaccard_mean"],
            "rbo_lag1": st[key]["lag_1"]["rbo_mean"],
            "rbo_cross": st[key]["cross_row"]["rbo_mean"],
        }

    # core/tail headline
    out["core_tail"] = {}
    for name in ("mass_top_1pct", "mass_top_6p25", "mass_top_25"):
        e = core["latest"]["defs"][name]
        out["core_tail"][name] = {
            "core_size_fraction_K": e["core_size_fraction_K"],
            "per_key": {
                kk["key"]: {
                    "core_mass_fraction_mean": float(np.mean(
                        [l["mass_fraction"]["mean"] for l in kk["levels"]])),
                    "residual_top64_mean": float(np.mean(
                        [l["residual_topN"]["64"]["mean"]
                         for l in kk["levels"]])),
                    "coords_for_50pct_residual_mean": float(np.mean(
                        [l["residual_need_coords"]["p50"]["mean"]
                         for l in kk["levels"]])),
                } for kk in e["levels"]},
        }

    # frequency headline
    fnull = freq["latest"]["statistics"]
    fbands = freq["latest"]["band_null_intervals"]
    out["frequency"] = {}
    for key in ("x", "u"):
        st8 = fnull[key]["bands8"]
        cells = fbands["per_cell"][key]
        zs = [cells[f"L{l}_H{h}"]["max_share_z"]
              for l in range(8) for h in range(4)]
        out["frequency"][key] = {
            "pooled_max_share": st8["max_mass_share"]["observed"],
            "pooled_max_share_null": st8["max_mass_share"]["null_mean"],
            "pooled_max_share_z": st8["max_mass_share"]["z"],
            "pooled_entropy_z": st8["share_entropy"]["z"],
            "pooled_rho_freq_mass": st8[
                "spearman_freq_vs_mass_share"]["observed"],
            "pooled_rho_freq_mass_z": st8[
                "spearman_freq_vs_mass_share"]["z"],
            "per_cell_max_share_z_min": float(np.min(zs)),
            "per_cell_max_share_z_max": float(np.max(zs)),
            "cells_significant_z_gt_3": int(sum(1 for z in zs if z > 3)),
            "n_cells": len(zs),
        }
    out["frequency"]["random_init_x"] = {
        "pooled_max_share":
            freq["random_init"]["statistics"]["x"]["bands8"][
                "max_mass_share"]["observed"],
        "pooled_max_share_null":
            freq["random_init"]["statistics"]["x"]["bands8"][
                "max_mass_share"]["null_mean"],
        "z": freq["random_init"]["statistics"]["x"]["bands8"][
            "max_mass_share"]["z"],
    }

    # external reference
    out["external"] = {
        "gemma_encoder_norm_pr_over_n": {
            w: sae[w]["spectrum"]["encoder"]["participation_ratio"] / 640.0
            for w in ("small", "medium")},
        "gemma_effective_rank_encoder": {
            w: sae[w]["spectrum"]["encoder"]["effective_rank_entropy"]
            for w in ("small", "medium")},
        "gemma_decoder_nn1_median": {
            w: sae[w]["decoder_similarity"]["nn1_top1"]["p50"]
            for w in ("small", "medium")},
        "gemma_encoder_nn1_median": {
            w: sae[w]["encoder_similarity"]["nn1_top1"]["p50"]
            for w in ("small", "medium")},
        "gemma_decoder_mutual_nn_rate": {
            w: sae[w]["decoder_similarity"]["mutual_nn_rate_top1"]
            for w in ("small", "medium")},
        "anthropic_density_gini":
            anthro["monosemantic_2023"]["density_gini"],
        "anthropic_density_median":
            anthro["monosemantic_2023"]["density"]["p50"],
        "cross_system_gini": {
            "arm_a_x_mass_pooled":
                cross["systems"]["arm_a_latest"]["x_mass_pooled"]["gini"],
            "arm_a_u_mass_pooled":
                cross["systems"]["arm_a_latest"]["u_mass_pooled"]["gini"],
            "gemma_small_encoder_norm_sq":
                cross["systems"]["gemma_small"]["encoder_norm_sq"]["gini"],
            "anthropic_density":
                cross["systems"]["anthropic_public"]["density"]["gini"],
        },
    }
    out["controls_random_init"] = {
        "x_zero": ctrl["random_init"]["x_zero_fraction"],
        "u_zero": ctrl["random_init"]["u_zero_fraction"],
        "x_neff_over_K": ctrl["random_init"]["x_neff_over_K"],
        "u_neff_over_K": ctrl["random_init"]["u_neff_over_K"],
        "x_jaccard64_lag1":
            ctrl["random_init"]["x_jaccard64_lag1"],
        "u_local64_head0":
            ctrl["random_init"]["u_local64_head0"],
        "u_global64_head0":
            ctrl["random_init"]["u_global64_head0"],
    }
    save_json(RESULTS_DIR / "headline_stats.json", out)
    print(json.dumps({"ok": True}, indent=2))


if __name__ == "__main__":
    main()
