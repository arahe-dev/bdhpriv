"""Control comparison: canonical random init vs trained checkpoints.

Writes results/arm_a_science/controls.json. The random-init run uses the same
sampling and capture code paths (E1 diagnostic control, seed-matched data).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import KEYS, RESULTS_DIR, save_json  # noqa: E402


def main():
    mat = json.loads((RESULTS_DIR / "sparsity_maturation.json").read_text(
        encoding="utf-8"))["checkpoints"]
    topn = json.loads((RESULTS_DIR / "topn_global_local.json").read_text(
        encoding="utf-8"))["checkpoints"]
    freq = json.loads((RESULTS_DIR / "frequency_bands.json").read_text(
        encoding="utf-8"))["checkpoints"]
    stab = json.loads((RESULTS_DIR / "population_stability.json").read_text(
        encoding="utf-8"))["checkpoints"]

    def summarize(c):
        m = mat[c]
        out = {"step": m["step"]}
        for key in KEYS:
            out[f"{key}_zero_fraction"] = m["zero_fraction"][key]["mean"]
            out[f"{key}_pair_zero_fraction"] = \
                m["pair_zero_fraction"][key]["mean"]
            out[f"{key}_neff_over_K"] = \
                m["concentration"][key]["neff_over_K"]["mean"]
            out[f"{key}_gini"] = m["concentration"][key]["gini"]["mean"]
            out[f"{key}_pr_over_K"] = \
                m["concentration"][key]["participation_ratio_over_K"]["mean"]
            out[f"{key}_top1pct_share"] = \
                m["concentration"][key]["top1pct_mass_share"]["mean"]
            out[f"{key}_top5pct_share"] = \
                m["concentration"][key]["top5pct_mass_share"]["mean"]
        # pooled x and per-cell u global-vs-local at N=64 and 256
        lv = topn[c]["pooled"]["x"]["levels"][4]
        out["x_pooled_local64"] = lv["local"]["64"]["mean"]
        out["x_pooled_global64"] = lv["global"]["64"]["mean"]
        u = topn[c]["per_cell"]["u"]["levels"][4]["heads"][0]
        out["u_local64_head0"] = u["local"]["64"]["mean"]
        out["u_global64_head0"] = u["global"]["64"]["mean"]
        out["u_local256_head0"] = u["local"]["256"]["mean"]
        out["u_global256_head0"] = u["global"]["256"]["mean"]
        # band profile (x/u, 8 bands, level 0 head 0)
        e = freq[c]["x"]["levels"][0]["heads"][0]["bands8"]
        out["x_band_mass_share_8"] = [round(v, 4)
                                      for v in e["mass_share"]]
        out["x_band_occupancy_8"] = [round(v, 4) for v in e["occupancy"]]
        e = freq[c]["x"]["levels"][0]["heads"][0]["bands8"]
        out["x_top64_band_share_8"] = [round(v, 4) for v in
                                       e["top64_membership_share"]]
        e = freq[c]["u"]["levels"][0]["heads"][0]["bands8"]
        out["u_band_mass_share_8"] = [round(v, 4)
                                      for v in e["mass_share"]]
        # stability summary (x, lag_1 vs cross_row, at N=64)
        sc = stab[c]["summary"]["x"]
        out["x_jaccard64_lag1"] = sc["lag_1"]["jaccard_mean_per_N"]["64"]
        out["x_jaccard64_cross"] = sc["cross_row"]["jaccard_mean_per_N"]["64"]
        out["x_spearman_lag1"] = sc["lag_1"]["spearman_mean"]
        out["x_spearman_cross"] = sc["cross_row"]["spearman_mean"]
        return out

    out = {
        "meta": {
            "evidence_class": "E1 diagnostic control (canonical random init, "
                              "same synthetic batches)",
            "note": "random-init numbers are NOT production evidence; they "
                    "calibrate which structures training created",
        },
        "checkpoints": {c: summarize(c) for c in mat},
    }
    init = out["checkpoints"]["random_init"]
    trained = out["checkpoints"]["latest"]
    deltas = {}
    for k, v in init.items():
        if k == "step" or isinstance(v, list):
            continue
        t = trained.get(k)
        if isinstance(v, (int, float)) and isinstance(t, (int, float)):
            deltas[k] = {"init": v, "trained": t, "delta": t - v,
                         "ratio": (t / v) if v else None}
    out["init_vs_trained"] = deltas
    save_json(RESULTS_DIR / "controls.json", out)
    print(json.dumps({
        "init": {k: init[k] for k in
                 ("x_zero_fraction", "u_zero_fraction", "x_gini",
                  "x_neff_over_K", "u_gini", "u_neff_over_K")},
        "trained": {k: trained[k] for k in
                    ("x_zero_fraction", "u_zero_fraction", "x_gini",
                     "x_neff_over_K", "u_gini", "u_neff_over_K")},
    }, indent=2))


if __name__ == "__main__":
    main()
