"""Core/tail sensitivity ladder: 1% / 3.125% / 6.25% / 12.5% / 25%.

Reads the dedicated collect run (raw/raw_ladder) whose core definitions are
exactly that predeclared ladder, and asks whether the core+tail decomposition
has a stable operating region rather than one cherry-picked threshold.

Writes results/arm_a_science/core_ladder.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import LADDER, RAW_DIR, RESULTS_DIR, save_json  # noqa: E402

FRACS = {
    "mass_top_01": 0.01,
    "mass_top_03125": 0.03125,
    "mass_top_0625": 0.0625,
    "mass_top_125": 0.125,
    "mass_top_25": 0.25,
}
RAW = RAW_DIR / "raw_ladder"


def _q(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {"mean": float(v.mean()), "p10": float(np.quantile(v, 0.1)),
            "p50": float(np.quantile(v, 0.5)),
            "p90": float(np.quantile(v, 0.9)), "n": int(v.size)}


def main():
    p2 = np.load(RAW / "pass2_latest.npz")
    ranks = np.load(RAW / "ranks_latest.npz")
    meta = json.loads((RAW / "meta_latest.json").read_text(encoding="utf-8"))
    K = meta["model_config"]["N"] // meta["model_config"]["H"]
    out = {
        "meta": {
            "evidence_class": "E2 (synthetic packed batches, latest "
                              "checkpoint, one trajectory)",
            "ladder": [FRACS[n] for n in FRACS],
            "residual_quantiles": meta["res_quantiles"],
            "core_definition": "cross-fitted global mass top fraction per "
                               "(level, head); mask from the other batch "
                               "split",
            "note": "quality under matched active width requires the E4 "
                    "training intervention; this file reports coverage and "
                    "tail geometry (see e4_forward_ablation.json for the "
                    "forward-only fidelity pilot)",
        },
        "ladder": {},
    }
    for name, frac in FRACS.items():
        entry = {"core_fraction_of_K": frac,
                 "core_coords_per_head": int(round(frac * K))}
        # measured mask size (should match the fraction)
        sizes = [ranks[f"core_{name}_x_L{level}_{split}"].sum()
                 for level in range(meta["model_config"]["L"])
                 for split in ("A", "B")]
        entry["measured_core_fraction"] = float(
            np.mean(sizes) / (K * meta["model_config"]["H"]))
        entry["per_key"] = {}
        for key in ("x", "u"):
            frac_arr = p2[f"core_{name}_frac_{key}"]
            cpos = p2[f"core_{name}_cpos_{key}"]
            res = p2[f"core_{name}_res_ratio_{key}"]
            need = p2[f"core_{name}_res_need_{key}"]
            entry["per_key"][key] = {
                "core_mass_fraction": _q(frac_arr),
                "tokens_with_core_active": float((cpos > 0).mean()),
                "residual_top64": float(np.nanmean(res[..., 2])),
                "residual_top256": float(np.nanmean(res[..., 4])),
                "coords_for_50pct_residual": _q(need[..., 0]),
                "coords_for_80pct_residual": _q(need[..., 1]),
                "coords_for_90pct_residual": _q(need[..., 2]),
            }
            # matched-width coverage proxy: core + tail coords needed for 90%
            # of the residual, expressed as fraction of K per head
            c90 = float(np.nanmean(need[..., 2]))
            entry["per_key"][key]["active_width_for_90pct_total"] = {
                "coords": float(round(frac * K) + c90),
                "fraction_of_K": float((round(frac * K) + c90) / K),
            }
        # p_act-defined cores as a scale check
        out["ladder"][name] = entry
    # p_act definitions
    for name in ("pact_ge_0p5", "pact_ge_0p9"):
        entry = {"per_key": {}}
        for key in ("x", "u"):
            frac_arr = p2[f"core_{name}_frac_{key}"]
            cpos = p2[f"core_{name}_cpos_{key}"]
            res = p2[f"core_{name}_res_ratio_{key}"]
            entry["per_key"][key] = {
                "core_mass_fraction": _q(frac_arr),
                "tokens_with_core_active": float((cpos > 0).mean()),
                "residual_top64": float(np.nanmean(res[..., 2])),
            }
        out["ladder"][name] = entry
    save_json(RESULTS_DIR / "core_ladder.json", out)
    print(json.dumps({"ok": True}, indent=2))


if __name__ == "__main__":
    main()
