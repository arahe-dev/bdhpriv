"""P3 analysis: training maturation of the sparse population.

Writes
  results/arm_a_science/sparsity_maturation.json
  results/arm_a_science/training_trajectory.json

Sparsity/concentration statistics are computed from pass-1 NPZ (synthetic
packed batches, E1/E2). Training-trajectory statistics are parsed from the
run log and are real training telemetry (E3-like for the log itself, but do
not measure population structure).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (  # noqa: E402
    CHECKPOINT_STEPS, KEYS, RAW_DIR, RESULTS_DIR, ROOT, save_json,
)


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64).ravel())
    if x.size == 0 or x[-1] <= 0:
        return float("nan")
    n = x.size
    i = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (i * x).sum()) / (n * x.sum()) - (n + 1.0) / n)


def neff(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return float("nan")
    return float(np.exp(-(p * np.log(p)).sum()))


def pratio(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    return float(1.0 / (p * p).sum()) if p.sum() > 0 else float("nan")


def checkpoint_metrics(ckpt: str, raw: Path) -> dict:
    p1 = np.load(raw / f"pass1_{ckpt}.npz")
    meta = json.loads((raw / f"meta_{ckpt}.json").read_text(encoding="utf-8"))
    L = p1["mass_sum_x"].shape[1]
    H = p1["mass_sum_x"].shape[2]
    K = p1["mass_sum_x"].shape[3]
    n_batches = p1["mass_sum_x"].shape[0]
    rows = meta["specs"][0]["rows"]
    n_tokens = n_batches * rows * meta["model_config"]["T"]
    out = {
        "step": 0 if ckpt == "random_init" else CHECKPOINT_STEPS[ckpt],
        "n_levels": int(L), "n_heads": int(H), "K": int(K),
        "n_tokens_population": int(n_tokens),
        "zero_fraction": {}, "pair_zero_fraction": {},
        "concentration": {}, "activation": {},
    }
    for key in KEYS:
        pos = p1[f"pos_count_{key}"].sum(0)          # (L,H,K)
        mass = p1[f"mass_sum_{key}"].sum(0)
        pair_pos = p1[f"pair_pos_count_{key}"].sum(0)  # (L,H,K/2)
        zero = 1.0 - pos / float(n_tokens)
        pzero = 1.0 - pair_pos / float(n_tokens)
        out["zero_fraction"][key] = {
            "mean": float(zero.mean()), "p10": float(np.quantile(zero, 0.1)),
            "p50": float(np.quantile(zero, 0.5)),
            "p90": float(np.quantile(zero, 0.9)),
            "per_level_mean": zero.mean(axis=(1, 2)).tolist(),
        }
        out["pair_zero_fraction"][key] = {
            "mean": float(pzero.mean()),
            "per_level_mean": pzero.mean(axis=(1, 2)).tolist(),
        }
        neff_frac, ginis, prs, top1, top5, dead, active = [], [], [], [], [], [], []
        for level in range(L):
            for h in range(H):
                m = mass[level, h]
                p = m / max(m.sum(), 1e-30)
                neff_frac.append(neff(p) / K)
                ginis.append(gini(m))
                prs.append(pratio(p) / K)
                order = np.sort(p)[::-1]
                take1 = max(1, K // 100)
                take5 = max(1, K // 20)
                top1.append(float(order[:take1].sum()))
                top5.append(float(order[:take5].sum()))
                dead.append(float((pos[level, h] == 0).mean()))
                active.append(float((pos[level, h] / n_tokens > 1e-3).mean()))
        out["concentration"][key] = {
            "neff_over_K": {"mean": float(np.mean(neff_frac)),
                            "min": float(np.min(neff_frac)),
                            "max": float(np.max(neff_frac))},
            "gini": {"mean": float(np.mean(ginis)),
                     "min": float(np.min(ginis)),
                     "max": float(np.max(ginis))},
            "participation_ratio_over_K": {"mean": float(np.mean(prs))},
            "top1pct_mass_share": {"mean": float(np.mean(top1))},
            "top5pct_mass_share": {"mean": float(np.mean(top5))},
            "dead_coordinate_fraction": {"mean": float(np.mean(dead))},
            "active_pact_gt_1e_3_fraction": {"mean": float(np.mean(active))},
        }
        # mean positive magnitude per coordinate
        mp = mass / np.maximum(pos, 1)
        out["activation"][key] = {
            "mean_positive_magnitude_per_coordinate": {
                "mean": float(mp.mean()),
                "p50": float(np.quantile(mp, 0.5)),
                "p90": float(np.quantile(mp, 0.9)),
                "p99": float(np.quantile(mp, 0.99)),
            },
            "total_mass": float(mass.sum()),
        }
    return out


def training_trajectory(run_dir: Path) -> dict:
    log = run_dir / "logs" / "train.jsonl"
    updates = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("event") == "update":
            updates.append(rec)
    if not updates:
        return {"error": "no updates"}
    steps = np.array([u["step"] for u in updates])
    tokens = np.array([u["tokens_consumed"] for u in updates])
    loss = np.array([u["loss"] for u in updates])
    grad = np.array([u["grad_norm"] for u in updates])
    lr = np.array([u["lr"] for u in updates])
    toks = np.array([u["tok_s"] for u in updates])
    peak = np.array([u.get("mem_peak_GiB", np.nan) for u in updates])

    def rolling(v, w=51, fn=np.median):
        out = np.empty_like(v, dtype=np.float64)
        half = w // 2
        for i in range(len(v)):
            out[i] = fn(v[max(0, i - half):i + half + 1])
        return out

    loss_med = rolling(loss)
    spikes = np.where(loss > loss_med + 3 * 1.4826 * rolling(
        np.abs(loss - loss_med), 51, np.median))[0]
    return {
        "source": str(log),
        "n_updates": int(len(updates)),
        "steps": [int(steps[0]), int(steps[-1])],
        "tokens": [int(tokens[0]), int(tokens[-1])],
        "loss_first": float(loss[0]), "loss_last": float(loss[-1]),
        "loss_min": float(loss.min()),
        "loss_last50_mean": float(loss[-50:].mean()),
        "loss_at_checkpoint_steps": {
            str(s): float(loss[np.argmin(np.abs(steps - s))])
            for s in (2000, 18000, 19000, 19074)},
        "grad_norm_last": float(grad[-1]),
        "grad_norm_median": float(np.median(grad)),
        "lr_last": float(lr[-1]),
        "tok_s_median": float(np.median(toks)),
        "peak_mem_GiB": float(np.nanmax(peak)),
        "spike_count_3mad": int(len(spikes)),
        "spike_steps": steps[spikes].astype(int).tolist(),
        "curves_sampled": {
            "steps": steps[::10].tolist(),
            "loss": loss[::10].tolist(),
            "loss_rolling_median": loss_med[::10].tolist(),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", nargs="+",
                    default=["step2000", "step18000", "step19000", "latest"])
    ap.add_argument("--raw", default=str(RAW_DIR))
    args = ap.parse_args()
    raw = Path(args.raw)
    mat = {"meta": {
        "evidence_class": "E1/E2 synthetic packed batches, one trajectory",
        "note": "checkpoints are from ONE training run; not independent seeds",
        "n_tokens_population_per_checkpoint": "4 batches x 4 rows x 2048",
    }, "checkpoints": {}}
    for ckpt in args.ckpts:
        p = raw / f"pass1_{ckpt}.npz"
        if p.exists():
            mat["checkpoints"][ckpt] = checkpoint_metrics(ckpt, raw)
    # attach trajectory metrics + cross-checkpoint deltas
    mat["training_trajectory"] = training_trajectory(
        ROOT / "runs" / "arm_a_2p5b_opt3c_all")
    steps = [mat["checkpoints"][c]["step"] for c in mat["checkpoints"]]
    for key in KEYS:
        mat.setdefault("deltas", {})[key] = {}
        for metric in ("zero_fraction", "pair_zero_fraction"):
            vals = [mat["checkpoints"][c][metric][key]["mean"]
                    for c in mat["checkpoints"]]
            mat["deltas"][key][metric] = {
                "steps": steps, "values": vals,
                "first_to_last": float(vals[-1] - vals[0]),
            }
        vals = [mat["checkpoints"][c]["concentration"][key]["neff_over_K"]
                ["mean"] for c in mat["checkpoints"]]
        mat["deltas"][key]["neff_over_K"] = {"steps": steps, "values": vals}
        vals = [mat["checkpoints"][c]["concentration"][key]["top5pct_mass_share"]
                ["mean"] for c in mat["checkpoints"]]
        mat["deltas"][key]["top5pct_mass_share"] = {"steps": steps,
                                                    "values": vals}
    save_json(RESULTS_DIR / "sparsity_maturation.json", mat)
    traj = {
        "meta": {
            "note": "training telemetry from the Arm-A production run log; "
                    "single run, single seed",
        },
        **mat["training_trajectory"],
    }
    save_json(RESULTS_DIR / "training_trajectory.json", traj)
    print(json.dumps({"checkpoints": list(mat["checkpoints"]),
                      "trajectory_updates": traj["n_updates"]}, indent=2))


if __name__ == "__main__":
    main()
