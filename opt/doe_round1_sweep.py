"""Round-1 DOE sweep driver: one process per config, randomized order.

Runs inside the container:
  python opt/doe_round1_sweep.py --out results/doe_round1_sweep.json
Aggregates per-config JSONs, repeats, controls, response-surface fit and
residual-runtime model.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]


def configs():
    runs = []
    for top_r in (2, 1, 4):
        for group in (64, 128, 256, 512):
            runs.append(dict(top_r=top_r, G=group, cap=1.0))
    for factor in (1.25, 1.5, 2.0):
        runs.append(dict(top_r=2, G=128, cap=factor))
    repeats = [
        dict(top_r=2, G=128, cap=1.0),
        dict(top_r=2, G=128, cap=1.0),
        dict(top_r=2, G=128, cap=1.0),
    ]
    schedule = runs + repeats
    random.Random(1337).shuffle(schedule)
    return schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/doe_round1_sweep.json")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()

    schedule = configs()
    records = []
    tmp_dir = Path("results/doe_cfg")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for index, run in enumerate(schedule):
        run_id = f"top{run['top_r']}_G{run['G']}_cap{run['cap']}_r{index}"
        out_path = tmp_dir / f"{run_id}.json"
        cmd = [
            sys.executable, "opt/doe_config.py",
            "--top-r", str(run["top_r"]), "--G", str(run["G"]),
            "--cap", str(run["cap"]), "--id", run_id,
            "--steps", str(args.steps), "--warmups", str(args.warmups),
            "--out", str(out_path),
        ]
        print(json.dumps({"launch": run_id}), flush=True)
        subprocess.run(cmd, check=True, cwd=str(SCRIPT_ROOT))
        records.append(json.loads(out_path.read_text(encoding="utf-8")))

    baseline_ms = [r["baseline"]["median_ms"] for r in records]
    summary = {
        "configs": len(records),
        "baseline_median_ms_all": baseline_ms,
        "baseline_median_of_medians": float(np.median(baseline_ms)),
        "baseline_noise_ms": float(np.std(baseline_ms)),
        "speedups": {r["id"]: round(r["speedup_same_session"], 3)
                     for r in records},
        "fit": fit(records),
        "residual_runtime_model": residual(records),
    }
    out = Path(args.out)
    out.write_text(json.dumps({"summary": summary, "records": records},
                              indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return 0


def fit(records):
    def feature(r):
        width = r["top_r"] * 512
        group = r["G"]
        return [1.0, width / 1024.0, group / 128.0, (group / 128.0) ** 2,
                r["route"]["capacity_tokens"] / 512.0,
                r["route"]["padding_fraction"],
                (width / 1024.0) * (group / 128.0)]
    x = np.array([feature(r) for r in records])
    y = np.array([r["candidate"]["median_ms"] for r in records])
    coeff, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coeff
    return {
        "terms": ["intercept", "active_K/1024", "G/128", "G2",
                  "capacity/512", "padding_fraction", "activeK:G"],
        "coefficients": coeff.tolist(),
        "r2": float(1 - ((y - pred) ** 2).sum()
                    / max(1e-12, ((y - y.mean()) ** 2).sum())),
        "n": len(records),
    }


def residual(records):
    base = [r for r in records if r["route"]["padding_fraction"] < 0.05]
    k = np.array([r["top_r"] * 512 for r in base], dtype=np.float64)
    t = np.array([r["candidate"]["median_ms"] for r in base],
                 dtype=np.float64)
    a = np.polyfit(k, t, 1)
    return {
        "records": [r["id"] for r in base],
        "Kactive": k.tolist(),
        "median_ms": t.tolist(),
        "T_fixed_ms": float(a[1]),
        "ms_per_Kactive_head": float(a[0]),
        "fit_r2": float(1 - ((t - np.polyval(a, k)) ** 2).sum()
                        / max(1e-12, ((t - t.mean()) ** 2).sum())),
        "note": "T(Kactive) = T_fixed + a*Kactive; dispatch folded into T_fixed",
    }


if __name__ == "__main__":
    raise SystemExit(main())
