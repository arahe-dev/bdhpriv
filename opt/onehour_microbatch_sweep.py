"""Microbatch sweep: one process per (arm, microbatch) at constant global
batch. Records OOM/failed geometries explicitly.

python opt/onehour_microbatch_sweep.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    ("dense", 1), ("dense", 2), ("dense", 4),
    ("sparse", 1), ("sparse", 2), ("sparse", 4), ("sparse", 8),
    ("sparse", 16),
]


def main():
    tmp = Path("results/onehour_mb_cfg")
    tmp.mkdir(parents=True, exist_ok=True)
    records = []
    for arm, mb in CONFIGS:
        out = tmp / f"{arm}_mb{mb}.json"
        cmd = [sys.executable, "opt/onehour_microbatch.py", "--arm", arm,
               "--microbatch", str(mb), "--out", str(out)]
        print(json.dumps({"launch": f"{arm}_mb{mb}"}), flush=True)
        proc = subprocess.run(cmd, cwd=str(SCRIPT_ROOT),
                              capture_output=True, text=True)
        if proc.returncode != 0:
            records.append({"arm": arm, "microbatch": mb, "status": "failed",
                            "stderr_tail": proc.stderr[-400:]})
            print(json.dumps({"failed": f"{arm}_mb{mb}"}), flush=True)
            continue
        rec = json.loads(out.read_text(encoding="utf-8"))
        rec["status"] = "ok"
        records.append(rec)

    ok = [r for r in records if r.get("status") == "ok"]
    dense = {r["microbatch"]: r for r in ok if r["arm"] == "dense"}
    sparse = {r["microbatch"]: r for r in ok if r["arm"] == "sparse"}
    matched = []
    for mb in sorted(set(dense) & set(sparse)):
        matched.append({
            "microbatch": mb,
            "dense_ms": dense[mb]["median_ms"],
            "sparse_ms": sparse[mb]["median_ms"],
            "speedup": dense[mb]["median_ms"] / sparse[mb]["median_ms"],
            "dense_tok_s": dense[mb]["tok_per_s"],
            "sparse_tok_s": sparse[mb]["tok_per_s"],
            "dense_peak_GiB": dense[mb]["peak_mem_GiB"],
            "sparse_peak_GiB": sparse[mb]["peak_mem_GiB"],
        })
    best_dense = max(ok, key=lambda r: r["tok_per_s"] if r["arm"] == "dense"
                     else -1, default=None)
    best_sparse = max(ok, key=lambda r: r["tok_per_s"] if r["arm"] == "sparse"
                      else -1, default=None)
    summary = {
        "protocol": "constant global batch 64x2048; one process per config",
        "matched_geometry": matched,
        "best_feasible": {
            "dense": ({k: best_dense[k] for k in
                       ("microbatch", "median_ms", "tok_per_s",
                        "peak_mem_GiB", "kernels_per_update")}
                      if best_dense else None),
            "sparse": ({k: best_sparse[k] for k in
                        ("microbatch", "median_ms", "tok_per_s",
                         "peak_mem_GiB", "kernels_per_update")}
                       if best_sparse else None),
            "speedup": (best_dense["median_ms"] / best_sparse["median_ms"]
                        if best_dense and best_sparse else None),
        },
        "records": records,
    }
    out_path = Path("results/onehour_microbatch.json")
    out_path.write_text(json.dumps(summary, indent=2, default=str),
                        encoding="utf-8")
    print(json.dumps({"out": str(out_path),
                      "matched": matched,
                      "best_feasible": summary["best_feasible"]},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
