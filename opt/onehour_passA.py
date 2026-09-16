"""Pass A sweep: active-width frontier for the one-hour mission.

One fresh process per config (doe_config.py), same-process Arm-A control,
fixed cyclic contiguous windows, compact minimum capacity, static shapes.

python opt/onehour_passA.py   (runs inside the container)
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]


def configs():
    runs = [
        dict(id="A0_M8_Ke512_top1_G128", experts=8, ke=512, top_r=1, G=128),
        dict(id="A1_M16_Ke256_top1_G128", experts=16, ke=256, top_r=1, G=128),
        dict(id="A2_M32_Ke128_top1_G64", experts=32, ke=128, top_r=1, G=64),
        dict(id="A3_M64_Ke64_top1_G32", experts=64, ke=64, top_r=1, G=32),
        dict(id="A0_M8_Ke512_top1_G128_rep", experts=8, ke=512, top_r=1,
             G=128),
        dict(id="A1_M16_Ke256_top1_G128_rep", experts=16, ke=256, top_r=1,
             G=128),
    ]
    random.Random(4242).shuffle(runs)
    return runs


def main():
    out_path = Path("results/onehour_active_width.json")
    tmp_dir = Path("results/onehour_cfg")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for run in configs():
        out = tmp_dir / f"{run['id']}.json"
        cmd = [
            sys.executable, "opt/doe_config.py",
            "--top-r", str(run["top_r"]), "--G", str(run["G"]),
            "--experts", str(run["experts"]),
            "--expert-width", str(run["ke"]),
            "--id", run["id"], "--steps", "6", "--warmups", "2",
            "--out", str(out),
        ]
        print(json.dumps({"launch": run["id"]}), flush=True)
        subprocess.run(cmd, check=True, cwd=str(SCRIPT_ROOT))
        records.append(json.loads(out.read_text(encoding="utf-8")))

    baselines = [r["baseline"]["median_ms"] for r in records]
    summary = {
        "protocol": "one process per config; same-process Arm-A control",
        "baseline_medians_ms": baselines,
        "baseline_median_of_medians_ms": float(np.median(baselines)),
        "baseline_noise_ms": float(np.std(baselines)),
        "records": [
            {
                "id": r["id"],
                "M": r["route"]["active_experts"].__len__(),
                "top_r": r["top_r"],
                "G": r["G"],
                "active_K_per_head": r["top_r"] * int(
                    r["candidate"]["ledger"]["Ke"]),
                "stored_K_per_head": r["candidate"]["ledger"][
                    "stored_K_per_head"],
                "candidate_ms": r["candidate"]["median_ms"],
                "p10_ms": r["candidate"]["p10_ms"],
                "p90_ms": r["candidate"]["p90_ms"],
                "speedup": r["speedup_same_session"],
                "peak_mem_GiB": r["candidate"]["peak_mem_GiB"],
                "capacity_tokens": r["route"]["capacity_tokens"],
                "padding_fraction": r["route"]["padding_fraction"],
                "active_experts": r["route"]["active_experts"],
                "compile_s": r["compile_seconds"]["candidate"],
                "ms_array": r["candidate"]["ms"],
            }
            for r in records
        ],
    }
    out_path.write_text(json.dumps(summary, indent=2, default=str),
                        encoding="utf-8")
    print(json.dumps({
        "out": str(out_path),
        "baseline_medians_ms": [round(x, 1) for x in baselines],
        "table": [
            {k: (round(v, 2) if isinstance(v, float) else v)
             for k, v in rec.items()
             if k in ("id", "candidate_ms", "speedup", "capacity_tokens",
                      "padding_fraction", "peak_mem_GiB", "compile_s")}
            for rec in summary["records"]
        ],
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
