"""E3 finalize wrapper (PREPARED — not executed).

Runs the standard analysis scripts against a frozen-corpus census output
directory and writes the mission-required E3 file names without clobbering
the E2 artifacts:

  results/arm_a_science/e3_topn_global_local.json
  results/arm_a_science/e3_population_stability.json
  results/arm_a_science/e3_core_tail.json
  results/arm_a_science/e3_frequency_bands.json
  results/arm_a_science/e3_frequency.json

Usage (after e3_census.py has produced raw_frozen/):

  python analysis/arm_a_science/e3_finalize.py --tag v1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "arm_a_science"
RAW_FROZEN = RESULTS / "raw_frozen"


def run(cmd):
    print("RUN", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    ckpt = f"frozen_{args.tag}"
    if not (RAW_FROZEN / f"pass1_{ckpt}.npz").exists():
        raise SystemExit(f"missing {RAW_FROZEN / f'pass1_{ckpt}.npz'}; run "
                         f"e3_census.py --tag {args.tag} first")
    suffix = "" if args.tag == "v1" else f"_{args.tag}"
    names = {
        "topn": f"e3_topn_global_local{suffix}.json",
        "stab": f"e3_population_stability{suffix}.json",
        "core": f"e3_core_tail{suffix}.json",
        "freqb": f"e3_frequency_bands{suffix}.json",
        "freq": f"e3_frequency{suffix}.json",
    }
    run([args.python, "analysis/arm_a_science/analyze_topn.py",
         "--ckpts", ckpt, "--raw", str(RAW_FROZEN),
         "--out", str(RESULTS / names["topn"])])
    run([args.python, "analysis/arm_a_science/analyze_population.py",
         "--ckpts", ckpt, "--raw", str(RAW_FROZEN),
         "--out-stability", str(RESULTS / names["stab"]),
         "--out-core", str(RESULTS / names["core"])])
    run([args.python, "analysis/arm_a_science/frequency.py",
         "--ckpts", ckpt, "--raw", str(RAW_FROZEN),
         "--out-bands", str(RESULTS / names["freqb"]),
         "--out-null", str(RESULTS / names["freq"])])
    report = {
        "tag": args.tag,
        "checkpoint_key": ckpt,
        "outputs": list(names.values()),
        "comparison_targets": {
            "Delta_u(64)": "headline_stats.json p1_global_vs_local.per_head.u.64",
            "core_top6p25_u_mass": "core_tail.json mass_top_6p25 per_key.u",
            "per_head_band_z": "frequency_null.json ... band_null_intervals"
                               ".per_cell",
            "context_series": "run e3_census.py --context {256,512,1024} "
                              "into separate tags and compare with "
                              "context_length.json",
        },
        "note": "Predeclared primary statistics must be reported as measured; "
                "no metric redesign after seeing the results.",
    }
    (RESULTS / f"e3_summary_{args.tag}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
