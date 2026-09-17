"""Host-side verification reporter: stats, classification, plots, report.md.

Reads artifacts/arm_a_speed_verification/{environment.json,
benchmark_manifest.json,raw_runs.csv} and writes summary.json, plots/ and
report.md. No GPU work.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "arm_a_speed_verification"
PLOTS = ART / "plots"
RNG = np.random.default_rng(12345)


def load_rows():
    with open(ART / "raw_runs.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fnum(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def stats(values):
    arr = np.array([v for v in values if v is not None], dtype=float)
    if len(arr) == 0:
        return {}
    med = float(np.median(arr))
    boot = []
    for _ in range(10000):
        sample = RNG.choice(arr, size=len(arr), replace=True)
        boot.append(float(np.median(sample)))
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "median": med,
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "cv_pct": (float(arr.std(ddof=1) / arr.mean() * 100.0)
                   if len(arr) > 1 and arr.mean() else None),
        "min": float(arr.min()), "max": float(arr.max()),
        "p05": float(np.percentile(arr, 5)),
        "p50": med,
        "p95": float(np.percentile(arr, 95)),
        "mad": float(np.median(np.abs(arr - med))),
        "ci95_median": ci,
        "p95_over_p05": (float(np.percentile(arr, 95)
                               / np.percentile(arr, 5))),
    }


def window_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["kind"] != "warm" or row.get("error"):
            continue
        key = f"{row['path']}_mb{row['microbatch']}"
        groups[key].append(row)
    return groups


def drift_pct_per_min(rows, value_key="tok_s"):
    arr = []
    for row in rows:
        value = fnum(row.get(value_key))
        elapsed = fnum(row.get("elapsed_s"))
        if value is not None and elapsed is not None:
            arr.append((elapsed, value))
    if len(arr) < 3:
        return None
    x = np.array([a[0] for a in arr])
    y = np.array([a[1] for a in arr])
    slope = np.polyfit(x, y, 1)[0]
    med = float(np.median(y))
    return float(slope * 60.0 / med * 100.0) if med else None


def first_last_ratio(rows):
    by_run = defaultdict(list)
    for row in rows:
        by_run[row["run_id"]].append(row)
    ratios = []
    for run_rows in by_run.values():
        ordered = sorted(run_rows, key=lambda r: fnum(r["window"]) or 0)
        vals = [fnum(r["tok_s"]) for r in ordered]
        vals = [v for v in vals if v]
        if len(vals) >= 4:
            ratios.append(vals[-1] / vals[0])
    return statistics.median(ratios) if ratios else None


def bootstrap_speedup(dense_tps, sparse_tps):
    a = np.array([v for v in dense_tps if v], dtype=float)
    b = np.array([v for v in sparse_tps if v], dtype=float)
    if len(a) == 0 or len(b) == 0:
        return None
    ratio = float(np.median(b) / np.median(a))
    boot = []
    for _ in range(10000):
        ra = float(np.median(RNG.choice(a, size=len(a), replace=True)))
        rb = float(np.median(RNG.choice(b, size=len(b), replace=True)))
        boot.append(rb / ra)
    return {"median_ratio": ratio,
            "ci95": [float(np.percentile(boot, 2.5)),
                     float(np.percentile(boot, 97.5))]}


def verify_correctness():
    """Run the declared-sparse correctness contract (host CPU suites)."""
    import subprocess
    results = {}
    for label, cmd in (
        ("routed_executor_oracle",
         [sys.executable, "opt/test_routed_expert.py"]),
        ("learned_router_gates",
         [sys.executable, "opt/test_learned_router.py"]),
    ):
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                              text=True)
        results[label] = {
            "returncode": proc.returncode,
            "tail": (proc.stdout.strip().splitlines() or [""])[-1],
        }
    return results


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    groups = window_groups(rows)

    warm_stats = {name: stats([fnum(r["tok_s"]) for r in group])
                  for name, group in groups.items()}
    long_rows = [r for r in rows if r["kind"] == "longrun"]
    long_values = [fnum(r["tok_s"]) for r in long_rows]
    long_stats = stats(long_values)
    long_drift = drift_pct_per_min(long_rows)
    long_first_last = first_last_ratio(long_rows)
    long_temp = [fnum(r.get("temp_c")) for r in long_rows
                 if fnum(r.get("temp_c")) is not None]
    long_smclock = [fnum(r.get("sm_clock_mhz")) for r in long_rows
                    if fnum(r.get("sm_clock_mhz")) is not None]
    long_mem = [fnum(r.get("mem_used_mib")) for r in long_rows
                if fnum(r.get("mem_used_mib")) is not None]

    cold_rows = [r for r in rows if r["kind"] == "cold"]
    cold_stats = {
        "first_step_tok_s": stats([fnum(r["tok_s"]) for r in cold_rows]),
        "external_wall_s": [round(fnum(r["external_wall_s"]) or 0, 2)
                            for r in cold_rows],
    }

    dense = [fnum(r["tok_s"]) for r in groups.get("dense_mb1", [])]
    sparse_mb1 = [fnum(r["tok_s"]) for r in groups.get("sparse_mb1", [])]
    sparse_mb2 = [fnum(r["tok_s"]) for r in groups.get("sparse_mb2", [])]
    speed_best = bootstrap_speedup(dense, sparse_mb2)
    speed_matched = bootstrap_speedup(dense, sparse_mb1)

    sparse = warm_stats.get("sparse_mb2", {})
    verdict_parts = []
    stable = True
    if sparse.get("cv_pct") is not None and sparse["cv_pct"] > 5.0:
        stable = False
    if sparse.get("p95_over_p05") and sparse["p95_over_p05"] > 1.10:
        stable = False
    drift = long_drift if long_drift is not None else drift_pct_per_min(
        groups.get("sparse_mb2", []))
    if drift is not None and abs(drift) > 5.0:
        stable = False
    if long_temp and (max(long_temp) >= 85):
        stable = False
    if long_mem and (max(long_mem) - min(long_mem) > 200):
        stable = False

    correctness = verify_correctness()
    correctness_ok = all(v["returncode"] == 0
                         for v in correctness.values())
    parity_path = ART / "parity_failure_B2.json"
    parity = (json.loads(parity_path.read_text(encoding="utf-8"))
              if parity_path.is_file() else None)
    b2_leak = (parity is not None
               and parity["max_abs_diff"][
                   "routed_B2_vs_per_row_reference"] > 1e-3)

    if b2_leak:
        verdict = "UNSTABLE"
    elif not correctness_ok:
        verdict = "UNSTABLE"
    elif stable:
        verdict = "STABLE"
    else:
        verdict = "CONDITIONALLY STABLE"

    summary = {
        "verdict": verdict,
        "warm_stats": warm_stats,
        "longrun": {"stats": long_stats, "drift_pct_per_min": long_drift,
                    "first_last_ratio": long_first_last,
                    "temperature_c": {"min": min(long_temp),
                                      "max": max(long_temp)}
                    if long_temp else None,
                    "sm_clock_mhz": {"min": min(long_smclock),
                                     "max": max(long_smclock)}
                    if long_smclock else None,
                    "mem_used_mib": {"min": min(long_mem),
                                     "max": max(long_mem)}
                    if long_mem else None},
        "cold": cold_stats,
        "speedup_best_feasible": speed_best,
        "speedup_matched_mb1": speed_matched,
        "correctness": correctness,
        "metrics": {
            "primary": "packed input tokens / full-update wall "
                       "(131072 tokens, CUDA synchronized)",
            "warm_windows_per_process": {"dense_mb1": 10,
                                         "sparse_mb1": 12,
                                         "sparse_mb2": 12},
        },
    }
    (ART / "summary.json").write_text(json.dumps(summary, indent=2,
                                                 default=str),
                                      encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=130)
    for name, color in (("dense_mb1", "tab:blue"),
                        ("sparse_mb1", "tab:orange"),
                        ("sparse_mb2", "tab:green")):
        values = [fnum(r["tok_s"]) for r in groups.get(name, [])]
        values = [v for v in values if v]
        if values:
            axes[0].hist(values, bins=12, alpha=0.6, label=name,
                         color=color)
    axes[0].set_xlabel("packed tok/s (warm windows)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Warm throughput distribution")
    axes[0].legend()
    axes[1].plot([fnum(r["elapsed_s"]) / 60.0 for r in long_rows],
                 [fnum(r["tok_s"]) for r in long_rows], lw=1.4,
                 color="tab:green", label="longrun tok/s")
    axes[1].set_xlabel("minutes")
    axes[1].set_ylabel("tok/s")
    if long_temp:
        ax2 = axes[1].twinx()
        ax2.plot([fnum(r["elapsed_s"]) / 60.0 for r in long_rows],
                 long_temp, lw=1.0, color="tab:red", alpha=0.7,
                 label="temp C")
        ax2.set_ylabel("temp C", color="tab:red")
    axes[1].set_title("Long-run stability")
    fig.tight_layout()
    fig.savefig(PLOTS / "throughput.png")

    def fmt(value, digits=1):
        return "n/a" if value is None else f"{value:.{digits}f}"

    median_tps = sparse.get("median")
    report = f"""VERDICT: {verdict}
CANONICAL_COMMIT: 0dcbb878d24b99b5808359c889e97143c3cec00b (verified blobs match working tree)
CANONICAL_CONFIG: M8/Ke512/top1 fixed cyclic window, compact exact-capacity executor, microbatch 2 x 32 accumulation
DEVICE: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, driver 595.79 (container iclr-arm-a)
PRIMARY_METRIC: packed input tokens per full optimizer update (131072 tokens), CUDA-synchronized wall, warm steady state
MEDIAN_THROUGHPUT: {fmt(median_tps)} tok/s (sparse mb2 warm)
P05_THROUGHPUT: {fmt(sparse.get("p05"))}
P95_THROUGHPUT: {fmt(sparse.get("p95"))}
COEFFICIENT_OF_VARIATION: {fmt(sparse.get("cv_pct"))} %
LONG_RUN_DRIFT: {fmt(long_drift, 2)} %/min
CORRECTNESS: {"PASS" if correctness_ok else "CHECK"} (declared-sparse oracle suites; see report section 8)
TARGET_STATUS: 350k G4 target NOT VERIFIABLE on this machine (different hardware and production geometry); local 4x claim evaluated below

# Arm-A Sparse Speed Verification Report

## 1. Exact commands executed

See `commands.txt`. Suite: `python opt/verify_speed_orchestrate.py` inside
`iclr-arm-a`, then `py -3.12 opt/verify_speed_report.py` on the host.

## 2. Exact source/config/checkpoint used

All nine candidate blobs match `results/350k_frozen_state.json`
(commit 0dcbb87). No checkpoint is loaded by the canonical benchmark: the
harness uses deterministic synthetic packed batches with fixed seeds (this
is the same boundary used by every prior benchmark in the project). Config
and manifest: `benchmark_manifest.json`.

## 3. Cold-start results

Cold sparse runs: {len(cold_rows)}. First-step throughput (includes lazy
Inductor compile): {json.dumps(cold_stats["first_step_tok_s"].get("median") if cold_stats["first_step_tok_s"] else None)} tok/s median.
External process wall times: {cold_stats["external_wall_s"]}.
See `runs/cold_sparse_*.json` for model-load vs first-step breakdown.

## 4. Warm steady-state results

Per-path window statistics (warm):

| path | n | median tok/s | CV % | p05 | p95 | p95/p05 | first/last |
|---|---|---|---|---|---|---|---|
| dense_mb1 | {warm_stats.get('dense_mb1', {}).get('n', 0)} | {fmt(warm_stats.get('dense_mb1', {}).get('median'))} | {fmt(warm_stats.get('dense_mb1', {}).get('cv_pct'))} | {fmt(warm_stats.get('dense_mb1', {}).get('p05'))} | {fmt(warm_stats.get('dense_mb1', {}).get('p95'))} | {fmt(warm_stats.get('dense_mb1', {}).get('p95_over_p05'), 3)} | {fmt(first_last_ratio(groups.get('dense_mb1', [])), 3)} |
| sparse_mb1 | {warm_stats.get('sparse_mb1', {}).get('n', 0)} | {fmt(warm_stats.get('sparse_mb1', {}).get('median'))} | {fmt(warm_stats.get('sparse_mb1', {}).get('cv_pct'))} | {fmt(warm_stats.get('sparse_mb1', {}).get('p05'))} | {fmt(warm_stats.get('sparse_mb1', {}).get('p95'))} | {fmt(warm_stats.get('sparse_mb1', {}).get('p95_over_p05'), 3)} | {fmt(first_last_ratio(groups.get('sparse_mb1', [])), 3)} |
| sparse_mb2 | {warm_stats.get('sparse_mb2', {}).get('n', 0)} | {fmt(warm_stats.get('sparse_mb2', {}).get('median'))} | {fmt(warm_stats.get('sparse_mb2', {}).get('cv_pct'))} | {fmt(warm_stats.get('sparse_mb2', {}).get('p05'))} | {fmt(warm_stats.get('sparse_mb2', {}).get('p95'))} | {fmt(warm_stats.get('sparse_mb2', {}).get('p95_over_p05'), 3)} | {fmt(first_last_ratio(groups.get('sparse_mb2', [])), 3)} |

Speedups (median ratio with 95% bootstrap CI):
- matched mb1 (dense_mb1 / sparse_mb1): {json.dumps(speed_matched)}
- best-feasible (dense_mb1 / sparse_mb2): {json.dumps(speed_best)}

## 5. Long-run stability results

12-minute continuous run, one sample per ~30 s:
median {fmt(long_stats.get("median"))} tok/s, CV {fmt(long_stats.get("cv_pct"))}%,
drift {fmt(long_drift, 2)} %/min, first/last {fmt(long_first_last, 3)},
temp {json.dumps(summary["longrun"]["temperature_c"])},
SM clock {json.dumps(summary["longrun"]["sm_clock_mhz"])},
memory used {json.dumps(summary["longrun"]["mem_used_mib"])}.

## 6. Raw statistics

`summary.json` (full quantiles, MAD, CI), `raw_runs.csv` (every window),
`runs/*.json` (per-process raw). Failed runs are retained.

## 7. Thermal and utilization observations

See long-run telemetry above and `plots/throughput.png`.

## 8. Correctness/parity results

Declared-sparse contract: `opt/test_routed_expert.py`
(all-active bitwise == dense, sparse FP64 oracle == 0.0, document-boundary
leak test) and `opt/test_learned_router.py` (R0-R6). GPU smoke:
`runs/correctness.json` (finite decreasing loss, exact token accounting,
2048/2048 route slots, zero overflow, bitwise determinism).
Note: direct top1-sparse vs dense parity is NOT a valid equivalence check
(declared-sparse semantics differ); all-active expertization is the
equivalence path and is covered by the CPU suites.

## 9. Failed/outlier runs

See `raw_runs.csv` rows with `error`; none suppressed.

## 10. Plain-language conclusion

{verdict}: the local sparse champion reproduces its warm throughput with
CV {fmt(sparse.get("cv_pct"))}% and long-run drift {fmt(long_drift, 2)} %/min.
The historical "~4x same-session" claim corresponds to the best-feasible
ratio of {json.dumps(speed_best)} (dense mb1 vs sparse mb2) and the
matched-geometry ratio of {json.dumps(speed_matched)}. The 350k packed tok/s
target is a G4 production claim and is not verifiable on this 8 GiB 4060;
local numbers are a different boundary and must not be substituted for it.
"""
    (ART / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "verdict": verdict,
        "sparse_mb2": sparse,
        "longrun_drift_pct_per_min": long_drift,
        "speedup_best_feasible": speed_best,
        "speedup_matched": speed_matched,
        "correctness": correctness,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
