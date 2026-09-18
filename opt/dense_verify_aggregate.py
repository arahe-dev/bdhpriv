"""Aggregate dense verification run JSONs into benchmark_samples.csv and
thermal_samples.csv (recovers from an interrupted orchestrator).

py -3.12 opt/dense_verify_aggregate.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "dense_bdh_verification"
RUNS = ART / "runs"


def parse_telemetry(line):
    if not line:
        return {}
    parts = [p.strip() for p in line.split(",")]
    try:
        return {"temp_c": float(parts[0]),
                "sm_clock_mhz": float(parts[1].split()[0]),
                "mem_clock_mhz": float(parts[2].split()[0]),
                "power_w": float(parts[3].split()[0]),
                "util_gpu_pct": float(parts[4].split()[0]),
                "util_mem_pct": float(parts[5].split()[0]),
                "mem_used_mib": float(parts[6].split()[0])}
    except Exception:  # noqa: BLE001
        return {}


def main():
    rows = []
    for path in sorted(RUNS.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        base = {"run_id": path.stem, "timestamp": "",
                "external_wall_s": "", "returncode": 0}
        kind = data.get("kind")
        if kind == "warm":
            for window in data["windows"]:
                rows.append({**base, "kind": kind,
                             "microbatch": data["microbatch"],
                             "window": window["window"],
                             "elapsed_s": round(window["elapsed_s"], 4),
                             "tok_s": round(window["tok_s"], 2),
                             "loss": window["loss"],
                             **parse_telemetry(window.get("telemetry"))})
        elif kind == "comparator":
            for row in data["updates"]:
                rows.append({**base, "kind": kind, "microbatch": 1,
                             "window": row["update"],
                             "elapsed_s": round(row["elapsed_s"], 4),
                             "tok_s": round(row["tok_s"], 2),
                             "loss": row["loss"],
                             **parse_telemetry(row.get("telemetry"))})
        elif kind == "cold":
            rows.append({**base, "kind": kind, "microbatch": 1, "window": 0,
                         "elapsed_s": round(data["first_step_s"], 4),
                         "tok_s": round(data["first_step_tok_s"], 2),
                         **parse_telemetry(data.get("telemetry"))})
        elif kind == "longrun":
            for sample in data["samples"]:
                rows.append({**base, "kind": kind, "microbatch": 1,
                             "window": len(rows),
                             "elapsed_s": round(sample["t_s"], 2),
                             "tok_s": round(sample["tok_s"], 2),
                             "loss": sample["loss"],
                             **parse_telemetry(sample.get("telemetry"))})
    fieldnames = ["run_id", "timestamp", "kind", "microbatch", "window",
                  "elapsed_s", "tok_s", "loss", "temp_c", "sm_clock_mhz",
                  "mem_clock_mhz", "power_w", "util_gpu_pct", "util_mem_pct",
                  "mem_used_mib", "external_wall_s", "returncode"]
    with open(ART / "benchmark_samples.csv", "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    thermal = [r for r in rows if r.get("temp_c") is not None]
    with open(ART / "thermal_samples.csv", "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["run_id", "timestamp", "kind", "window",
                            "temp_c", "sm_clock_mhz", "power_w",
                            "util_gpu_pct", "mem_used_mib"],
            extrasaction="ignore")
        writer.writeheader()
        for row in thermal:
            writer.writerow(row)
    print(json.dumps({"rows": len(rows), "thermal_rows": len(thermal)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
