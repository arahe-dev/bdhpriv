"""Dense BDH verification orchestrator (dense only).

Suite: correctness (already run or re-run), warm B=1 x3 processes,
warm B=2 x1 limited, comparator x1, cold x3 with isolated caches,
long-run 12 min. Writes raw runs + benchmark_samples.csv +
thermal_samples.csv into artifacts/dense_bdh_verification/.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "dense_bdh_verification"
LOGS = ART / "logs"
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


def run_one(name, args, env_extra=None):
    out_path = RUNS / f"{name}.json"
    log_path = LOGS / f"{name}.log"
    cmd = [sys.executable, "opt/dense_verify.py"] + args + \
        ["--out", str(out_path)]
    env = None
    if env_extra:
        import os
        env = dict(os.environ)
        env.update(env_extra)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                          env=env)
    external = time.perf_counter() - t0
    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr,
                        encoding="utf-8")
    record = {"run_id": name, "external_wall_s": external,
              "returncode": proc.returncode,
              "timestamp": dt.datetime.now(dt.UTC).isoformat()}
    if proc.returncode == 0 and out_path.is_file():
        record["data"] = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        record["error"] = proc.stderr[-500:]
    return record


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    records = []

    schedule = []
    for rep in range(3):
        schedule.append(("warm_b1_rep%d" % rep,
                         ["--kind", "warm", "--microbatch", "1",
                          "--windows", "12"], None))
    schedule.append(("warm_b2_limited",
                     ["--kind", "warm", "--microbatch", "2",
                      "--windows", "4"], None))
    schedule.append(("comparator_reference",
                     ["--kind", "comparator", "--updates", "3"], None))
    for i in range(3):
        schedule.append((f"cold_true_{i}", ["--kind", "cold"],
                         {"TORCHINDUCTOR_CACHE_DIR": f"/tmp/dense_cold_ind_{i}",
                          "TRITON_CACHE_DIR": f"/tmp/dense_cold_tri_{i}"}))
    schedule.append(("longrun_dense_control",
                     ["--kind", "longrun", "--minutes", "12"], None))
    random.Random(2026).shuffle(schedule)
    # correctness first regardless of shuffle
    correctness = run_one("correctness", ["--kind", "correctness"])
    records.append(correctness)
    print(json.dumps({"done": "correctness",
                      "rc": correctness["returncode"]}), flush=True)

    for name, args, env_extra in schedule:
        print(json.dumps({"launch": name}), flush=True)
        record = run_one(name, args, env_extra)
        records.append(record)
        print(json.dumps({"done": name, "rc": record["returncode"],
                          "wall_s": round(record["external_wall_s"], 1)}),
              flush=True)

    sample_rows = []
    for record in records:
        base = {"run_id": record["run_id"], "timestamp": record["timestamp"],
                "external_wall_s": round(record["external_wall_s"], 3),
                "returncode": record["returncode"]}
        data = record.get("data")
        if not data:
            sample_rows.append({**base, "kind": "failed", "window": -1})
            continue
        kind = data.get("kind")
        if kind == "warm":
            for window in data["windows"]:
                sample_rows.append({**base, "kind": kind,
                                    "microbatch": data["microbatch"],
                                    "window": window["window"],
                                    "elapsed_s": round(
                                        window["elapsed_s"], 4),
                                    "tok_s": round(window["tok_s"], 2),
                                    "loss": window["loss"],
                                    **parse_telemetry(
                                        window.get("telemetry"))})
        elif kind == "comparator":
            for row in data["updates"]:
                sample_rows.append({**base, "kind": kind, "microbatch": 1,
                                    "window": row["update"],
                                    "elapsed_s": round(row["elapsed_s"], 4),
                                    "tok_s": round(row["tok_s"], 2),
                                    "loss": row["loss"],
                                    **parse_telemetry(
                                        row.get("telemetry"))})
        elif kind == "cold":
            sample_rows.append({**base, "kind": kind, "microbatch": 1,
                                "window": 0,
                                "elapsed_s": round(data["first_step_s"], 4),
                                "tok_s": round(data["first_step_tok_s"], 2),
                                **parse_telemetry(data.get("telemetry"))})
        elif kind == "longrun":
            for sample in data["samples"]:
                sample_rows.append({**base, "kind": kind, "microbatch": 1,
                                    "window": len(sample_rows),
                                    "elapsed_s": round(sample["t_s"], 2),
                                    "tok_s": round(sample["tok_s"], 2),
                                    "loss": sample["loss"],
                                    **parse_telemetry(
                                        sample.get("telemetry"))})

    fieldnames = ["run_id", "timestamp", "kind", "microbatch", "window",
                  "elapsed_s", "tok_s", "loss", "temp_c", "sm_clock_mhz",
                  "mem_clock_mhz", "power_w", "util_gpu_pct", "util_mem_pct",
                  "mem_used_mib", "external_wall_s", "returncode"]
    with open(ART / "benchmark_samples.csv", "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        for row in sample_rows:
            writer.writerow(row)

    thermal_rows = [r for r in sample_rows if r.get("temp_c") is not None]
    with open(ART / "thermal_samples.csv", "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["run_id", "timestamp", "kind", "window",
                            "temp_c", "sm_clock_mhz", "power_w",
                            "util_gpu_pct", "mem_used_mib"],
            extrasaction="ignore")
        writer.writeheader()
        for row in thermal_rows:
            writer.writerow(row)

    (ART / "run_manifest.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "data"} for r in records],
        indent=2, default=str), encoding="utf-8")
    print(json.dumps({"artifacts": str(ART), "rows": len(sample_rows)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
