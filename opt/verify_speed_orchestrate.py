"""Speed-verification orchestrator (runs inside the container).

Executes the full cold/warm/long-run suite as fresh subprocesses, captures
logs, telemetry and raw runs, writes environment.json, benchmark_manifest.json
and raw_runs.csv into artifacts/arm_a_speed_verification/.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "arm_a_speed_verification"
LOGS = ART / "logs"
RUNS = ART / "runs"
ARTIFACTS = dict((k, str(v)) for k, v in {
    "repo": ROOT,
    "python": sys.version,
    "platform": platform.platform(),
    "torch": None, "cuda": None, "triton": None, "cudnn": None,
    "nvidia_smi": None, "cpu": None, "memory": None,
}.items())


def sh(cmd):
    try:
        return subprocess.check_output(cmd, text=True,
                                       stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # noqa: BLE001
        return f"error: {type(exc).__name__}: {exc}"


def capture_environment():
    import torch
    env = dict(ARTIFACTS)
    env["torch"] = torch.__version__
    env["cuda"] = torch.version.cuda
    try:
        import triton
        env["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        env["triton"] = None
    env["cudnn"] = torch.backends.cudnn.version()
    env["gpu_name"] = torch.cuda.get_device_name(0)
    env["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    env["gpu_memory_MiB"] = torch.cuda.get_device_properties(0).total_memory \
        // 2**20
    env["nvidia_smi_full"] = sh([
        "nvidia-smi",
        "--query-gpu=name,driver_version,vbios_version,memory.total,"
        "clocks.max.sm,clocks.max.mem,power.limit,temperature.gpu,"
        "temperature.gpu.tlimit,pstate",
        "--format=csv,noheader"])
    env["nvidia_smi"] = sh([
        "nvidia-smi",
        "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,"
        "utilization.gpu,utilization.memory,memory.used",
        "--format=csv,noheader"])
    env["cpu"] = sh(["sh", "-c",
                     "grep -m1 'model name' /proc/cpuinfo && "
                     "nproc"])
    env["memory"] = sh(["sh", "-c",
                        "grep -E 'MemTotal|MemAvailable' /proc/meminfo"])
    env["os"] = sh(["uname", "-a"])
    env["gcc"] = sh(["gcc", "--version"])[:200]
    return env


def run_one(kind, path, microbatch, label, windows, out_name, minutes=0):
    out_path = RUNS / f"{out_name}.json"
    log_path = LOGS / f"{out_name}.log"
    cmd = [sys.executable, "opt/verify_speed_run.py", "--kind", kind,
           "--path", path, "--microbatch", str(microbatch), "--label", label,
           "--windows", str(windows), "--out", str(out_path)]
    if minutes:
        cmd += ["--minutes", str(minutes)]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    external = time.perf_counter() - t0
    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr,
                        encoding="utf-8")
    record = {"run_id": out_name, "kind": kind, "path": path,
              "microbatch": microbatch, "label": label,
              "external_wall_s": external,
              "returncode": proc.returncode,
              "timestamp": dt.datetime.now(dt.UTC).isoformat()}
    if proc.returncode == 0 and out_path.is_file():
        data = json.loads(out_path.read_text(encoding="utf-8"))
        record["data"] = data
    else:
        record["error"] = proc.stderr[-500:]
    return record


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
    LOGS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    env = capture_environment()
    (ART / "environment.json").write_text(
        json.dumps(env, indent=2, default=str), encoding="utf-8")

    schedule = [("correctness", "sparse", 1, "correctness")]
    for rep in range(3):
        order = [("warm", "dense", 1),
                 ("warm", "sparse", 1),
                 ("warm", "sparse", 2)]
        random.Random(1337 + rep).shuffle(order)
        for kind, path, mb in order:
            windows = 10 if path == "dense" else 12
            schedule.append((kind, path, mb,
                             f"warm_{path}_mb{mb}_rep{rep}", windows))
    for cold_rep in range(5):
        schedule.append(("cold", "sparse", 2, f"cold_sparse_{cold_rep}"))
    schedule.append(("longrun", "sparse", 2, "longrun_sparse", 0,
                     12.0))

    records = []
    for item in schedule:
        kind, path, mb, name = item[0], item[1], item[2], item[3]
        windows = item[4] if len(item) > 4 else 0
        minutes = item[5] if len(item) > 5 else 0
        print(json.dumps({"launch": name}), flush=True)
        record = run_one(kind, path, mb, name, windows, name,
                         minutes=minutes)
        records.append(record)
        print(json.dumps({"done": name,
                          "rc": record["returncode"],
                          "wall_s": round(record["external_wall_s"], 1)}),
              flush=True)

    rows = []
    for record in records:
        base = {"run_id": record["run_id"], "timestamp": record["timestamp"],
                "kind": record["kind"], "path": record["path"],
                "microbatch": record["microbatch"],
                "external_wall_s": round(record["external_wall_s"], 3),
                "returncode": record["returncode"]}
        data = record.get("data")
        if not data:
            rows.append({**base, "window": -1, "tok_s": "", "error": "failed"})
            continue
        if data.get("kind") == "warm":
            for window in data["windows"]:
                tel = parse_telemetry(window.get("telemetry"))
                rows.append({**base, "window": window["window"],
                             "elapsed_s": round(window["elapsed_s"], 4),
                             "tok_s": round(window["tok_s"], 2),
                             "loss": window["loss"], **tel})
        elif data.get("kind") == "cold":
            tel = parse_telemetry(data.get("telemetry"))
            rows.append({**base, "window": 0,
                         "elapsed_s": round(data["first_step_s"], 4),
                         "tok_s": round(data["first_step_tok_s"], 2),
                         **tel})
        elif data.get("kind") == "longrun":
            for sample in data["samples"]:
                tel = parse_telemetry(sample.get("telemetry"))
                rows.append({**base, "window": len(rows),
                             "elapsed_s": round(sample["t_s"], 2),
                             "tok_s": round(sample["tok_s"], 2),
                             "loss": sample["loss"], **tel})

    fieldnames = ["run_id", "timestamp", "kind", "path", "microbatch",
                  "window", "elapsed_s", "tok_s", "loss", "temp_c",
                  "sm_clock_mhz", "mem_clock_mhz", "power_w",
                  "util_gpu_pct", "util_mem_pct", "mem_used_mib",
                  "external_wall_s", "returncode", "error"]
    with open(ART / "raw_runs.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    manifest = {
        "purpose": "Arm-A sparse speed verification (cold/warm/long-run)",
        "commit_head_at_run": sh(["git", "rev-parse", "HEAD"]),
        "frozen_commit": "0dcbb878d24b99b5808359c889e97143c3cec00b",
        "candidate_config": {
            "architecture": "M8/Ke512/top1 fixed cyclic window, compact "
                            "exact-capacity executor",
            "source": "opt/routed_expert.py RoutedExpertArmA",
            "microbatch_modes": {"sparse_mb1": 1, "sparse_mb2": 2},
            "dense_baseline": "opt/model_opt.py OptArmA opt3c flags, mb1",
            "seed": 1337, "global_sequences": 64, "T": 2048,
            "global_tokens_per_update": 131072,
            "metric": "packed input tokens / elapsed wall of a full "
                      "optimizer update (131072 tokens), CUDA-synchronized",
        },
        "protocol": {"cold_runs": 5, "warm_processes_per_path": 3,
                     "warm_windows": {"dense_mb1": 10, "sparse_mb1": 12,
                                      "sparse_mb2": 12},
                     "longrun_minutes": 12, "sample_s": 30},
        "records": [{k: v for k, v in r.items() if k != "data"}
                    for r in records],
    }
    (ART / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (ART / "commands.txt").write_text(
        "\n".join([
            "docker start iclr-arm-a",
            "docker exec -w /workspace/iclr-oc iclr-arm-a python "
            "opt/verify_speed_orchestrate.py",
            "host: py -3.12 opt/verify_speed_report.py",
        ]) + "\n", encoding="utf-8")
    print(json.dumps({"artifacts": str(ART), "csv_rows": len(rows)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
