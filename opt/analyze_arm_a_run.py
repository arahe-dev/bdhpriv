"""Analyze an exported Arm-A production run (logs + checkpoint metadata).

Usage:
  py -3.12 opt/analyze_arm_a_run.py --run-dir runs/arm_a_2p5b_opt3c_all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

SCRIPT_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path):
    records = []
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def checkpoint_metadata(path: Path, include_state: bool = False):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    progress = ckpt.get("progress", {})
    out = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "progress": progress,
        "saved_at": ckpt.get("saved_at"),
        "implementation": ckpt.get("implementation"),
        "code_sha256": ckpt.get("code_sha256"),
        "torch": ckpt.get("torch"),
        "cuda": ckpt.get("cuda"),
        "flags": ckpt.get("config", {}).get("flags"),
        "session_stats": ckpt.get("session_stats"),
    }
    if include_state:
        out["_state"] = ckpt
    return out


def extract_log_summary(records):
    updates = [r for r in records if r.get("event") == "update"]
    events = {}
    for record in records:
        events[record.get("event")] = events.get(record.get("event"), 0) + 1
    summary = {"event_counts": events, "updates": len(updates)}
    if updates:
        losses = [u["loss"] for u in updates]
        tok_s = [u["tok_s"] for u in updates if u.get("tok_s")]
        step_ms = [u["step_ms"] for u in updates if u.get("step_ms")]
        mem = [u.get("mem_peak_GiB", 0) for u in updates]
        summary.update({
            "first_update": updates[0],
            "last_update": updates[-1],
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "loss_min": min(losses),
            "loss_last50_mean": statistics.mean(losses[-50:]),
            "loss_last50_std": (statistics.pstdev(losses[-50:])
                                if len(losses) >= 2 else 0.0),
            "tok_s_median": statistics.median(tok_s) if tok_s else None,
            "tok_s_last": tok_s[-1] if tok_s else None,
            "step_ms_median": statistics.median(step_ms) if step_ms else None,
            "step_ms_last": step_ms[-1] if step_ms else None,
            "mem_peak_GiB_max": max(mem) if mem else None,
            "tokens_consumed_last": updates[-1].get("tokens_consumed"),
            "wall_seconds": updates[-1].get("elapsed_s"),
        })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/arm_a_2p5b_opt3c_all")
    parser.add_argument("--out", default="results/arm_a_2p5b_run_summary.json")
    parser.add_argument("--full-state", action="store_true",
                        help="also load model/optimizer tensors (slow, RAM)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    logs = run_dir / "logs"
    train = read_jsonl(logs / "train.jsonl")
    smoke = read_jsonl(logs / "smoke.jsonl")
    startup = read_jsonl(logs / "startup.jsonl")

    ckpt_paths = sorted((run_dir / "ckpt").glob("*.pt"))
    census_dir = run_dir / "census_ckpts"
    if census_dir.is_dir():
        ckpt_paths += sorted(census_dir.glob("*.pt"))
    checkpoints = [checkpoint_metadata(p, include_state=args.full_state)
                   for p in ckpt_paths]

    trainer_path = run_dir / "arm_a_2p5b_trainer.py"
    local_fp = None
    if trainer_path.is_file():
        local_fp = hashlib.sha256(
            trainer_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    code_fp_match = {
        c["path"]: (c["code_sha256"] == local_fp if local_fp else None)
        for c in checkpoints
    }

    summary = {
        "run_dir": str(run_dir),
        "train_log": extract_log_summary(train),
        "startup": startup[:1],
        "smoke": smoke[:3] + smoke[-3:],
        "checkpoints": checkpoints,
        "local_trainer_code_fingerprint": local_fp,
        "checkpoint_code_fingerprint_matches_local_trainer": code_fp_match,
        "checkpoint_sha256": {c["path"]: sha256_file(Path(c["path"]))
                              for c in checkpoints},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str),
                   encoding="utf-8")

    log = summary["train_log"]
    print(json.dumps({
        "out": str(out),
        "updates_logged": log.get("updates"),
        "loss_first": log.get("loss_first"),
        "loss_last": log.get("loss_last"),
        "loss_min": log.get("loss_min"),
        "tok_s_median": log.get("tok_s_median"),
        "step_ms_median": log.get("step_ms_median"),
        "mem_peak_GiB_max": log.get("mem_peak_GiB_max"),
        "wall_hours": (log.get("wall_seconds") or 0) / 3600.0,
        "tokens_consumed_last": log.get("tokens_consumed_last"),
        "checkpoint_progress": [
            {Path(c["path"]).name: c["progress"]} for c in checkpoints
        ],
        "code_fp_matches": code_fp_match,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
