"""Autoresearch loop driver for Arm-A SFT optimization.

Commands:
  python -m training.autoresearch.ar init          materialize experiment specs
  python -m training.autoresearch.ar run --spec <path> [--generation]
  python -m training.autoresearch.ar status
  python -m training.autoresearch.ar finalize

Every run writes results/autoresearch/experiment_<id>.json, appends to
research_log.md, and updates best.json according to the mission keep rule:

  KEEP  iff instruction score improves
        and Akasha parity TRUE
        and hidden cosine >= 0.90
        and relative weight drift <= 0.25
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.sft_probe import common  # noqa: E402

AR_DIR = common.OUT_DIR.parent / "autoresearch"
SPEC_DIR = AR_DIR / "specs"
RUN_DIR = ROOT / "runs" / "autoresearch"
EVAL_DIR = AR_DIR / "eval"
GEN_DIR = AR_DIR / "generations"
DATA_DIR = AR_DIR / "data"
REPLAY_SOURCE = common.DATA_DIR / "replay_rows.jsonl"
PROXY_SOURCE = common.DATA_DIR / "proxy_rows.jsonl"
BEST_PATH = AR_DIR / "best.json"
LOG_PATH = AR_DIR / "research_log.md"
PY = sys.executable

DOSE = {
    "0.01": 173_755,
    "0.03": 521_265,
    "0.10": 1_737_549,
    "0.30": 5_212_647,
    "0.40": 6_950_196,
}

MIN_DELTA = 0.002
HIDDEN_COS_FLOOR = 0.90
DRIFT_CEILING = 0.25


def ensure_replay(mix_dir: Path):
    for name, source in (
        ("replay_rows.jsonl", REPLAY_SOURCE),
        ("proxy_rows.jsonl", PROXY_SOURCE),
    ):
        target = mix_dir / name
        if not target.is_file():
            if not source.is_file():
                raise RuntimeError(f"missing asset: {source}")
            shutil.copyfile(source, target)


def run_command(args, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(str(a) for a in args)}\n")
        handle.flush()
        process = subprocess.Popen(
            [str(a) for a in args], cwd=str(ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace",
        )
        for line in process.stdout:
            handle.write(line)
            handle.flush()
        process.wait()
    return process.returncode


def load_spec(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checkpoint_path(spec: dict) -> Path:
    dose = spec["train"]["checkpoint_at"]
    return RUN_DIR / spec["experiment_id"] / f"dose_{dose}.pt"


def eval_path(experiment_id: str) -> Path:
    return EVAL_DIR / f"{experiment_id}.json"


def gen_path(experiment_id: str) -> Path:
    return GEN_DIR / f"{experiment_id}.json"


def train_experiment(spec: dict, force=False) -> int:
    mix_dir = DATA_DIR / spec["train"]["mix"]
    ensure_replay(mix_dir)
    out_dir = RUN_DIR / spec["experiment_id"]
    final = out_dir / "final.pt"
    if final.is_file() and not force:
        print(f"[ar] {spec['experiment_id']}: final.pt exists, skip train")
        return 0
    train = spec["train"]
    checkpoint_at = train["checkpoint_at"]
    checkpoint_tokens = (
        DOSE[str(checkpoint_at)] if str(checkpoint_at) in DOSE
        else int(checkpoint_at)
    )
    target_tokens = (
        DOSE[str(train["dose_tpp"])] if str(train["dose_tpp"]) in DOSE
        else int(train["dose_tpp"])
    )
    args = [
        PY, "-m", "training.autoresearch.train_ar",
        "--arm", spec["experiment_id"],
        "--lr", str(train["lr"]),
        "--target-tokens", str(target_tokens),
        "--checkpoint-at", str(checkpoint_tokens),
        "--replay-pct", str(train.get("replay_pct", 0.0)),
        "--scheduler", train.get("scheduler", "constant"),
        "--warmup-frac", str(train.get("warmup_frac", 0.02)),
        "--tokens-per-update", str(train.get("tokens_per_update", 8192)),
        "--microbatch-rows", str(train.get("microbatch_rows", 1)),
        "--compile",
        "--out-dir", str(out_dir),
        "--data-dir", str(mix_dir),
    ]
    if train.get("init_from"):
        args += ["--init-from", train["init_from"]]
    return run_command(args, out_dir / "launch.log")


def eval_experiment(spec: dict, generation=False) -> int:
    experiment_id = spec["experiment_id"]
    mix_dir = DATA_DIR / (spec.get("train") or {}).get("mix", "mix_base")
    ensure_replay(mix_dir)
    if "checkpoint" in spec.get("eval", {}):
        checkpoint = spec["eval"]["checkpoint"]
    else:
        checkpoint = str(checkpoint_path(spec))
    args = [
        PY, "-m", "training.sft_probe.evaluate",
        "--tag", experiment_id,
        "--data-dir", str(mix_dir),
        "--out-dir", str(EVAL_DIR),
        "--native",
    ]
    if checkpoint != "base":
        args += ["--checkpoint", checkpoint, "--params"]
    code = run_command(args, EVAL_DIR / f"{experiment_id}.log")
    if code != 0:
        return code
    if generation:
        gen_args = [
            PY, "-m", "training.sft_probe.generation_eval",
            "--tag", experiment_id,
            "--out-dir", str(GEN_DIR),
            "--metrics", str(AR_DIR / "generation_metrics.json"),
        ]
        if checkpoint != "base":
            gen_args += ["--checkpoint", checkpoint]
        code = run_command(gen_args, GEN_DIR / f"{experiment_id}.log")
    return code


def metrics_from_eval(experiment_id: str, spec: dict) -> dict:
    data = common.load_json(eval_path(experiment_id))
    lm = data.get("lm", {})
    native = data.get("native", {})
    levels = native.get("levels") or []

    def mean(field):
        values = [level.get(field) for level in levels
                  if level.get(field) is not None]
        return float(sum(values) / len(values)) if values else None

    hidden = None
    if levels and "hidden_vs_base" in levels[0]:
        hidden = mean_hidden(levels)
    params = data.get("params") or {}
    total_drift = (
        (params.get("groups", {}).get("TOTAL") or {}).get(
            "relative_update_norm"
        )
    )
    is_base = spec["experiment_id"] == "ar_000_base"
    if is_base and total_drift is None:
        total_drift = 0.0
    gen = gen_path(experiment_id)
    gen_summary = None
    if gen.is_file():
        gen_data = common.load_json(gen)
        gen_summary = {
            "greedy": gen_data.get("greedy_summary"),
            "sampled": gen_data.get("sampled_summary"),
            "parity": gen_data.get("FULL_VS_RECURRENT_GREEDY_MATCH"),
            "first_divergence": gen_data.get("FIRST_DIVERGENCE_TOKEN"),
        }
    return {
        "validation_nll": lm.get("sft_val_nll"),
        "proxy_nll": lm.get("proxy_nll"),
        "sft_val_target_tokens": lm.get("sft_val_target_tokens"),
        "hidden_cosine": hidden if hidden is not None else (
            1.0 if spec["experiment_id"] == "ar_000_base" else None
        ),
        "weight_drift": total_drift,
        "top64_overlap": mean("x_top64_overlap_vs_base"),
        "u_top64_overlap": mean("u_top64_overlap_vs_base"),
        "akasha_parity": (
            gen_summary["parity"] if gen_summary else
            (True if spec["experiment_id"] == "ar_000_base" else None)
        ),
        "generation": gen_summary,
        "sft_meta": data.get("sft_meta"),
    }


def mean_hidden(levels) -> float:
    values = [
        level["hidden_vs_base"]["cosine_mean"] for level in levels
        if level.get("hidden_vs_base")
    ]
    return float(sum(values) / len(values)) if values else None


def read_best() -> dict:
    if BEST_PATH.is_file():
        return common.load_json(BEST_PATH)
    return {"best_experiment": None, "validation_nll": None}


def decide(experiment: dict, best: dict) -> dict:
    val = experiment.get("validation_nll")
    hidden = experiment.get("hidden_cosine")
    drift = experiment.get("weight_drift")
    parity = experiment.get("akasha_parity")
    reasons = []
    keep = True
    if val is None or not (val == val):
        keep = False
        reasons.append("no finite validation NLL")
    if parity is not True:
        keep = False
        reasons.append("Akasha parity not TRUE")
    if hidden is None or hidden < HIDDEN_COS_FLOOR:
        keep = False
        reasons.append(
            f"hidden cosine {hidden} below floor {HIDDEN_COS_FLOOR}"
        )
    if drift is None or drift > DRIFT_CEILING:
        keep = False
        reasons.append(f"weight drift {drift} above ceiling {DRIFT_CEILING}")
    best_val = best.get("validation_nll")
    if keep and best_val is not None and val is not None:
        if val > best_val - MIN_DELTA:
            keep = False
            reasons.append(
                f"no improvement over best {best_val:.6f} "
                f"(delta threshold {MIN_DELTA})"
            )
    decision = "KEEP" if keep else "REVERT"
    return {
        "decision": decision,
        "decision_reasons": reasons or ["all keep criteria met"],
    }


def append_log(experiment: dict):
    header = (
        "# Arm-A SFT autoresearch log\n\n"
        if not LOG_PATH.is_file() else ""
    )
    with open(LOG_PATH, "a", encoding="utf-8") as handle:
        if header:
            handle.write(header)
        handle.write(
            f"\n## {experiment['experiment_id']} - "
            f"{experiment['decision']}\n\n"
        )
        handle.write(f"- phase: {experiment.get('phase')}\n")
        handle.write(f"- hypothesis: {experiment.get('hypothesis')}\n")
        handle.write(
            f"- changed: {experiment.get('changed_variable')} "
            f"({experiment.get('old_value')} -> "
            f"{experiment.get('new_value')})\n"
        )
        handle.write(
            f"- val NLL: {experiment.get('validation_nll')}; proxy NLL: "
            f"{experiment.get('proxy_nll')}; hidden cosine: "
            f"{experiment.get('hidden_cosine')}; weight drift: "
            f"{experiment.get('weight_drift')}; parity: "
            f"{experiment.get('akasha_parity')}\n"
        )
        handle.write(
            f"- reasons: {'; '.join(experiment.get('decision_reasons', []))}\n"
        )
        handle.write(f"- notes: {experiment.get('notes', '')}\n")


def update_metrics_table(experiment: dict):
    path = AR_DIR / "metrics.json"
    metrics = common.load_json(path) if path.is_file() else {}
    metrics[experiment["experiment_id"]] = {
        key: experiment.get(key)
        for key in (
            "phase", "hypothesis", "changed_variable", "old_value",
            "new_value", "tokens_trained", "validation_nll", "proxy_nll",
            "hidden_cosine", "weight_drift", "top64_overlap",
            "akasha_parity", "decision", "config",
        )
    }
    common.save_json(path, metrics)


def cmd_run(args) -> int:
    spec = load_spec(Path(args.spec))
    experiment_id = spec["experiment_id"]
    print(f"[ar] running {experiment_id}: {spec['hypothesis']}")
    started = time.perf_counter()
    if spec.get("train"):
        code = train_experiment(spec, force=args.force)
        if code != 0:
            record = {
                "experiment_id": experiment_id,
                "hypothesis": spec["hypothesis"],
                "changed_variable": spec["changed_variable"],
                "old_value": spec["old_value"],
                "new_value": spec["new_value"],
                "tokens_trained": 0,
                "validation_nll": None,
                "instruction_score": None,
                "hidden_cosine": None,
                "weight_drift": None,
                "akasha_parity": None,
                "decision": "REVERT",
                "notes": f"training failed with exit code {code}",
                "phase": spec.get("phase"),
                "config": spec.get("train"),
            }
            common.save_json(
                AR_DIR / f"experiment_{experiment_id}.json", record
            )
            append_log(record)
            return 1
    code = eval_experiment(spec, generation=args.generation)
    if code != 0:
        raise RuntimeError(f"evaluation failed for {experiment_id}")
    metrics = metrics_from_eval(experiment_id, spec)
    best = read_best()
    decision = decide(metrics, best)
    trained = None
    if spec.get("train"):
        trained = metrics.get("sft_meta", {}).get(
            "instruction_target_tokens"
        )
    record = {
        "experiment_id": experiment_id,
        "phase": spec.get("phase"),
        "hypothesis": spec["hypothesis"],
        "changed_variable": spec["changed_variable"],
        "old_value": spec["old_value"],
        "new_value": spec["new_value"],
        "tokens_trained": trained,
        "validation_nll": metrics["validation_nll"],
        "instruction_score": (
            None if metrics["validation_nll"] is None
            else -metrics["validation_nll"]
        ),
        "proxy_nll": metrics["proxy_nll"],
        "hidden_cosine": metrics["hidden_cosine"],
        "weight_drift": metrics["weight_drift"],
        "top64_overlap": metrics["top64_overlap"],
        "u_top64_overlap": metrics["u_top64_overlap"],
        "akasha_parity": metrics["akasha_parity"],
        "generation": metrics["generation"],
        "decision": decision["decision"],
        "decision_reasons": decision["decision_reasons"],
        "notes": spec.get("notes", ""),
        "config": spec.get("train"),
        "eval": spec.get("eval"),
        "wall_seconds": time.perf_counter() - started,
        "checkpoint": (
            str(checkpoint_path(spec)) if spec.get("train") else "base"
        ),
    }
    common.save_json(
        AR_DIR / f"experiment_{experiment_id}.json", record
    )
    append_log(record)
    update_metrics_table(record)
    if decision["decision"] == "KEEP" and spec.get("train"):
        common.save_json(BEST_PATH, {
            "best_experiment": experiment_id,
            "validation_nll": metrics["validation_nll"],
            "proxy_nll": metrics["proxy_nll"],
            "hidden_cosine": metrics["hidden_cosine"],
            "weight_drift": metrics["weight_drift"],
            "akasha_parity": metrics["akasha_parity"],
            "checkpoint": str(checkpoint_path(spec)),
            "config": spec.get("train"),
            "updated_at": common.iso_now(),
        })
        print(f"[ar] {experiment_id}: KEEP (new best {metrics['validation_nll']:.6f})")
    else:
        print(f"[ar] {experiment_id}: {decision['decision']} "
              f"({'; '.join(decision['decision_reasons'])})")
    if spec.get("train") and not args.keep_checkpoints:
        if decision["decision"] != "KEEP":
            for path in (RUN_DIR / experiment_id).glob("*.pt"):
                path.unlink()
    return 0


def cmd_status(args) -> int:
    metrics = common.load_json(AR_DIR / "metrics.json") \
        if (AR_DIR / "metrics.json").is_file() else {}
    best = read_best()
    print(f"best: {best.get('best_experiment')} "
          f"val={best.get('validation_nll')}")
    print(f"{'id':<12}{'phase':<12}{'var':<22}{'val_nll':>10}"
          f"{'hidden':>9}{'drift':>9}{'parity':>8}  decision")
    for experiment_id, row in metrics.items():
        print(
            f"{experiment_id:<12}{str(row.get('phase')):<12}"
            f"{str(row.get('changed_variable'))[:21]:<22}"
            f"{format_float(row.get('validation_nll')):>10}"
            f"{format_float(row.get('hidden_cosine')):>9}"
            f"{format_float(row.get('weight_drift')):>9}"
            f"{str(row.get('akasha_parity')):>8}  {row.get('decision')}"
        )
    return 0


def format_float(value):
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--spec", required=True)
    run.add_argument("--force", action="store_true")
    run.add_argument("--generation", action="store_true")
    run.add_argument("--keep-checkpoints", action="store_true")
    sub.add_parser("status")
    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "status":
        return cmd_status(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
