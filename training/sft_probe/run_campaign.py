"""Arm-A SFT calibration campaign orchestrator.

Stages (resumable; completed stages are skipped unless --force):

  env       environment.json + frozen-asset verification
  baseline  evaluate the untouched 2.5B checkpoint (dose 0)
  a         Phase A LR micro-sweep (3e-5, 1e-4, 3e-4) to 0.03 TPP
  b         Phase B replay test (0% vs 10%) at the selected LR to 0.10 TPP
  c         Phase C dose extension to 0.30 TPP with the selected recipe
  report    consolidate JSON artifacts + final markdown report

Every stage writes under results/sft_probe/ and runs/sft_probe/. Training and
evaluation are invoked as subprocesses so each phase is independently
resumable and the frozen assets are re-verified per process.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.sft_probe import common, report  # noqa: E402

PY = sys.executable
N_PARAM = common.N_PARAM
TARGET_001 = 173_755
TARGET_003 = 521_265
TARGET_010 = 1_737_549
TARGET_030 = 5_212_647

PHASE_A_ARMS = (
    ("a1_3e5", 3e-5),
    ("a2_1e4", 1e-4),
    ("a3_3e4", 3e-4),
)

EVAL_DIR = common.OUT_DIR / "eval"
GEN_DIR = common.OUT_DIR / "generations"
CKPT_DIR = common.CKPT_DIR


def log(message: str) -> None:
    print(f"[campaign] {common.iso_now()} {message}", flush=True)


def run_command(args, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(str(a) for a in args)}\n")
        handle.flush()
        process = subprocess.Popen(
            [str(a) for a in args],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in process.stdout:
            handle.write(line)
            handle.flush()
        process.wait()
    return process.returncode


def train(arm, lr, target, checkpoint_at, replay_pct=0.0, init_from=None,
          force=False):
    out_dir = CKPT_DIR / arm
    final = out_dir / "final.pt"
    if final.is_file() and not force:
        log(f"train {arm}: final.pt exists, skipping")
        return
    args = [
        PY, "-m", "training.sft_probe.train_sft",
        "--arm", arm,
        "--lr", str(lr),
        "--target-tokens", str(target),
        "--checkpoint-at", ",".join(str(x) for x in checkpoint_at),
        "--replay-pct", str(replay_pct),
        "--tokens-per-update", "8192",
        "--microbatch-rows", "1",
        "--warmup-steps", "20",
        "--compile",
        "--out-dir", str(out_dir),
    ]
    if init_from:
        args += ["--init-from", str(init_from)]
    log(f"train {arm}: lr={lr} target={target} replay={replay_pct}")
    code = run_command(args, out_dir / "launch.log")
    if code != 0:
        raise RuntimeError(f"training failed for {arm} (exit {code})")


def evaluate(tag, checkpoint, native=True, params=True, force=False,
             skip_generation=False):
    eval_path = EVAL_DIR / f"{tag}.json"
    if eval_path.is_file() and not force:
        log(f"evaluate {tag}: exists, skipping")
        return
    args = [
        PY, "-m", "training.sft_probe.evaluate",
        "--tag", tag,
        "--out-dir", str(EVAL_DIR),
    ]
    if checkpoint:
        args += ["--checkpoint", str(checkpoint)]
    if native:
        args += ["--native"]
    if params:
        args += ["--params"]
    log(f"evaluate {tag}: lm/native/params")
    code = run_command(args, EVAL_DIR / f"{tag}.log")
    if code != 0:
        raise RuntimeError(f"evaluation failed for {tag} (exit {code})")
    if not skip_generation:
        gen_args = [
            PY, "-m", "training.sft_probe.generation_eval",
            "--tag", tag,
            "--out-dir", str(GEN_DIR),
            "--metrics", str(common.OUT_DIR / "generation_metrics.json"),
        ]
        if checkpoint:
            gen_args += ["--checkpoint", str(checkpoint)]
        log(f"evaluate {tag}: generation suite")
        code = run_command(gen_args, GEN_DIR / f"{tag}.log")
        if code != 0:
            raise RuntimeError(f"generation failed for {tag} (exit {code})")


def write_environment(force=False):
    path = common.OUT_DIR / "environment.json"
    if path.is_file() and not force:
        return
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True,
        text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(ROOT),
        capture_output=True, text=True,
    ).stdout
    assets = common.verify_frozen_assets(verify_checkpoint=True)
    import platform
    import torch

    environment = {
        "format": "arm_a_sft_probe_environment_v1",
        "created_at": common.iso_now(),
        "git_head": git_head,
        "git_status_porcelain": git_status.splitlines(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "gpu_total_memory_gib": (
            torch.cuda.get_device_properties(0).total_memory / 2**30
            if torch.cuda.is_available()
            else None
        ),
        "frozen_assets": assets,
        "source_commit": "0dcbb87",
        "base_checkpoint": {
            "path": str(common.BASE_CKPT),
            "sha256": common.BASE_CKPT_SHA256,
            "training_tokens": common.BASE_CKPT_TOKENS,
            "updates": common.BASE_CKPT_UPDATES,
            "weights_fingerprint": common.BASE_WEIGHTS_FINGERPRINT,
        },
        "parameter_count": N_PARAM,
        "dose_landmarks": {
            "0.01_tpp": TARGET_001,
            "0.03_tpp": TARGET_003,
            "0.10_tpp": TARGET_010,
            "0.30_tpp": TARGET_030,
        },
        "sae_search": {
            "arm_a_sae_found": False,
            "searched": [
                "results/sae (Gemma Scope 2 + Anthropic external only)",
                "data/sae (external SAE bundles only)",
                "analysis/sae (external comparisons only)",
                "repository-wide grep for BDH/Arm-A SAE training",
            ],
            "conclusion": (
                "no frozen Arm-A/Akasha SAE exists locally; native BDH "
                "sparse-neuron probes are used instead"
            ),
        },
        "base_corpus_status": "BLOCKED_ARTIFACT_NOT_LOCAL",
    }
    common.save_json(path, environment)


def read_eval(tag):
    return common.load_json(EVAL_DIR / f"{tag}.json")


def read_gen(tag):
    return common.load_json(GEN_DIR / f"{tag}.json")


def read_train_meta(arm):
    payload = torch_load_meta(CKPT_DIR / arm / "final.pt")
    return payload


def torch_load_meta(path):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload.get("sft", {})


def train_log_summary(arm):
    path = CKPT_DIR / arm / "log.jsonl"
    updates = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") == "update":
                updates.append(record)
    if not updates:
        return {}
    sequence_tokens = sum(u["sequence_tokens"] for u in updates)
    seconds = sum(u["step_seconds"] for u in updates)
    return {
        "updates": len(updates),
        "sequence_tokens": sequence_tokens,
        "seconds": seconds,
        "tok_s": sequence_tokens / max(seconds, 1e-9),
        "first_loss": updates[0]["loss"],
        "last_loss": updates[-1]["loss"],
        "max_peak_mem_gib": max(
            (u.get("peak_mem_gib") or 0) for u in updates
        ),
    }


def phase_a(force=False):
    for arm, lr in PHASE_A_ARMS:
        train(arm, lr, TARGET_003, [TARGET_001, TARGET_003],
              replay_pct=0.0, force=force)
        out_dir = CKPT_DIR / arm
        evaluate(f"{arm}_dose001", out_dir / "dose_173755.pt",
                 native=True, params=False, force=force)
        evaluate(f"{arm}_dose003", out_dir / "dose_521265.pt",
                 native=True, params=True, force=force)
    selection = report.select_phase_a(PHASE_A_ARMS)
    common.save_json(common.OUT_DIR / "phase_a_lr_sweep.json", selection)
    log(f"Phase A selection: {selection['SELECTED_LR']} "
        f"({selection['SELECTED_ARM']})")


def phase_b(force=False):
    selection = common.load_json(common.OUT_DIR / "phase_a_lr_sweep.json")
    lr = selection["SELECTED_LR"]
    selected_arm = selection["SELECTED_ARM"]
    for arm, replay in (("b1_rep0", 0.0), ("b2_rep10", 10.0)):
        train(arm, lr, TARGET_010, [TARGET_003, TARGET_010],
              replay_pct=replay, force=force)
        out_dir = CKPT_DIR / arm
        evaluate(f"{arm}_dose003", out_dir / "dose_521265.pt",
                 native=True, params=False, force=force)
        evaluate(f"{arm}_dose010", out_dir / "dose_1737549.pt",
                 native=True, params=True, force=force)
    result = report.select_phase_b(lr, selected_arm)
    common.save_json(common.OUT_DIR / "phase_b_replay.json", result)
    log(f"Phase B selection: replay={result['SELECTED_REPLAY']}%")


def phase_c(force=False):
    phase_b_result = common.load_json(common.OUT_DIR / "phase_b_replay.json")
    lr = phase_b_result["SELECTED_LR"]
    replay = float(str(phase_b_result["SELECTED_REPLAY"]).rstrip("%"))
    arm = "c_dose030"
    source_arm = "b1_rep0" if replay == 0 else "b2_rep10"
    init_from = CKPT_DIR / source_arm / "dose_1737549.pt"
    train(f"{arm}_from_{source_arm}", lr, TARGET_030, [TARGET_030],
          replay_pct=replay, init_from=init_from, force=force)
    out_dir = CKPT_DIR / f"{arm}_from_{source_arm}"
    evaluate("c_dose030", out_dir / "dose_5212647.pt",
             native=True, params=True, force=force)
    result = report.build_phase_c(lr, replay, source_arm)
    common.save_json(common.OUT_DIR / "phase_c_dose.json", result)
    log("Phase C complete")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        default="all",
        choices=["all", "env", "baseline", "a", "b", "c", "report"],
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    write_environment(force=args.force)
    if args.phase in ("all", "baseline"):
        evaluate("base", None, native=True, params=False, force=args.force)
    if args.phase in ("all", "a"):
        phase_a(force=args.force)
    if args.phase in ("all", "b"):
        phase_b(force=args.force)
    if args.phase in ("all", "c"):
        phase_c(force=args.force)
    if args.phase in ("all", "report", "env", "baseline", "a", "b", "c"):
        report.build_reports()
    log(f"campaign phase '{args.phase}' done in "
        f"{time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
