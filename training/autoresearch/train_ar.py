"""Autoresearch trainer: the frozen SFT trainer plus a pluggable LR schedule.

The only difference from ``training/sft_probe/train_sft.py`` is that the
module-global ``lr_for_update`` of the frozen trainer is replaced (monkey-
patched) with a schedule implementing:

  * constant (with linear warmup over a fraction of the run), or
  * cosine decay from the peak LR over the remaining updates.

Everything else -- model, packing, loss mask, optimizer, gradient clipping,
checkpoint format, replay -- is the validated pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.sft_probe import common, packing, train_sft  # noqa: E402

T = 2048


def estimate_total_updates(data_dir: Path, target_tokens: int,
                           tokens_per_update: int, init_from=None) -> dict:
    plan = packing.load_row_plan_jsonl(data_dir / "train_rows.jsonl")
    targets = [entry["target_tokens"] for entry in plan]
    avg_target = sum(targets) / max(1, len(targets))
    rows_per_update = max(1, tokens_per_update // T)
    done = 0
    done_tokens = 0
    if init_from:
        import torch

        payload = torch.load(init_from, map_location="cpu",
                             weights_only=False)
        meta = payload.get("sft", {})
        done = int(meta.get("updates_done", 0))
        done_tokens = int(meta.get("instruction_target_tokens", 0))
    remaining = max(0, target_tokens - done_tokens)
    per_update = max(1.0, avg_target * rows_per_update)
    updates = int(math.ceil(remaining / per_update))
    return {
        "avg_target_tokens_per_row": avg_target,
        "rows_per_update": rows_per_update,
        "estimated_updates": updates + done,
        "updates_done_before": done,
    }


def make_schedule(mode: str, peak_lr: float, warmup_updates: int,
                  total_updates: int):
    warmup_updates = max(0, int(warmup_updates))
    total_updates = max(warmup_updates + 1, int(total_updates))

    def schedule(update_1based: int, cfg) -> float:
        step = int(update_1based)
        if warmup_updates > 0 and step <= warmup_updates:
            return peak_lr * step / warmup_updates
        if mode == "cosine":
            progress = (step - warmup_updates) / max(
                1, total_updates - warmup_updates
            )
            progress = min(1.0, max(0.0, progress))
            return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        return peak_lr

    return schedule


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--checkpoint-at", default="")
    parser.add_argument("--replay-pct", type=float, default=0.0)
    parser.add_argument("--scheduler", default="constant",
                        choices=["constant", "cosine"])
    parser.add_argument("--warmup-frac", type=float, default=0.02)
    parser.add_argument("--tokens-per-update", type=int, default=8192)
    parser.add_argument("--microbatch-rows", type=int, default=1)
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--reset-data-cursor", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--max-updates", type=int, default=0)
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    estimate = estimate_total_updates(
        data_dir, args.target_tokens, args.tokens_per_update,
        init_from=args.init_from,
    )
    warmup_updates = int(round(args.warmup_frac * estimate["estimated_updates"]))
    schedule = make_schedule(
        args.scheduler, args.lr, warmup_updates, estimate["estimated_updates"]
    )
    trainer = common.load_trainer()
    trainer.lr_for_update = schedule

    print(json.dumps({
        "autoresearch_train": args.arm,
        "scheduler": args.scheduler,
        "warmup_frac": args.warmup_frac,
        "warmup_updates": warmup_updates,
        "estimated_total_updates": estimate["estimated_updates"],
        "avg_target_tokens_per_row": estimate["avg_target_tokens_per_row"],
    }), flush=True)

    inner = [
        "--arm", args.arm,
        "--lr", str(args.lr),
        "--target-tokens", str(args.target_tokens),
        "--checkpoint-at", args.checkpoint_at,
        "--replay-pct", str(args.replay_pct),
        "--tokens-per-update", str(args.tokens_per_update),
        "--microbatch-rows", str(args.microbatch_rows),
        "--warmup-steps", "0",
        "--out-dir", str(args.out_dir),
        "--data-dir", str(data_dir),
    ]
    if args.init_from:
        inner += ["--init-from", args.init_from]
    if args.reset_data_cursor:
        inner += ["--reset-data-cursor"]
    if args.device:
        inner += ["--device", args.device]
    if args.compile:
        inner += ["--compile"]
    if args.max_updates:
        inner += ["--max-updates", str(args.max_updates)]
    return train_sft.main(inner)


if __name__ == "__main__":
    sys.exit(main())
