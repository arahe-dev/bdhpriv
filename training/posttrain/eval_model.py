"""Evaluate a post-training checkpoint on the frozen verifiable suite plus
the identity guards (native BDH drift, parameter drift, Akasha parity).

Usage:
  python -m training.posttrain.eval_model --tag pt_000_base --checkpoint base
  python -m training.posttrain.eval_model --tag pt_001 --checkpoint runs/...

Writes results/posttraining/eval_<tag>.json and appends to
results/posttraining/eval_index.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.posttrain import eval_suite  # noqa: E402
from training.sft_probe import common, native_probe  # noqa: E402

OUT_DIR = common.OUT_DIR.parent / "posttraining" / "eval"
INDEX_PATH = common.OUT_DIR.parent / "posttraining" / "eval_index.json"


def load_state(checkpoint):
    if checkpoint in (None, "base"):
        return common.load_base_state(), None
    payload = torch.load(checkpoint, map_location="cpu",
                         weights_only=False)
    return payload["model"], payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--checkpoint", default="base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--story-limit", type=int, default=12)
    parser.add_argument("--story-max-new", type=int, default=160)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--skip-native", action="store_true")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    device = torch.device(args.device)
    tokenizer = common.load_tokenizer()
    suite = eval_suite.build_suite()
    state, payload = load_state(args.checkpoint)
    model = common.build_model(state, device=device.type)
    model.to(device)
    model.eval()

    from akasha.checkpoint.loader import map_trainer_state_dict
    from akasha.models.arma.config import production_config

    akasha_cfg = production_config()
    weights = map_trainer_state_dict(state, akasha_cfg, strict=True)
    weights = weights.to(device=device, dtype=torch.float32)

    result = {
        "tag": args.tag,
        "checkpoint": str(args.checkpoint),
        "created_at": common.iso_now(),
        "suite_path": str(eval_suite.SUITE_PATH),
    }
    if payload is not None:
        result["sft_meta"] = payload.get("sft")

    suite_result = eval_suite.evaluate_suite(
        model, tokenizer, device, suite,
        story_max_new=args.story_max_new,
        story_limit=args.story_limit,
        weights=weights,
        akasha_cfg=akasha_cfg,
        task_limit=args.task_limit,
    )
    result["suite"] = suite_result

    task_accuracies = [
        entry["accuracy"] for entry in suite_result["tasks"].values()
    ]
    result["summary"] = {
        "task_macro_accuracy": float(np.mean(task_accuracies)),
        "task_details": {
            name: entry["accuracy"]
            for name, entry in suite_result["tasks"].items()
        },
        "blimp_macro_length_normalized": suite_result.get(
            "blimp_macro_length_normalized"
        ),
        "story_pass_rate": suite_result["stories"]["pass_rate"],
        "seconds": time.perf_counter() - started,
    }

    if not args.skip_native:
        native = native_probe.run_probe(state, device=args.device)
        if payload is not None:
            base = common.load_json(
                common.OUT_DIR / "native" / "base.json"
            )
            base_hidden = dict(np.load(
                common.OUT_DIR / "native" / "base_hidden.npz"
            ))
            native = native_probe.compare_to_base(
                native, base, base_hidden
            )
        native.pop("_hidden", None)
        result["native"] = native
        levels = native["levels"]
        result["summary"]["hidden_cosine_vs_base"] = float(np.mean([
            level["hidden_vs_base"]["cosine_mean"]
            for level in levels if level.get("hidden_vs_base")
        ])) if payload is not None else 1.0
        result["summary"]["x_top64_overlap_vs_base"] = float(np.mean([
            level["x_top64_overlap_vs_base"]
            for level in levels if level.get("x_top64_overlap_vs_base")
        ])) if payload is not None else 1.0

    if payload is not None:
        base_state = common.load_base_state()
        drift = common.parameter_drift(base_state, state)
        result["params"] = drift
        result["summary"]["weight_drift"] = drift["groups"]["TOTAL"][
            "relative_update_norm"
        ]
    else:
        result["summary"]["weight_drift"] = 0.0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    common.save_json(OUT_DIR / f"{args.tag}.json", result)
    index = common.load_json(INDEX_PATH) if INDEX_PATH.is_file() else {}
    index[args.tag] = result["summary"]
    common.save_json(INDEX_PATH, index)
    print(json.dumps({"tag": args.tag, **result["summary"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
