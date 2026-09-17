"""Rejection-sampling fine-tuning (RFT) data builder.

Samples completions from a checkpoint on *training* prompts, keeps only
outputs that pass the programmatic verifier (tasks) or the story rules, and
writes a new training data directory. Story sampling uses temperature 0.8
for diversity; task sampling is greedy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.posttrain import eval_suite, tasks  # noqa: E402
from training.sft_probe import common, packing  # noqa: E402

DATA_ROOT = common.OUT_DIR.parent / "posttraining" / "data"
TASK_ORDER = ("copy", "reverse", "sort_asc", "add", "count_words",
              "first_letter")


def load_state(checkpoint):
    payload = torch.load(checkpoint, map_location="cpu",
                         weights_only=False)
    return payload["model"]


def build_rft(name: str, checkpoint: str, per_task: int, story_prompts: int,
              story_max_new: int = 80, seed: int = 9393):
    from akasha.checkpoint.loader import map_trainer_state_dict
    from akasha.models.arma.config import production_config
    from akasha.sampling.sampler import Sampler, SamplerMethod

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = common.load_tokenizer()
    state = load_state(checkpoint)
    model = common.build_model(state, device=device.type)
    model.to(device)
    model.eval()
    akasha_cfg = production_config()
    weights = map_trainer_state_dict(state, akasha_cfg, strict=True)
    weights = weights.to(device=device, dtype=torch.float32)

    sampler = Sampler(
        method=SamplerMethod.MULTINOMIAL, temperature=0.8, top_k=50,
        seed=seed,
    )

    out_dir = DATA_ROOT / name
    train_records = []
    stats = {task: {"sampled": 0, "kept": 0} for task in TASK_ORDER}
    for task in TASK_ORDER:
        for entry in tasks.generate_task(task, "train", per_task, seed + 77):
            text = eval_suite.generate_text(
                model, tokenizer, device, entry["prompt"], max_new=16,
                weights=weights, akasha_cfg=akasha_cfg,
            )
            stats[task]["sampled"] += 1
            if not tasks.grade(task, text, entry["answer"]):
                continue
            stats[task]["kept"] += 1
            result = packing.tokenize_example(
                tokenizer, entry["prompt"], "", text.strip() or " ", max_len=2048
            )
            if result["skipped"]:
                continue
            train_records.append(
                {
                    "p": result["prefix_ids"],
                    "r": result["response_ids"],
                    "tr": bool(result["truncated"]),
                }
            )

    story_stats = {"sampled": 0, "kept": 0}
    if story_prompts > 0:
        story_dir = DATA_ROOT / "pt_stories"
        story_tokens = packing.load_tokens_jsonl(
            story_dir / "train_tokens.jsonl"
        )
        indices = sorted(story_tokens)[:story_prompts]
        for index in indices:
            record = story_tokens[index]
            # decode the prompt prefix back is not stored; use generic prompt
            prompt = "Tell me a short story."
            text = eval_suite.generate_text(
                model, tokenizer, device, prompt, max_new=story_max_new,
                sampler=sampler, weights=weights, akasha_cfg=akasha_cfg,
            )
            story_stats["sampled"] += 1
            grade = tasks.grade_story(text, "")
            passed = all(
                value for key, value in grade["checks"].items()
                if key != "contains_required_word"
            )
            if not passed:
                continue
            story_stats["kept"] += 1
            result = packing.tokenize_example(
                tokenizer, prompt, "", text.strip() or " ", max_len=2048
            )
            if not result["skipped"]:
                train_records.append(
                    {
                        "p": result["prefix_ids"],
                        "r": result["response_ids"],
                        "tr": bool(result["truncated"]),
                    }
                )

    # validation: reuse the source task dev split
    val_tokens = packing.load_tokens_jsonl(
        DATA_ROOT / "pt_tasks_big" / "val_tokens.jsonl"
    )
    val_records = [val_tokens[i] for i in sorted(val_tokens)]

    for index, record in enumerate(train_records):
        record["i"] = index
    for index, record in enumerate(val_records):
        record["i"] = index
    from training.posttrain.make_data import pack_and_write

    manifest = {
        "format": "arm_a_posttrain_data_v1",
        "name": name,
        "kind": "rft",
        "source_checkpoint": checkpoint,
        "per_task": per_task,
        "story_prompts": story_prompts,
        "sampler_temperature": 0.8,
        "seed": seed,
        "task_stats": stats,
        "story_stats": story_stats,
    }
    pack_and_write(out_dir, train_records, val_records, manifest)
    return out_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--per-task", type=int, default=4000)
    parser.add_argument("--story-prompts", type=int, default=0)
    parser.add_argument("--story-max-new", type=int, default=80)
    args = parser.parse_args(argv)
    build_rft(
        args.name, args.checkpoint, args.per_task, args.story_prompts,
        story_max_new=args.story_max_new,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
