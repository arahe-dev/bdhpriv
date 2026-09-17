"""Build post-training datasets (programmatic tasks and/or TinyStories).

Task data uses only the 'train' wording templates; the frozen evaluation
suite uses disjoint 'dev'/'test' templates. Story data is a deterministic
prefix subset of the pinned TinyStories corpus.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from typing import List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.posttrain import tasks  # noqa: E402
from training.sft_probe import common, packing  # noqa: E402

DATA_ROOT = common.OUT_DIR.parent / "posttraining" / "data"
TINYSTORIES_REPO = "roneneldan/TinyStories"
TINYSTORIES_REVISION = "f54c09fd2331"
TASK_ORDER = ("copy", "reverse", "sort_asc", "add", "count_words",
              "first_letter")
STORY_TEMPLATES = (
    "Tell me a short story.",
    "Write a short story.",
    "Write a story for children.",
)
STORY_TRAIN_NOUNS = (
    "a small bird", "a red ball", "a happy dog", "a quiet pond",
    "a brave rabbit", "a shiny rock", "a green frog", "a tall tower",
    "a sleepy cat", "a warm fire", "a kind farmer", "a wooden boat",
    "a little star", "a busy bee", "a soft cloud", "a silver bell",
)


def _story_instruction(rng, required_words) -> tuple:
    mode = rng.randrange(6)
    if mode == 0:
        return STORY_TEMPLATES[0], None
    if mode == 1:
        return STORY_TEMPLATES[1], None
    if mode == 2:
        return STORY_TEMPLATES[2], None
    noun = rng.choice(STORY_TRAIN_NOUNS)
    if mode == 3:
        return f"Write a short story about {noun}.", None
    word = rng.choice(required_words)
    if mode == 4:
        return f"Tell me a story that includes the word '{word}'.", word
    return (
        f"Write a short story about {noun}. Include the word '{word}'.",
        word,
    )


def tokenize_examples(records: List[dict]) -> List[dict]:
    tokenizer = common.load_tokenizer()
    tokenized = []
    for index, record in enumerate(records):
        result = packing.tokenize_example(
            tokenizer, record["prompt"], "", record["answer"], max_len=2048
        )
        if result["skipped"]:
            continue
        tokenized.append(
            {
                "i": index,
                "p": result["prefix_ids"],
                "r": result["response_ids"],
                "tr": bool(result["truncated"]),
            }
        )
    return tokenized


def tokenize_story(text: str, template: str) -> dict:
    tokenizer = common.load_tokenizer()
    result = packing.tokenize_example(
        tokenizer, template, "", text, max_len=2048
    )
    return result


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def pack_and_write(out_dir: Path, train_tokens, val_tokens, manifest):
    train_rows = packing.pack_examples(
        [
            {
                "index": r["i"],
                "prefix_ids": r["p"],
                "response_ids": r["r"],
            }
            for r in train_tokens
        ],
        T=2048,
    )
    val_rows = packing.pack_examples(
        [
            {
                "index": r["i"],
                "prefix_ids": r["p"],
                "response_ids": r["r"],
            }
            for r in val_tokens
        ],
        T=2048,
    )
    write_jsonl(out_dir / "train_tokens.jsonl", train_tokens)
    write_jsonl(out_dir / "val_tokens.jsonl", val_tokens)
    write_jsonl(
        out_dir / "train_rows.jsonl",
        [
            {
                "row": j,
                "examples": row.example_indices,
                "target_tokens": row.target_tokens,
                "sequence_tokens": row.sequence_tokens,
            }
            for j, row in enumerate(train_rows)
        ],
    )
    write_jsonl(
        out_dir / "val_rows.jsonl",
        [
            {
                "row": j,
                "examples": row.example_indices,
                "target_tokens": row.target_tokens,
                "sequence_tokens": row.sequence_tokens,
            }
            for j, row in enumerate(val_rows)
        ],
    )
    manifest.update(
        {
            "train_examples": len(train_tokens),
            "train_target_tokens": int(
                sum(r.target_tokens for r in train_rows)
            ),
            "train_sequence_tokens": int(
                sum(r.sequence_tokens for r in train_rows)
            ),
            "train_rows": len(train_rows),
            "val_examples": len(val_tokens),
            "val_target_tokens": int(
                sum(r.target_tokens for r in val_rows)
            ),
            "val_rows": len(val_rows),
        }
    )
    common.save_json(out_dir / "mixture_manifest.json", manifest)
    print(json.dumps(manifest, indent=1))


def build_tasks(name: str, per_task_train: int, per_task_dev: int,
                seed: int = 4242):
    out_dir = DATA_ROOT / name
    records = []
    for index, task in enumerate(TASK_ORDER):
        records.extend(
            tasks.generate_task(
                task, "train", per_task_train, seed + index * 101
            )
        )
    random.Random(seed).shuffle(records)
    tokenized = tokenize_examples(records)
    dev_records = []
    for index, task in enumerate(TASK_ORDER):
        dev_records.extend(
            tasks.generate_task(
                task, "dev", per_task_dev, seed + 5000 + index * 101
            )
        )
    dev_tokenized = tokenize_examples(dev_records)
    manifest = {
        "format": "arm_a_posttrain_data_v1",
        "created_at": common.iso_now(),
        "name": name,
        "kind": "programmatic_tasks",
        "tasks": list(TASK_ORDER),
        "per_task_train": per_task_train,
        "per_task_dev": per_task_dev,
        "train_seed": seed,
        "note": (
            "train wording templates only; dev/test templates are disjoint "
            "and defined in training/posttrain/tasks.py"
        ),
    }
    pack_and_write(out_dir, tokenized, dev_tokenized, manifest)
    return out_dir


def iter_tinystories(limit: int):
    from datasets import load_dataset

    dataset = load_dataset(
        TINYSTORIES_REPO, split="train", streaming=True,
        revision=TINYSTORIES_REVISION,
    )
    count = 0
    for record in dataset:
        text = (record.get("text") or "").strip()
        if not text:
            continue
        yield text
        count += 1
        if count >= limit:
            break


def build_stories(name: str, n_stories: int, seed: int = 4242):
    out_dir = DATA_ROOT / name
    tokenizer = common.load_tokenizer()
    rng = random.Random(seed)
    suite_path = (common.OUT_DIR.parent / "posttraining" / "eval_suite"
                  / "suite.json")
    suite_prompts = set()
    if suite_path.is_file():
        suite = common.load_json(suite_path)
        suite_prompts = {entry["prompt"] for entry in suite["stories"]}
    train_tokens = []
    val_tokens = []
    for index, story in enumerate(iter_tinystories(n_stories + 400)):
        instruction, _ = _story_instruction(rng, tasks.WORDS)
        if instruction in suite_prompts:
            continue
        result = tokenize_story(story, instruction)
        if result["skipped"]:
            continue
        record = {
            "i": index,
            "p": result["prefix_ids"],
            "r": result["response_ids"],
            "tr": bool(result["truncated"]),
        }
        if index < 400:
            val_tokens.append(record)
        else:
            train_tokens.append(record)
    for new_index, record in enumerate(train_tokens):
        record["i"] = new_index
    for new_index, record in enumerate(val_tokens):
        record["i"] = new_index
    manifest = {
        "format": "arm_a_posttrain_data_v1",
        "created_at": common.iso_now(),
        "name": name,
        "kind": "tinystories",
        "dataset": TINYSTORIES_REPO,
        "revision": TINYSTORIES_REVISION,
        "license": "cdla-sharing-1.0",
        "n_stories_requested": n_stories,
        "templates": list(STORY_TEMPLATES),
        "instruction_modes": 6,
        "suite_prompt_overlap_filtered": True,
        "train_nouns": list(STORY_TRAIN_NOUNS),
        "seed": seed,
    }
    pack_and_write(out_dir, train_tokens, val_tokens, manifest)
    return out_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True,
                        choices=["tasks", "stories", "mix"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--per-task-train", type=int, default=25000)
    parser.add_argument("--per-task-dev", type=int, default=200)
    parser.add_argument("--n-stories", type=int, default=20000)
    parser.add_argument("--mix-tasks", default="pt_tasks")
    parser.add_argument("--mix-stories", default="pt_stories")
    parser.add_argument("--task-fraction", type=float, default=0.4)
    args = parser.parse_args(argv)

    if args.kind == "tasks":
        build_tasks(args.name, args.per_task_train, args.per_task_dev)
    elif args.kind == "stories":
        build_stories(args.name, args.n_stories)
    else:
        mix_datasets(args)
    return 0


def mix_datasets(args):
    """Interleave task and story dirs by target-token fraction."""
    task_dir = DATA_ROOT / args.mix_tasks
    story_dir = DATA_ROOT / args.mix_stories
    out_dir = DATA_ROOT / args.name
    task_tokens = packing.load_tokens_jsonl(task_dir / "train_tokens.jsonl")
    story_tokens = packing.load_tokens_jsonl(story_dir / "train_tokens.jsonl")
    task_list = [task_tokens[i] for i in sorted(task_tokens)]
    story_list = [story_tokens[i] for i in sorted(story_tokens)]
    rng = random.Random(4242)
    rng.shuffle(task_list)
    rng.shuffle(story_list)
    fraction = args.task_fraction
    interleaved = []
    task_cursor = 0
    story_cursor = 0
    task_added = 0
    story_added = 0
    while task_cursor < len(task_list) and story_cursor < len(story_list):
        total = task_added + story_added
        want_task = (task_added / total) < fraction if total else True
        if want_task:
            record = task_list[task_cursor]
            task_cursor += 1
            task_added += len(record["r"])
        else:
            record = story_list[story_cursor]
            story_cursor += 1
            story_added += len(record["r"])
        interleaved.append(record)
    for index, record in enumerate(interleaved):
        record["i"] = index
    task_val_tokens = packing.load_tokens_jsonl(
        task_dir / "val_tokens.jsonl"
    )
    story_val_tokens = packing.load_tokens_jsonl(
        story_dir / "val_tokens.jsonl"
    )
    val_list = []
    task_val_cursor = 0
    story_val_cursor = 0
    while (task_val_cursor < len(task_val_tokens)
           or story_val_cursor < len(story_val_tokens)):
        want_task = (len(val_list) % 2) == 0 or story_val_cursor >= len(
            story_val_tokens
        )
        if want_task and task_val_cursor < len(task_val_tokens):
            val_list.append(task_val_tokens[task_val_cursor])
            task_val_cursor += 1
        elif story_val_cursor < len(story_val_tokens):
            val_list.append(story_val_tokens[story_val_cursor])
            story_val_cursor += 1
        elif task_val_cursor < len(task_val_tokens):
            val_list.append(task_val_tokens[task_val_cursor])
            task_val_cursor += 1
        else:
            break
    for index, record in enumerate(val_list):
        record["i"] = index
    manifest = {
        "format": "arm_a_posttrain_data_v1",
        "name": args.name,
        "kind": "mix",
        "task_fraction_by_target_tokens": fraction,
        "task_source": args.mix_tasks,
        "story_source": args.mix_stories,
        "task_target_tokens": task_added,
        "story_target_tokens": story_added,
    }
    pack_and_write(out_dir, interleaved, val_list, manifest)


if __name__ == "__main__":
    sys.exit(main())
