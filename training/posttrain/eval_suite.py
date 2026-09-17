"""Frozen verifiable evaluation suite for post-training cycles.

Contains three independent instruments:
  * programmatic tasks (held-out wording templates, exact match)
  * BLiMP minimal pairs (grammaticality by length-normalized log-prob)
  * constrained story prompts graded by rules

The suite is frozen once (build_suite) and never trained on; test templates
and seeds are disjoint from all training data.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.posttrain import tasks  # noqa: E402
from training.sft_probe import common, packing  # noqa: E402

SUITE_PATH = common.OUT_DIR.parent / "posttraining" / "eval_suite" / "suite.json"
BLIMP_DIR = common.OUT_DIR.parent / "posttraining" / "raw" / "blimp"
TASKS = ("copy", "reverse", "sort_asc", "add", "count_words", "first_letter")
TEST_PER_TASK = 80
DEV_PER_TASK = 80
TEST_SEED = 7_000_001
DEV_SEED = 7_000_002
BLIMP_PARADIGMS = (
    "anaphor_gender_agreement",
    "anaphor_number_agreement",
    "determiner_noun_agreement_1",
    "determiner_noun_agreement_2",
    "regular_plural_subject_verb_agreement_1",
    "regular_plural_subject_verb_agreement_2",
    "irregular_plural_subject_verb_agreement_1",
    "irregular_plural_subject_verb_agreement_2",
    "principle_A_case_1",
    "principle_A_case_2",
    "wh_questions_object_gap",
    "wh_questions_subject_gap",
    "sentential_negation_npi_licensor_present",
    "npi_present_1",
    "distractor_agreement_relational_noun",
)
BLIMP_PAIRS_PER_PARADIGM = 200
STORY_PROMPTS = 40
STORY_SEED = 8_000_003
STORY_NOUNS = (
    "a lost puppy", "a brave mouse", "a small robot", "a magic garden",
    "a kind giant", "a sleepy dragon", "a red boat", "a shiny star",
    "a lonely bird", "a clever fox", "a muddy pig", "a tiny castle",
    "a talking cat", "a happy family", "a windy day", "a snowy hill",
    "a golden key", "a quiet library", "a busy farm", "a deep ocean",
    "a broken toy", "a new friend", "a secret door", "a loud party",
    "a green forest", "a warm blanket", "a fast train", "a tall tree",
    "a blue whale", "a little seed", "a hungry bear", "a bright light",
    "a wooden bridge", "a silver fish", "a purple hat", "a funny clown",
    "a snowy morning", "a nice teacher", "a giant pumpkin", "a brave girl",
)
BLIMP_STOP = {".", "?", "!", ","}


def build_suite(force: bool = False) -> dict:
    if SUITE_PATH.is_file() and not force:
        return common.load_json(SUITE_PATH)
    suite = {
        "format": "arm_a_posttrain_suite_v1",
        "created_at": common.iso_now(),
        "test_seed": TEST_SEED,
        "dev_seed": DEV_SEED,
        "tasks": {},
        "blimp": {},
        "stories": [],
    }
    for task in TASKS:
        salt = zlib.crc32(task.encode()) % 1000
        suite["tasks"][task] = {
            "test": tasks.generate_task(task, "test", TEST_PER_TASK,
                                        TEST_SEED + salt),
            "dev": tasks.generate_task(task, "dev", DEV_PER_TASK,
                                       DEV_SEED + salt),
        }
    import pyarrow.parquet as pq

    for paradigm in BLIMP_PARADIGMS:
        path = BLIMP_DIR / paradigm / "train-00000-of-00001.parquet"
        if not path.is_file():
            continue
        table = pq.read_table(path).slice(0, BLIMP_PAIRS_PER_PARADIGM)
        pairs = table.select(["sentence_good", "sentence_bad"]).to_pylist()
        suite["blimp"][paradigm] = pairs
    rng = random.Random(STORY_SEED)
    words = list(tasks.WORDS)
    for index in range(STORY_PROMPTS):
        noun = STORY_NOUNS[index % len(STORY_NOUNS)]
        required = rng.choice(words)
        suite["stories"].append(
            {
                "id": f"story_{index:02d}",
                "prompt": (
                    f"Write a short story about {noun}. "
                    f"Include the word '{required}'."
                ),
                "required_word": required,
            }
        )
    SUITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    common.save_json(SUITE_PATH, suite)
    return suite


def prompt_prefix(instruction: str) -> str:
    return f"User: {instruction}\nAssistant: "


@torch.no_grad()
def generate_text(model, tokenizer, device, instruction: str, max_new: int,
                  sampler=None, weights=None, akasha_cfg=None) -> str:
    if weights is not None and akasha_cfg is not None:
        from akasha.models.arma.reference_recurrent import (
            create_state, prefill_tokens, step,
        )

        prefix_ids = tokenizer.encode(prompt_prefix(instruction)).ids
        ids = torch.tensor(prefix_ids, dtype=torch.long, device=device)
        state = create_state(weights, akasha_cfg)
        logits = prefill_tokens(weights, akasha_cfg, state, ids)
        generated = []
        for _ in range(max_new):
            if sampler is None:
                next_id = int(torch.argmax(logits).item())
            else:
                next_id = sampler.sample(logits.detach().to("cpu"))
            generated.append(next_id)
            if next_id == tokenizer.encode("\n").ids[0] and len(generated) > 1:
                break
            logits = step(weights, akasha_cfg, state, next_id)
        return tokenizer.decode(generated)

    prefix_ids = tokenizer.encode(prompt_prefix(instruction)).ids
    row = {
        "x": np.full(2048, packing.PAD_ID, dtype=np.uint16),
        "pos": np.zeros(2048, dtype=np.int32),
        "segpos": np.zeros(2048, dtype=np.int32),
        "start": np.zeros(2048, dtype=np.int32),
        "input_valid": np.zeros(2048, dtype=np.bool_),
    }
    n = min(len(prefix_ids), 2048)
    row["x"][:n] = np.asarray(prefix_ids[:n], dtype=np.uint16)
    row["pos"][:n] = np.arange(n, dtype=np.int32)
    row["segpos"][:n] = np.arange(n, dtype=np.int32)
    row["input_valid"][:n] = True
    row["start"][n:] = 2049
    cpu = packing.rows_to_cpu_batch(
        [packing.PackedRow(
            x=row["x"], y=np.zeros(2048, dtype=np.uint16), pos=row["pos"],
            segpos=row["segpos"], start=row["start"],
            input_valid=row["input_valid"],
            valid=np.zeros(2048, dtype=np.bool_),
            example_indices=[], target_tokens=0, sequence_tokens=n,
        )]
    )
    batch = packing.cpu_batch_to_device(cpu, device)
    generated = []
    tokens = list(prefix_ids[:n])
    for _ in range(max_new):
        logits = model.forward_packed(
            batch["x"], batch["pos"], batch["segpos"], batch["full_mask"],
            batch["start"],
        )
        next_logits = logits[0, -1]
        if sampler is None:
            next_id = int(torch.argmax(next_logits).item())
        else:
            next_id = sampler.sample(next_logits.detach().to("cpu"))
        generated.append(next_id)
        tokens.append(next_id)
        if next_id == tokenizer.encode("\n").ids[0] and len(generated) > 1:
            break
        position = len(tokens) - 1
        if position >= 2048:
            break
        batch["x"][0, position] = next_id
        batch["pos"][0, position] = position
        batch["segpos"][0, position] = position
        batch["start"][0, position] = 0
        batch["input_valid"][0, position] = True
        batch["full_mask"][0, position, :position] = True
    return tokenizer.decode(generated)


@torch.no_grad()
def sentence_logprob_batch(model, device, list_of_token_ids,
                           max_len: int = 2048) -> list:
    """Length-normalized and raw log-prob for many sentences in one forward.

    Each sentence is its own segment inside one packed row; targets are only
    scored within a segment.
    """
    results = []
    chunk_ids = []
    chunk_lengths = []

    def flush():
        if not chunk_ids:
            return
        length = sum(len(ids) for ids in chunk_ids)
        x = np.full(max_len, packing.PAD_ID, dtype=np.uint16)
        y = np.zeros(max_len, dtype=np.uint16)
        pos = np.zeros(max_len, dtype=np.int32)
        segpos = np.zeros(max_len, dtype=np.int32)
        start = np.zeros(max_len, dtype=np.int32)
        input_valid = np.zeros(max_len, dtype=np.bool_)
        valid = np.zeros(max_len, dtype=np.bool_)
        cursor = 0
        spans = []
        for ids in chunk_ids:
            n = len(ids)
            x[cursor:cursor + n] = np.asarray(ids, dtype=np.uint16)
            pos[cursor:cursor + n] = np.arange(n, dtype=np.int32)
            segpos[cursor:cursor + n] = np.arange(n, dtype=np.int32)
            start[cursor:cursor + n] = cursor
            input_valid[cursor:cursor + n] = True
            if n > 1:
                y[cursor:cursor + n - 1] = x[cursor + 1:cursor + n]
                valid[cursor:cursor + n - 1] = True
            spans.append((cursor, n))
            cursor += n
        start[cursor:] = max_len + 1
        cpu = packing.rows_to_cpu_batch(
            [packing.PackedRow(
                x=x, y=y, pos=pos, segpos=segpos, start=start,
                input_valid=input_valid, valid=valid, example_indices=[],
                target_tokens=int(valid.sum()), sequence_tokens=cursor,
            )]
        )
        batch = packing.cpu_batch_to_device(cpu, device)
        logits = model.forward_packed(
            batch["x"], batch["pos"], batch["segpos"], batch["full_mask"],
            batch["start"],
        )
        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        picked = log_probs.gather(
            -1, batch["y"][0].unsqueeze(-1)
        ).squeeze(-1)
        for begin, n in spans:
            tokens = picked[begin:begin + n - 1]
            total = float(tokens.sum().item())
            results.append({
                "sum": total,
                "tokens": int(n - 1),
                "normalized": total / max(1, n - 1),
            })
        chunk_ids.clear()
        chunk_lengths.clear()

    for ids in list_of_token_ids:
        if len(chunk_ids) >= 128 or (
            chunk_ids and sum(len(i) for i in chunk_ids) + len(ids) > max_len
        ):
            flush()
        chunk_ids.append(ids)
    flush()
    return results


def evaluate_suite(model, tokenizer, device, suite: dict,
                   story_max_new: int = 80, story_limit: int = 12,
                   weights=None, akasha_cfg=None, task_limit=None) -> dict:
    result = {
        "tasks": {},
        "blimp": {},
        "stories": {},
        "audit": {},
    }
    for task in TASKS:
        entries = suite["tasks"][task]["test"]
        if task_limit is not None:
            entries = entries[:task_limit]
        correct = 0
        samples = []
        for entry in entries:
            text = generate_text(
                model, tokenizer, device, entry["prompt"], max_new=16,
                weights=weights, akasha_cfg=akasha_cfg,
            )
            ok = tasks.grade(task, text, entry["answer"])
            correct += int(ok)
            if len(samples) < 5:
                samples.append({
                    "prompt": entry["prompt"],
                    "expected": entry["answer"],
                    "generated": text,
                    "correct": ok,
                })
        result["tasks"][task] = {
            "accuracy": correct / max(1, len(entries)),
            "correct": correct,
            "total": len(entries),
        }
        result["audit"][task] = samples

    for paradigm, pairs in suite["blimp"].items():
        good_ids = [
            tokenizer.encode(pair["sentence_good"]).ids for pair in pairs
        ]
        bad_ids = [
            tokenizer.encode(pair["sentence_bad"]).ids for pair in pairs
        ]
        good_lps = sentence_logprob_batch(model, device, good_ids)
        bad_lps = sentence_logprob_batch(model, device, bad_ids)
        good_wins = sum(
            1 for good, bad in zip(good_lps, bad_lps)
            if good["normalized"] > bad["normalized"]
        )
        raw_good_wins = sum(
            1 for good, bad in zip(good_lps, bad_lps)
            if good["sum"] > bad["sum"]
        )
        result["blimp"][paradigm] = {
            "length_normalized_accuracy": good_wins / max(1, len(pairs)),
            "raw_accuracy": raw_good_wins / max(1, len(pairs)),
            "pairs": len(pairs),
        }
    if result["blimp"]:
        result["blimp_macro_length_normalized"] = float(
            np.mean([v["length_normalized_accuracy"]
                     for v in result["blimp"].values()])
        )
        result["blimp_macro_raw"] = float(
            np.mean([v["raw_accuracy"] for v in result["blimp"].values()])
        )

    story_pass = 0
    fluency_pass = 0
    constraint_pass = 0
    stories = suite["stories"][:story_limit]
    audit = []
    for entry in stories:
        text = generate_text(
            model, tokenizer, device, entry["prompt"], max_new=story_max_new,
            weights=weights, akasha_cfg=akasha_cfg,
        )
        grade = tasks.grade_story(text, entry["required_word"])
        fluency = tasks.grade_story_fluency(text)
        story_pass += int(grade["passed"])
        fluency_pass += int(fluency["passed"])
        constraint_pass += int(grade["checks"]["contains_required_word"])
        if len(audit) < 5:
            audit.append({
                "id": entry["id"],
                "prompt": entry["prompt"],
                "generated": text,
                "grade": grade,
                "fluency": fluency,
            })
    total = max(1, len(stories))
    result["stories"] = {
        "pass_rate": story_pass / total,
        "fluency_pass_rate": fluency_pass / total,
        "constraint_follow_rate": constraint_pass / total,
        "passed": story_pass,
        "total": len(stories),
    }
    result["audit"]["stories"] = audit
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    suite = build_suite(force=args.force)
    summary = {
        "path": str(SUITE_PATH),
        "tasks": {task: len(suite["tasks"][task]["test"]) for task in TASKS},
        "blimp_paradigms": len(suite["blimp"]),
        "blimp_pairs": sum(len(v) for v in suite["blimp"].values()),
        "stories": len(suite["stories"]),
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
