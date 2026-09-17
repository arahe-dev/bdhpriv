"""Frozen capability battery for the post-trained Arm-A model.

Exploratory capability mapping, NOT a model-selection instrument: the
battery is frozen (hash-recorded) before any model is evaluated, all items
are scored programmatically, and every generated output is stored for
inspection.

Families:
  arith_train / arith_carry / arith_ood   arithmetic (in and out of the SFT
                                          distribution)
  copy / reverse / sort / count / first   trained programmatic skills
  transfer                                same skills, unseen wordings
  context_extract                         use a fact stated in the prompt
  facts                                   world knowledge
  listing                                 category enumeration
  format                                  output-format compliance
  completion                              sentence completion (LM prior)
  story                                   constrained free generation
  nonsense                                non-degeneracy on odd inputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.posttrain import eval_suite, tasks  # noqa: E402
from training.sft_probe import common  # noqa: E402

PROBE_PATH = (common.OUT_DIR.parent / "posttraining" / "capability_probe"
              / "battery.json")
OUT_DIR = common.OUT_DIR.parent / "posttraining" / "capability"
SEED = 31_337

COLORS = (
    "red", "blue", "green", "yellow", "black", "white", "purple",
    "orange", "pink", "brown", "gray", "grey",
)
ANIMALS = (
    "cat", "dog", "bird", "fish", "horse", "cow", "pig", "sheep", "lion",
    "tiger", "bear", "mouse", "rabbit", "fox", "duck", "frog", "snake",
    "whale", "shark", "elephant", "monkey", "owl", "deer", "goat",
)
NUMBER_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}

FACT_ITEMS = (
    ("What is the capital of France?", ("paris",)),
    ("What is the capital of India?", ("new delhi", "delhi")),
    ("What is the capital of the United States?", ("washington", "washington dc", "washington d.c.")),
    ("What is the capital of Japan?", ("tokyo",)),
    ("What color is the sky on a clear day?", ("blue",)),
    ("What color is grass?", ("green",)),
    ("How many days are in a week?", ("7", "seven")),
    ("Which animal says meow?", ("cat",)),
    ("Which company makes the iPhone?", ("apple",)),
    ("What is 2 plus 2?", ("4", "four")),
)

COMPLETION_ITEMS = (
    ("The cat sat on the", ("mat", "floor", "couch", "chair", "bed", "ground", "table", "lap", "roof", "windowsill")),
    ("The sun rises in the", ("east", "morning", "sky")),
    ("Water freezes when it gets", ("cold", "freezing", "colder")),
    ("A dog says", ("woof", "bark", "ruff", "bow")),
    ("We read a", ("book", "story", "novel", "paper", "magazine", "poem", "letter")),
    ("She opened the door and", ("walked", "went", "entered", "stepped", "ran", "saw", "left", "closed")),
    ("The boy kicked the", ("ball", "can", "stone", "bucket", "rock")),
    ("I drink water from a", ("cup", "glass", "bottle", "mug", "straw")),
    ("The fish swims in the", ("water", "pond", "sea", "lake", "ocean", "river")),
    ("He wrote his name with a", ("pen", "pencil", "crayon", "marker", "stick", "chalk")),
)

LISTING_ITEMS = (
    ("List three colors.", "colors"),
    ("Name two colors.", "colors"),
    ("Give me a color.", "colors"),
    ("List three animals.", "animals"),
    ("Name two animals.", "animals"),
    ("Name an animal that lives in water.", "animals"),
    ("List three fruits.", "fruits"),
    ("Name two numbers.", "numbers"),
)

FORMAT_ITEMS = (
    ("Answer yes or no: Is the sky blue?", ("yes", "no")),
    ("Answer yes or no: Is fire cold?", ("yes", "no")),
    ("Answer with one word: What color is snow?", ("white", "snow")),
    ("Answer with one word: What color is coal?", ("black",)),
    ("Reply with a single number: 3", ("3", "three")),
    ("Reply with a single number: 8", ("8", "eight")),
)

NONSENSE_ITEMS = (
    "asdfgh qwerty zxcvb",
    "???! ***",
    "1234567890",
    "the the the the the",
    "x",
)


def normalize(text: str) -> str:
    text = text.strip().splitlines()[0] if text.strip() else ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_battery(force=False):
    if PROBE_PATH.is_file() and not force:
        return common.load_json(PROBE_PATH)
    rng = random.Random(SEED)
    battery = {
        "format": "arm_a_capability_battery_v1",
        "created_at": common.iso_now(),
        "seed": SEED,
        "families": {},
    }

    def add(family, prompt, expected, meta=None):
        battery["families"].setdefault(family, []).append(
            {"prompt": prompt, "expected": expected, "meta": meta or {}}
        )

    for _ in range(20):
        a, b = rng.randint(10, 89), rng.randint(10, 89)
        carry = (a % 10) + (b % 10) >= 10
        add("arith_train", f"Calculate: {a} + {b} =", str(a + b),
            {"carry": carry})
    for _ in range(8):
        a, b = rng.randint(100, 899), rng.randint(100, 899)
        add("arith_ood_add", f"Calculate: {a} + {b} =", str(a + b))
    for _ in range(6):
        a, b = rng.randint(20, 99), rng.randint(1, 19)
        add("arith_ood_sub", f"Calculate: {a} - {b} =", str(a - b))
    for _ in range(6):
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        add("arith_ood_mul", f"Calculate: {a} * {b} =", str(a * b))
    for _ in range(6):
        b = rng.randint(2, 9)
        a = b * rng.randint(2, 9)
        add("arith_ood_div", f"What is {a} divided by {b}?", str(a // b))
    for _ in range(10):
        words = [rng.choice(tasks.WORDS) for _ in range(rng.randint(3, 5))]
        payload = " ".join(words)
        add("copy", f"Copy this text: {payload}", payload)
    for _ in range(10):
        word = rng.choice([w for w in tasks.WORDS if len(w) >= 3])
        add("reverse", f"Reverse the word {word}:", word[::-1])
    for _ in range(10):
        values = [rng.randint(1, 99) for _ in range(3)]
        payload = " ".join(str(v) for v in values)
        add("sort", f"Sort these numbers: {payload}",
            " ".join(str(v) for v in sorted(values)))
    for _ in range(10):
        words = [rng.choice(tasks.WORDS) for _ in range(rng.randint(2, 7))]
        add("count", f"Count the words: {' '.join(words)}", str(len(words)))
    for _ in range(10):
        word = rng.choice(tasks.WORDS)
        add("first_letter", f"First letter of {word}:", word[0])
    for _ in range(3):
        word = rng.choice([w for w in tasks.WORDS if len(w) >= 3])
        add("transfer", f"Spell {word} backwards:", word[::-1])
    for _ in range(3):
        values = [rng.randint(1, 99) for _ in range(3)]
        add("transfer", f"Write the numbers in order: "
                        f"{' '.join(map(str, values))}",
            " ".join(str(v) for v in sorted(values)))
    for _ in range(3):
        words = [rng.choice(tasks.WORDS) for _ in range(rng.randint(2, 6))]
        add("transfer", f"How many items are in the list: "
                        f"{' '.join(words)}", str(len(words)))
    for _ in range(3):
        word = rng.choice(tasks.WORDS)
        add("transfer", f"The word {word} starts with which letter?", word[0])
    for _ in range(10):
        word = rng.choice(tasks.WORDS)
        add("context_extract",
            f"The secret word is {word}. Question: What is the secret word?",
            word)
    for prompt, answers in FACT_ITEMS:
        add("facts", prompt, list(answers))
    for prompt, kind in LISTING_ITEMS:
        add("listing", prompt, kind)
    for prompt, answers in FORMAT_ITEMS:
        add("format", prompt, list(answers))
    for prompt, answers in COMPLETION_ITEMS:
        add("completion", prompt, list(answers))
    for index in range(5):
        noun = ("a dog", "a little cat", "a happy child", "a bird",
                "a small house")[index]
        add("story", f"Tell me a short story about {noun}.", "fluency")
    for prompt in NONSENSE_ITEMS:
        add("nonsense", prompt, "nondegenerate")

    payload = json.dumps(battery, sort_keys=True).encode()
    battery["battery_sha256"] = hashlib.sha256(payload).hexdigest()
    PROBE_PATH.parent.mkdir(parents=True, exist_ok=True)
    common.save_json(PROBE_PATH, battery)
    return battery


def grade_item(family, entry, generated):
    text = generated.strip()
    norm = normalize(text)
    expected = entry["expected"]
    if family in ("copy", "reverse", "sort", "count", "first_letter",
                  "transfer", "context_extract",
                  "arith_train", "arith_ood_add", "arith_ood_sub",
                  "arith_ood_mul", "arith_ood_div"):
        return norm == normalize(expected)
    if family == "facts":
        return any(normalize(a) in norm.split(" ")
                   or norm == normalize(a) or normalize(a) in norm
                   for a in expected)
    if family == "format":
        return any(norm == normalize(a) or norm.split(" ")[0] ==
                   normalize(a) for a in expected)
    if family == "completion":
        first = norm.split(" ")[0] if norm else ""
        return first in [normalize(a) for a in expected]
    if family == "listing":
        words = set(norm.split(" "))
        allowed = set(COLORS if expected == "colors" else
                      ANIMALS if expected == "animals" else
                      ("apple", "banana", "orange", "grape", "pear",
                       "peach", "mango", "plum") if expected == "fruits"
                      else tuple(NUMBER_WORDS.values()) + tuple(NUMBER_WORDS))
        hits = words & allowed
        return len(hits) >= 1
    if family == "story":
        grade = tasks.grade_story_fluency(text)
        return bool(grade["passed"])
    if family == "nonsense":
        if not text:
            return False
        tokens = re.findall(r"[a-zA-Z']+", text.lower())
        if len(set(tokens)) <= 1 and len(tokens) > 1:
            return False
        run = 1
        best = 1
        for previous, current in zip(tokens, tokens[1:]):
            run = run + 1 if current == previous else 1
            best = max(best, run)
        return best <= 6
    return False


def run_family(model, tokenizer, device, weights, cfg, family, entries,
               max_new):
    results = []
    for entry in entries:
        generated = eval_suite.generate_text(
            model, tokenizer, device, entry["prompt"], max_new=max_new,
            weights=weights, akasha_cfg=cfg,
        )
        correct = grade_item(family, entry, generated)
        results.append({
            "prompt": entry["prompt"],
            "expected": entry["expected"],
            "generated": generated,
            "correct": correct,
            "meta": entry["meta"],
        })
    return results


MAX_NEW = {
    "copy": 16, "reverse": 12, "sort": 16, "count": 10, "first_letter": 8,
    "transfer": 16, "context_extract": 12, "facts": 16, "listing": 24,
    "format": 12, "completion": 8, "story": 160, "nonsense": 24,
    "arith_train": 10, "arith_ood_add": 10, "arith_ood_sub": 10,
    "arith_ood_mul": 10, "arith_ood_div": 12,
}


def main(argv=None) -> int:
    from akasha.checkpoint.loader import map_trainer_state_dict
    from akasha.models.arma.config import production_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--checkpoint", default="base")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    battery = build_battery()
    device = torch.device(args.device)
    tokenizer = common.load_tokenizer()
    if args.checkpoint in (None, "base"):
        state = common.load_base_state()
    else:
        payload = torch.load(args.checkpoint, map_location="cpu",
                             weights_only=False)
        state = payload["model"]
    model = common.build_model(state, device=device.type)
    model.to(device)
    model.eval()
    cfg = production_config()
    weights = map_trainer_state_dict(state, cfg, strict=True)
    weights = weights.to(device=device, dtype=torch.float32)

    report = {
        "tag": args.tag,
        "checkpoint": str(args.checkpoint),
        "battery_sha256": battery.get("battery_sha256"),
        "families": {},
        "summary": {},
    }
    for family, entries in battery["families"].items():
        results = run_family(
            model, tokenizer, device, weights, cfg, family, entries,
            MAX_NEW.get(family, 16),
        )
        accuracy = sum(r["correct"] for r in results) / len(results)
        report["families"][family] = {
            "accuracy": accuracy,
            "items": results,
        }
        report["summary"][family] = accuracy
    report["summary"]["macro"] = sum(report["summary"].values()) / len(
        report["summary"]
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    common.save_json(OUT_DIR / f"{args.tag}.json", report)
    print(json.dumps(report["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
