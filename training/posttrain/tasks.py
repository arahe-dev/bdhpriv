"""Programmatic verifiable tasks for the post-training campaign.

Every task has:
  * disjoint train/dev/test wording templates (test wording never trained),
  * deterministic sampling from a seed per split,
  * a programmatic answer and a strict grader.

Tasks are deliberately short-answer (<= 8 tokens) so a 17M model can
plausibly learn them; broad instruction following is not attempted.
"""

from __future__ import annotations

import random
import re
from typing import Callable, Dict, List

WORDS = (
    "cat", "dog", "sun", "moon", "tree", "book", "house", "water", "fire",
    "bird", "fish", "star", "cloud", "rain", "snow", "wind", "road", "town",
    "farm", "hill", "lake", "rock", "sand", "ship", "train", "plane", "bike",
    "apple", "bread", "milk", "cake", "soup", "rice", "corn", "bean", "leaf",
    "rose", "grass", "stone", "table", "chair", "door", "window", "garden",
    "river", "forest", "island", "castle", "dragon", "knight", "queen",
    "king", "prince", "witch", "giant", "mouse", "horse", "sheep", "goat",
    "duck", "frog", "snake", "tiger", "lion", "bear", "wolf", "fox", "deer",
    "rabbit", "monkey", "panda", "whale", "shark", "eagle", "owl", "robin",
    "happy", "sad", "big", "small", "fast", "slow", "hot", "cold", "bright",
    "dark", "loud", "quiet", "soft", "hard", "sweet", "bitter", "young",
    "old", "new", "clean", "dirty", "empty", "full", "open", "closed",
    "run", "jump", "sing", "dance", "read", "write", "draw", "paint", "build",
    "climb", "swim", "fly", "walk", "laugh", "smile", "dream", "sleep",
)

TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "copy": {
        "train": [
            "Copy this text: {payload}",
            "Repeat after me: {payload}",
            "Write the same words again: {payload}",
        ],
        "dev": ["Copy the phrase: {payload}"],
        "test": [
            "Echo the following: {payload}",
            "Please copy: {payload}",
            "Type the same words: {payload}",
        ],
    },
    "reverse": {
        "train": [
            "Reverse the word {payload}:",
            "Write {payload} backwards:",
            "Backwards spelling of {payload}:",
        ],
        "dev": ["Reverse this word: {payload}"],
        "test": ["Spell {payload} in reverse:", "Reverse: {payload}"],
    },
    "sort_asc": {
        "train": [
            "Sort these numbers: {payload}",
            "Put in order: {payload}",
            "Arrange from smallest to largest: {payload}",
        ],
        "dev": ["Sort ascending: {payload}"],
        "test": ["Order these: {payload}", "Sort: {payload}"],
    },
    "add": {
        "train": [
            "Calculate: {payload} =",
            "Add: {payload} =",
            "What is {payload}?",
        ],
        "dev": ["Sum: {payload} ="],
        "test": ["Compute: {payload} =", "{payload} ="],
    },
    "count_words": {
        "train": [
            "Count the words: {payload}",
            "How many words are here: {payload}",
            "Number of words: {payload}",
        ],
        "dev": ["Count words: {payload}"],
        "test": ["How many words: {payload}", "Count: {payload}"],
    },
    "first_letter": {
        "train": [
            "First letter of {payload}:",
            "What letter does {payload} start with?",
            "Initial letter of {payload}:",
        ],
        "dev": ["First letter: {payload}"],
        "test": ["Starts with what letter: {payload}", "{payload} begins with:"],
    },
}


def _payload(task: str, rng: random.Random) -> str:
    if task == "copy":
        words = [rng.choice(WORDS) for _ in range(rng.randint(3, 5))]
        return " ".join(words)
    if task == "reverse":
        return rng.choice([w for w in WORDS if len(w) >= 3])
    if task == "sort_asc":
        values = [rng.randint(1, 99) for _ in range(rng.randint(3, 4))]
        return " ".join(str(v) for v in values)
    if task == "add":
        return f"{rng.randint(1, 99)} + {rng.randint(1, 99)}"
    if task == "count_words":
        return " ".join(rng.choice(WORDS) for _ in range(rng.randint(2, 7)))
    if task == "first_letter":
        return rng.choice(WORDS)
    raise KeyError(task)


def _answer(task: str, payload: str) -> str:
    if task == "copy":
        return payload
    if task == "reverse":
        return payload[::-1]
    if task == "sort_asc":
        return " ".join(str(v) for v in sorted(int(x) for x in payload.split()))
    if task == "add":
        left, right = payload.split(" + ")
        return str(int(left) + int(right))
    if task == "count_words":
        return str(len(payload.split()))
    if task == "first_letter":
        return payload[0]
    raise KeyError(task)


def generate_task(task: str, split: str, count: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    templates = TEMPLATES[task][split]
    examples = []
    for _ in range(count):
        payload = _payload(task, rng)
        template = rng.choice(templates)
        prompt = template.format(payload=payload)
        answer = _answer(task, payload)
        examples.append(
            {
                "task": task,
                "split": split,
                "prompt": prompt,
                "payload": payload,
                "answer": answer,
            }
        )
    return examples


def normalize(text: str) -> str:
    text = text.strip().splitlines()[0] if text.strip() else ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def grade(task: str, prediction: str, answer: str) -> bool:
    return normalize(prediction) == normalize(answer)


def grade_story(text: str, required_word: str) -> dict:
    words = text.split()
    lower = text.lower()
    token_repeats = {}
    tokens = re.findall(r"[a-zA-Z']+", lower)
    for index in range(len(tokens) - 5):
        gram = tuple(tokens[index:index + 6])
        token_repeats[gram] = token_repeats.get(gram, 0) + 1
    max_gram_repeat = max(token_repeats.values()) if token_repeats else 0
    max_run = 1
    run = 1
    for previous, current in zip(tokens, tokens[1:]):
        run = run + 1 if current == previous else 1
        max_run = max(max_run, run)
    checks = {
        "word_count_20_200": 20 <= len(words) <= 200,
        "ends_with_punctuation": text.rstrip().endswith((".", "!", "?")),
        "contains_required_word": required_word.lower() in lower,
        "no_heavy_5gram_repeat": max_gram_repeat <= 3,
        "no_long_token_run": max_run <= 4,
        "not_echoing_prompt": len(words) >= 10,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "word_count": len(words),
        "max_6gram_repeat": max_gram_repeat,
        "max_token_run": max_run,
    }


FLUENCY_EXCLUDE = {"contains_required_word"}


def grade_story_fluency(text: str) -> dict:
    """Fluency-only story grading (constraint following is scored separately)."""
    grade = grade_story(text, "")
    checks = {
        key: value for key, value in grade["checks"].items()
        if key not in FLUENCY_EXCLUDE
    }
    checks["min_40_words"] = grade["word_count"] >= 40
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "word_count": grade["word_count"],
        "max_6gram_repeat": grade["max_6gram_repeat"],
        "max_token_run": grade["max_token_run"],
    }
