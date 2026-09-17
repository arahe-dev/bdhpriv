"""Build a frozen instruction-mixture training stream and validation split.

Usage:
  python -m training.autoresearch.prepare_mixture --name mix_base \
      --weights bespoke:25,oasst:25,tulu:25,openhermes:25

Produces under results/autoresearch/data/<name>/:
  train_tokens.jsonl  global instruction examples in deterministic order
  train_rows.jsonl    packed 2048-token row plan
  val_tokens.jsonl    frozen 1000-example validation split (250/source)
  val_rows.jsonl      packed validation rows
  mixture_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.sft_probe import common, packing  # noqa: E402

DATA_DIR = common.OUT_DIR.parent / "autoresearch" / "data"
VAL_PER_SOURCE = 250
MAX_LEN = 2048


def load_source_tokens(name: str):
    train = packing.load_tokens_jsonl(DATA_DIR / f"src_{name}_tokens.jsonl")
    val = packing.load_tokens_jsonl(DATA_DIR / f"src_{name}_val.jsonl")
    return train, val


def quota_pattern(weights: dict) -> list:
    """Deterministic round-robin pattern hitting the exact quota ratios."""
    total = sum(weights.values())
    counts = {
        name: int(round(weight / total * 100))
        for name, weight in weights.items()
    }
    drift = 100 - sum(counts.values())
    if drift:
        first = max(counts, key=counts.get)
        counts[first] += drift
    remaining = dict(counts)
    pattern = []
    order = list(weights)
    while sum(remaining.values()) > 0:
        for name in order:
            if remaining[name] > 0:
                pattern.append(name)
                remaining[name] -= 1
    return pattern


def build_mixture(name: str, weights: dict) -> dict:
    out_dir = DATA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {source: load_source_tokens(source) for source in weights}

    pattern = quota_pattern(weights)
    cursors = {source: 0 for source in weights}
    train_records = []
    source_counts = {source: 0 for source in weights}
    source_tokens = {source: 0 for source in weights}
    for position in range(len(pattern) * 10000):
        source = pattern[position % len(pattern)]
        pool = sources[source][0]
        cursor = cursors[source]
        if cursor >= len(pool):
            continue
        example = pool[cursor]
        cursors[source] = cursor + 1
        train_records.append(
            {
                "i": len(train_records),
                "p": example["p"],
                "r": example["r"],
                "tr": example["tr"],
                "src": source,
            }
        )
        source_counts[source] += 1
        source_tokens[source] += len(example["r"])
        if all(cursors[s] >= len(sources[s][0]) for s in weights):
            break
        if len(train_records) >= 4_000_000:
            break

    val_records = []
    source_val_counts = {}
    for source in weights:
        val_pool = sources[source][1]
        taken = 0
        for example in val_pool.values():
            if taken >= VAL_PER_SOURCE:
                break
            val_records.append(
                {
                    "i": len(val_records),
                    "p": example["p"],
                    "r": example["r"],
                    "tr": example["tr"],
                    "src": source,
                }
            )
            taken += 1
        source_val_counts[source] = taken

    train_rows = packing.pack_examples(
        [
            {"index": r["i"], "prefix_ids": r["p"], "response_ids": r["r"]}
            for r in train_records
        ],
        T=MAX_LEN,
    )
    val_rows = packing.pack_examples(
        [
            {"index": r["i"], "prefix_ids": r["p"], "response_ids": r["r"]}
            for r in val_records
        ],
        T=MAX_LEN,
    )

    def write_jsonl(path, records):
        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    write_jsonl(out_dir / "train_tokens.jsonl", train_records)
    write_jsonl(out_dir / "val_tokens.jsonl", val_records)
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

    manifest = {
        "format": "arm_a_autoresearch_mixture_v1",
        "created_at": common.iso_now(),
        "name": name,
        "weights": weights,
        "quota_pattern": pattern,
        "seed": common.SPLIT_SEED,
        "deterministic_order": (
            "round-robin quota pattern over per-source token pools; no "
            "reshuffling"
        ),
        "train_examples": len(train_records),
        "train_target_tokens": int(
            sum(r.target_tokens for r in train_rows)
        ),
        "train_sequence_tokens": int(
            sum(r.sequence_tokens for r in train_rows)
        ),
        "train_rows": len(train_rows),
        "source_train_counts": source_counts,
        "source_train_target_tokens": source_tokens,
        "val_examples": len(val_records),
        "val_target_tokens": int(
            sum(r.target_tokens for r in val_rows)
        ),
        "val_rows": len(val_rows),
        "source_val_counts": source_val_counts,
        "val_frozen_note": (
            "the validation split is identical across mixture variants "
            "(250 examples per source); only the training mixture changes"
        ),
    }
    common.save_json(out_dir / "mixture_manifest.json", manifest)
    print(json.dumps({
        "name": name,
        "train_examples": manifest["train_examples"],
        "train_target_tokens": manifest["train_target_tokens"],
        "train_sequence_tokens": manifest["train_sequence_tokens"],
        "train_rows": manifest["train_rows"],
        "source_train_counts": source_counts,
        "val_target_tokens": manifest["val_target_tokens"],
    }, indent=1))
    return manifest


def parse_weights(text: str) -> dict:
    weights = {}
    for item in text.split(","):
        source, value = item.split(":")
        weights[source.strip()] = float(value)
    return weights


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--weights", default="bespoke:25,oasst:25,tulu:25,openhermes:25"
    )
    args = parser.parse_args(argv)
    build_mixture(args.name, parse_weights(args.weights))
    return 0


if __name__ == "__main__":
    sys.exit(main())
