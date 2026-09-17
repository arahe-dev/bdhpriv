"""Instruction-dataset sources for the SFT autoresearch mission.

Pinned datasets (license noted from dataset cards):
  bespoke     HuggingFaceH4/Bespoke-Stratos-17k    Apache-2.0
  oasst       OpenAssistant/oasst1                 Apache-2.0
  tulu        allenai/tulu-3-sft-mixture           ODC-BY 1.0 (mixture)
  openhermes  teknium/OpenHermes-2.5               Apache-2.0

Each source is converted to our single-turn format (first user turn as the
instruction, first assistant turn as the response), tokenized with the exact
Arm-A tokenizer, deterministically stride-sampled to a target-token cap, and
written as per-source token pools. Metadata is recorded in
``results/autoresearch/data/sources_manifest.json``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.sft_probe import common, packing  # noqa: E402

RAW_DIR = common.OUT_DIR.parent / "autoresearch" / "raw"
DATA_DIR = common.OUT_DIR.parent / "autoresearch" / "data"

SPECS = {
    "bespoke": {
        "repo": "HuggingFaceH4/Bespoke-Stratos-17k",
        "revision": "384140b6bba79a3f50697536c3b4192b86ddce1d",
        "license": "Apache-2.0",
        "files": [
            "data/train-00000-of-00002.parquet",
            "data/train-00001-of-00002.parquet",
        ],
    },
    "oasst": {
        "repo": "OpenAssistant/oasst1",
        "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
        "license": "Apache-2.0",
        "files": ["data/train-00000-of-00001-b42a775f407cee45.parquet"],
    },
    "tulu": {
        "repo": "allenai/tulu-3-sft-mixture",
        "revision": "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be",
        "license": "ODC-BY 1.0 (mixture; per-source licenses apply)",
        "files": ["data/train-00000-of-00006.parquet"],
    },
    "openhermes": {
        "repo": "teknium/OpenHermes-2.5",
        "revision": "b82037821055c377bed0d495e72e46de3bc72e84",
        "license": "Apache-2.0",
        "files": ["openhermes2_5.json"],
        "stream": True,
    },
}

SCAN_LIMIT = 40_000
TARGET_TOKENS = 1_800_000
MAX_EXAMPLES = 30_000
VAL_PER_SOURCE = 250
MAX_LEN = 2048


def download_source(name: str) -> dict:
    from huggingface_hub import hf_hub_download

    spec = SPECS[name]
    local = {}
    for fname in spec["files"]:
        if spec.get("stream") and fname.endswith(".json"):
            local[fname] = None
            continue
        path = hf_hub_download(
            spec["repo"],
            fname,
            revision=spec["revision"],
            repo_type="dataset",
            local_dir=str(RAW_DIR),
        )
        local[fname] = {
            "path": str(path),
            "bytes": Path(path).stat().st_size,
            "sha256": common.sha256_file(Path(path)),
        }
    return local


def _first_turn(messages, role_key="role", content_key="content",
                user_values=("user", "human"),
                assistant_values=("assistant", "gpt")):
    instruction = None
    for message in messages:
        role = str(message.get(role_key, "")).lower()
        content = message.get(content_key)
        if not isinstance(content, str) or not content.strip():
            continue
        if role in user_values and instruction is None:
            instruction = content.strip()
            continue
        if instruction is not None and role in assistant_values:
            return instruction, content.strip()
    return None, None


def iter_bespoke():
    import pyarrow.parquet as pq

    for fname in SPECS["bespoke"]["files"]:
        path = RAW_DIR / fname
        table = pq.read_table(path, columns=["messages"])
        for row in table.column("messages").to_pylist():
            if not row:
                continue
            instruction, response = _first_turn(row)
            if instruction and response:
                yield instruction, response


def iter_tulu():
    import pyarrow.parquet as pq

    path = RAW_DIR / SPECS["tulu"]["files"][0]
    table = pq.read_table(path, columns=["messages"])
    for row in table.column("messages").to_pylist():
        if not row:
            continue
        instruction, response = _first_turn(row)
        if instruction and response:
            yield instruction, response


def iter_oasst():
    import pyarrow.parquet as pq

    path = RAW_DIR / SPECS["oasst"]["files"][0]
    table = pq.read_table(
        path,
        columns=["message_id", "parent_id", "text", "role", "lang",
                 "deleted", "rank"],
    )
    data = table.to_pydict()
    children: Dict[Optional[str], List[int]] = {}
    for index, parent in enumerate(data["parent_id"]):
        children.setdefault(parent, []).append(index)
    pairs = []
    for index, role in enumerate(data["role"]):
        if role != "prompter" or data["deleted"][index]:
            continue
        if data["lang"][index] != "en":
            continue
        candidates = [
            child for child in children.get(data["message_id"][index], [])
            if data["role"][child] == "assistant"
            and not data["deleted"][child]
            and data["lang"][child] == "en"
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda c: (data["rank"][c] or 10**6, c))
        prompt = (data["text"][index] or "").strip()
        response = (data["text"][best] or "").strip()
        if prompt and response:
            pairs.append((prompt, response))
    return pairs


def iter_openhermes(scan_limit: int = SCAN_LIMIT):
    from datasets import load_dataset

    spec = SPECS["openhermes"]
    dataset = load_dataset(
        spec["repo"], split="train", streaming=True,
        revision=spec["revision"],
    )
    for index, record in enumerate(dataset):
        if index >= scan_limit:
            break
        conversations = record.get("conversations") or []
        instruction, response = _first_turn(
            conversations, role_key="from", content_key="value",
            user_values=("human",), assistant_values=("gpt",),
        )
        if instruction and response:
            yield instruction, response


ITERATORS = {
    "bespoke": iter_bespoke,
    "oasst": iter_oasst,
    "tulu": iter_tulu,
    "openhermes": iter_openhermes,
}


def tokenize_pool(name, tokenizer, scan_limit=SCAN_LIMIT):
    """Scan source examples; stride-select train pool and a disjoint val set."""
    scanned = []
    iterator = ITERATORS[name]()
    for index, (instruction, response) in enumerate(iterator):
        if index >= scan_limit:
            break
        result = packing.tokenize_example(
            tokenizer, instruction, "", response, max_len=MAX_LEN
        )
        if result["skipped"]:
            continue
        scanned.append(
            {
                "instruction": instruction,
                "response": response,
                "prefix_ids": result["prefix_ids"],
                "response_ids": result["response_ids"],
                "truncated": result["truncated"],
            }
        )
    if not scanned:
        raise RuntimeError(f"no usable examples for source {name}")

    total_tokens = sum(len(ex["response_ids"]) for ex in scanned)
    mean_tokens = total_tokens / len(scanned)
    # reserve val examples from the scanned order first (stride spread)
    val_stride = max(1, len(scanned) // (VAL_PER_SOURCE * 4))
    val_indices = list(range(0, min(len(scanned), VAL_PER_SOURCE * val_stride),
                             val_stride))[:VAL_PER_SOURCE]
    val_set = set(val_indices)
    candidate_indices = [i for i in range(len(scanned)) if i not in val_set]

    # stride-sample the candidates to the target token budget
    budget_count = int(TARGET_TOKENS / max(mean_tokens, 1.0))
    budget_count = min(MAX_EXAMPLES, max(1, budget_count))
    stride = max(1, len(candidate_indices) // budget_count)
    selected = candidate_indices[::stride]
    if len(selected) > MAX_EXAMPLES:
        selected = selected[:MAX_EXAMPLES]

    train_records = []
    tokens = 0
    kept_indices = []
    for new_index, source_index in enumerate(selected):
        if tokens >= TARGET_TOKENS:
            break
        ex = scanned[source_index]
        train_records.append(
            {
                "i": new_index,
                "p": ex["prefix_ids"],
                "r": ex["response_ids"],
                "tr": bool(ex["truncated"]),
            }
        )
        tokens += len(ex["response_ids"])
        kept_indices.append(source_index)

    val_records = []
    for new_index, source_index in enumerate(val_indices):
        ex = scanned[source_index]
        val_records.append(
            {
                "i": new_index,
                "p": ex["prefix_ids"],
                "r": ex["response_ids"],
                "tr": bool(ex["truncated"]),
            }
        )

    return {
        "scan_examples": len(scanned),
        "scan_tokens": total_tokens,
        "train_examples": len(train_records),
        "train_target_tokens": tokens,
        "val_examples": len(val_records),
        "val_target_tokens": sum(len(r["r"]) for r in val_records),
        "val_source_indices": val_indices,
        "stride": stride,
        "_train": train_records,
        "_val": val_records,
    }


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def build_source(name: str, tokenizer) -> dict:
    spec = SPECS[name]
    local = download_source(name)
    result = tokenize_pool(name, tokenizer)
    train_records = result.pop("_train")
    val_records = result.pop("_val")
    write_jsonl(DATA_DIR / f"src_{name}_tokens.jsonl", train_records)
    write_jsonl(DATA_DIR / f"src_{name}_val.jsonl", val_records)
    manifest = {
        "dataset": spec["repo"],
        "revision": spec["revision"],
        "license": spec["license"],
        "files": local,
        "format_adapter": (
            "first user turn -> instruction, first assistant turn -> "
            "response; system turns dropped; single turn only"
        ),
        "scan_limit": SCAN_LIMIT,
        "target_token_cap": TARGET_TOKENS,
        "max_examples": MAX_EXAMPLES,
        **result,
        "train_tokens_path": str(DATA_DIR / f"src_{name}_tokens.jsonl"),
        "val_tokens_path": str(DATA_DIR / f"src_{name}_val.jsonl"),
    }
    print(json.dumps({k: manifest[k] for k in (
        "dataset", "revision", "scan_examples", "train_examples",
        "train_target_tokens", "val_examples")}))
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="bespoke,oasst,tulu,openhermes")
    parser.add_argument("--out", default=str(DATA_DIR / "sources_manifest.json"))
    args = parser.parse_args(argv)

    tokenizer = common.load_tokenizer()
    manifest = {"format": "arm_a_autoresearch_sources_v1",
                "created_at": common.iso_now(),
                "tokenizer_identity": common.TOKENIZER_IDENTITY,
                "tokenizer_sha256": common.TOKENIZER_SHA256,
                "sources": {}}
    for name in args.sources.split(","):
        name = name.strip()
        if not name:
            continue
        print(f"[sources] {name}: downloading")
        manifest["sources"][name] = build_source(name, tokenizer)
    common.save_json(Path(args.out), manifest)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
