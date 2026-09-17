"""Prepare the frozen SFT calibration data (Alpaca-GPT4) and BASE_TEXT_PROXY.

Deterministic, seed 1337. Produces:
  results/sft_probe/data/raw/...            downloaded dataset (hash-checked)
  results/sft_probe/data/train_tokens.jsonl tokenized train split
  results/sft_probe/data/val_tokens.jsonl   tokenized held-out split
  results/sft_probe/data/train_rows.jsonl   frozen packed-row plan (train)
  results/sft_probe/data/val_rows.jsonl     frozen packed-row plan (val)
  results/sft_probe/data/proxy_rows.jsonl   BASE_TEXT_PROXY windows
  results/sft_probe/data/proxy_tokens.jsonl
  results/sft_probe/data_manifest.json      full manifest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from training.sft_probe import common, packing
else:
    from . import common, packing

DATASET_REPO = "vicgalle/alpaca-gpt4"
DATASET_REVISION = "f7e3ded725cb81e8e564e32feb12860f376f2b51"
DATASET_FILE = "data/train-00000-of-00001-6ef3991c06080e14.parquet"
DATASET_SHA256 = (
    "bdd9b3f1aa3688ee2015550974c3a14b27fec20e4cfb459fd6800dc14030b9e6"
)
RAW_DIR = common.DATA_DIR / "raw"
VALIDATION_EXAMPLES = 1000
MAX_LEN = 2048
PROXY_EXCLUDE_PARTS = (
    ".git", ".pytest_cache", "workspace_template", "raw", "sft_probe",
)
PROXY_EXCLUDE_FILES = ("campaigns/ARM_A_SFT_CALIBRATION_REPORT.md",)
REPLAY_EXCLUDE_PARTS = (
    ".git", ".pytest_cache", "workspace_template", "results", "runs",
    "data", "sft_probe", "__pycache__",
)


def _dataset_path() -> Path:
    candidate = RAW_DIR / DATASET_FILE
    if not candidate.is_file():
        candidate = RAW_DIR / Path(DATASET_FILE).name
    if not candidate.is_file():
        raise FileNotFoundError(
            f"dataset parquet not found under {RAW_DIR}; run with "
            "--download first"
        )
    return candidate


def download_dataset() -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        DATASET_REPO,
        DATASET_FILE,
        revision=DATASET_REVISION,
        repo_type="dataset",
        local_dir=str(RAW_DIR),
    )
    return Path(path)


def build_splits(rows: int) -> dict:
    rng = np.random.default_rng(common.SPLIT_SEED)
    perm = rng.permutation(rows)
    val = perm[:VALIDATION_EXAMPLES]
    train = perm[VALIDATION_EXAMPLES:]
    return {
        "permutation_seed": common.SPLIT_SEED,
        "raw_example_count": int(rows),
        "validation_example_indices": [int(i) for i in val],
        "train_example_count": int(train.size),
        "validation_example_count": int(val.size),
        "train_indices": [int(i) for i in train],
    }


def tokenize_split(tokenizer, table, indices) -> list:
    instructions = table["instruction"].to_pylist()
    inputs = table["input"].to_pylist()
    outputs = table["output"].to_pylist()
    records = []
    for index in indices:
        result = packing.tokenize_example(
            tokenizer,
            instructions[index],
            inputs[index],
            outputs[index],
            max_len=MAX_LEN,
        )
        records.append(
            {
                "i": int(index),
                "p": result["prefix_ids"],
                "r": result["response_ids"],
                "tr": bool(result["truncated"]),
                "skip": result["skipped"],
            }
        )
    return records


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def summarize_tokens(records, label: str) -> dict:
    kept = [r for r in records if r["skip"] is None]
    response_lengths = [len(r["r"]) for r in kept]
    total = [len(r["p"]) + len(r["r"]) for r in kept]
    return {
        f"{label}_examples": len(records),
        f"{label}_kept": len(kept),
        f"{label}_skipped_prompt_too_long": len(records) - len(kept),
        f"{label}_truncated_responses": sum(1 for r in kept if r["tr"]),
        f"{label}_target_tokens": int(sum(response_lengths)),
        f"{label}_total_sequence_tokens": int(sum(total)),
        f"{label}_response_len_mean": float(np.mean(response_lengths)),
        f"{label}_response_len_p50": float(np.percentile(response_lengths, 50)),
        f"{label}_response_len_p95": float(np.percentile(response_lengths, 95)),
        f"{label}_response_len_max": int(np.max(response_lengths)),
        f"{label}_total_sequence_len_max": int(np.max(total)),
    }


def proxy_files() -> list:
    root = common.ROOT
    files = []
    for path in sorted(root.glob("**/*.md")):
        rel = path.relative_to(root)
        parts = set(rel.parts)
        if any(part in parts for part in PROXY_EXCLUDE_PARTS):
            continue
        if rel.as_posix() in PROXY_EXCLUDE_FILES:
            continue
        files.append(path)
    return files


def replay_files() -> list:
    root = common.ROOT
    files = []
    for path in sorted(root.glob("**/*.py")):
        parts = set(path.relative_to(root).parts)
        if any(part in parts for part in REPLAY_EXCLUDE_PARTS):
            continue
        files.append(path)
    return files


def build_text_rows(tokenizer, files) -> tuple:
    """Tokenize text files and split each into independent 2048-token rows."""
    tokens_records = []
    rows = []
    row_index = 0
    token_offset = 0
    for path in files:
        rel = path.relative_to(common.ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        ids = tokenizer.encode(text).ids
        tokens_records.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "tokens": len(ids),
                "token_offset": token_offset,
            }
        )
        for start in range(0, len(ids), MAX_LEN):
            chunk = ids[start:start + MAX_LEN]
            row = _proxy_row(chunk, row_index)
            rows.append(row)
            row_index += 1
        token_offset += len(ids)
    return tokens_records, rows


def build_proxy(tokenizer) -> tuple:
    return build_text_rows(tokenizer, proxy_files())


def content_sha256(files) -> str:
    import hashlib

    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(common.ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _proxy_row(chunk, row_index: int) -> dict:
    T = MAX_LEN
    x = np.full(T, packing.PAD_ID, dtype=np.uint16)
    y = np.zeros(T, dtype=np.uint16)
    pos = np.zeros(T, dtype=np.int32)
    segpos = np.zeros(T, dtype=np.int32)
    start = np.zeros(T, dtype=np.int32)
    input_valid = np.zeros(T, dtype=np.bool_)
    valid = np.zeros(T, dtype=np.bool_)
    n = len(chunk)
    x[:n] = np.asarray(chunk, dtype=np.uint16)
    pos[:n] = np.arange(n, dtype=np.int32)
    segpos[:n] = np.arange(n, dtype=np.int32)
    input_valid[:n] = True
    if n > 1:
        y[:n - 1] = x[1:n]
        valid[:n - 1] = True
    start[n:] = np.int32(T + row_index + 1)
    return {
        "row": int(row_index),
        "x": x.tolist(),
        "y": y.tolist(),
        "pos": pos.tolist(),
        "segpos": segpos.tolist(),
        "start": start.tolist(),
        "input_valid": input_valid.tolist(),
        "valid": valid.tolist(),
        "target_tokens": int(max(0, n - 1)),
        "sequence_tokens": int(n),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--proxy-only", action="store_true")
    args = parser.parse_args(argv)

    common.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = common.load_tokenizer()
    proxy_tokens, proxy_rows = build_proxy(tokenizer)
    write_jsonl(common.DATA_DIR / "proxy_rows.jsonl", proxy_rows)
    replay_tokens, replay_rows = build_text_rows(tokenizer, replay_files())
    write_jsonl(common.DATA_DIR / "replay_rows.jsonl", replay_rows)
    replay_meta = {
        "replay_label": "BASE_TEXT_PROXY_REPLAY",
        "replay_definition": (
            "All repository .py source files (excluding results/, runs/, "
            "data/, workspace_template/, __pycache__ and the sft_probe "
            "package itself), sorted by relative path, tokenized with the "
            "exact Arm-A tokenizer; each 2048-token window is an "
            "independent segment with ordinary next-token loss. Disjoint "
            "from the retention/probe proxy (markdown files)."
        ),
        "replay_files": replay_tokens,
        "replay_windows": len(replay_rows),
        "replay_target_tokens": int(
            sum(r["target_tokens"] for r in replay_rows)
        ),
        "replay_sequence_tokens": int(
            sum(r["sequence_tokens"] for r in replay_rows)
        ),
        "replay_content_sha256": content_sha256(replay_files()),
        "file_count": len(replay_tokens),
    }
    print(json.dumps({
        "replay_windows": replay_meta["replay_windows"],
        "replay_target_tokens": replay_meta["replay_target_tokens"],
    }))
    proxy_meta = {
        "base_corpus_retention_status": "BLOCKED_ARTIFACT_NOT_LOCAL",
        "proxy_label": "BASE_TEXT_PROXY",
        "proxy_definition": (
            "All repository .md files (tracked docs), sorted by relative "
            "path, tokenized with the exact Arm-A tokenizer; each 2048-token "
            "window is an independent segment. NOT the frozen corpus."
        ),
        "proxy_files": proxy_tokens,
        "proxy_target_tokens": int(
            sum(r["target_tokens"] for r in proxy_rows)
        ),
        "proxy_sequence_tokens": int(
            sum(r["sequence_tokens"] for r in proxy_rows)
        ),
        "proxy_windows": len(proxy_rows),
        "proxy_content_sha256": content_sha256(proxy_files()),
        "proxy_excluded_files": list(PROXY_EXCLUDE_FILES),
    }
    print(json.dumps({k: proxy_meta[k] for k in (
        "proxy_windows", "proxy_target_tokens", "proxy_sequence_tokens")}))

    if args.proxy_only:
        common.save_json(common.DATA_DIR / "proxy_manifest.json", {
            "proxy": proxy_meta, "replay": replay_meta,
        })
        return 0

    path = _dataset_path()
    sha = common.sha256_file(path)
    if sha != DATASET_SHA256:
        raise RuntimeError(
            f"dataset SHA-256 mismatch: {sha} != {DATASET_SHA256}"
        )
    table = pq.read_table(path)
    splits = build_splits(table.num_rows)

    train_records = tokenize_split(
        tokenizer, table, splits["train_indices"]
    )
    val_records = tokenize_split(
        tokenizer, table, splits["validation_example_indices"]
    )
    write_jsonl(common.DATA_DIR / "train_tokens.jsonl", train_records)
    write_jsonl(common.DATA_DIR / "val_tokens.jsonl", val_records)

    train_kept = [r for r in train_records if r["skip"] is None]
    val_kept = [r for r in val_records if r["skip"] is None]

    train_rows = packing.pack_examples(
        [
            {
                "index": r["i"],
                "prefix_ids": r["p"],
                "response_ids": r["r"],
            }
            for r in train_kept
        ],
        T=MAX_LEN,
    )
    val_rows = packing.pack_examples(
        [
            {
                "index": r["i"],
                "prefix_ids": r["p"],
                "response_ids": r["r"],
            }
            for r in val_kept
        ],
        T=MAX_LEN,
    )
    write_jsonl(
        common.DATA_DIR / "train_rows.jsonl",
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
        common.DATA_DIR / "val_rows.jsonl",
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
        "format": "arm_a_sft_probe_data_manifest_v1",
        "created_at": common.iso_now(),
        "dataset_name": "Alpaca-GPT4 (vicgalle/alpaca-gpt4)",
        "dataset_source": f"https://huggingface.co/datasets/{DATASET_REPO}",
        "dataset_revision": DATASET_REVISION,
        "dataset_file": DATASET_FILE,
        "dataset_file_sha256": sha,
        "dataset_file_bytes": path.stat().st_size,
        "tokenizer_identity": common.TOKENIZER_IDENTITY,
        "tokenizer_sha256": common.TOKENIZER_SHA256,
        "format_template": {
            "no_input": "User: {instruction}\\nAssistant: {response}\\n",
            "with_input": (
                "User: {instruction}\\nInput: {input}\\n"
                "Assistant: {response}\\n"
            ),
            "loss_applies_to": (
                "assistant response tokens + terminating newline (target "
                "positions only)"
            ),
            "special_tokens_added": False,
            "max_seq_len": MAX_LEN,
            "truncation": (
                "response tail only, deterministic; full prompt preserved; "
                "over-long prompts skipped"
            ),
        },
        "split": {
            "seed": common.SPLIT_SEED,
            "method": "numpy default_rng(1337).permutation once",
            "raw_example_count": int(table.num_rows),
            "validation_example_count": VALIDATION_EXAMPLES,
            "train_example_count": splits["train_example_count"],
            **summarize_tokens(train_records, "train"),
            **summarize_tokens(val_records, "validation"),
        },
        "packing": {
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "train_packed_target_tokens": int(
                sum(r.target_tokens for r in train_rows)
            ),
            "train_packed_sequence_tokens": int(
                sum(r.sequence_tokens for r in train_rows)
            ),
            "train_fill_ratio": float(
                sum(r.sequence_tokens for r in train_rows)
                / (len(train_rows) * MAX_LEN)
            ),
            "val_packed_target_tokens": int(
                sum(r.target_tokens for r in val_rows)
            ),
            "val_packed_sequence_tokens": int(
                sum(r.sequence_tokens for r in val_rows)
            ),
            "examples_per_row_mean": float(
                np.mean([len(r.example_indices) for r in train_rows])
            ),
        },
        "proxy": proxy_meta,
        "replay": replay_meta,
        "dose_landmarks": {
            key: int(round(value))
            for key, value in common.TARGET_TPP.items()
        },
        "parameter_count": common.N_PARAM,
    }
    common.save_json(common.DATA_DIR / "data_manifest.json", manifest)
    print(json.dumps(
        {
            "train_target_tokens": manifest["split"]["train_target_tokens"],
            "train_total_sequence_tokens": manifest["split"][
                "train_total_sequence_tokens"
            ],
            "val_target_tokens": manifest["split"][
                "validation_target_tokens"
            ],
            "train_rows": manifest["packing"]["train_rows"],
            "train_fill_ratio": manifest["packing"]["train_fill_ratio"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
