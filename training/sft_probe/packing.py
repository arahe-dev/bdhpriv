"""Deterministic instruction packing for the Arm-A SFT calibration.

Format (no tokenizer special tokens):

    User: <instruction>\n
    Input: <input>\n            (only when input is non-empty)
    Assistant: <response>\n

Loss is applied only to response tokens plus the terminating newline, which is
the termination text in a corpus/tokenizer contract that has no EOS token
(vocab 8192, single special token ``<unk>`` id 0).

Packing rules:
  * complete examples per row; an example never spans two rows;
  * every example is its own attention segment (``start`` = its first token
    index in the row), so no cross-example information leakage exists;
  * padding positions are masked out of attention and loss;
  * maximum sequence length 2048; response tail truncation is deterministic
    and recorded (full prompt is always preserved; over-long prompts are
    skipped instead of truncated).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

PAD_ID = 0
PROMPT_TEMPLATE = "User: {instruction}\nAssistant: "
PROMPT_TEMPLATE_INPUT = "User: {instruction}\nInput: {input}\nAssistant: "
TERMINATION = "\n"


@dataclass
class PackedRow:
    x: np.ndarray
    y: np.ndarray
    pos: np.ndarray
    segpos: np.ndarray
    start: np.ndarray
    input_valid: np.ndarray
    valid: np.ndarray
    example_indices: List[int]
    target_tokens: int
    sequence_tokens: int

    def as_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "pos": self.pos,
            "segpos": self.segpos,
            "start": self.start,
            "input_valid": self.input_valid,
            "valid": self.valid,
            "example_indices": list(self.example_indices),
            "target_tokens": int(self.target_tokens),
            "sequence_tokens": int(self.sequence_tokens),
        }


def format_example(instruction: str, input_text: str, response: str):
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    response = response or ""
    if input_text:
        prefix = PROMPT_TEMPLATE_INPUT.format(
            instruction=instruction, input=input_text
        )
    else:
        prefix = PROMPT_TEMPLATE.format(instruction=instruction)
    return prefix, response + TERMINATION


def tokenize_example(tokenizer, instruction, input_text, response, max_len=2048):
    prefix, block = format_example(instruction, input_text, response)
    prefix_ids = tokenizer.encode(prefix).ids
    block_ids = tokenizer.encode(block).ids
    truncated = False
    if len(prefix_ids) >= max_len:
        return {
            "prefix_ids": None,
            "response_ids": None,
            "truncated": False,
            "skipped": "prompt_too_long",
        }
    room = max_len - len(prefix_ids)
    if len(block_ids) > room:
        block_ids = block_ids[:room]
        truncated = True
    if not block_ids:
        block_ids = tokenizer.encode(TERMINATION).ids[:1] or [PAD_ID]
        truncated = True
    return {
        "prefix_ids": prefix_ids,
        "response_ids": block_ids,
        "truncated": truncated,
        "skipped": None,
    }


def pack_examples(
    examples: Sequence[dict],
    T: int = 2048,
    max_rows: Optional[int] = None,
) -> List[PackedRow]:
    """Greedily pack tokenized examples into 2048-token rows.

    ``examples`` items need ``index``, ``prefix_ids`` and ``response_ids``.
    """
    rows: List[PackedRow] = []
    x = np.full(T, PAD_ID, dtype=np.uint16)
    y = np.zeros(T, dtype=np.uint16)
    pos = np.zeros(T, dtype=np.int32)
    segpos = np.zeros(T, dtype=np.int32)
    start = np.zeros(T, dtype=np.int32)
    input_valid = np.zeros(T, dtype=np.bool_)
    valid = np.zeros(T, dtype=np.bool_)
    example_indices: List[int] = []
    target_tokens = 0
    fill = 0

    def flush(row_index: int):
        nonlocal x, y, pos, segpos, start, input_valid, valid
        nonlocal example_indices, target_tokens, fill
        if fill == 0:
            return
        start[fill:] = np.int32(T + row_index + 1)
        rows.append(
            PackedRow(
                x=x,
                y=y,
                pos=pos,
                segpos=segpos,
                start=start,
                input_valid=input_valid,
                valid=valid,
                example_indices=example_indices,
                target_tokens=target_tokens,
                sequence_tokens=fill,
            )
        )
        x = np.full(T, PAD_ID, dtype=np.uint16)
        y = np.zeros(T, dtype=np.uint16)
        pos = np.zeros(T, dtype=np.int32)
        segpos = np.zeros(T, dtype=np.int32)
        start = np.zeros(T, dtype=np.int32)
        input_valid = np.zeros(T, dtype=np.bool_)
        valid = np.zeros(T, dtype=np.bool_)
        example_indices = []
        target_tokens = 0
        fill = 0

    for ex in examples:
        prefix_ids = ex["prefix_ids"]
        response_ids = ex["response_ids"]
        total = len(prefix_ids) + len(response_ids)
        if total > T:
            raise ValueError("example longer than T; truncate before packing")
        if fill + total > T:
            flush(len(rows))
        p0 = fill
        ids = list(prefix_ids) + list(response_ids)
        end = p0 + total
        x[p0:end] = np.asarray(ids, dtype=np.uint16)
        pos[p0:end] = np.arange(total, dtype=np.int32)
        segpos[p0:end] = np.arange(total, dtype=np.int32)
        start[p0:end] = np.int32(p0)
        input_valid[p0:end] = True
        if total > 1:
            y[p0:end - 1] = x[p0 + 1:end]
            loss_lo = p0 + len(prefix_ids) - 1
            if loss_lo < p0:
                loss_lo = p0
            valid[loss_lo:end - 1] = True
        example_indices.append(int(ex["index"]))
        target_tokens += len(response_ids)
        fill = end
        if max_rows is not None and len(rows) >= max_rows:
            break
    flush(len(rows))
    if max_rows is not None:
        return rows[:max_rows]
    return rows


def rows_to_cpu_batch(rows: Sequence[PackedRow]) -> Dict[str, torch.Tensor]:
    def pin(array):
        return torch.from_numpy(array).pin_memory()

    return {
        "x": pin(np.stack([r.x for r in rows])),
        "y": pin(np.stack([r.y for r in rows])),
        "pos": pin(np.stack([r.pos for r in rows]).astype(np.int32)),
        "segpos": pin(np.stack([r.segpos for r in rows]).astype(np.int32)),
        "start": pin(np.stack([r.start for r in rows]).astype(np.int32)),
        "input_valid": pin(np.stack([r.input_valid for r in rows])),
        "valid": pin(np.stack([r.valid for r in rows])),
    }


_CAUSAL_CACHE: Dict[int, torch.Tensor] = {}


def load_tokens_jsonl(path) -> Dict[int, dict]:
    import json

    examples: Dict[int, dict] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            examples[int(record["i"])] = record
    return examples


def load_row_plan_jsonl(path) -> List[dict]:
    import json

    plan = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            plan.append(json.loads(line))
    return plan


def build_rows_from_plan(
    plan: Sequence[dict],
    examples_by_index: Dict[int, dict],
    T: int = 2048,
    start_row: int = 0,
    max_rows: Optional[int] = None,
) -> List[PackedRow]:
    """Rebuild packed rows from a frozen row plan (example ids per row)."""
    rows: List[PackedRow] = []
    count = 0
    for entry in plan[start_row:]:
        if max_rows is not None and count >= max_rows:
            break
        examples = [examples_by_index[i] for i in entry["examples"]]
        built = pack_examples(examples, T=T)
        if len(built) != 1:
            raise ValueError(
                "row plan entry did not pack into exactly one row: "
                f"{entry['row']}"
            )
        rows.append(built[0])
        count += 1
    return rows


def causal_full(device, t: int) -> torch.Tensor:
    key = (str(device), int(t))
    mask = _CAUSAL_CACHE.get(key)
    if mask is None or mask.device != torch.device(device):
        mask = torch.ones((t, t), dtype=torch.bool, device=device).tril(
            diagonal=-1
        )
        _CAUSAL_CACHE[key] = mask
    return mask


def cpu_batch_to_device(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    out = {}
    out["x"] = batch["x"].to(device, dtype=torch.long, non_blocking=True)
    out["y"] = batch["y"].to(device, dtype=torch.long, non_blocking=True)
    out["pos"] = batch["pos"].to(device, dtype=torch.int32, non_blocking=True)
    out["segpos"] = batch["segpos"].to(
        device, dtype=torch.int32, non_blocking=True
    )
    out["start"] = batch["start"].to(device, dtype=torch.int32, non_blocking=True)
    out["input_valid"] = batch["input_valid"].to(device, non_blocking=True)
    out["valid"] = batch["valid"].to(device, non_blocking=True)
    start = out["start"]
    iv = out["input_valid"]
    out["full_mask"] = (
        (start[:, :, None] == start[:, None, :])
        & iv[:, :, None]
        & iv[:, None, :]
        & causal_full(device, batch["x"].shape[1]).unsqueeze(0)
    )
    return out
