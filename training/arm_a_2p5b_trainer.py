#!/usr/bin/env python
"""Arm-A 2.5B production trainer -- opt3c_all_b1024, frozen semantics.

Freeze (do not change):
  T=2048, N=16384, D=256, H=4, L=8, K=4096, V=8192, writer_hidden=1040,
  B16x4 (global B64), BF16 autocast + FP32 master params, fused AdamW
  (lr 1e-3, betas 0.9/0.95, eps 1e-8, wd 0.1, clip 1.0, linear warmup
  10M tokens then constant), torch.compile(mode="default"),
  branch-free packed state update, zero-carry skip for the t0==0 chunk,
  direct paper_y layout, cached RoPE phase, dense canonical coordinator,
  no activation checkpointing, frozen corpus semantics (packed documents,
  strict-past same-document attention, exact cross-sequence targets,
  no replay, padding positions masked).

Modes:
  smoke  Production smoke gate: real packed updates, atomic checkpoint,
         fresh-model reload, one identical update on both paths, then
         loss/parameter/corpus-cursor equivalence. Prints the final
         machine-readable line ARM_A_2P5B_TRAINER_READY=true|false.
  train  Full training: streams the frozen corpus in deterministic order,
         stops at --target-tokens (default 2.5e9; use "full" to stream every
         batch of the whole corpus, including the final 63-row partial batch
         that closes out all 2,441,407 sequences -- no replay, no padding).
         Auto-resumes from the newest valid checkpoint.
         The checkpoint carries optimizer/LR/token schedule state, so a
         later 5B continuation restarts nothing.

Fail closed on: corpus-contract mismatch, non-finite loss/gradients,
graph breaks, OOM, invalid resume state, token-accounting errors.

Single file; no repo imports. Run with:
  python arm_a_2p5b_trainer.py --mode smoke
  python arm_a_2p5b_trainer.py --mode train
or in Colab:
  %run arm_a_2p5b_trainer.py --mode smoke
"""

from __future__ import annotations

import argparse
import bisect
import gc
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterator, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pyarrow is required only for the corpus reader
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None

# ---------------------------------------------------------------------------
# frozen constants
# ---------------------------------------------------------------------------

IMPLEMENTATION_VERSION = "arm_a_2p5b_trainer_v1_opt3c_all_b1024"
CKPT_FORMAT = "arm_a_2p5b_ckpt_v1"

TARGET_2P5B = 2_500_000_000
TOKENS_PER_UPDATE = 64 * 2048

PROD_CORPUS_ROOT = Path(
    "/content/drive/Shareddrives/ICLR PHASE BDH/"
    "phase_bdh/corpus/stage2/frozen_5b_v1"
)
PROD_RUN_DIR = Path(
    "/content/drive/Shareddrives/ICLR PHASE BDH/"
    "phase_bdh/runs/arm_a_2p5b_opt3c_all"
)

FROZEN_FLAGS = {
    "scan": "chunkwise",
    "scan_block": 1024,
    "packed_update": "branchfree",
    "zero_carry": True,
    "paper_layout": "direct",
    "cache_rope": True,
    "coordinator": "dense",
    "checkpointing": "none",
    "precision": "bf16_autocast_fp32_master",
    "optimizer": "adamw_fused",
}


@dataclass(frozen=True)
class ArmAConfig:
    T: int = 2048
    V: int = 8192
    D: int = 256
    N: int = 16384
    H: int = 4
    L: int = 8
    HIDDEN: int = 1040
    SEED: int = 1337
    INIT_STD: float = 0.02
    THETA: float = 2**16
    READ_BLOCK: int = 256
    PEAK_LR: float = 1e-3
    BETAS: tuple = (0.9, 0.95)
    EPS: float = 1e-8
    WEIGHT_DECAY: float = 0.1
    CLIP_NORM: float = 1.0
    GLOBAL_BATCH: int = 64
    MICROBATCH: int = 16
    SCAN_BLOCK: int = 1024
    WARMUP_TOKENS: int = 10_000_000

    @property
    def K(self):
        return self.N // self.H


PROD_CFG = ArmAConfig()


def frozen_config_dict(cfg: ArmAConfig) -> dict:
    return {
        "T": int(cfg.T),
        "V": int(cfg.V),
        "D": int(cfg.D),
        "N": int(cfg.N),
        "H": int(cfg.H),
        "K": int(cfg.K),
        "L": int(cfg.L),
        "HIDDEN": int(cfg.HIDDEN),
        "SEED": int(cfg.SEED),
        "INIT_STD": float(cfg.INIT_STD),
        "THETA": float(cfg.THETA),
        "READ_BLOCK": int(cfg.READ_BLOCK),
        "PEAK_LR": float(cfg.PEAK_LR),
        "BETAS": [float(b) for b in cfg.BETAS],
        "EPS": float(cfg.EPS),
        "WEIGHT_DECAY": float(cfg.WEIGHT_DECAY),
        "CLIP_NORM": float(cfg.CLIP_NORM),
        "GLOBAL_BATCH": int(cfg.GLOBAL_BATCH),
        "MICROBATCH": int(cfg.MICROBATCH),
        "SCAN_BLOCK": int(cfg.SCAN_BLOCK),
        "WARMUP_TOKENS": int(cfg.WARMUP_TOKENS),
        "flags": dict(FROZEN_FLAGS),
    }


# ---------------------------------------------------------------------------
# errors / small helpers
# ---------------------------------------------------------------------------

class CorpusContractError(RuntimeError):
    pass


class TrainerError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise CorpusContractError(message)


def _pin(t: torch.Tensor) -> torch.Tensor:
    return t.pin_memory() if torch.cuda.is_available() else t


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_file(path: Path, block: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _safe_corpus_path(root: Path, relative) -> Path:
    rel = PurePosixPath(str(relative))
    require(
        not rel.is_absolute() and ".." not in rel.parts and "\\" not in str(relative),
        f"unsafe/noncanonical corpus path: {relative!r}",
    )
    return root.joinpath(*rel.parts)


def code_fingerprint() -> str:
    try:
        src = Path(__file__).resolve().read_bytes()
    except Exception:
        src = IMPLEMENTATION_VERSION.encode()
    return hashlib.sha256(src.replace(b"\r\n", b"\n")).hexdigest()


class RunLogger:
    def __init__(self, log_path: Optional[Path]):
        self.log_path = Path(log_path) if log_path else None
        self._f = None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._f = open(self.log_path, "a", encoding="utf-8")

    def log(self, event: str, **fields):
        rec = {"event": event, "ts": _iso_now(), **fields}
        line = json.dumps(rec, sort_keys=True, default=str)
        print(line, flush=True)
        if self._f is not None:
            self._f.write(line + "\n")
            self._f.flush()

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None


# ---------------------------------------------------------------------------
# frozen corpus contract + deterministic packed stream
# ---------------------------------------------------------------------------

PROVENANCE_COLUMNS = (
    "sequence_index",
    "sequence_token_start",
    "sequence_token_end",
    "selected_document_index",
    "document_token_start",
    "document_token_end",
)


@dataclass(frozen=True)
class CorpusContract:
    corpus_id: str
    context_length: int
    tokenizer_sha256: str
    logical_replay_sha256: str
    artifact_hashes_sha256: str
    real_training_tokens: int
    sequences: int
    padding_tokens_masked: int
    selected_documents: int
    physical_tokens: int
    shard_count: int


PROD_CONTRACT = CorpusContract(
    corpus_id="phase_bdh_stage2_5b_v1",
    context_length=2048,
    tokenizer_sha256=(
        "9a05bca4c065d9952a01ee1e3f4b6e23e39200e1f33da65e5dad1819419f71c3"
    ),
    logical_replay_sha256=(
        "c60bd7df4a8ef7329d20ec5b0f022f4579fca20c95657a90f79a2e9e67edb043"
    ),
    artifact_hashes_sha256=(
        "2fe45d14ff8ae36ded7f1b57af9e7e200233fcd3726354e1a6186af7ee0ea6a3"
    ),
    real_training_tokens=5_000_000_000,
    sequences=2_441_407,
    padding_tokens_masked=1_536,
    selected_documents=3_498_441,
    physical_tokens=5_000_001_536,
    shard_count=25,
)


class _ShardData:
    __slots__ = ("index", "nseq", "seq_base", "lengths", "tokens", "rows",
                 "first_row", "first_token", "token_path", "length_path", "prov_path")

    def __init__(self, index, nseq, seq_base, lengths, tokens, rows, first_row,
                 first_token, token_path, length_path, prov_path):
        self.index = index
        self.nseq = nseq
        self.seq_base = seq_base
        self.lengths = lengths
        self.tokens = tokens
        self.rows = rows
        self.first_row = first_row
        self.first_token = first_token
        self.token_path = token_path
        self.length_path = length_path
        self.prov_path = prov_path

    def close(self):
        self.lengths = None
        self.tokens = None
        self.rows = None


class FrozenPackedCorpus:
    """Streams the frozen packed corpus in deterministic order.

    Each yielded batch is 64 fully-reconstructed packed rows (x, y, pos,
    segpos, start, input_valid, valid) exactly as the certified benchmark
    reconstructs them, including cross-sequence lookahead targets and
    masked padding.
    """

    def __init__(self, root: Path, cfg: ArmAConfig,
                 contract: Optional[CorpusContract] = None,
                 verify_files: bool = True, fast_verify: bool = False,
                 verify_workers: int = 4):
        require(pq is not None, "pyarrow is required for the corpus reader")
        self.root = Path(root)
        self.cfg = cfg
        self.contract = contract if contract is not None else PROD_CONTRACT
        self._cache: Dict[int, _ShardData] = {}
        self.frozen, self.manifest, self.shards = self._read_contract()
        self.total_sequences = self.contract.sequences
        self.bases: List[int] = []
        base = 0
        for shard in self.shards:
            self.bases.append(base)
            base += int(shard["sequences"])
        require(base == self.total_sequences,
                "shard sequence counts do not sum to the frozen total")
        self.artifact_hashes = self._read_json(self.root / "artifact_hashes.json")
        if verify_files:
            self._verify_all_files(fast=fast_verify, workers=verify_workers)

    # -- contract ----------------------------------------------------------

    @staticmethod
    def _read_json(path: Path) -> dict:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def _read_contract(self):
        require(self.root.is_dir(), f"frozen corpus directory missing: {self.root}")
        frozen = self._read_json(self.root / "FROZEN.json")
        manifest = self._read_json(self.root / "corpus_manifest.json")
        artifact = self._read_json(self.root / "artifact_hashes.json")
        c = self.contract

        require(frozen.get("status") == "FROZEN", "FROZEN.json is not FROZEN")
        require(frozen.get("corpus_id") == c.corpus_id, "wrong frozen corpus id")
        require(int(frozen.get("context_length")) == c.context_length,
                "frozen context length differs")
        require(frozen.get("tokenizer_sha256") == c.tokenizer_sha256,
                "tokenizer hash mismatch")
        require(frozen.get("logical_replay_sha256") == c.logical_replay_sha256,
                "logical replay hash mismatch")
        require(frozen.get("artifact_hashes_sha256") == c.artifact_hashes_sha256,
                "artifact set hash mismatch")
        require(int(frozen.get("real_training_tokens")) == c.real_training_tokens,
                "real training token count mismatch")
        require(int(frozen.get("sequences")) == c.sequences,
                "sequence count mismatch")
        require(int(frozen.get("padding_tokens_masked")) == c.padding_tokens_masked,
                "padding token count mismatch")

        require(manifest.get("status") == "FROZEN", "manifest is not FROZEN")
        require(int(manifest.get("real_tokens")) == c.real_training_tokens,
                "manifest real-token count mismatch")
        require(int(manifest.get("physical_tokens")) == c.physical_tokens,
                "manifest physical-token count mismatch")
        require(int(manifest.get("padding_tokens")) == c.padding_tokens_masked,
                "manifest padding-token count mismatch")
        require(int(manifest.get("selected_documents")) == c.selected_documents,
                "manifest selected-document count mismatch")
        require(int(manifest.get("sequences")) == c.sequences,
                "manifest sequence count mismatch")
        require(manifest.get("logical_replay_sha256") == c.logical_replay_sha256,
                "manifest logical replay hash differs")

        require(artifact.get("algorithm") == "sha256",
                "artifact index does not declare sha256")
        require(sha256_bytes(Path(self.root / "artifact_hashes.json").read_bytes())
                == c.artifact_hashes_sha256,
                "artifact_hashes.json digest differs from the frozen contract")

        records = artifact.get("files", [])
        by_path = {}
        for record in records:
            rel = record.get("path")
            require(isinstance(rel, str) and rel not in by_path,
                    f"missing or duplicate artifact-index path: {rel!r}")
            by_path[rel] = record

        manifest_record = by_path.get("corpus_manifest.json")
        require(manifest_record is not None,
                "corpus_manifest.json absent from the artifact index")
        manifest_bytes = (self.root / "corpus_manifest.json").read_bytes()
        require(len(manifest_bytes) == int(manifest_record["bytes"]),
                "corpus_manifest.json byte count mismatch")
        require(sha256_bytes(manifest_bytes) == manifest_record["sha256"],
                "corpus_manifest.json SHA-256 mismatch")

        shards = manifest.get("shards", [])
        require(len(shards) == c.shard_count,
                f"expected {c.shard_count} shards, found {len(shards)}")
        require(sum(int(s["sequences"]) for s in shards) == c.sequences,
                "shard sequence counts do not sum to the frozen total")
        require(sum(int(v) for v in manifest.get("stream_tokens", {}).values())
                == c.real_training_tokens,
                "manifest stream token counts do not sum to the frozen total")

        declared_tokens, declared_lengths, declared_prov = set(), set(), set()
        for shard_number, shard in enumerate(shards):
            stem = f"shard_{shard_number:06d}"
            require(shard.get("stem") == stem,
                    f"unexpected shard order/stem at index {shard_number}")
            require(shard.get("tokens_file") == f"train/{stem}.tokens.bin",
                    f"unexpected token path for {stem}")
            require(shard.get("valid_lengths_file") == f"train/{stem}.valid_lengths.bin",
                    f"unexpected valid-length path for {stem}")
            require(shard.get("provenance_file") == f"train/{stem}.provenance.parquet",
                    f"unexpected provenance path for {stem}")
            require(int(shard["sequences"]) > 0 and int(shard["provenance_rows"]) > 0,
                    f"empty shard declaration for {stem}")
            declared_tokens.add(shard["tokens_file"])
            declared_lengths.add(shard["valid_lengths_file"])
            declared_prov.add(shard["provenance_file"])
        require({p for p in by_path if p.endswith(".tokens.bin")} == declared_tokens,
                "artifact index token shards differ from the manifest")
        require({p for p in by_path if p.endswith(".valid_lengths.bin")} == declared_lengths,
                "artifact index valid-length shards differ from the manifest")
        require({p for p in by_path if p.endswith(".provenance.parquet")} == declared_prov,
                "artifact index provenance shards differ from the manifest")
        self._artifact_index = by_path
        return frozen, manifest, shards

    def _verify_all_files(self, fast: bool, workers: int):
        jobs = []
        for shard_number, shard in enumerate(self.shards):
            nseq = int(shard["sequences"])
            for rel, expected_size in (
                (shard["tokens_file"], nseq * self.cfg.T * 2),
                (shard["valid_lengths_file"], nseq * 2),
                (shard["provenance_file"], None),
            ):
                path = _safe_corpus_path(self.root, rel)
                require(path.is_file(), f"missing declared artifact: {path}")
                actual = path.stat().st_size
                require(actual == expected_size if expected_size is not None
                        else actual == int(self._artifact_index[rel]["bytes"]),
                        f"byte count mismatch: {path}")
                jobs.append((rel, path))
        if fast:
            return
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = {ex.submit(_sha256_file, p): rel for rel, p in jobs}
            for fut, rel in futs.items():
                digest = fut.result()
                expected = self._artifact_index[rel]["sha256"]
                require(digest == expected,
                        f"SHA-256 mismatch: {rel}")

    # -- shards ------------------------------------------------------------

    def _shard_paths(self, k: int):
        shard = self.shards[k]
        nseq = int(shard["sequences"])
        return (
            nseq,
            _safe_corpus_path(self.root, shard["tokens_file"]),
            _safe_corpus_path(self.root, shard["valid_lengths_file"]),
            _safe_corpus_path(self.root, shard["provenance_file"]),
        )

    def _load_shard(self, k: int) -> _ShardData:
        if k in self._cache:
            return self._cache[k]
        require(0 <= k < len(self.shards), f"shard index out of range: {k}")
        nseq, token_path, length_path, prov_path = self._shard_paths(k)
        lengths = np.memmap(length_path, mode="r", dtype=np.uint16, shape=(nseq,))
        require(bool(np.all((lengths > 0) & (lengths <= self.cfg.T))),
                f"invalid valid_lengths values in {length_path}")
        tokens = np.memmap(token_path, mode="r", dtype=np.uint16,
                           shape=(nseq, self.cfg.T))
        table = pq.read_table(prov_path, columns=list(PROVENANCE_COLUMNS))
        require(table.num_rows == int(self.shards[k]["provenance_rows"]),
                f"provenance row count mismatch: {prov_path}")
        cols = {}
        for name in PROVENANCE_COLUMNS:
            col = table.column(name).combine_chunks()
            require(col.null_count == 0, f"null provenance values in {name}")
            cols[name] = col.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        order = np.lexsort((cols["sequence_token_start"], cols["sequence_index"]))
        seq_base = self.bases[k]
        rows: Dict[int, List[Tuple[int, int, int, int, int]]] = {}
        for ri in order:
            sg = int(cols["sequence_index"][ri])
            require(seq_base <= sg < seq_base + nseq,
                    f"sequence index outside shard in {prov_path}")
            rows.setdefault(sg - seq_base, []).append(
                (
                    int(cols["sequence_token_start"][ri]),
                    int(cols["sequence_token_end"][ri]),
                    int(cols["selected_document_index"][ri]),
                    int(cols["document_token_start"][ri]),
                    int(cols["document_token_end"][ri]),
                )
            )
        require(set(rows) == set(range(nseq)),
                f"provenance does not cover every sequence in {prov_path}")
        rows = {s: sorted(v, key=lambda r: r[0]) for s, v in rows.items()}
        first_row = rows[0][0]
        first_token = int(tokens[0, first_row[0]])
        data = _ShardData(k, nseq, seq_base, lengths, tokens, rows, first_row,
                          first_token, token_path, length_path, prov_path)
        self._cache[k] = data
        for old in [i for i in self._cache if i < k - 1]:
            self._cache.pop(old).close()
        return data

    def _shard_for(self, seq_global: int) -> int:
        k = bisect.bisect_right(self.bases, seq_global) - 1
        return max(0, min(k, len(self.shards) - 1))

    # -- row construction --------------------------------------------------

    def _build_row(self, seq_global, tok, length, rows, next_first, next_tok0):
        t = self.cfg.T
        length = int(length)

        def continues_inside(left, right):
            return (left[2] == right[2] and left[4] == right[3]
                    and left[1] == right[0])

        def continues_across(left, right):
            return left[2] == right[2] and left[4] == right[3]

        x = np.asarray(tok, dtype=np.uint16).copy()
        y = np.zeros(t, dtype=np.uint16)
        pos = np.zeros(t, dtype=np.int64)
        segpos = np.zeros(t, dtype=np.int32)
        start = np.zeros(t, dtype=np.int32)
        input_valid = np.zeros(t, dtype=np.bool_)
        valid = np.zeros(t, dtype=np.bool_)
        require(0 < length <= t, f"invalid valid length at sequence {seq_global}")

        cursor = 0
        seen_doc_ids = set()
        previous_row = None
        previous_segment_start = None
        for row_index, row in enumerate(rows):
            a, e, doc_id, doc_start, doc_end = row
            require(0 <= a < e <= length and a == cursor,
                    f"packed provenance gap/overlap at sequence {seq_global}")
            require(doc_start >= 0 and doc_end > doc_start
                    and e - a == doc_end - doc_start,
                    f"packed document span mismatch at sequence {seq_global}")
            cursor = e
            if previous_row is None or doc_id != previous_row[2]:
                require(doc_id not in seen_doc_ids,
                        f"document reappeared within sequence {seq_global}")
                seen_doc_ids.add(doc_id)
                segment_start = a
            else:
                require(doc_start == previous_row[4],
                        f"non-contiguous document spans at sequence {seq_global}")
                segment_start = previous_segment_start
            start[a:e] = segment_start
            segpos[a:e] = np.arange(a, e, dtype=np.int32) - segment_start
            pos[a:e] = doc_start + np.arange(e - a, dtype=np.int64)
            input_valid[a:e] = True
            if e - a > 1:
                y[a:e - 1] = x[a + 1:e]
                valid[a:e - 1] = True
            if row_index + 1 < len(rows):
                nxt = rows[row_index + 1]
                if continues_inside(row, nxt):
                    y[e - 1] = x[e]
                    valid[e - 1] = True
            elif next_first is not None:
                if (e == length and next_first[0] == 0
                        and continues_across(row, next_first)):
                    y[e - 1] = next_tok0
                    valid[e - 1] = True
            previous_row = row
            previous_segment_start = segment_start

        require(cursor == length,
                f"provenance does not cover sequence {seq_global}")
        start[length:] = np.int32(t + seq_global + 1)
        return {
            "x": x, "y": y, "pos": pos, "segpos": segpos, "start": start,
            "input_valid": input_valid, "valid": valid,
        }

    # -- streaming ---------------------------------------------------------

    def iter_rows(self, start_sequence: int = 0) -> Iterator[Tuple[int, dict]]:
        require(0 <= start_sequence <= self.total_sequences,
                f"start_sequence out of range: {start_sequence}")
        k = self._shard_for(start_sequence)
        shard = self._load_shard(k)
        nxt = self._load_shard(k + 1) if k + 1 < len(self.shards) else None
        while shard is not None:
            for s in range(shard.nseq):
                seq_global = shard.seq_base + s
                if seq_global < start_sequence:
                    continue
                rows = shard.rows[s]
                if s + 1 < shard.nseq:
                    next_first = shard.rows[s + 1][0]
                    next_tok0 = int(shard.tokens[s + 1, next_first[0]])
                elif nxt is not None:
                    next_first = nxt.first_row
                    next_tok0 = nxt.first_token
                else:
                    next_first = None
                    next_tok0 = None
                yield seq_global, self._build_row(
                    seq_global, shard.tokens[s], shard.lengths[s], rows,
                    next_first, next_tok0,
                )
            shard = nxt
            nxt = (self._load_shard(shard.index + 1)
                   if shard is not None and shard.index + 1 < len(self.shards)
                   else None)

    def stream_batches(self, start_sequence: int, batch_size: int,
                       allow_partial_tail: bool = False
                       ) -> Iterator[Tuple[int, Dict[str, torch.Tensor]]]:
        """Yield (first_sequence_of_batch, pinned CPU batch).

        Only full batches are yielded unless allow_partial_tail is set, in
        which case the final short batch (rows < batch_size, consumed once
        at the end of the corpus) is yielded as-is: never padded, never
        replayed.
        """
        xs, ys, poss, segs, starts, ivs, vs = [], [], [], [], [], [], []
        cursor = start_sequence
        for _, row in self.iter_rows(start_sequence):
            xs.append(row["x"])
            ys.append(row["y"])
            poss.append(row["pos"])
            segs.append(row["segpos"])
            starts.append(row["start"])
            ivs.append(row["input_valid"])
            vs.append(row["valid"])
            if len(xs) == batch_size:
                yield cursor, self._assemble(xs, ys, poss, segs, starts, ivs, vs)
                cursor += batch_size
                xs, ys, poss, segs, starts, ivs, vs = [], [], [], [], [], [], []
        if allow_partial_tail and xs:
            yield cursor, self._assemble(xs, ys, poss, segs, starts, ivs, vs)

    @staticmethod
    def _assemble(xs, ys, poss, segs, starts, ivs, vs):
        return {
            "x": _pin(torch.from_numpy(np.stack(xs))),
            "y": _pin(torch.from_numpy(np.stack(ys))),
            "pos": _pin(torch.from_numpy(np.stack(poss))),
            "segpos": _pin(torch.from_numpy(np.stack(segs))),
            "start": _pin(torch.from_numpy(np.stack(starts))),
            "input_valid": _pin(torch.from_numpy(np.stack(ivs))),
            "valid": _pin(torch.from_numpy(np.stack(vs))),
        }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# canonical Arm-A (frozen; copied from the certified implementation)
# ---------------------------------------------------------------------------

CAUSAL_CACHE = {}


def _causal_full(dev, t):
    key = ("full", str(dev), t)
    m = CAUSAL_CACHE.get(key)
    if m is None or m.device != dev:
        m = torch.ones((t, t), dtype=torch.bool, device=dev).tril(diagonal=-1)
        CAUSAL_CACHE[key] = m
    return m


def _causal_mask(dev, w):
    key = ("scan", str(dev), w)
    m = CAUSAL_CACHE.get(key)
    if m is None or m.device != dev:
        m = torch.ones((w, w), dtype=torch.bool, device=dev).tril(diagonal=-1)
        CAUSAL_CACHE[key] = m
    return m


def scan_chunkwise_candidate(qh, vh, segment_start, block=128, zero_carry=True):
    """Certified production scan: branch-free state + zero-carry skip."""
    b, h, t, k = qh.shape
    dv = vh.shape[-1]
    dev = qh.device
    state = torch.zeros((b, h, k, dv), dtype=qh.dtype, device=dev)
    outs = []
    for t0 in range(0, t, block):
        t1 = min(t0 + block, t)
        w = t1 - t0
        qb = qh[:, :, t0:t1]
        vb = vh[:, :, t0:t1]
        seg = segment_start[:, t0:t1]
        scores = qb @ qb.transpose(-1, -2)
        causal = _causal_mask(dev, w)
        samedoc = seg[:, :, None] == seg[:, None, :]
        local = scores.masked_fill(~(samedoc.unsqueeze(1) & causal), 0.0) @ vb
        if zero_carry and t0 == 0:
            outs.append(local)
        else:
            cont = (seg < t0).to(qb.dtype).view(b, 1, w, 1)
            carry = torch.einsum("bhwk,bhkd->bhwd", qb, state) * cont
            outs.append(local + carry)
        if t1 < t:
            segb = segment_start[:, t1]
            contb = (segb < t0).to(qb.dtype).view(b, 1, 1, 1)
            j = (segb - t0).clamp_min(0)
            keep = (
                (torch.arange(w, device=dev).unsqueeze(0) >= j.unsqueeze(1))
                .to(qb.dtype)
                .view(b, 1, w, 1)
            )
            fresh = torch.einsum("bhwk,bhwd->bhkd", qb * keep, vb)
            state = state * contb + fresh
    return torch.cat(outs, dim=2)


def canonical_init(cfg: ArmAConfig):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(cfg.SEED)

    def rnd(shape):
        return torch.randn(shape, generator=generator,
                           dtype=torch.float32) * cfg.INIT_STD

    return {
        "embedding": rnd((cfg.V, cfg.D)),
        "encoder": rnd((cfg.N, cfg.D)),
        "decoder_x": rnd((cfg.H, cfg.D, cfg.K)),
        "decoder_y": rnd((cfg.H, cfg.D, cfg.K)),
        "readout": rnd((cfg.D, cfg.V)),
        "coord_Wc": rnd((cfg.D, cfg.D)),
        "coord_bc": torch.zeros(cfg.D, dtype=torch.float32),
        "coord_alpha": torch.zeros((), dtype=torch.float32),
        "writer_W1": rnd((cfg.D, cfg.HIDDEN)),
        "writer_W2": rnd((cfg.HIDDEN, cfg.D)),
    }


def rope_pair_freq(cfg: ArmAConfig, device):
    return (
        1.0
        / (cfg.THETA ** ((2.0 * torch.arange(cfg.K // 2, dtype=torch.float32,
                                              device=device)) / cfg.K))
        / (2.0 * math.pi)
    )


def rope_phase(pos, freq):
    phase = pos.float().unsqueeze(-1).unsqueeze(-1) * freq.view(1, 1, 1, -1)
    phase = torch.remainder(phase, 1.0) * (2.0 * math.pi)
    return torch.cos(phase), torch.sin(phase)


class DenseWriter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(cfg.D, cfg.HIDDEN))
        self.W2 = nn.Parameter(torch.empty(cfg.HIDDEN, cfg.D))

    def forward(self, x):
        return F.relu(x @ self.W1) @ self.W2


class Coordinator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.Wc = nn.Parameter(torch.empty(cfg.D, cfg.D))
        self.bc = nn.Parameter(torch.zeros(cfg.D))
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, v, segpos, full_mask):
        z = v @ self.Wc + self.bc
        prev_sum = torch.bmm(full_mask.to(dtype=z.dtype), z)
        den = segpos.clamp_min(1).to(z.dtype).unsqueeze(-1)
        c = prev_sum / den - z
        rho = torch.sigmoid(self.alpha)
        return 1.0 + rho.to(c.dtype) * torch.tanh(c)


class OptArmA(nn.Module):
    """Production packed model, opt3c_all_b1024 execution (frozen)."""

    def __init__(self, cfg: ArmAConfig, device, scan_block: int = 1024):
        super().__init__()
        self.cfg = cfg
        self.scan_block = int(scan_block)
        self.embedding = nn.Embedding(cfg.V, cfg.D)
        self.encoder = nn.Parameter(torch.empty(cfg.N, cfg.D))
        self.decoder_x = nn.Parameter(torch.empty(cfg.H, cfg.D, cfg.K))
        self.decoder_y = nn.Parameter(torch.empty(cfg.H, cfg.D, cfg.K))
        self.readout = nn.Parameter(torch.empty(cfg.D, cfg.V))
        self.coordinator = Coordinator(cfg)
        self.writer = DenseWriter(cfg)
        self.ln = nn.LayerNorm(cfg.D, elementwise_affine=False, bias=False)
        self.rope_freq = rope_pair_freq(cfg, device)

    def project_x_native(self, v):
        cfg = self.cfg
        w_wide = self.decoder_x.permute(1, 0, 2).reshape(cfg.D, cfg.N)
        return F.relu(
            (v.reshape(v.shape[0] * cfg.T, cfg.D) @ w_wide).reshape(
                v.shape[0], cfg.T, cfg.H, cfg.K
            )
        )

    def attention_scan(self, x_bt, v, pos, segment_start, cs, sn):
        cfg = self.cfg
        if cs is not None:
            b, t, h, k = x_bt.shape
            qp = x_bt.reshape(b, t, h, k // 2, 2)
            csq = cs.to(x_bt.dtype)
            snq = sn.to(x_bt.dtype)
            qe, qo = qp[..., 0], qp[..., 1]
            qh = torch.stack(
                (qe * csq - qo * snq, qo * csq + qe * snq), dim=-1
            ).reshape_as(x_bt).permute(0, 2, 1, 3)
        else:
            raise TrainerError("cached RoPE phase is required (frozen flag)")
        vh = v.unsqueeze(1).expand(-1, cfg.H, -1, -1)
        return scan_chunkwise_candidate(
            qh, vh, segment_start, block=self.scan_block,
            zero_carry=FROZEN_FLAGS["zero_carry"],
        )

    def level(self, v, pos, segpos, full_mask, segment_start, cs, sn):
        cfg = self.cfg
        x_bt = self.project_x_native(v)
        a = self.ln(self.attention_scan(x_bt, v, pos, segment_start, cs, sn))
        ypre = F.relu(a @ self.decoder_y)
        prod = x_bt * ypre.permute(0, 2, 1, 3)
        paper_y_flat = prod.reshape(v.shape[0], cfg.T, cfg.N)
        base = self.ln(paper_y_flat @ self.encoder)
        g = self.coordinator(v, segpos, full_mask)
        delta = self.writer(g * base)
        return self.ln(v + delta)

    def forward_packed(self, idx, pos, segpos, full_mask, segment_start):
        cfg = self.cfg
        v = self.ln(self.embedding(idx))
        cs, sn = rope_phase(pos, self.rope_freq)
        for _ in range(cfg.L):
            v = self.level(v, pos, segpos, full_mask, segment_start, cs, sn)
        return v @ self.readout


def load_init(model, init, device):
    with torch.no_grad():
        model.embedding.weight.copy_(init["embedding"].to(device))
        model.encoder.copy_(init["encoder"].to(device))
        model.decoder_x.copy_(init["decoder_x"].to(device))
        model.decoder_y.copy_(init["decoder_y"].to(device))
        model.readout.copy_(init["readout"].to(device))
        model.coordinator.Wc.copy_(init["coord_Wc"].to(device))
        model.coordinator.bc.copy_(init["coord_bc"].to(device))
        model.coordinator.alpha.copy_(init["coord_alpha"].to(device))
        model.writer.W1.copy_(init["writer_W1"].to(device))
        model.writer.W2.copy_(init["writer_W2"].to(device))


def make_optimizer(model, cfg: ArmAConfig, device_type: str):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.PEAK_LR,
        betas=cfg.BETAS,
        eps=cfg.EPS,
        fused=(device_type == "cuda"),
    )


def lr_for_update(update_1based: int, cfg: ArmAConfig) -> float:
    return cfg.PEAK_LR * min(
        (update_1based * cfg.GLOBAL_BATCH * cfg.T) / cfg.WARMUP_TOKENS, 1.0
    )


def ce_sum(logits, targets, valid, vocab):
    per_token = F.cross_entropy(
        logits.reshape(-1, vocab), targets.reshape(-1), reduction="none"
    )
    return per_token[valid.reshape(-1)].sum(dtype=torch.float32)


def is_genuine_cuda_oom(error) -> bool:
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, torch.cuda.OutOfMemoryError):
            return True
        message = str(current).lower()
        if ("cuda out of memory" in message
                or "cuda error: out of memory" in message
                or "cublas_status_alloc_failed" in message):
            return True
        current = current.__cause__ or current.__context__
    return False


def one_full_update(cpu, model, compiled_model, optimizer, update_index,
                    cfg: ArmAConfig, device, check_grads: bool = True):
    """One complete training update: fwd + CE + bwd (+accum) + clip + AdamW.

    Works for any packed batch size up to GLOBAL_BATCH, so the final
    short batch of the corpus is a normal update without padding or replay.
    """
    rows = int(cpu["x"].shape[0])
    if rows <= 0:
        raise TrainerError(f"update {update_index}: empty packed batch")
    denom = int(cpu["valid"].sum().item())
    if denom <= 0:
        raise TrainerError(f"update {update_index}: no valid targets")
    lr = lr_for_update(update_index + 1, cfg)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    autocast_enabled = device.type == "cuda"
    for lo in range(0, rows, cfg.MICROBATCH):
        hi = min(lo + cfg.MICROBATCH, rows)
        x = cpu["x"][lo:hi].to(device, dtype=torch.long, non_blocking=True)
        y = cpu["y"][lo:hi].to(device, dtype=torch.long, non_blocking=True)
        pos = cpu["pos"][lo:hi].to(device, dtype=torch.int32, non_blocking=True)
        valid = cpu["valid"][lo:hi].to(device, dtype=torch.bool, non_blocking=True)
        input_valid = cpu["input_valid"][lo:hi].to(
            device, dtype=torch.bool, non_blocking=True)
        start = cpu["start"][lo:hi].to(device, dtype=torch.int32, non_blocking=True)
        segpos = cpu["segpos"][lo:hi].to(device, dtype=torch.int32, non_blocking=True)
        full_mask = (
            (start[:, :, None] == start[:, None, :])
            & input_valid[:, :, None]
            & input_valid[:, None, :]
            & _causal_full(device, cfg.T).unsqueeze(0)
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            cache_enabled=False, enabled=autocast_enabled):
            logits = compiled_model(x, pos, segpos, full_mask, start)
            loss = ce_sum(logits, y, valid, cfg.V) / denom
        if not bool(torch.isfinite(loss.detach())):
            raise TrainerError(f"update {update_index}: non-finite loss")
        loss.backward()
        loss_total += float(loss.detach())
        del x, y, pos, valid, input_valid, start, segpos, full_mask, logits, loss
    if check_grads:
        for pname, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            if not bool(torch.isfinite(parameter.grad).all()):
                raise TrainerError(f"non-finite gradient: {pname}")
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_NORM)
    if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
        raise TrainerError("non-finite gradient norm")
    optimizer.step()
    return {
        "loss": loss_total,
        "lr": float(lr),
        "grad_norm": float(grad_norm),
        "valid_pairs": denom,
    }


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------

def _to_cpu_tree(obj):
    if torch.is_tensor(obj):
        return obj.detach().to("cpu")
    if isinstance(obj, dict):
        return {k: _to_cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu_tree(v) for v in obj)
    return obj


def save_checkpoint_atomic(path: Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def build_checkpoint(cfg, corpus, model, optimizer, updates_done, tokens_consumed,
                     next_sequence, target_tokens, code_fp, session_stats=None):
    payload = {
        "format": CKPT_FORMAT,
        "implementation": IMPLEMENTATION_VERSION,
        "code_sha256": code_fp,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "config": frozen_config_dict(cfg),
        "corpus": {
            "corpus_id": corpus.contract.corpus_id,
            "artifact_hashes_sha256": corpus.contract.artifact_hashes_sha256,
            "logical_replay_sha256": corpus.contract.logical_replay_sha256,
            "total_sequences": int(corpus.total_sequences),
        },
        "progress": {
            "updates_done": int(updates_done),
            "tokens_consumed": int(tokens_consumed),
            "target_tokens": (None if target_tokens is None else int(target_tokens)),
            "next_sequence": int(next_sequence),
        },
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None),
        },
        "saved_at": _iso_now(),
        "session_stats": session_stats or {},
    }
    return _to_cpu_tree(payload)


def validate_and_load_checkpoint(path: Path, cfg, corpus, model, optimizer,
                                 device, code_fp, allow_code_change=False):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or ckpt.get("format") != CKPT_FORMAT:
        raise TrainerError(f"not a {CKPT_FORMAT} checkpoint: {path}")
    if ckpt.get("implementation") != IMPLEMENTATION_VERSION and not allow_code_change:
        raise TrainerError(
            f"checkpoint implementation mismatch at {path}: "
            f"{ckpt.get('implementation')!r} != {IMPLEMENTATION_VERSION!r}"
        )
    if ckpt.get("code_sha256") != code_fp and not allow_code_change:
        raise TrainerError(
            f"checkpoint code fingerprint mismatch at {path}; "
            "set ARM_A_ALLOW_CODE_CHANGE=1 (or --allow-code-change) to override"
        )
    if ckpt.get("config") != frozen_config_dict(cfg):
        raise TrainerError(f"checkpoint config differs from the frozen config: {path}")
    ccorpus = ckpt.get("corpus", {})
    if (ccorpus.get("corpus_id") != corpus.contract.corpus_id
            or ccorpus.get("artifact_hashes_sha256")
            != corpus.contract.artifact_hashes_sha256):
        raise TrainerError(f"checkpoint corpus fingerprint mismatch: {path}")
    progress = ckpt.get("progress", {})
    for key in ("updates_done", "tokens_consumed", "next_sequence"):
        if key not in progress:
            raise TrainerError(f"checkpoint missing progress.{key}: {path}")
    updates_done = int(progress["updates_done"])
    tokens_consumed = int(progress["tokens_consumed"])
    next_sequence = int(progress["next_sequence"])
    if next_sequence < 0 or tokens_consumed < 0 or updates_done < 0:
        raise TrainerError(f"negative progress field in {path}")
    if not (0 <= next_sequence <= corpus.total_sequences):
        raise TrainerError(f"corpus cursor out of range in {path}")
    if tokens_consumed != next_sequence * cfg.T:
        raise TrainerError(
            f"token accounting inconsistent in {path}: "
            f"tokens_consumed={tokens_consumed} next_sequence={next_sequence}"
        )
    if updates_done == 0:
        if next_sequence != 0:
            raise TrainerError(f"corpus cursor inconsistent in {path}")
    elif not ((updates_done - 1) * cfg.GLOBAL_BATCH < next_sequence
              <= updates_done * cfg.GLOBAL_BATCH):
        raise TrainerError(
            f"update/cursor inconsistent in {path}: "
            f"updates_done={updates_done} next_sequence={next_sequence}"
        )
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    for name, p in model.named_parameters():
        if not bool(torch.isfinite(p).all()):
            raise TrainerError(f"non-finite parameter after resume: {name}")
    opt_state = optimizer.state_dict()
    for state in opt_state.get("state", {}).values():
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state and not bool(torch.isfinite(state[key]).all()):
                raise TrainerError(f"non-finite optimizer state ({key}) after resume")
    rng = ckpt.get("rng", {})
    if "torch" in rng and rng["torch"] is not None:
        torch.set_rng_state(rng["torch"])
    if device.type == "cuda" and rng.get("cuda"):
        torch.cuda.set_rng_state_all(rng["cuda"])
    return {
        "updates_done": updates_done,
        "tokens_consumed": tokens_consumed,
        "next_sequence": next_sequence,
        "saved_at": ckpt.get("saved_at"),
        "path": str(path),
    }


def _candidate_checkpoints(ckpt_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    latest = ckpt_dir / "latest.pt"
    if latest.is_file():
        candidates.append(latest)
    archives = sorted(ckpt_dir.glob("step_*.pt"), reverse=True)
    candidates.extend(archives)
    return candidates


def find_latest_valid_checkpoint(ckpt_dir, cfg, corpus, model, optimizer, device,
                                 code_fp, logger, allow_code_change=False):
    candidates = _candidate_checkpoints(Path(ckpt_dir))
    if not candidates:
        return None
    failures = []
    for path in candidates:
        try:
            info = validate_and_load_checkpoint(
                path, cfg, corpus, model, optimizer, device, code_fp,
                allow_code_change=allow_code_change,
            )
            logger.log("resume_selected", **info)
            return info
        except Exception as exc:  # noqa: BLE001
            failures.append({"path": str(path),
                             "error": f"{type(exc).__name__}: {exc}"})
            logger.log("resume_candidate_rejected", path=str(path),
                       error=f"{type(exc).__name__}: {str(exc)[:300]}")
    raise TrainerError(
        "checkpoints exist but none is valid; refusing to start fresh: "
        + json.dumps(failures)
    )


# ---------------------------------------------------------------------------
# startup gates
# ---------------------------------------------------------------------------

def production_probe_b1(batch, device, cfg):
    """B=1 production-shaped probe (T, scan block) with the packed full mask.

    Mirrors the certified G4 preflight graph-hygiene gate: one real packed
    row, full_mask rebuilt from the sliced start/input_valid, no FP32 B64
    forward (which would OOM the probe).
    """
    x = batch["x"][:1].to(device, dtype=torch.long, non_blocking=True)
    pos = batch["pos"][:1].to(device, dtype=torch.int32, non_blocking=True)
    segpos = batch["segpos"][:1].to(device, dtype=torch.int32,
                                   non_blocking=True)
    start = batch["start"][:1].to(device, dtype=torch.int32,
                                  non_blocking=True)
    input_valid = batch["input_valid"][:1].to(device, dtype=torch.bool,
                                              non_blocking=True)
    full_mask = (
        (start[:, :, None] == start[:, None, :])
        & input_valid[:, :, None]
        & input_valid[:, None, :]
        & _causal_full(device, cfg.T).unsqueeze(0)
    )
    return x, pos, segpos, full_mask, start


def gate_graph_breaks(model, batch, device, logger) -> bool:
    if device.type != "cuda":
        logger.log("graph_break_gate", status="SKIP", reason="cpu")
        return True
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()
    probe = production_probe_b1(batch, device, model.cfg)
    explained = torch._dynamo.explain(model.forward_packed)(*probe)
    breaks = int(getattr(explained, "graph_break_count", -1))
    logger.log("graph_break_gate", status="PASS" if breaks == 0 else "FAIL",
               graph_break_count=breaks,
               graph_count=int(getattr(explained, "graph_count", -1)),
               context="production T/block, B=1 packed",
               break_reasons=[str(r)[:200]
                              for r in getattr(explained, "break_reasons", [])])
    del probe, explained
    gc.collect()
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()
    return breaks == 0


def gate_cpu_batch_equivalence(batch_a, batch_b, logger) -> bool:
    ok = True
    for key in ("x", "y", "pos", "segpos", "start", "input_valid", "valid"):
        if not torch.equal(batch_a[key], batch_b[key]):
            ok = False
            logger.log("batch_equivalence", status="FAIL", tensor=key)
    logger.log("batch_equivalence", status="PASS" if ok else "FAIL")
    return ok


def _prepare_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


PROD_GPU_NAME = "RTX PRO 6000 Blackwell"
PROD_GPU_CAPABILITY = (12, 0)
PROD_TORCH_VERSION = "2.11.0+cu128"
PROD_CUDA_VERSION = "12.8"


def enforce_production_runtime(device, logger):
    """Fail closed unless this is the certified G4 runtime."""
    if device.type != "cuda":
        raise TrainerError(
            f"production runtime requires a CUDA device "
            f"({PROD_GPU_NAME} sm_120); found {device.type}"
        )
    if not torch.cuda.is_bf16_supported():
        raise TrainerError("CUDA device does not support BF16")
    gpu_name = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    if PROD_GPU_NAME not in gpu_name or capability != PROD_GPU_CAPABILITY:
        raise TrainerError(
            f"unexpected GPU {gpu_name!r} capability={capability}; "
            f"require {PROD_GPU_NAME} sm_120"
        )
    if (torch.__version__ != PROD_TORCH_VERSION
            or torch.version.cuda != PROD_CUDA_VERSION):
        raise TrainerError(
            f"unexpected runtime torch={torch.__version__} "
            f"CUDA={torch.version.cuda}; require torch "
            f"{PROD_TORCH_VERSION} / CUDA {PROD_CUDA_VERSION}"
        )
    logger.log("runtime_gate", status="PASS", gpu=gpu_name,
               capability=list(capability), torch=torch.__version__,
               cuda=torch.version.cuda)


def _mount_drive_if_needed(corpus_root: Path):
    if corpus_root.exists():
        return
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive", force_remount=False)
    except Exception:
        pass


def _configure_dynamo():
    try:
        # Keep every compile static: the production model reshapes the time
        # dimension and Dynamo must not generalize it to a symbolic shape.
        torch._dynamo.config.automatic_dynamic_shapes = False
    except Exception:
        pass


def _build_model(cfg, device, use_compile):
    _configure_dynamo()
    torch.manual_seed(cfg.SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.SEED)
    model = OptArmA(cfg, device, scan_block=cfg.SCAN_BLOCK).to(device)
    if use_compile:
        entry = torch.compile(model.forward_packed, mode="default")
    else:
        entry = model.forward_packed
    return model, entry


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------

def run_smoke(cfg, corpus, run_dir: Path, device, use_compile=True,
              check_graph_breaks=True, allow_code_change=False):
    logger = RunLogger(Path(run_dir) / "logs" / "smoke.jsonl")
    smoke_dir = Path(run_dir) / "smoke_ckpt"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    code_fp = code_fingerprint()
    status = {"ok": False, "reasons": []}

    def fail(reason):
        status["reasons"].append(reason)
        logger.log("smoke_failure", reason=reason)

    try:
        logger.log("session_start", mode="smoke", device=str(device),
                   torch=torch.__version__, cuda=torch.version.cuda,
                   implementation=IMPLEMENTATION_VERSION,
                   code_sha256=code_fp,
                   corpus={"id": corpus.contract.corpus_id,
                           "sequences": corpus.total_sequences})

        model_a, entry_a = _build_model(cfg, device, use_compile)
        load_init(model_a, canonical_init(cfg), device)
        model_a.train()
        opt_a = make_optimizer(model_a, cfg, device.type)

        smoke_batches = []
        stream_a = corpus.stream_batches(0, cfg.GLOBAL_BATCH)
        for _ in range(3):
            try:
                smoke_batches.append(next(stream_a)[1])
            except StopIteration:
                break
        if len(smoke_batches) < 3:
            fail("corpus too small for the 3-update smoke gate")
        else:
            if check_graph_breaks and not gate_graph_breaks(
                model_a, smoke_batches[0], device, logger
            ):
                fail("graph breaks detected")
            for update_index in range(2):
                result = one_full_update(smoke_batches[update_index], model_a,
                                         entry_a, opt_a, update_index, cfg,
                                         device)
                logger.log("smoke_update", update_index=update_index,
                           loss=result["loss"], lr=result["lr"],
                           grad_norm=result["grad_norm"],
                           valid_pairs=result["valid_pairs"])

            ckpt_path = smoke_dir / "latest.pt"
            payload = build_checkpoint(cfg, corpus, model_a, opt_a,
                                       updates_done=2,
                                       tokens_consumed=2 * cfg.GLOBAL_BATCH * cfg.T,
                                       next_sequence=2 * cfg.GLOBAL_BATCH,
                                       target_tokens=None, code_fp=code_fp)
            save_checkpoint_atomic(ckpt_path, payload)
            load_info = validate_and_load_checkpoint(
                ckpt_path, cfg, corpus, model_a, opt_a, device, code_fp,
                allow_code_change=allow_code_change)
            logger.log("smoke_checkpoint_reloaded", **load_info)

            batch_for_update2 = smoke_batches[2]
            result_a = one_full_update(batch_for_update2, model_a, entry_a,
                                       opt_a, 2, cfg, device)
            params_a = {n: p.detach().clone()
                        for n, p in model_a.named_parameters()}
            cursor_a = 3 * cfg.GLOBAL_BATCH

            model_b, entry_b = _build_model(cfg, device, use_compile)
            opt_b = make_optimizer(model_b, cfg, device.type)
            validate_and_load_checkpoint(ckpt_path, cfg, corpus, model_b,
                                         opt_b, device, code_fp,
                                         allow_code_change=allow_code_change)
            stream_b = corpus.stream_batches(2 * cfg.GLOBAL_BATCH,
                                             cfg.GLOBAL_BATCH)
            _, batch_b = next(stream_b)
            if not gate_cpu_batch_equivalence(batch_for_update2, batch_b,
                                              logger):
                fail("resumed batch differs from the uninterrupted batch")
            result_b = one_full_update(batch_b, model_b, entry_b, opt_b,
                                       2, cfg, device)
            params_b = {n: p.detach().clone()
                        for n, p in model_b.named_parameters()}
            cursor_b = 3 * cfg.GLOBAL_BATCH

            loss_diff = abs(result_a["loss"] - result_b["loss"])
            param_diff = 0.0
            for name in params_a:
                if name not in params_b:
                    fail(f"parameter set mismatch on resume: missing {name}")
                    continue
                param_diff = max(
                    param_diff,
                    float((params_a[name] - params_b[name]).abs().max()),
                )
            if cursor_a != cursor_b:
                fail(f"corpus cursor mismatch: {cursor_a} != {cursor_b}")
            if loss_diff > 1e-5:
                fail(f"loss mismatch after resume: {loss_diff:.3e}")
            if param_diff > 1e-5:
                fail(f"parameter mismatch after resume: {param_diff:.3e}")
            if result_a["valid_pairs"] != result_b["valid_pairs"]:
                fail("valid-pair accounting mismatch after resume")
            logger.log("smoke_comparison", loss_a=result_a["loss"],
                   loss_b=result_b["loss"], loss_diff=loss_diff,
                   param_max_abs_diff=param_diff,
                   cursor_a=cursor_a, cursor_b=cursor_b)
        status["ok"] = not status["reasons"]
    except Exception as exc:  # noqa: BLE001
        fail(f"{type(exc).__name__}: {str(exc)[:500]}")
    finally:
        logger.log("smoke_end", status="PASS" if status["ok"] else "FAIL",
                   reasons=status["reasons"])
        print("SMOKE_PASS=" + str(status["ok"]).lower(), flush=True)
        print("ARM_A_2P5B_TRAINER_READY=" + str(status["ok"]).lower(),
              flush=True)
        logger.close()
    if status["ok"]:
        try:
            for f in smoke_dir.glob("*.pt*"):
                f.unlink()
            smoke_dir.rmdir()
        except OSError:
            pass
    return status["ok"]


def run_train(cfg, corpus, run_dir: Path, device, target_tokens,
              save_every: int, archive_every: int, log_every: int,
              use_compile=True, check_graph_breaks=True,
              allow_code_change=False, keep_archives=2):
    logger = RunLogger(Path(run_dir) / "logs" / "train.jsonl")
    ckpt_dir = Path(run_dir) / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    code_fp = code_fingerprint()
    ok = False

    try:
        logger.log("session_start", mode="train", device=str(device),
                   torch=torch.__version__, cuda=torch.version.cuda,
                   implementation=IMPLEMENTATION_VERSION,
                   code_sha256=code_fp,
                   target_tokens=target_tokens,
                   corpus={"id": corpus.contract.corpus_id,
                           "sequences": corpus.total_sequences,
                           "artifact_hashes_sha256":
                               corpus.contract.artifact_hashes_sha256})

        model, entry = _build_model(cfg, device, use_compile)
        optimizer = make_optimizer(model, cfg, device.type)

        resume = find_latest_valid_checkpoint(
            ckpt_dir, cfg, corpus, model, optimizer, device, code_fp, logger,
            allow_code_change=allow_code_change)
        if resume is None:
            load_init(model, canonical_init(cfg), device)
            updates_done = 0
            tokens_consumed = 0
            next_sequence = 0
            logger.log("fresh_start")
        else:
            updates_done = resume["updates_done"]
            tokens_consumed = resume["tokens_consumed"]
            next_sequence = resume["next_sequence"]
            logger.log("resumed", updates_done=updates_done,
                       tokens_consumed=tokens_consumed,
                       next_sequence=next_sequence)
        model.train()

        if target_tokens is not None and tokens_consumed >= target_tokens:
            logger.log("already_complete", tokens_consumed=tokens_consumed,
                       target_tokens=target_tokens)
        else:
            stream = corpus.stream_batches(next_sequence, cfg.GLOBAL_BATCH,
                                           allow_partial_tail=True)
            first_batch = None
            try:
                _, first_batch = next(stream)
            except StopIteration:
                first_batch = None
            if first_batch is None:
                logger.log("corpus_exhausted_before_target",
                           tokens_consumed=tokens_consumed,
                           next_sequence=next_sequence,
                           target_tokens=target_tokens)
            else:
                if check_graph_breaks:
                    if not gate_graph_breaks(model, first_batch, device, logger):
                        raise TrainerError("graph breaks detected; aborting")

                batch = first_batch
                session_start = time.perf_counter()
                while True:
                    rows = int(batch["x"].shape[0])
                    step_t0 = time.perf_counter()
                    result = one_full_update(batch, model, entry, optimizer,
                                             updates_done, cfg, device)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    step_ms = (time.perf_counter() - step_t0) * 1000.0
                    updates_done += 1
                    next_sequence += rows
                    tokens_consumed += rows * cfg.T
                    if device.type == "cuda":
                        mem_alloc = torch.cuda.memory_allocated(device)
                        mem_peak = torch.cuda.max_memory_allocated(device)
                        mem_reserved = torch.cuda.memory_reserved(device)
                    else:
                        mem_alloc = mem_peak = mem_reserved = 0
                    if log_every and updates_done % log_every == 0:
                        tok_s = (rows * cfg.T) / (step_ms / 1000.0)
                        logger.log("update", step=updates_done, rows=rows,
                                   tokens_consumed=tokens_consumed,
                                   next_sequence=next_sequence,
                                   loss=result["loss"], lr=result["lr"],
                                   grad_norm=result["grad_norm"],
                                   valid_pairs=result["valid_pairs"],
                                   step_ms=step_ms, tok_s=tok_s,
                                   valid_pairs_s=result["valid_pairs"] /
                                   (step_ms / 1000.0),
                                   mem_alloc_GiB=mem_alloc / 2**30,
                                   mem_peak_GiB=mem_peak / 2**30,
                                   mem_reserved_GiB=mem_reserved / 2**30,
                                   elapsed_s=time.perf_counter() - session_start)
                    if save_every and updates_done % save_every == 0:
                        payload = build_checkpoint(
                            cfg, corpus, model, optimizer, updates_done,
                            tokens_consumed, next_sequence, target_tokens,
                            code_fp,
                            session_stats={"last_loss": result["loss"]})
                        save_checkpoint_atomic(ckpt_dir / "latest.pt", payload)
                        logger.log("checkpoint", step=updates_done)
                    if (archive_every and updates_done % archive_every == 0):
                        payload = build_checkpoint(
                            cfg, corpus, model, optimizer, updates_done,
                            tokens_consumed, next_sequence, target_tokens,
                            code_fp)
                        save_checkpoint_atomic(
                            ckpt_dir / f"step_{updates_done:010d}.pt", payload)
                        archives = sorted(ckpt_dir.glob("step_*.pt"))
                        for old in archives[:-keep_archives]:
                            old.unlink(missing_ok=True)
                        logger.log("checkpoint_archive", step=updates_done)
                    if target_tokens is not None and tokens_consumed >= target_tokens:
                        break
                    try:
                        _, batch = next(stream)
                    except StopIteration:
                        logger.log("corpus_exhausted",
                                   tokens_consumed=tokens_consumed,
                                   next_sequence=next_sequence,
                                   total_sequences=corpus.total_sequences)
                        break

        payload = build_checkpoint(cfg, corpus, model, optimizer, updates_done,
                                   tokens_consumed, next_sequence, target_tokens,
                                   code_fp)
        save_checkpoint_atomic(ckpt_dir / "latest.pt", payload)
        logger.log("checkpoint_final", step=updates_done)
        logger.log("session_end", status="COMPLETE",
                   updates_done=updates_done,
                   tokens_consumed=tokens_consumed)
        print("TRAIN_STATUS=COMPLETE")
        print(f"UPDATES_DONE={updates_done}")
        print(f"TOKENS_CONSUMED={tokens_consumed}")
        ok = True
    except torch.cuda.OutOfMemoryError as exc:  # noqa: PERF203
        logger.log("session_end", status="ABORTED", reason="OOM",
                   error=str(exc)[:300])
        print("TRAIN_STATUS=ABORTED")
        print("ABORT_REASON=OOM")
    except TrainerError as exc:
        logger.log("session_end", status="ABORTED", reason="trainer_error",
                   error=str(exc)[:300])
        print("TRAIN_STATUS=ABORTED")
        print(f"ABORT_REASON={type(exc).__name__}: {str(exc)[:200]}")
    except Exception as exc:  # noqa: BLE001
        if is_genuine_cuda_oom(exc):
            logger.log("session_end", status="ABORTED", reason="OOM",
                       error=str(exc)[:300])
            print("TRAIN_STATUS=ABORTED")
            print("ABORT_REASON=OOM")
        else:
            logger.log("session_end", status="ABORTED", reason="exception",
                       error=f"{type(exc).__name__}: {str(exc)[:300]}")
            print("TRAIN_STATUS=ABORTED")
            print(f"ABORT_REASON={type(exc).__name__}: {str(exc)[:200]}")
    finally:
        logger.close()
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_target(value: str):
    if value is None or str(value).lower() in ("full", "none", "all"):
        return None
    return int(value)


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "train"), required=True)
    parser.add_argument("--corpus-root", default=str(PROD_CORPUS_ROOT))
    parser.add_argument("--run-dir", default=str(PROD_RUN_DIR))
    parser.add_argument("--target-tokens", default=str(TARGET_2P5B),
                        help="input-token budget for this run; 'full' streams "
                             "the whole corpus, ending with the final "
                             "partial (63-row) batch")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--archive-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-compile", action="store_true",
                        help="debug only; production uses torch.compile")
    parser.add_argument("--no-graph-check", action="store_true")
    parser.add_argument("--allow-code-change", action="store_true")
    parser.add_argument("--fast-verify", action="store_true",
                        help="skip per-file SHA-256 (sizes and index digests only)")
    args = parser.parse_args(argv)

    corpus_root = Path(args.corpus_root)
    run_dir = Path(args.run_dir)
    _mount_drive_if_needed(corpus_root)

    cfg = PROD_CFG
    device = _prepare_device()
    _configure_dynamo()
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    logger = RunLogger(run_dir / "logs" / "startup.jsonl")
    try:
        enforce_production_runtime(device, logger)
        logger.log("startup", mode=args.mode, corpus_root=str(corpus_root),
                   run_dir=str(run_dir), device=str(device),
                   implementation=IMPLEMENTATION_VERSION,
                   code_sha256=code_fingerprint(),
                   flags=dict(FROZEN_FLAGS))
        started = time.perf_counter()
        corpus = FrozenPackedCorpus(corpus_root, cfg,
                                    verify_files=True,
                                    fast_verify=args.fast_verify)
        logger.log("corpus_validated",
                   seconds=time.perf_counter() - started,
                   sequences=corpus.total_sequences,
                   artifact_hashes_sha256=corpus.contract.artifact_hashes_sha256)
        if args.mode == "smoke":
            ok = run_smoke(cfg, corpus, run_dir, device,
                           use_compile=not args.no_compile,
                           check_graph_breaks=not args.no_graph_check,
                           allow_code_change=args.allow_code_change)
            return 0 if ok else 2
        else:
            ok = run_train(cfg, corpus, run_dir, device,
                           target_tokens=parse_target(args.target_tokens),
                           save_every=args.save_every,
                           archive_every=args.archive_every,
                           log_every=args.log_every,
                           use_compile=not args.no_compile,
                           check_graph_breaks=not args.no_graph_check,
                           allow_code_change=args.allow_code_change)
            return 0 if ok else 3
    except CorpusContractError as exc:
        logger.log("startup_failed", reason="corpus_contract",
                   error=str(exc)[:500])
        print("ARM_A_2P5B_TRAINER_READY=false")
        print(f"STARTUP_FAILURE=corpus_contract: {str(exc)[:300]}")
        return 2
    except Exception as exc:  # noqa: BLE001
        logger.log("startup_failed", reason="exception",
                   error=f"{type(exc).__name__}: {str(exc)[:500]}")
        print("ARM_A_2P5B_TRAINER_READY=false")
        print(f"STARTUP_FAILURE={type(exc).__name__}: {str(exc)[:300]}")
        return 1
    finally:
        logger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
