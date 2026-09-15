# ARM-A AUTORESEARCH III — ONE-CELL G4 PREFLIGHT + CONFIRMATION
#
# Run as the ONLY cell in a freshly restarted G4 Colab runtime.
# This cell is intentionally not executed by the authoring environment.
#
# It answers two questions in order:
#   1. Is opt3c_all robust enough to launch? (exactness, determinism,
#      checkpoint/resume, graph breaks, frozen-corpus smoke train; the
#      benchmark below only runs when every gate passes.)
#   2. Does it beat the certified packed b1024 baseline?
#      (~69,183 tok/s, 1894.56 ms/update, 62.55 GiB)
#
# Output ends with one machine-readable verdict:
#   FAILED_GATES=[...]
#   OPT3C_ALL_ROBUST=true|false
#   CANDIDATE_BEATS_CERTIFIED_ANCHOR_69183=true|false
#
# Certified baseline: opt3c chunkwise score-free scan, b1024, dense
# coordinator, no checkpoint, B16x4, compiled, where-based packed state
# update (the exact code that produced the certified number).
#
# Candidate: identical math with four exact execution changes:
#   1. branch-free packed state update (one bmm + mul-add, no full-bmm,
#      no where-select, no host sync)  [locally: bitwise identical outputs]
#   2. zero-carry skip for the chunk at t0==0 (state is exactly zero, so
#      the carry bmm is provably dead work)  [exact]
#   3. paper_y direct layout (no transpose copy)   [exact]
#   4. cached RoPE phase (pos is constant across levels)  [exact]
#
# Same frozen packed 13 global batches, same init/optimizer/LR schedule,
# same global B64, B16x4, 3 warmups + 10 timed full updates, compiled.

import os
import sys

if "torch" in sys.modules:
    raise RuntimeError("Restart the Colab runtime before running this cell.")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import math
import random
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from google.colab import drive

T, VOCAB = 2048, 8192
D, N, H = 256, 16_384, 4
K, L, HIDDEN = N // H, 8, 1040
SEED, INIT_STD, THETA = 1337, 0.02, 2**16
READ_BLOCK = 256
PEAK_LR, WARMUP_TOKENS = 1e-3, 10_000_000
BETAS, EPS, WEIGHT_DECAY, CLIP_NORM = (0.9, 0.95), 1e-8, 0.1, 1.0
GLOBAL_BATCH, WARMUPS, SAMPLES = 64, 3, 10
GLOBAL_TOKENS = GLOBAL_BATCH * T
UNIQUE_BATCHES = WARMUPS + SAMPLES
REQUIRED_ROWS = GLOBAL_BATCH * UNIQUE_BATCHES
DEVICE = torch.device("cuda")

# The preflight compiles both tiny gate models (T=64) and production models
# (T=2048) in one process. Without this, Dynamo generalizes the time
# dimension to a symbolic shape after seeing both sizes, which breaks the
# static reshapes inside project_x_native. Keep every compile static.
torch._dynamo.config.automatic_dynamic_shapes = False

CERTIFIED_TOK_S = 69_183.0
CERTIFIED_MS = 1894.56
CERTIFIED_GIB = 62.55

# Candidate execution flags. Set from local same-session A/B evidence.
CANDIDATE_FLAGS = {
    "packed_update": "branchfree",
    "zero_carry": True,
    "paper_layout": "direct",  # "flat" to disable
    "cache_rope": True,        # False to disable
}

CORPUS = Path(
    "/content/drive/Shareddrives/ICLR PHASE BDH/"
    "phase_bdh/corpus/stage2/frozen_5b_v1"
)

PROVENANCE_COLUMNS = (
    "sequence_index",
    "sequence_token_start",
    "sequence_token_end",
    "selected_document_index",
    "document_token_start",
    "document_token_end",
)

drive.mount("/content/drive", force_remount=False)

if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise RuntimeError("This cell requires a BF16-capable CUDA GPU.")

GPU_NAME = torch.cuda.get_device_name(0)
GPU_CAPABILITY = torch.cuda.get_device_capability(0)

if "RTX PRO 6000 Blackwell" not in GPU_NAME or GPU_CAPABILITY != (12, 0):
    raise RuntimeError(
        f"Expected RTX PRO 6000 Blackwell sm_120; found "
        f"{GPU_NAME!r}, capability={GPU_CAPABILITY}."
    )

if torch.__version__ != "2.11.0+cu128" or torch.version.cuda != "12.8":
    raise RuntimeError(
        f"Expected torch 2.11.0+cu128 / CUDA 12.8; found "
        f"torch={torch.__version__}, CUDA={torch.version.cuda}."
    )

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)

CAUSAL = torch.ones((T, T), device=DEVICE, dtype=torch.bool).tril(diagonal=-1)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def safe_corpus_path(relative_path):
    rel = PurePosixPath(str(relative_path))
    require(
        not rel.is_absolute()
        and ".." not in rel.parts
        and "\\" not in str(relative_path),
        f"Unsafe/noncanonical corpus path: {relative_path!r}",
    )
    return CORPUS.joinpath(*rel.parts)


def read_frozen_manifest():
    require(CORPUS.is_dir(), f"Frozen corpus directory is missing: {CORPUS}")

    frozen = json.loads((CORPUS / "FROZEN.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (CORPUS / "corpus_manifest.json").read_text(encoding="utf-8")
    )

    require(frozen.get("status") == "FROZEN", "FROZEN.json is not FROZEN.")
    require(
        frozen.get("corpus_id") == "phase_bdh_stage2_5b_v1",
        "Wrong frozen corpus id.",
    )
    require(int(frozen.get("context_length")) == T, "Context length differs.")
    require(int(frozen.get("sequences")) == 2_441_407, "Sequence count differs.")
    require(manifest.get("status") == "FROZEN", "Manifest is not FROZEN.")
    require(
        int(manifest.get("real_tokens")) == 5_000_000_000,
        "Real-token count differs.",
    )
    require(
        int(manifest.get("physical_tokens")) == 5_000_001_536,
        "Physical-token count differs.",
    )
    require(
        int(manifest.get("padding_tokens")) == 1_536,
        "Padding-token count differs.",
    )

    shards = manifest.get("shards", [])
    require(len(shards) == 25, f"Expected 25 frozen shards; found {len(shards)}.")
    return frozen, manifest, shards


def shard_paths(shard_number, shard):
    stem = f"shard_{shard_number:06d}"
    require(shard.get("stem") == stem, f"Unexpected shard stem at {shard_number}.")
    require(
        shard.get("tokens_file") == f"train/{stem}.tokens.bin",
        f"Unexpected token path for {stem}.",
    )
    require(
        shard.get("valid_lengths_file") == f"train/{stem}.valid_lengths.bin",
        f"Unexpected length path for {stem}.",
    )
    require(
        shard.get("provenance_file") == f"train/{stem}.provenance.parquet",
        f"Unexpected provenance path for {stem}.",
    )
    return (
        int(shard["sequences"]),
        safe_corpus_path(shard["tokens_file"]),
        safe_corpus_path(shard["valid_lengths_file"]),
        safe_corpus_path(shard["provenance_file"]),
    )


def provenance_columns(table):
    out = {}
    for name in PROVENANCE_COLUMNS:
        col = table.column(name).combine_chunks()
        require(col.null_count == 0, f"Null provenance values in {name}.")
        out[name] = col.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    return out


def reconstruct_packed_batches():
    """Materialize 832 physical sequences plus one required target lookahead."""
    _, _, shards = read_frozen_manifest()

    last_needed = REQUIRED_ROWS
    tokens_by_seq = {}
    lengths_by_seq = {}
    rows_by_seq = {}

    sequence_base = 0

    for shard_number, shard in enumerate(shards):
        (
            nseq,
            token_path,
            length_path,
            provenance_path,
        ) = shard_paths(shard_number, shard)

        shard_last = sequence_base + nseq - 1
        if sequence_base > last_needed:
            break

        require(
            token_path.stat().st_size == nseq * T * 2,
            f"Unexpected token-shard size: {token_path}",
        )
        require(
            length_path.stat().st_size == nseq * 2,
            f"Unexpected length-shard size: {length_path}",
        )

        lengths = np.memmap(length_path, mode="r", dtype=np.uint16, shape=(nseq,))
        token_map = np.memmap(token_path, mode="r", dtype=np.uint16, shape=(nseq, T))

        table = pq.read_table(provenance_path, columns=list(PROVENANCE_COLUMNS))
        require(
            table.num_rows == int(shard["provenance_rows"]),
            f"Provenance row count mismatch: {provenance_path}",
        )

        cols = provenance_columns(table)
        order = np.lexsort(
            (cols["sequence_token_start"], cols["sequence_index"])
        )

        for row_index in order:
            seq_global = int(cols["sequence_index"][row_index])
            if seq_global < sequence_base or seq_global > last_needed:
                continue
            require(
                seq_global < sequence_base + nseq,
                "Packed sequence index outside shard.",
            )
            seq = seq_global - sequence_base

            if seq_global not in tokens_by_seq:
                tokens_by_seq[seq_global] = np.asarray(
                    token_map[seq], dtype=np.uint16
                ).copy()
                lengths_by_seq[seq_global] = int(lengths[seq])

            rows_by_seq.setdefault(seq_global, []).append(
                (
                    int(cols["sequence_token_start"][row_index]),
                    int(cols["sequence_token_end"][row_index]),
                    int(cols["selected_document_index"][row_index]),
                    int(cols["document_token_start"][row_index]),
                    int(cols["document_token_end"][row_index]),
                )
            )

        del table, cols, order, token_map, lengths
        gc.collect()

        if shard_last >= last_needed:
            break
        sequence_base += nseq

    expected = set(range(last_needed + 1))
    require(
        set(tokens_by_seq) == expected,
        "Packed prefix did not read exactly the required sequence ids.",
    )
    require(
        set(rows_by_seq) == expected,
        "Packed prefix has missing provenance sequences.",
    )

    x_np = np.zeros((REQUIRED_ROWS, T), dtype=np.uint16)
    y_np = np.zeros((REQUIRED_ROWS, T), dtype=np.uint16)
    pos_np = np.zeros((REQUIRED_ROWS, T), dtype=np.int64)
    segpos_np = np.zeros((REQUIRED_ROWS, T), dtype=np.int32)
    start_np = np.zeros((REQUIRED_ROWS, T), dtype=np.int32)
    input_valid_np = np.zeros((REQUIRED_ROWS, T), dtype=np.bool_)
    valid_np = np.zeros((REQUIRED_ROWS, T), dtype=np.bool_)
    doc_np = np.full((REQUIRED_ROWS, T), -1, dtype=np.int64)

    cross_sequence_pairs = 0

    def continues_inside(left, right):
        return (
            left[2] == right[2]
            and left[4] == right[3]
            and left[1] == right[0]
        )

    def continues_across(left, right):
        return left[2] == right[2] and left[4] == right[3]

    for seq_global in range(REQUIRED_ROWS):
        tok = tokens_by_seq[seq_global]
        length = lengths_by_seq[seq_global]
        rows = sorted(rows_by_seq[seq_global], key=lambda r: r[0])

        require(
            0 < length <= T,
            f"Invalid packed valid length at sequence {seq_global}.",
        )

        cursor = 0
        seen_doc_ids = set()
        previous_segment_start = None
        previous_row = None

        for row_index, row in enumerate(rows):
            a, e, doc_id, doc_start, doc_end = row

            require(
                0 <= a < e <= length,
                f"Invalid packed span at sequence {seq_global}.",
            )
            require(
                a == cursor,
                f"Packed provenance gap/overlap at sequence {seq_global}.",
            )
            require(
                doc_start >= 0
                and doc_end > doc_start
                and e - a == doc_end - doc_start,
                "Packed document span mismatch.",
            )

            cursor = e

            if previous_row is None or doc_id != previous_row[2]:
                require(
                    doc_id not in seen_doc_ids,
                    "A document reappeared within one physical sequence.",
                )
                seen_doc_ids.add(doc_id)
                segment_start = a
            else:
                require(
                    doc_start == previous_row[4],
                    "Same-document packed spans are not contiguous.",
                )
                segment_start = previous_segment_start

            start_np[seq_global, a:e] = segment_start
            segpos_np[seq_global, a:e] = (
                np.arange(a, e, dtype=np.int32) - segment_start
            )
            pos_np[seq_global, a:e] = doc_start + np.arange(e - a, dtype=np.int64)
            doc_np[seq_global, a:e] = doc_id
            input_valid_np[seq_global, a:e] = True

            if e - a > 1:
                y_np[seq_global, a:e - 1] = tok[a + 1:e]
                valid_np[seq_global, a:e - 1] = True

            if row_index + 1 < len(rows):
                nxt = rows[row_index + 1]
                if continues_inside(row, nxt):
                    y_np[seq_global, e - 1] = tok[e]
                    valid_np[seq_global, e - 1] = True
            elif seq_global < REQUIRED_ROWS:
                nxt = sorted(rows_by_seq[seq_global + 1], key=lambda r: r[0])[0]
                if (
                    e == length
                    and nxt[0] == 0
                    and continues_across(row, nxt)
                ):
                    y_np[seq_global, e - 1] = tokens_by_seq[seq_global + 1][nxt[0]]
                    valid_np[seq_global, e - 1] = True
                    cross_sequence_pairs += 1

            previous_row = row
            previous_segment_start = segment_start

        require(
            cursor == length,
            f"Packed provenance does not cover sequence {seq_global}.",
        )
        start_np[seq_global, length:] = T + seq_global + 1
        x_np[seq_global] = tok

    for seq_global in range(REQUIRED_ROWS - 1):
        last = sorted(rows_by_seq[seq_global], key=lambda r: r[0])[-1]
        nxt = sorted(rows_by_seq[seq_global + 1], key=lambda r: r[0])[0]

        expected_pair = (
            last[1] == lengths_by_seq[seq_global]
            and nxt[0] == 0
            and continues_across(last, nxt)
        )
        observed_pair = bool(
            valid_np[seq_global, lengths_by_seq[seq_global] - 1]
        )
        require(
            observed_pair == bool(expected_pair),
            "Packed cross-sequence target semantics mismatch.",
        )
        if expected_pair:
            require(
                int(y_np[seq_global, lengths_by_seq[seq_global] - 1])
                == int(tokens_by_seq[seq_global + 1][nxt[0]]),
                "Packed lookahead target mismatch.",
            )

    require(
        int(input_valid_np.sum())
        == sum(lengths_by_seq[i] for i in range(REQUIRED_ROWS)),
        "Packed valid-length accounting failed.",
    )
    require(
        int(pos_np[input_valid_np].max()) < 2**31,
        "Packed RoPE positions exceed int32 transfer range.",
    )

    cpu = {
        "x": torch.from_numpy(np.ascontiguousarray(x_np)).pin_memory(),
        "y": torch.from_numpy(np.ascontiguousarray(y_np)).pin_memory(),
        "pos": torch.from_numpy(np.ascontiguousarray(pos_np)).pin_memory(),
        "valid": torch.from_numpy(np.ascontiguousarray(valid_np)).pin_memory(),
        "input_valid": torch.from_numpy(
            np.ascontiguousarray(input_valid_np)
        ).pin_memory(),
        "start": torch.from_numpy(np.ascontiguousarray(start_np)).pin_memory(),
        "segpos": torch.from_numpy(np.ascontiguousarray(segpos_np)).pin_memory(),
        "document_id": torch.from_numpy(np.ascontiguousarray(doc_np)).pin_memory(),
    }

    valid_pairs = [
        int(
            valid_np[
                i * GLOBAL_BATCH:(i + 1) * GLOBAL_BATCH
            ].sum()
        )
        for i in range(UNIQUE_BATCHES)
    ]
    require(all(n > 0 for n in valid_pairs), "A packed update has no valid targets.")

    metadata = {
        "unique_global_batches": UNIQUE_BATCHES,
        "measured_physical_sequences": REQUIRED_ROWS,
        "lookahead_sequence_index": REQUIRED_ROWS,
        "replayed": False,
        "cross_sequence_target_pairs": int(cross_sequence_pairs),
        "padding_positions_masked": True,
    }

    del (x_np, y_np, pos_np, segpos_np, start_np, input_valid_np, valid_np, doc_np)
    return cpu, valid_pairs, metadata


# ---------------------------------------------------------------------------
# Scan implementations (exact; both are the full production math)
# ---------------------------------------------------------------------------

CAUSAL_CACHE = {}


def _causal_mask(dev, w):
    key = (str(dev), w)
    m = CAUSAL_CACHE.get(key)
    if m is None or m.device != dev:
        m = torch.ones((w, w), dtype=torch.bool, device=dev).tril(diagonal=-1)
        CAUSAL_CACHE[key] = m
    return m


def scan_chunkwise_certified(qh, vh, segment_start, block=128):
    """Certified mathematical execution path (A/B control arm).

    The certified Stage-2 packed recurrences: unconditional full-block-sum
    bmm plus where-select state update. The only difference from the
    historical certified run is the shared non-semantic causal-mask cache
    (the original built the triangular mask inside the scan; the cached
    mask is mathematically identical and constant-folds). This control
    reproduces the certified production behavior in-session; the historical
    69,183 tok/s anchor is printed separately.
    """
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
        cont = (seg < t0).to(qb.dtype).view(b, 1, w, 1)
        carry = torch.einsum("bhwk,bhkd->bhwd", qb, state) * cont
        outs.append(local + carry)
        full = state + torch.einsum("bhwk,bhwd->bhkd", qb, vb)
        if t1 < t:
            segb = segment_start[:, t1]
            contb = segb < t0
            j = (segb - t0).clamp_min(0)
            keep = (
                (torch.arange(w, device=dev).unsqueeze(0) >= j.unsqueeze(1))
                .to(qb.dtype)
                .view(b, 1, w, 1)
            )
            fresh = torch.einsum("bhwk,bhwd->bhkd", qb * keep, vb)
            state = torch.where(contb.view(b, 1, 1, 1), full, fresh)
    return torch.cat(outs, dim=2)


def scan_chunkwise_candidate(qh, vh, segment_start, block=128,
                             zero_carry=True):
    """Candidate packed path: branch-free state + zero-carry skip.

    Exact same math as the certified path:
      contb=1 -> S + full block sum; contb=0 -> rows >= seg(t1) only.
    Removes the full-bmm, the where-select, the dead final-chunk update,
    and (zero_carry) the provably-zero carry bmm at t0==0.
    """
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


@dataclass(frozen=True)
class ArmAConfig:
    T: int = T
    V: int = VOCAB
    D: int = D
    N: int = N
    H: int = H
    L: int = L
    HIDDEN: int = HIDDEN
    SEED: int = SEED
    INIT_STD: float = INIT_STD
    THETA: float = THETA
    READ_BLOCK: int = READ_BLOCK
    PEAK_LR: float = PEAK_LR
    BETAS: tuple = BETAS
    EPS: float = EPS
    WEIGHT_DECAY: float = WEIGHT_DECAY
    CLIP_NORM: float = CLIP_NORM

    @property
    def K(self):
        return self.N // self.H


CFG = ArmAConfig()


def canonical_init(cfg):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(cfg.SEED)

    def rnd(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32) * cfg.INIT_STD

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


def rope_pair_freq(cfg, device):
    return (
        1.0
        / (
            cfg.THETA
            ** ((2.0 * torch.arange(cfg.K // 2, dtype=torch.float32, device=device)) / cfg.K)
        )
        / (2.0 * math.pi)
    )


def rope_phase(pos, freq):
    phase = pos.float().unsqueeze(-1).unsqueeze(-1) * freq.view(1, 1, 1, -1)
    phase = torch.remainder(phase, 1.0) * (2.0 * math.pi)
    return torch.cos(phase), torch.sin(phase)


def rope_bthk(q, pos, freq):
    cs, sn = rope_phase(pos, freq)
    cs = cs.to(q.dtype)
    sn = sn.to(q.dtype)
    b, t, h, k = q.shape
    qp = q.reshape(b, t, h, k // 2, 2)
    qe, qo = qp[..., 0], qp[..., 1]
    return torch.stack((qe * cs - qo * sn, qo * cs + qe * sn), dim=-1).reshape_as(q)


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
    """Production packed model with certified/candidate execution switch."""

    def __init__(self, cfg, device, scan_block=1024, variant="candidate"):
        super().__init__()
        require(variant in ("certified", "candidate"), "bad variant")
        self.cfg = cfg
        self.scan_block = int(scan_block)
        self.variant = variant
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
            qh = rope_bthk(x_bt, pos, self.rope_freq).permute(0, 2, 1, 3)
        vh = v.unsqueeze(1).expand(-1, cfg.H, -1, -1)
        if self.variant == "certified":
            return scan_chunkwise_certified(
                qh, vh, segment_start, block=self.scan_block
            )
        return scan_chunkwise_candidate(
            qh, vh, segment_start, block=self.scan_block,
            zero_carry=CANDIDATE_FLAGS["zero_carry"],
        )

    def level(self, v, pos, segpos, full_mask, segment_start, cs, sn):
        cfg = self.cfg
        x_bt = self.project_x_native(v)
        a = self.ln(self.attention_scan(x_bt, v, pos, segment_start, cs, sn))
        ypre = F.relu(a @ self.decoder_y)
        if self.variant == "candidate" and CANDIDATE_FLAGS["paper_layout"] == "direct":
            prod = x_bt * ypre.permute(0, 2, 1, 3)
            paper_y_flat = prod.reshape(v.shape[0], cfg.T, cfg.N)
        else:
            paper_y = x_bt.permute(0, 2, 1, 3) * ypre
            paper_y_flat = paper_y.transpose(1, 2).reshape(
                v.shape[0], cfg.T, cfg.N
            )
        base = self.ln(paper_y_flat @ self.encoder)
        g = self.coordinator(v, segpos, full_mask)
        delta = self.writer(g * base)
        return self.ln(v + delta)

    def forward_packed(self, idx, pos, segpos, full_mask, segment_start):
        cfg = self.cfg
        v = self.ln(self.embedding(idx))
        cs = sn = None
        if self.variant == "candidate" and CANDIDATE_FLAGS["cache_rope"]:
            cs, sn = rope_phase(pos, self.rope_freq)
        for _ in range(cfg.L):
            v = self.level(v, pos, segpos, full_mask, segment_start, cs, sn)
        return v @ self.readout


def load_init(model, init):
    with torch.no_grad():
        model.embedding.weight.copy_(init["embedding"].to(DEVICE))
        model.encoder.copy_(init["encoder"].to(DEVICE))
        model.decoder_x.copy_(init["decoder_x"].to(DEVICE))
        model.decoder_y.copy_(init["decoder_y"].to(DEVICE))
        model.readout.copy_(init["readout"].to(DEVICE))
        model.coordinator.Wc.copy_(init["coord_Wc"].to(DEVICE))
        model.coordinator.bc.copy_(init["coord_bc"].to(DEVICE))
        model.coordinator.alpha.copy_(init["coord_alpha"].to(DEVICE))
        model.writer.W1.copy_(init["writer_W1"].to(DEVICE))
        model.writer.W2.copy_(init["writer_W2"].to(DEVICE))


def make_optimizer(model):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=PEAK_LR,
        betas=BETAS,
        eps=EPS,
        fused=True,
    )


def lr_for_update(update_1based):
    return PEAK_LR * min((update_1based * GLOBAL_TOKENS) / WARMUP_TOKENS, 1.0)


def ce_sum(logits, targets, valid, vocab=VOCAB):
    per_token = F.cross_entropy(
        logits.reshape(-1, vocab), targets.reshape(-1), reduction="none"
    )
    return per_token[valid.reshape(-1)].sum(dtype=torch.float32)


def one_full_update(cpu, model, compiled_model, optimizer, update_index,
                    check_finite=False):
    row0 = update_index * GLOBAL_BATCH
    denom = int(cpu["valid"][row0:row0 + GLOBAL_BATCH].sum().item())
    require(denom > 0, f"Update {update_index + 1} has no valid targets.")

    for group in optimizer.param_groups:
        group["lr"] = lr_for_update(update_index + 1)

    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0

    for lo in range(row0, row0 + GLOBAL_BATCH, 16):
        hi = lo + 16
        x = cpu["x"][lo:hi].to(DEVICE, dtype=torch.long, non_blocking=True)
        y = cpu["y"][lo:hi].to(DEVICE, dtype=torch.long, non_blocking=True)
        pos = cpu["pos"][lo:hi].to(DEVICE, dtype=torch.int32, non_blocking=True)
        valid = cpu["valid"][lo:hi].to(DEVICE, dtype=torch.bool, non_blocking=True)
        input_valid = cpu["input_valid"][lo:hi].to(
            DEVICE, dtype=torch.bool, non_blocking=True
        )
        start = cpu["start"][lo:hi].to(DEVICE, dtype=torch.int32, non_blocking=True)
        segpos = cpu["segpos"][lo:hi].to(DEVICE, dtype=torch.int32, non_blocking=True)

        full_mask = (
            (start[:, :, None] == start[:, None, :])
            & input_valid[:, :, None]
            & input_valid[:, None, :]
            & CAUSAL.unsqueeze(0)
        )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False):
            logits = compiled_model(x, pos, segpos, full_mask, start)
            loss = ce_sum(logits, y, valid) / denom

        loss.backward()
        loss_total += float(loss.detach())
        del x, y, pos, valid, input_valid, start, segpos, full_mask, logits, loss

    if check_finite:
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            if not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError(f"Non-finite gradient in {name}")

    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
    optimizer.step()
    return loss_total


def is_genuine_cuda_oom(error):
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, torch.cuda.OutOfMemoryError):
            return True
        message = str(current).lower()
        if (
            "cuda out of memory" in message
            or "cuda error: out of memory" in message
            or "cublas_status_alloc_failed" in message
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


INIT = canonical_init(CFG)
PARAMETER_COUNT = 17_375_489


def run_treatment(variant, cpu, valid_pairs, run_order):
    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()

    model = None
    optimizer = None
    compiled_model = None

    try:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        model = OptArmA(CFG, DEVICE, scan_block=1024, variant=variant).to(DEVICE)
        load_init(model, INIT)
        model.train()

        parameter_count = sum(p.numel() for p in model.parameters())
        require(
            parameter_count == PARAMETER_COUNT,
            f"Unexpected parameter count: {parameter_count}.",
        )
        require(
            all(p.dtype == torch.float32 for p in model.parameters()),
            "Master parameters are not FP32.",
        )

        optimizer = make_optimizer(model)
        compiled_model = torch.compile(model.forward_packed, mode="default")

        for update_index in range(WARMUPS):
            one_full_update(cpu, model, compiled_model, optimizer, update_index)

        torch.cuda.synchronize(DEVICE)
        torch.cuda.reset_peak_memory_stats(DEVICE)

        timings_ms = []
        for sample_index in range(SAMPLES):
            update_index = WARMUPS + sample_index
            torch.cuda.synchronize(DEVICE)
            t0 = time.perf_counter()
            one_full_update(cpu, model, compiled_model, optimizer, update_index)
            torch.cuda.synchronize(DEVICE)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)

        median_ms = float(np.percentile(timings_ms, 50))
        return {
            "status": "MEASURED",
            "variant": variant,
            "run_order": run_order,
            "scan_block": 1024,
            "microbatch_sequences": 16,
            "gradient_accumulation_steps": 4,
            "candidate_flags": CANDIDATE_FLAGS if variant == "candidate" else None,
            "warmup_updates": WARMUPS,
            "timed_updates": SAMPLES,
            "median_ms": median_ms,
            "p10_ms": float(np.percentile(timings_ms, 10)),
            "p90_ms": float(np.percentile(timings_ms, 90)),
            "step_times_ms": [float(v) for v in timings_ms],
            "input_tokens_per_update": GLOBAL_TOKENS,
            "input_tok_s": float(GLOBAL_TOKENS / (median_ms / 1000.0)),
            "peak_allocated_GiB": float(
                torch.cuda.max_memory_allocated(DEVICE) / 2**30
            ),
            "peak_reserved_GiB": float(
                torch.cuda.max_memory_reserved(DEVICE) / 2**30
            ),
        }
    finally:
        model = None
        optimizer = None
        compiled_model = None
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()


def tiny_packed_correctness():
    """Dense-oracle + certified-vs-candidate gate on documented boundaries."""
    block = 8
    test_t = 16
    test_cfg = ArmAConfig(T=test_t, V=64, D=16, N=64, H=2, L=2, HIDDEN=32)
    layouts = [
        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8]] * 2),
        np.array([[0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12]] * 2),
        # document starts at column 6 (inside the first 8-token block) and
        # continues across the block boundary at 8; value must be 6, not 5.
        np.array([[0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]] * 2),
        np.arange(test_t)[None, :].repeat(2, axis=0),
    ]

    gen = torch.Generator(device="cpu")
    gen.manual_seed(4242)
    q0 = torch.randn((2, 2, test_t, 4), generator=gen, dtype=torch.float64)
    v0 = torch.randn((2, test_t, 3), generator=gen, dtype=torch.float64)

    worst = 0.0
    for seg_np in layouts:
        seg = torch.from_numpy(seg_np.astype(np.int64))
        same = seg[:, :, None] == seg[:, None, :]
        strict = torch.ones((test_t, test_t), dtype=torch.bool).tril(diagonal=-1)
        allowed = (same & strict.unsqueeze(0)).unsqueeze(1)
        probe = torch.randn((2, 2, test_t, 3), generator=gen, dtype=torch.float64)

        def grads(fn):
            q = q0.clone().requires_grad_(True)
            v = v0.clone().requires_grad_(True)
            vh = v.unsqueeze(1).expand(-1, q.shape[1], -1, -1)
            out = fn(q, vh, seg, block)
            gq, gv = torch.autograd.grad((out * probe).sum(), (q, v))
            return out.detach(), gq, gv

        q = q0.clone().requires_grad_(True)
        v = v0.clone().requires_grad_(True)
        scores = q @ q.transpose(-1, -2)
        scores = scores.masked_fill(~allowed, 0.0)
        vh = v.unsqueeze(1).expand(-1, 2, -1, -1)
        dense = scores @ vh
        gq, gv = torch.autograd.grad((dense * probe).sum(), (q, v))
        ref = (dense.detach(), gq, gv)

        for name, fn in (
            ("certified", scan_chunkwise_certified),
            ("candidate", lambda qq, vv, ss, bb: scan_chunkwise_candidate(qq, vv, ss, bb, zero_carry=True)),
        ):
            got = grads(fn)
            for a, c in zip(got, ref):
                d = float((a - c).abs().max())
                worst = max(worst, d)
                torch.testing.assert_close(a, c, rtol=1e-10, atol=1e-10)

    return {"status": "PASS", "worst_abs_error": worst, "layouts": len(layouts)}


def tiny_model_gate():
    """Tiny same-state packed full-model check: certified vs candidate stack.

    Exercises the real production path (OptArmA.forward_packed), so the
    direct-paper layout and cached-RoPE changes are covered as well as the
    branch-free/zero-carry scan. One tiny multi-document case with a
    document starting inside the first block and continuing across the
    block boundary; compares logits, loss, and every parameter gradient.
    """
    block = 8
    test_t = 16
    dev = torch.device("cpu")
    test_cfg = ArmAConfig(T=test_t, V=64, D=16, N=64, H=2, L=2, HIDDEN=32)

    layout = np.array(
        [
            [0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12],
        ],
        dtype=np.int64,
    )
    seg = torch.from_numpy(layout).to(dev)
    pos = (
        torch.arange(test_t, dtype=torch.int64).unsqueeze(0).expand(2, -1) - seg
    ).to(torch.int32)
    segpos = pos.clone()
    same = seg[:, :, None] == seg[:, None, :]
    strict = torch.ones((test_t, test_t), dtype=torch.bool).tril(diagonal=-1)
    full_mask = (same & strict.unsqueeze(0)).to(dev)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(31337)
    x = torch.randint(0, test_cfg.V, (2, test_t), generator=gen, dtype=torch.long).to(dev)
    y = torch.randint(0, test_cfg.V, (2, test_t), generator=gen, dtype=torch.long).to(dev)
    valid = torch.ones((2, test_t), dtype=torch.bool, device=dev)

    init = canonical_init(test_cfg)
    mapping = {
        "embedding.weight": "embedding",
        "encoder": "encoder",
        "decoder_x": "decoder_x",
        "decoder_y": "decoder_y",
        "readout": "readout",
        "coordinator.Wc": "coord_Wc",
        "coordinator.bc": "coord_bc",
        "coordinator.alpha": "coord_alpha",
        "writer.W1": "writer_W1",
        "writer.W2": "writer_W2",
    }

    def run(variant):
        model = OptArmA(test_cfg, dev, scan_block=block, variant=variant).to(dev)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(init[mapping[name]].to(dev))
        model.train()
        model.zero_grad(set_to_none=True)
        logits = model.forward_packed(x, pos, segpos, full_mask, seg)
        loss = ce_sum(logits, y, valid, test_cfg.V) / int(valid.sum().item())
        loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
        return logits.detach(), loss.detach(), grads

    ref_logits, ref_loss, ref_grads = run("certified")
    cand_logits, cand_loss, cand_grads = run("candidate")

    logit_err = float((cand_logits - ref_logits).abs().max())
    loss_err = float((cand_loss - ref_loss).abs().max())
    grad_err = max(
        float((cand_grads[n] - ref_grads[n]).abs().max()) for n in ref_grads
    )

    torch.testing.assert_close(cand_logits, ref_logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(cand_loss, ref_loss, rtol=1e-5, atol=1e-6)
    for name in ref_grads:
        torch.testing.assert_close(cand_grads[name], ref_grads[name], rtol=1e-5, atol=1e-6)

    return {
        "status": "PASS",
        "logit_max_abs_error": logit_err,
        "loss_abs_error": loss_err,
        "grad_max_abs_error": grad_err,
        "parameters_compared": len(ref_grads),
    }


# ---------------------------------------------------------------------------
# Robustness gates (merged from the standalone preflight): exactness,
# determinism, checkpoint/resume, graph breaks, and a corpus smoke train.
# The certified-vs-candidate benchmark only runs when no gate fails.
# ---------------------------------------------------------------------------

GATES = []


def gate(name, passed=None, skipped=False, **detail):
    status = "SKIP" if skipped else ("PASS" if passed else "FAIL")
    GATES.append({"name": name, "status": status, "detail": detail})
    print(
        "GATE " + json.dumps(GATES[-1], sort_keys=True, default=str),
        flush=True,
    )


def _packed_batch(cfg, device, layout, seed=0):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    b, t = len(layout), len(layout[0])
    seg = torch.tensor(layout, dtype=torch.long, device=device)
    pos = (
        torch.arange(t, dtype=torch.int64).unsqueeze(0).expand(b, -1).to(device) - seg
    ).to(torch.int32)
    segpos = pos.clone()
    same = seg[:, :, None] == seg[:, None, :]
    strict = torch.ones((t, t), dtype=torch.bool, device=device).tril(diagonal=-1)
    full_mask = same & strict.unsqueeze(0)
    x = torch.randint(0, cfg.V, (b, t), generator=gen).to(device)
    y = torch.randint(0, cfg.V, (b, t), generator=gen).to(device)
    valid = torch.ones((b, t), dtype=torch.bool, device=device)
    return {
        "x": x, "y": y, "pos": pos, "segpos": segpos,
        "valid": valid, "full_mask": full_mask, "segment_start": seg,
    }


def _model_forward_backward(model, entry, batch, bf16, vocab):
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                        cache_enabled=False, enabled=bf16):
        logits = entry(
            batch["x"], batch["pos"], batch["segpos"],
            batch["full_mask"], batch["segment_start"],
        )
        loss = ce_sum(logits, batch["y"], batch["valid"], vocab) / int(
            batch["valid"].sum().item()
        )
    loss.backward()
    grads = {
        n: (p.grad.detach().clone() if p.grad is not None else None)
        for n, p in model.named_parameters()
    }
    return logits.detach(), loss.detach(), grads


def gate_model_equivalence():
    """Certified vs candidate: fp32/bf16 x eager/compiled, packed+single."""
    if not torch.cuda.is_available():
        gate("model_equiv_*", skipped=True, reason="no CUDA")
        return
    cfg = ArmAConfig(T=64, V=256, D=32, N=128, H=2, L=2, HIDDEN=64)
    block = 32
    layouts = [
        [0] * 32 + [32] * 32,          # doc starts at block boundary
        [0] * 6 + [6] * 58,            # doc starts inside block 0
        list(range(64)),                # single-token documents
        [0] * 64,                       # single document
        [0] * 40 + [40] * 24,           # doc starts inside block 1
    ]
    init = canonical_init(cfg)
    for bf16 in (False, True):
        for compiled in (False, True):
            name = (
                f"model_equiv_packed_{'bf16' if bf16 else 'fp32'}_"
                f"{'compiled' if compiled else 'eager'}"
            )
            worst = {"logit": 0.0, "loss": 0.0, "grad": 0.0}
            try:
                for li, layout in enumerate(layouts):
                    batch = _packed_batch(cfg, DEVICE, [layout] * 2, seed=3 + li)
                    torch.manual_seed(cfg.SEED)
                    torch.cuda.manual_seed_all(cfg.SEED)
                    cert = OptArmA(cfg, DEVICE, scan_block=block,
                                   variant="certified").to(DEVICE)
                    load_init(cert, init)
                    cert.train()
                    cand = OptArmA(cfg, DEVICE, scan_block=block,
                                   variant="candidate").to(DEVICE)
                    load_init(cand, init)
                    cand.train()
                    cert_entry = (
                        torch.compile(cert.forward_packed, mode="default")
                        if compiled else cert.forward_packed
                    )
                    cand_entry = (
                        torch.compile(cand.forward_packed, mode="default")
                        if compiled else cand.forward_packed
                    )
                    r_logits, r_loss, r_grads = _model_forward_backward(
                        cert, cert_entry, batch, bf16, cfg.V
                    )
                    c_logits, c_loss, c_grads = _model_forward_backward(
                        cand, cand_entry, batch, bf16, cfg.V
                    )
                    worst["logit"] = max(
                        worst["logit"],
                        float((c_logits.float() - r_logits.float()).abs().max()),
                    )
                    worst["loss"] = max(
                        worst["loss"],
                        float((c_loss.float() - r_loss.float()).abs().max()),
                    )
                    worst["grad"] = max(
                        worst["grad"],
                        max(
                            float((c_grads[n].float() - r_grads[n].float()).abs().max())
                            for n in r_grads
                        ),
                    )
                    rtol, atol = (5e-2, 5e-2) if bf16 else (1e-5, 1e-6)
                    torch.testing.assert_close(
                        c_logits.float(), r_logits.float(), rtol=rtol, atol=atol
                    )
                    torch.testing.assert_close(
                        c_loss.float(), r_loss.float(), rtol=rtol, atol=atol
                    )
                    for n in r_grads:
                        torch.testing.assert_close(
                            c_grads[n].float(), r_grads[n].float(),
                            rtol=rtol, atol=atol,
                        )
                    del cert, cand
                    gc.collect()
                    torch.cuda.empty_cache()
                gate(name, True, layouts=len(layouts), **worst)
            except Exception as exc:
                gate(name, False, error=f"{type(exc).__name__}: {str(exc)[:400]}",
                     **worst)


def gate_graph_breaks_g4():
    if not torch.cuda.is_available():
        gate("graph_breaks_candidate", skipped=True, reason="no CUDA")
        return
    try:
        cfg = CFG
        model = OptArmA(cfg, DEVICE, scan_block=1024, variant="candidate").to(DEVICE)
        load_init(model, INIT)
        model.train()
        batch = _packed_batch(cfg, DEVICE, [[0] * (cfg.T // 2) + [cfg.T // 2] * (cfg.T // 2)])
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        exp = torch._dynamo.explain(model.forward_packed)(
            batch["x"], batch["pos"], batch["segpos"],
            batch["full_mask"], batch["segment_start"],
        )
        breaks = int(getattr(exp, "graph_break_count", -1))
        gate(
            "graph_breaks_candidate",
            breaks == 0,
            graph_break_count=breaks,
            graph_count=int(getattr(exp, "graph_count", -1)),
            break_reasons=[str(r)[:300] for r in getattr(exp, "break_reasons", [])],
            context="production T/block, B=1 packed",
        )
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as exc:
        gate("graph_breaks_candidate", False,
             error=f"{type(exc).__name__}: {str(exc)[:400]}")


def gate_determinism():
    """Repeated backward and repeated full updates on identical inputs."""
    if not torch.cuda.is_available():
        gate("determinism_repeat_backward", skipped=True, reason="no CUDA")
        gate("determinism_full_update_eager", skipped=True, reason="no CUDA")
        gate("determinism_full_update_compiled", skipped=True, reason="no CUDA")
        return
    cfg = CFG
    init = INIT
    layout = [0] * (cfg.T // 2) + [cfg.T // 2] * (cfg.T // 2)
    batch = _packed_batch(cfg, DEVICE, [layout], seed=7)

    # repeat backward: same model, same batch, warmup + 3 compared passes
    try:
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        model = OptArmA(cfg, DEVICE, scan_block=1024, variant="candidate").to(DEVICE)
        load_init(model, init)
        model.train()
        entry = torch.compile(model.forward_packed, mode="default")
        _model_forward_backward(model, entry, batch, True, cfg.V)  # warmup
        passes = [_model_forward_backward(model, entry, batch, True, cfg.V)
                  for _ in range(3)]
        diffs = []
        for i in range(len(passes)):
            for j in range(i + 1, len(passes)):
                for n in passes[i][2]:
                    diffs.append((
                        float((passes[i][2][n] - passes[j][2][n]).abs().max()), n
                    ))
        grad_d, worst_param = max(diffs)
        loss_d = max(
            abs(passes[i][1].item() - passes[j][1].item())
            for i in range(len(passes)) for j in range(i + 1, len(passes))
        )
        gate(
            "determinism_repeat_backward",
            grad_d <= 1e-5 and loss_d <= 1e-7,
            grad_max_abs_diff=grad_d,
            worst_parameter=worst_param,
            loss_abs_diff=loss_d,
            bitwise=(grad_d == 0.0 and loss_d == 0.0),
            tolerance=1e-5,
        )
        del model, entry
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as exc:
        gate("determinism_repeat_backward", False,
             error=f"{type(exc).__name__}: {str(exc)[:400]}")

    # full updates: two fresh models, same init/data, one AdamW step each
    def full_update_params(compiled):
        torch.manual_seed(cfg.SEED)
        torch.cuda.manual_seed_all(cfg.SEED)
        if compiled and hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        model = OptArmA(cfg, DEVICE, scan_block=1024, variant="candidate").to(DEVICE)
        load_init(model, init)
        model.train()
        optimizer = make_optimizer(model)
        for group in optimizer.param_groups:
            group["lr"] = PEAK_LR
        entry = (
            torch.compile(model.forward_packed, mode="default")
            if compiled else model.forward_packed
        )
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            cache_enabled=False):
            logits = entry(
                batch["x"], batch["pos"], batch["segpos"],
                batch["full_mask"], batch["segment_start"],
            )
            loss = ce_sum(logits, batch["y"], batch["valid"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()
        params = {n: p.detach().clone() for n, p in model.named_parameters()}
        del model, optimizer, entry
        gc.collect()
        torch.cuda.empty_cache()
        return params

    for compiled in (False, True):
        name = f"determinism_full_update_{'compiled' if compiled else 'eager'}"
        try:
            a = full_update_params(compiled)
            b = full_update_params(compiled)
            d = max(float((a[n] - b[n]).abs().max()) for n in a)
            tol = 1e-6 if compiled else 0.0
            gate(name, d <= tol, param_max_abs_diff=d, tolerance=tol,
                 bitwise=(d == 0.0))
        except Exception as exc:
            gate(name, False, error=f"{type(exc).__name__}: {str(exc)[:400]}")


def gate_checkpoint_resume():
    if not torch.cuda.is_available():
        gate("checkpoint_resume_equivalence", skipped=True, reason="no CUDA")
        return
    cfg = CFG
    first = cfg.T // 3
    layout = [0] * first + [first] * (cfg.T - first)
    batch = _packed_batch(cfg, DEVICE, [layout], seed=20)
    path = "/content/opt3c_all_preflight_ckpt.pt" if os.path.isdir("/content") \
        else "/tmp/opt3c_all_preflight_ckpt.pt"

    def make():
        torch.manual_seed(cfg.SEED)
        torch.cuda.manual_seed_all(cfg.SEED)
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        model = OptArmA(cfg, DEVICE, scan_block=1024, variant="candidate").to(DEVICE)
        load_init(model, INIT)
        model.train()
        optimizer = make_optimizer(model)
        entry = torch.compile(model.forward_packed, mode="default")
        return model, optimizer, entry

    def update(model, optimizer, entry, step_index):
        for group in optimizer.param_groups:
            group["lr"] = lr_for_update(step_index + 1)
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            cache_enabled=False):
            logits = entry(
                batch["x"], batch["pos"], batch["segpos"],
                batch["full_mask"], batch["segment_start"],
            )
            loss = ce_sum(logits, batch["y"], batch["valid"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()
        return float(loss.detach())

    try:
        model, optimizer, entry = make()
        update(model, optimizer, entry, 0)
        update(model, optimizer, entry, 1)
        torch.save(
            {
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "step": 2,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            },
            path,
        )
        loss_ref = update(model, optimizer, entry, 2)
        params_ref = {n: p.detach().clone() for n, p in model.named_parameters()}
        del model, optimizer, entry
        gc.collect()
        torch.cuda.empty_cache()

        model2, optimizer2, entry2 = make()
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model2.load_state_dict(ckpt["model"])
        optimizer2.load_state_dict(ckpt["opt"])
        loss_res = update(model2, optimizer2, entry2, 2)
        params_res = {n: p.detach().clone() for n, p in model2.named_parameters()}
        param_d = max(
            float((params_ref[n] - params_res[n]).abs().max()) for n in params_ref
        )
        loss_d = abs(loss_ref - loss_res)
        gate(
            "checkpoint_resume_equivalence",
            param_d <= 1e-6 and loss_d <= 1e-5,
            param_max_abs_diff=param_d,
            loss_max_abs_diff=loss_d,
            steps_before=2,
            steps_after=1,
        )
        del model2, optimizer2, entry2
        gc.collect()
        torch.cuda.empty_cache()
        try:
            os.unlink(path)
        except OSError:
            pass
    except Exception as exc:
        gate("checkpoint_resume_equivalence", False,
             error=f"{type(exc).__name__}: {str(exc)[:400]}")


def _smoke_update(cpu, model, entry, optimizer, update_index):
    row0 = update_index * GLOBAL_BATCH
    denom = int(cpu["valid"][row0:row0 + GLOBAL_BATCH].sum().item())
    require(denom > 0, f"smoke update {update_index} has no valid targets")
    for group in optimizer.param_groups:
        group["lr"] = lr_for_update(update_index + 1)
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for lo in range(row0, row0 + GLOBAL_BATCH, 16):
        hi = lo + 16
        x = cpu["x"][lo:hi].to(DEVICE, dtype=torch.long, non_blocking=True)
        y = cpu["y"][lo:hi].to(DEVICE, dtype=torch.long, non_blocking=True)
        pos = cpu["pos"][lo:hi].to(DEVICE, dtype=torch.int32, non_blocking=True)
        valid = cpu["valid"][lo:hi].to(DEVICE, dtype=torch.bool, non_blocking=True)
        input_valid = cpu["input_valid"][lo:hi].to(
            DEVICE, dtype=torch.bool, non_blocking=True
        )
        start = cpu["start"][lo:hi].to(DEVICE, dtype=torch.int32, non_blocking=True)
        segpos = cpu["segpos"][lo:hi].to(DEVICE, dtype=torch.int32, non_blocking=True)
        full_mask = (
            (start[:, :, None] == start[:, None, :])
            & input_valid[:, :, None]
            & input_valid[:, None, :]
            & CAUSAL.unsqueeze(0)
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            cache_enabled=False):
            logits = entry(x, pos, segpos, full_mask, start)
            loss = ce_sum(logits, y, valid) / denom
        loss.backward()
        total += float(loss.detach())
        del x, y, pos, valid, input_valid, start, segpos, full_mask, logits, loss
    for n, p in model.named_parameters():
        if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
            raise RuntimeError(f"non-finite gradient: {n}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
    optimizer.step()
    return total


def gate_smoke_train(cpu, valid_pairs):
    if not torch.cuda.is_available():
        gate("smoke_train_b16x4", skipped=True, reason="no CUDA")
        return
    try:
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        model = OptArmA(CFG, DEVICE, scan_block=1024, variant="candidate").to(DEVICE)
        load_init(model, INIT)
        model.train()
        optimizer = make_optimizer(model)
        entry = torch.compile(model.forward_packed, mode="default")
        torch.cuda.reset_peak_memory_stats(DEVICE)
        losses = []
        allocs = []
        for i in range(2):
            loss = _smoke_update(cpu, model, entry, optimizer, i)
            torch.cuda.synchronize(DEVICE)
            losses.append(loss)
            allocs.append(torch.cuda.memory_allocated(DEVICE))
        growth = (allocs[-1] - allocs[0]) / max(1, allocs[0])
        peak = torch.cuda.max_memory_allocated(DEVICE)
        total = torch.cuda.get_device_properties(0).total_memory
        finite = all(math.isfinite(l) for l in losses)
        gate(
            "smoke_train_b16x4",
            finite and growth <= 0.05 and peak < 0.95 * total,
            losses=losses,
            microbatch=16,
            gradient_accumulation_steps=4,
            data="frozen_packed_corpus",
            alloc_growth_fraction=growth,
            peak_alloc_GiB=peak / 2**30,
            total_mem_GiB=total / 2**30,
        )
        del model, optimizer, entry
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as exc:
        gate("smoke_train_b16x4", False,
             error=f"{type(exc).__name__}: {str(exc)[:500]}")


def main():
    # 1) scan oracle (dense fp64 vs certified/candidate)
    try:
        g = tiny_packed_correctness()
        gate("scan_oracle_dense_fp64", g["status"] == "PASS",
             worst_abs_error=g["worst_abs_error"], layouts=g["layouts"])
    except Exception as exc:
        gate("scan_oracle_dense_fp64", False,
             error=f"{type(exc).__name__}: {str(exc)[:400]}")

    # 2) tiny full-model gate (certified vs candidate, CPU fp32)
    try:
        mg = tiny_model_gate()
        gate("model_gate_cpu_fp32", mg["status"] == "PASS",
             logit_max_abs_error=mg["logit_max_abs_error"],
             grad_max_abs_error=mg["grad_max_abs_error"],
             parameters_compared=mg["parameters_compared"])
    except Exception as exc:
        gate("model_gate_cpu_fp32", False,
             error=f"{type(exc).__name__}: {str(exc)[:400]}")

    # 3) GPU equivalence: fp32/bf16 x eager/compiled x packed/single
    gate_model_equivalence()

    # 4) compiled graph hygiene
    gate_graph_breaks_g4()

    # 5) determinism
    gate_determinism()

    # 6) checkpoint save/resume
    gate_checkpoint_resume()

    # 7) frozen packed corpus + smoke train (B16x4, finite loss/grads)
    packed_cpu, packed_valid_pairs, packed_metadata = reconstruct_packed_batches()
    gate_smoke_train(packed_cpu, packed_valid_pairs)

    failed = [g["name"] for g in GATES if g["status"] == "FAIL"]
    print("FAILED_GATES=" + json.dumps(failed))
    print("OPT3C_ALL_ROBUST=" + str(not failed).lower())
    if failed:
        print("G4_CONFIRMATION_SKIPPED=true")
        return

    # 8) certified vs candidate packed benchmark (only when all gates pass)
    jobs = ["certified", "candidate"]
    order_rng = random.Random(20260915)
    order_rng.shuffle(jobs)

    results = {}
    oom_attempts = []

    for run_order, variant in enumerate(jobs, 1):
        print(f"G4_CONFIRM[{run_order}/2]={variant}")
        try:
            measured = run_treatment(variant, packed_cpu, packed_valid_pairs, run_order)
        except Exception as exc:
            if not is_genuine_cuda_oom(exc):
                raise
            oom = {
                "variant": variant,
                "run_order": run_order,
                "exception": f"{type(exc).__name__}: {str(exc)[:1200]}",
            }
            oom_attempts.append(oom)
            results[variant] = {"status": "OOM", **oom}
            print("G4_OOM=" + json.dumps(oom, sort_keys=True, separators=(",", ":")))
            del exc
            gc.collect()
            torch.cuda.empty_cache()
            continue

        results[variant] = measured
        print(
            "G4_MEASURED="
            + json.dumps(measured, sort_keys=True, separators=(",", ":"))
        )

    certified = results.get("certified")
    candidate = results.get("candidate")
    require(
        certified is not None and certified.get("status") == "MEASURED",
        "Certified treatment did not measure.",
    )
    require(
        candidate is not None and candidate.get("status") == "MEASURED",
        "Candidate treatment did not measure.",
    )

    speedup = certified["median_ms"] / candidate["median_ms"]
    beats = bool(candidate["input_tok_s"] > CERTIFIED_TOK_S)
    beats_baseline_repro = bool(candidate["median_ms"] < certified["median_ms"])

    print("CERTIFIED_TOK_S=" + f"{certified['input_tok_s']:.10g}")
    print("CERTIFIED_MS=" + f"{certified['median_ms']:.10g}")
    print("CERTIFIED_GIB=" + f"{certified['peak_allocated_GiB']:.10g}")
    print("CANDIDATE_TOK_S=" + f"{candidate['input_tok_s']:.10g}")
    print("CANDIDATE_MS=" + f"{candidate['median_ms']:.10g}")
    print("CANDIDATE_GIB=" + f"{candidate['peak_allocated_GiB']:.10g}")
    print("CANDIDATE_FLAGS=" + json.dumps(CANDIDATE_FLAGS, sort_keys=True, separators=(",", ":")))
    print("CANDIDATE_SPEEDUP_VS_CERTIFIED=" + f"{speedup:.10g}")
    print("CANDIDATE_BEATS_SESSION_BASELINE=" + str(beats_baseline_repro).lower())
    print("CANDIDATE_BEATS_CERTIFIED_ANCHOR_69183=" + str(beats).lower())
    print("PACKED_METADATA=" + json.dumps(packed_metadata, sort_keys=True, separators=(",", ":")))
    print("OOM_ATTEMPTS=" + json.dumps(oom_attempts, sort_keys=True, separators=(",", ":")))
    print("G4_CONFIRMATION_COMPLETE=true")


# ARM_A_GATE_ONLY=1 is used by the local host-side validator to import the
# correctness gates without launching the GPU benchmark. Colab runs it bare.
if os.environ.get("ARM_A_GATE_ONLY") != "1":
    main()
