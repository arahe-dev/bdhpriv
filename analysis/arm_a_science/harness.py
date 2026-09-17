"""CPU analysis harness for the Arm-A sparse-population science mission.

Read-only imports of the frozen `opt/` implementation. Everything this module
does is CPU + RAM only; no GPU calls, no modification of opt/, systems/, or
the one-hour optimization code.

Evidence discipline:
  * Synthetic packed batches (random tokens) are E1/E2 diagnostics, never E3.
  * Every sampler and seed is recorded in the emitted metadata.
"""

from __future__ import annotations

import ctypes
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opt.census_model import CensusArmA  # noqa: E402
from opt.model_ref import ArmAConfig, synthetic_packed_batch  # noqa: E402

CKPT_DIR = ROOT / "runs" / "arm_a_2p5b_opt3c_all"
CHECKPOINTS = {
    "step2000": CKPT_DIR / "census_ckpts" / "step_0000002000.pt",
    "step18000": CKPT_DIR / "ckpt" / "step_0000018000.pt",
    "step19000": CKPT_DIR / "ckpt" / "step_0000019000.pt",
    "latest": CKPT_DIR / "ckpt" / "latest.pt",
}
CHECKPOINT_STEPS = {
    "step2000": 2000,
    "step18000": 18000,
    "step19000": 19000,
    "latest": 19074,
}

KEYS = ("x", "y", "u")
LADDER = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
BANDS = 16  # highest resolution stored; 8 derived by summing adjacent bands

RANK_PASS_SEED = 20260916

RESULTS_DIR = ROOT / "results" / "arm_a_science"
FIG_DIR = ROOT / "figures" / "arm_a_science"
RAW_DIR = RESULTS_DIR / "raw"


# ---------------------------------------------------------------------------
# RAM accounting (no psutil dependency)
# ---------------------------------------------------------------------------

class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def rss_mb() -> float:
    try:
        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            return float("nan")
        return counters.WorkingSetSize / 1e6
    except Exception:
        return float("nan")


class PeakRam:
    """Samples process working set in a background thread; reports peak."""

    def __init__(self, interval_s: float = 0.25):
        self.interval_s = interval_s
        self.peak_mb = rss_mb()
        self._stop = False
        self._thread = None

    def __enter__(self):
        import threading

        def loop():
            while not self._stop:
                self.peak_mb = max(self.peak_mb, rss_mb())
                time.sleep(self.interval_s)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.peak_mb = max(self.peak_mb, rss_mb())


# ---------------------------------------------------------------------------
# Checkpoints / model
# ---------------------------------------------------------------------------

def load_state(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["model"], ckpt


def build_model(cfg: ArmAConfig, state: dict, threads: int = 12):
    torch.set_num_threads(threads)
    model = CensusArmA(cfg, torch.device("cpu"), scan_block=1024)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BatchSpec:
    """One synthetic packed forward batch of `rows` packed windows."""

    mode: str
    seed: int
    rows: int
    label: str
    mean_doc_len: float = 0.0


MAIN_SPECS = (
    BatchSpec("mixed", 11, 4, "mixed_a"),
    BatchSpec("mixed", 23, 4, "mixed_b"),
    BatchSpec("single", 7, 4, "single_a"),
    BatchSpec("heavy", 5, 4, "heavy_a"),
)

# 16 blocks of 128 tokens per 2048 window; two blocks per row are sampled.
CHUNK = 128
CHUNKS_PER_ROW = 2


@dataclass
class SampleLayout:
    """Deterministic token-level sampling for one BatchSpec.

    Tier A and Tier B are the same sampled tokens; pass 1 stores compact
    per-token statistics (Tier A) and pass 2 computes stability metrics
    in-line on the same tokens. The name `tierB_*` is retained for the
    stability-side usage.
    """

    spec: BatchSpec
    blocks: np.ndarray  # (rows, CHUNKS_PER_ROW) block indices
    tierA_flat: np.ndarray  # (rows * CHUNKS_PER_ROW * CHUNK,) flat row*T+pos
    tierB_flat: np.ndarray  # same ordering as tierA_flat
    tierA_local_pos: np.ndarray  # (S,) offset within chunk
    tierB_local_pos: np.ndarray
    tierA_chunk: np.ndarray  # (S,) chunk ordinal in [0, CHUNKS_PER_ROW)
    tierB_chunk: np.ndarray
    tierA_row: np.ndarray
    tierB_row: np.ndarray

    def to_meta(self) -> dict:
        return {
            "label": self.spec.label,
            "mode": self.spec.mode,
            "seed": self.spec.seed,
            "rows": self.spec.rows,
            "blocks": self.blocks.tolist(),
            "tierA_tokens_per_batch": int(self.tierA_flat.size),
        }


def make_sample_layout(spec: BatchSpec, cfg: ArmAConfig) -> SampleLayout:
    blocks_per_row = cfg.T // CHUNK
    rng = np.random.default_rng(RANK_PASS_SEED + spec.seed * 1000)
    blocks = np.stack(
        [
            rng.choice(blocks_per_row, size=CHUNKS_PER_ROW, replace=False)
            for _ in range(spec.rows)
        ]
    ).astype(np.int64)
    blocks.sort(axis=1)

    rows_a, pos_a, chunk_a = [], [], []
    for r in range(spec.rows):
        for c in range(CHUNKS_PER_ROW):
            base = blocks[r, c] * CHUNK
            for off in range(CHUNK):
                rows_a.append(r)
                pos_a.append(base + off)
                chunk_a.append(c)
    rows_a = np.asarray(rows_a)
    pos_a = np.asarray(pos_a)
    chunk_a = np.asarray(chunk_a)
    flat_a = rows_a * cfg.T + pos_a
    local_a = pos_a - blocks[rows_a, chunk_a] * CHUNK

    return SampleLayout(
        spec=spec,
        blocks=blocks,
        tierA_flat=flat_a,
        tierB_flat=flat_a.copy(),
        tierA_local_pos=local_a,
        tierB_local_pos=local_a.copy(),
        tierA_chunk=chunk_a,
        tierB_chunk=chunk_a.copy(),
        tierA_row=rows_a,
        tierB_row=rows_a.copy(),
    )


def make_batch(spec: BatchSpec, cfg: ArmAConfig):
    if spec.mode == "scaled_mixed":
        return scaled_packed_batch(cfg, spec.rows, "cpu", seed=spec.seed,
                                   mean_doc_len=spec.mean_doc_len)
    return synthetic_packed_batch(cfg, spec.rows, "cpu", seed=spec.seed,
                                  mode=spec.mode)


def _pack_from_cuts(t: int, cuts, rows: int, device):
    seg = np.zeros((rows, t), dtype=np.int64)
    for i in range(rows):
        col = []
        s = 0
        for c in list(cuts) + [t]:
            if c <= s:
                continue
            col += [s] * (c - s)
            s = c
        if len(col) < t:
            col += [s] * (t - len(col))
        seg[i, :len(col)] = np.asarray(col[:t], dtype=np.int64)
    return seg


def scaled_packed_batch(cfg: ArmAConfig, rows: int, device, seed: int = 0,
                        mean_doc_len: float | None = None):
    """Packed windows with exponential doc lengths scaled to `mean_doc_len`.

    Same semantics as opt.model_ref.synthetic_packed_batch but with a
    context-dependent doc-length mean so short-context evaluations (T=256,
    512, 1024) keep a packed multi-document layout instead of degenerating
    to single-document windows.
    """
    import random

    rng = random.Random(seed)
    if mean_doc_len is None:
        mean_doc_len = 0.7 * cfg.T
    mean_doc_len = max(8.0, float(mean_doc_len))
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 999)
    t = cfg.T
    seg = np.zeros((rows, t), dtype=np.int64)
    for i in range(rows):
        cuts = []
        s = 0
        while True:
            ln = max(1, int(rng.expovariate(1.0 / mean_doc_len)))
            s += ln
            if s >= t:
                break
            cuts.append(s)
        col = []
        s = 0
        for c in cuts + [t]:
            col += [s] * (c - s)
            s = c
        seg[i] = np.asarray(col[:t], dtype=np.int64)
    seg_t = torch.as_tensor(seg, dtype=torch.long)
    pos = (torch.arange(t).unsqueeze(0).expand(rows, -1) - seg_t
           ).to(torch.int32)
    segpos = pos.clone()
    same = seg_t[:, :, None] == seg_t[:, None, :]
    strict = torch.ones((t, t), dtype=torch.bool).tril(diagonal=-1)
    full_mask = (same & strict.unsqueeze(0)).contiguous()
    x = torch.randint(0, cfg.V, (rows, t), generator=g)
    y = torch.randint(0, cfg.V, (rows, t), generator=g)
    valid = torch.ones((rows, t), dtype=torch.bool)
    return {
        "x": x.to(device), "y": y.to(device),
        "pos": pos.to(device, dtype=torch.int32),
        "segpos": segpos.to(device, dtype=torch.int32),
        "valid": valid.to(device),
        "full_mask": full_mask.to(device),
        "segment_start": seg_t.to(device, dtype=torch.long),
    }


def repeat_motif_batch(cfg: ArmAConfig, rows: int, device, seed: int = 0,
                       block_len: int = 256, repeats: int = 0,
                       mode: str = "repeat"):
    """Single-document windows with a planted repeated motif.

    mode:
      repeat  : row = tile(base, repeats)[:T]        (repeated content)
      shuffle : same multiset, randomly permuted within the row
                (matched unigram distribution, no repetition)
      random  : independent uniform tokens (reference)
    """
    t = cfg.T
    if repeats <= 0:
        repeats = max(1, t // block_len)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 4242)
    x = torch.zeros((rows, t), dtype=torch.long)
    for i in range(rows):
        base = torch.randint(0, cfg.V, (block_len,), generator=g)
        row = base.repeat(repeats + 1)[:t]
        if mode == "shuffle":
            row = row[torch.randperm(t, generator=g)]
        elif mode == "random":
            row = torch.randint(0, cfg.V, (t,), generator=g)
        x[i] = row
    y = torch.randint(0, cfg.V, (rows, t), generator=g)
    seg = torch.zeros((rows, t), dtype=torch.long)
    pos = torch.arange(t).unsqueeze(0).expand(rows, -1).to(torch.int32)
    segpos = pos.clone()
    strict = torch.ones((t, t), dtype=torch.bool).tril(diagonal=-1)
    full_mask = strict.unsqueeze(0).expand(rows, -1, -1).contiguous()
    valid = torch.ones((rows, t), dtype=torch.bool)
    return {
        "x": x.to(device), "y": y.to(device),
        "pos": pos.to(device, dtype=torch.int32),
        "segpos": segpos.to(device, dtype=torch.int32),
        "valid": valid.to(device),
        "full_mask": full_mask.to(device),
        "segment_start": seg.to(device, dtype=torch.long),
    }


def token_meta(segment_start: np.ndarray, cfg: ArmAConfig) -> dict:
    """Document id / doc start / doc length / doc-relative position per token.

    segment_start: (rows, T) int array of document start positions.
    """
    rows, t = segment_start.shape
    seg = segment_start.astype(np.int64)
    doc_id = np.zeros((rows, t), dtype=np.int64)
    for r in range(rows):
        starts = seg[r]
        new = np.zeros(t, dtype=np.int64)
        new[1:] = starts[1:] != starts[:-1]
        doc_id[r] = np.cumsum(new)
    doc_start = seg
    # doc length: distance to next start (or T)
    doc_len = np.zeros((rows, t), dtype=np.int64)
    for r in range(rows):
        starts = np.unique(seg[r])
        for s in starts:
            idx = np.where(seg[r] == s)[0]
            doc_len[r, idx] = int(idx[-1] - idx[0] + 1)
    rel = np.arange(t, dtype=np.int64)[None, :] - seg
    return {
        "doc_id": doc_id,
        "doc_start": doc_start,
        "doc_len": doc_len,
        "doc_rel_pos": rel,
    }


def band_slices(k: int, bands: int):
    """Contiguous RoPE-pair bands, preserving pairs."""
    pairs = k // 2
    edges = np.linspace(0, pairs, bands + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(bands)]


# ---------------------------------------------------------------------------
# JSON utility
# ---------------------------------------------------------------------------

def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default),
                    encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serializable: {type(o)}")


def ckpt_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def quantiles(values: np.ndarray, qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> dict:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    out = {"n": int(v.size)}
    for q in qs:
        out[f"p{int(q * 100)}"] = float(np.quantile(v, q))
    out["mean"] = float(v.mean())
    out["std"] = float(v.std())
    return out


def bootstrap_ci(values: np.ndarray, stat=np.mean, n_boot: int = 2000,
                 seed: int = 12345, alpha: float = 0.05):
    """Percentile bootstrap CI with a fixed seed."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = stat(v[idx], axis=1)
    return {
        "point": float(stat(v)),
        "lo": float(np.quantile(boots, alpha / 2)),
        "hi": float(np.quantile(boots, 1 - alpha / 2)),
        "n": int(v.size),
        "n_boot": int(n_boot),
    }


def block_bootstrap_ci(values: np.ndarray, blocks: np.ndarray,
                       stat=np.mean, n_boot: int = 2000, seed: int = 12345,
                       alpha: float = 0.05):
    """Bootstrap resampling whole blocks (documents/batches)."""
    v = np.asarray(values, dtype=np.float64)
    b = np.asarray(blocks)
    uniq = np.unique(b)
    rng = np.random.default_rng(seed)
    per_block = {u: v[b == u] for u in uniq}
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        vals = np.concatenate([per_block[u] for u in pick])
        boots[i] = stat(vals)
    return {
        "point": float(stat(v)),
        "lo": float(np.quantile(boots, alpha / 2)),
        "hi": float(np.quantile(boots, 1 - alpha / 2)),
        "n": int(v.size),
        "n_blocks": int(uniq.size),
        "n_boot": int(n_boot),
    }
