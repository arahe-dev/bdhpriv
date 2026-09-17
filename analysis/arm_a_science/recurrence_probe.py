"""Structured-recurrence probe: does the fast-conditional tail track content?

The E2 "fast conditional" result came from random-token contexts, where every
token is unrelated to its predecessor. This probe plants repeated content in
the context and asks whether top-N population identity persists across
repetitions (content-driven) or not (context-independent rotation).

Design (predeclared):
  * T=2048, one document per row, base block of 256 tokens repeated 8x.
  * matched control: the SAME token multiset randomly permuted within the row
    (no repetition, identical unigram distribution, identical positions).
  * pair categories (per row, 32 sampled offsets):
      repeat_lag_256/512/1024  aligned positions across repeated copies
      shuffle_lag_256/512/1024 same positions in the shuffled control
      within_lag_1/8/32/64    local pairs inside one copy (both modes)
      cross_row               different rows
  * metrics per (level, head, key): top-64 Jaccard, support Jaccard,
    Spearman rank correlation of full K coordinates.

Primary prediction: if the conditional tail is content-driven, repeat_lag
overlap exceeds shuffle_lag overlap for u (and y); if it is a positional
artifact, they match. Level-0 x is reported but trivially 1.0 (x depends only
on the token embedding at level 0).

Writes results/arm_a_science/semantic_stability.json (synthetic probe; the
frozen-corpus version requires the E3 census).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import (  # noqa: E402
    CHECKPOINTS, RESULTS_DIR, ArmAConfig, build_model, load_state,
    repeat_motif_batch, rss_mb, save_json,
)

T = 2048
BLOCK = 256
REPEATS = T // BLOCK
OFFSETS = 32
MAX_PAIRS = 64
KEYS = ("x", "y", "u")
LAG_COPIES = (1, 2, 4)
WITHIN_LAGS = (1, 8, 32, 64)


def build_pairs(seed: int = 12):
    rng = np.random.default_rng(seed)
    offsets = rng.choice(BLOCK, size=OFFSETS, replace=False)
    rows = 4
    pairs = {"repeat": {}, "shuffle": {}}
    for c in LAG_COPIES:
        lag = c * BLOCK
        for row in range(rows):
            plist = [(row * T + o, row * T + c * BLOCK + o) for o in offsets]
            pairs["repeat"].setdefault(f"repeat_lag_{lag}", []).extend(plist)
            pairs["shuffle"].setdefault(f"shuffle_lag_{lag}", []).extend(plist)
    for mode in ("repeat", "shuffle"):
        for d in WITHIN_LAGS:
            plist = []
            for row in range(rows):
                for o in offsets:
                    if o + d < BLOCK:
                        plist.append((row * T + o, row * T + o + d))
            if len(plist) > MAX_PAIRS:
                sel = np.sort(rng.choice(len(plist), MAX_PAIRS, replace=False))
                plist = [plist[k] for k in sel]
            pairs[mode].setdefault(f"within_lag_{d}", []).extend(plist)
    cross = []
    for row in range(rows - 1):
        for o in offsets:
            cross.append((row * T + o, (row + 1) * T + o))
    pairs["repeat"]["cross_row"] = cross
    return pairs


class PairCapture:
    """Computes pair metrics on predeclared flat-index pairs."""

    def __init__(self, pairs, ladder=(16, 64, 256, 1024)):
        self.pairs = pairs
        self.ladder = np.asarray(ladder)
        self.rows = []
        self.K = None
        self.H = None

    @torch.no_grad()
    def __call__(self, level, x, y, u, segment_start):
        b, t, h, k = x.shape
        self.K, self.H = k, h
        views = {
            "x": x.reshape(b * t, h, k),
            "y": y.permute(0, 2, 1, 3).reshape(b * t, h, k),
            "u": u.reshape(b * t, h, k),
        }
        ar = np.arange(k)
        for key in KEYS:
            vals = views[key].numpy()
            order = np.argsort(-vals, axis=2)          # (N,H,K)
            inv = np.empty_like(order)
            np.put_along_axis(inv, order, np.broadcast_to(ar, order.shape),
                              axis=-1)
            support = vals > 0
            for hh in range(h):
                for cat, plist in self.pairs.items():
                    if not plist:
                        continue
                    t1 = np.asarray([p[0] for p in plist])
                    t2 = np.asarray([p[1] for p in plist])
                    o1 = order[t1, hh]
                    rob = np.take_along_axis(inv[t2, hh], o1, axis=1)
                    m = np.maximum(rob, ar[None, :])
                    P = len(plist)
                    occ = np.zeros((P, k + 1), dtype=np.int64)
                    rows = np.repeat(np.arange(P), k)
                    np.add.at(occ, (rows, m.ravel()), 1)
                    inter = np.cumsum(occ[:, :k], axis=1)
                    i_lad = inter[:, self.ladder - 1]
                    jac = i_lad / (2.0 * self.ladder - i_lad)
                    s1 = support[t1, hh]
                    s2 = support[t2, hh]
                    inter_s = np.logical_and(s1, s2).sum(1)
                    union_s = np.logical_or(s1, s2).sum(1)
                    supj = inter_s / np.maximum(union_s, 1)
                    r1 = inv[t1, hh].astype(np.float64)
                    r2 = inv[t2, hh].astype(np.float64)
                    r1 -= r1.mean(1, keepdims=True)
                    r2 -= r2.mean(1, keepdims=True)
                    den = np.sqrt((r1 * r1).sum(1) * (r2 * r2).sum(1))
                    rho = np.where(den > 0, (r1 * r2).sum(1) /
                                   np.maximum(den, 1e-30), 0.0)
                    self.rows.append({
                        "mode": "repeat" if not cat.startswith("shuffle")
                        else "shuffle",
                        "category": cat, "level": level, "head": hh,
                        "key": key,
                        "jaccard": jac.astype(np.float32),
                        "support_jaccard": supj.astype(np.float32),
                        "spearman": rho.astype(np.float32),
                    })


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="latest", choices=list(CHECKPOINTS))
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    cfg = ArmAConfig()
    state, _ = load_state(CHECKPOINTS[args.ckpt])
    model = build_model(cfg, state, threads=args.threads)
    del state

    pairs = build_pairs()
    batches = {
        mode: repeat_motif_batch(cfg, 4, "cpu", seed=12, block_len=BLOCK,
                                 repeats=REPEATS, mode=mode)
        for mode in ("repeat", "shuffle")
    }

    out = {
        "meta": {
            "design": "T=2048, base block 256 repeated 8x, 4 rows; matched "
                      "shuffled control keeps the unigram multiset and "
                      "positions but removes repetition",
            "offsets_per_row": OFFSETS,
            "max_pairs_per_category": MAX_PAIRS,
            "ladder": [16, 64, 256, 1024],
            "evidence_class": "E2 (synthetic structured contexts); the "
                              "frozen-corpus semantic version requires E3",
            "primary": "u top-64 Jaccard: repeat_lag_* vs shuffle_lag_*",
        },
        "cells": [],
        "seconds": None,
    }
    t0 = time.perf_counter()
    for mode in ("repeat", "shuffle"):
        batch = batches[mode]
        cap = PairCapture(pairs[mode])
        model.begin_forward(cap)
        with torch.no_grad():
            model.forward_packed(batch["x"], batch["pos"], batch["segpos"],
                                 batch["full_mask"], batch["segment_start"])
        for row in cap.rows:
            out["cells"].append({
                "mode": row["mode"], "category": row["category"],
                "level": row["level"], "head": row["head"],
                "key": row["key"],
                "n_pairs": int(row["jaccard"].shape[0]),
                "jaccard16": float(np.mean(row["jaccard"][:, 0])),
                "jaccard64": float(np.mean(row["jaccard"][:, 1])),
                "jaccard256": float(np.mean(row["jaccard"][:, 2])),
                "jaccard1024": float(np.mean(row["jaccard"][:, 3])),
                "support_jaccard": float(np.mean(row["support_jaccard"])),
                "spearman": float(np.mean(row["spearman"])),
            })
    out["seconds"] = time.perf_counter() - t0
    out["rss_mb"] = rss_mb()

    # aggregate: per category/key mode, mean over (level, head)
    agg = {}
    for cell in out["cells"]:
        key = f"{cell['mode']}|{cell['category']}|{cell['key']}"
        a = agg.setdefault(key, {"jac64": [], "jac16": [], "jac256": [],
                                 "support": [], "spearman": [],
                                 "levels": {}})
        a["jac16"].append(cell["jaccard16"])
        a["jac64"].append(cell["jaccard64"])
        a["jac256"].append(cell["jaccard256"])
        a["support"].append(cell["support_jaccard"])
        a["spearman"].append(cell["spearman"])
        a["levels"].setdefault(str(cell["level"]), []).append(
            cell["jaccard64"])
    summary = {}
    for k, a in agg.items():
        summary[k] = {
            "jaccard16_mean": float(np.mean(a["jac16"])),
            "jaccard64_mean": float(np.mean(a["jac64"])),
            "jaccard256_mean": float(np.mean(a["jac256"])),
            "support_mean": float(np.mean(a["support"])),
            "spearman_mean": float(np.mean(a["spearman"])),
            "level_mean_jaccard64": {
                lvl: float(np.mean(v)) for lvl, v in a["levels"].items()},
            "n_cells": len(a["jac64"]),
        }
    out["summary"] = summary
    save_json(RESULTS_DIR / "semantic_stability.json", out)
    # print the primary contrast
    for key in ("u", "y", "x"):
        for lag in (256, 512, 1024):
            r = summary.get(f"repeat|repeat_lag_{lag}|{key}", {})
            s = summary.get(f"shuffle|shuffle_lag_{lag}|{key}", {})
            print("%s lag %4d: repeat jac64=%.3f shuffle=%.3f (n=%d/%d)" % (
                key, lag, r.get("jaccard64_mean", float("nan")),
                s.get("jaccard64_mean", float("nan")),
                r.get("n_cells", 0), s.get("n_cells", 0)))


if __name__ == "__main__":
    main()
