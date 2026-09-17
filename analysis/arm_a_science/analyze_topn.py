"""P1 analysis: global vs local top-N concentration, per level/head/key.

Reads the raw pass-1/pass-2 NPZ collected by collect.py and writes
results/arm_a_science/topn_global_local.json.

Definitions (predeclared):
  M_local(N)  : per-token fraction of that token's positive mass captured by
                its OWN top-N coordinates (per level, per head).
  M_global(N) : per-token fraction captured by a FROZEN global top-N ranking
                built from the other batch split (cross-fitted).
  Delta(N)    = M_local(N) - M_global(N), paired per token.
  uniform(N)  = N / K is the uniform-mass reference for both.

Usage:
  py -3.12 analysis/arm_a_science/analyze_topn.py --ckpts latest step2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import LADDER, KEYS, RAW_DIR, RESULTS_DIR, save_json  # noqa: E402


def cell_stats(local: np.ndarray, glob: np.ndarray, totals: np.ndarray,
               blocks: np.ndarray, n_boot: int = 1000,
               seed: int = 11) -> dict:
    """local/glob: (n_tokens, n_ladder). Returns summary dict."""
    valid = totals > 0
    lv = local[valid]
    gv = glob[valid]
    out = {"n_tokens": int(valid.sum()),
           "n_zero_mass_tokens": int((~valid).sum()),
           "local": {}, "global": {}, "delta": {}}
    for i, n in enumerate(LADDER):
        l = lv[:, i].astype(np.float64)
        g = gv[:, i].astype(np.float64)
        d = l - g
        out["local"][str(n)] = _qs(l)
        out["global"][str(n)] = _qs(g)
        out["delta"][str(n)] = _qs(d)
    # document-block bootstrap for Delta at selected N (16, 256, 1024)
    out["delta_bootstrap"] = {}
    rng = np.random.default_rng(seed)
    uniq = np.unique(blocks)
    for n in (16, 256, 1024):
        i = LADDER.index(n)
        d = (lv[:, i] - gv[:, i]).astype(np.float64)
        b = blocks[valid]
        per_block = {u: d[b == u] for u in uniq}
        boots = np.empty(n_boot)
        for k in range(n_boot):
            pick = rng.choice(uniq, size=uniq.size, replace=True)
            vals = np.concatenate([per_block[u] for u in pick])
            boots[k] = vals.mean()
        out["delta_bootstrap"][str(n)] = {
            "mean": float(d.mean()),
            "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975)),
            "n_blocks": int(uniq.size),
        }
    return out


def _qs(v: np.ndarray) -> dict:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    out = {f"p{int(q*100)}": float(np.quantile(v, q)) for q in qs}
    out["mean"] = float(v.mean())
    out["std"] = float(v.std())
    out["n"] = int(v.size)
    return out


def load_ckpt(ckpt: str, raw: Path):
    p1 = np.load(raw / f"pass1_{ckpt}.npz")
    p2 = np.load(raw / f"pass2_{ckpt}.npz")
    return p1, p2


def analyze(ckpt: str, raw: Path) -> dict:
    p1, p2 = load_ckpt(ckpt, raw)
    ladder = list(LADDER)
    L = p1["mass_sum_x"].shape[1]
    H = p1["mass_sum_x"].shape[2]
    K = p1["mass_sum_x"].shape[3]
    B, S = p1["tok_meta"].shape[:2]
    doc_id = p1["tok_meta"][:, :, 2]  # (B, S)
    blocks = (np.arange(B)[:, None] * 100000 + doc_id).reshape(-1)

    out = {
        "checkpoint": ckpt,
        "ladder": ladder,
        "K_per_head": int(K),
        "n_heads": int(H),
        "n_levels": int(L),
        "n_sampled_tokens_per_batch": int(S),
        "n_batches": int(B),
        "definition": {
            "M_local": "per-token own top-N fraction of positive mass",
            "M_global": "per-token mass in cross-fitted global top-N ranking",
            "delta": "paired M_local - M_global",
            "uniform_baseline": "N / K",
        },
        "per_cell": {},
        "pooled": {},
    }

    for key in KEYS:
        local = p1[f"tA_topn_{key}"]  # (B, L, S, H, 9)
        glob = p2[f"g_topn_ratio_{key}"]  # (B, L, S, H, 9)
        totals = p1[f"tA_total_{key}"]  # (B, L, S, H)
        key_out = {"levels": []}
        for level in range(L):
            heads = []
            for h in range(H):
                lo = local[:, level, :, h, :].reshape(-1, len(LADDER))
                gl = glob[:, level, :, h, :].reshape(-1, len(LADDER))
                to = totals[:, level, :, h].reshape(-1)
                heads.append({"head": h, **cell_stats(
                    lo, gl, to, blocks)})
            key_out["levels"].append({"level": level, "heads": heads})
        out["per_cell"][key] = key_out

    for key in KEYS:
        lp = p2[f"l_topn_pooled_{key}"]  # (B, L, S, 9)
        gp = p2[f"g_topn_pooled_{key}"]
        # pooled totals: sum over heads of tA_total
        totals = p1[f"tA_total_{key}"].sum(-1)  # (B, L, S)
        levels = []
        for level in range(L):
            lo = lp[:, level, :, :].reshape(-1, len(LADDER))
            gl = gp[:, level, :, :].reshape(-1, len(LADDER))
            to = totals[:, level, :].reshape(-1)
            levels.append({"level": level, **cell_stats(lo, gl, to, blocks)})
        out["pooled"][key] = {"levels": levels, "N_total": int(K * H)}

    # per-mode breakdown for the primary checkpoint (using batch labels)
    meta_path = raw / f"meta_{ckpt}.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        labels = [s["label"] for s in meta["specs"]]
        out["per_mode"] = {}
        for key in ("x", "u"):
            per_mode = {}
            for bi, label in enumerate(labels):
                local = p1[f"tA_topn_{key}"][bi]  # (L,S,H,9)
                glob = p2[f"g_topn_ratio_{key}"][bi]
                to = p1[f"tA_total_{key}"][bi]
                valid = to > 0
                entry = {}
                for i, n in enumerate(LADDER):
                    l = local[..., i][valid].astype(np.float64)
                    g = glob[..., i][valid].astype(np.float64)
                    entry[str(n)] = {
                        "local_mean": float(l.mean()),
                        "global_mean": float(g.mean()),
                        "delta_mean": float((l - g).mean()),
                        "n": int(l.size),
                    }
                per_mode[label] = entry
            out["per_mode"][key] = per_mode
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", nargs="+", default=["latest"])
    ap.add_argument("--raw", default=str(RAW_DIR))
    ap.add_argument("--out", default=str(
        RESULTS_DIR / "topn_global_local.json"))
    args = ap.parse_args()
    raw = Path(args.raw)
    result = {
        "meta": {
            "analysis": "P1 global-vs-local top-N",
            "evidence_class": "E1/E2 synthetic packed batches, one trajectory",
            "uniform_baseline": "N/K per level-head; N/(H*K) pooled",
            "note": "M_global uses cross-fitted rankings (other batch split)",
        },
        "checkpoints": {},
    }
    for ckpt in args.ckpts:
        try:
            result["checkpoints"][ckpt] = analyze(ckpt, raw)
        except FileNotFoundError as e:
            print(f"skip {ckpt}: {e}")
    save_json(Path(args.out), result)
    print(json.dumps({"out": str(args.out), "checkpoints": list(
        result["checkpoints"])}, indent=2))


if __name__ == "__main__":
    main()
