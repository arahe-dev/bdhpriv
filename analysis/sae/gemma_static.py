"""Gemma Scope 2 270M MLP-out layer-12 SAE static structural analysis (CPU).

Analyzes the downloaded small-L0 (20) and medium-L0 (60) `jump_relu` SAEs.
Weights/thresholds are structural reference data only: they do not reveal
actual firing sparsity without activation data.

Usage:
  py -3.12 analysis/sae/gemma_static.py --which small
  py -3.12 analysis/sae/gemma_static.py --which medium
  py -3.12 analysis/sae/gemma_static.py --compare
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ASSET = ROOT / "data" / "sae" / "gemma-scope-2-270m-pt" / "mlp_out"
OUT = ROOT / "results" / "sae"


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64).ravel())
    if x.size == 0 or x[-1] <= 0:
        return float("nan")
    n = x.size
    i = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (i * x).sum()) / (n * x.sum()) - (n + 1.0) / n)


def lorenz_top_share(x: np.ndarray, fracs=(0.001, 0.01, 0.05, 0.1, 0.25,
                                           0.5)) -> dict:
    v = np.sort(np.asarray(x, dtype=np.float64).ravel())[::-1]
    total = v.sum()
    out = {}
    n = v.size
    for f in fracs:
        c = max(1, int(round(f * n)))
        out[str(f)] = float(v[:c].sum() / total) if total > 0 else float("nan")
    return out


def entropic_effective(x: np.ndarray) -> float:
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[v > 0]
    if v.size == 0:
        return float("nan")
    p = v / v.sum()
    h = -(p * np.log(p + 1e-300)).sum()
    return float(np.exp(h))


def participation_ratio(eigs: np.ndarray) -> float:
    v = np.asarray(eigs, dtype=np.float64)
    v = v[v > 0]
    return float(v.sum() ** 2 / (v * v).sum())


def quantile_table(x: np.ndarray) -> dict:
    v = np.asarray(x, dtype=np.float64).ravel()
    qs = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
    return {f"p{q*100:g}": float(np.quantile(v, q)) for q in qs} | {
        "mean": float(v.mean()), "std": float(v.std()),
        "min": float(v.min()), "max": float(v.max()), "n": int(v.size),
    }


def load(path: Path) -> dict:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k).float().numpy() for k in f.keys()}


def spike_stats(w_enc, w_dec, threshold, b_enc, b_dec):
    we = w_enc.T  # (16384, 640) encoder directions (feature-major)
    wd = w_dec  # (16384, 640) decoder directions
    enc_norm = np.linalg.norm(we, axis=1)
    dec_norm = np.linalg.norm(wd, axis=1)
    cos_align = (we * wd).sum(1) / np.maximum(enc_norm * dec_norm, 1e-30)
    return enc_norm, dec_norm, cos_align


def cosine_chunk_stats(w: np.ndarray, chunk: int = 512, topk: int = 11,
                       n_random_pairs: int = 200000, seed: int = 7):
    """NN cosine stats (chunked) + random-pair cosine distribution."""
    x = w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-30)
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    ii = rng.integers(0, n, size=n_random_pairs)
    jj = rng.integers(0, n, size=n_random_pairs)
    keep = ii != jj
    rnd = (x[ii[keep]] * x[jj[keep]]).sum(1)
    nn1 = np.empty(n, dtype=np.float32)
    nn10 = np.empty((n, topk - 1), dtype=np.float32)
    nn1_idx = np.empty(n, dtype=np.int32)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        sims = x[lo:hi] @ x.T
        sims[np.arange(hi - lo), np.arange(lo, hi)] = -np.inf
        part = np.partition(sims, -(topk - 1), axis=1)[:, -(topk - 1):]
        part = np.sort(part, axis=1)[:, ::-1]
        nn10[lo:hi] = part[:, :topk - 1]
        nn1[lo:hi] = part[:, 0]
        nn1_idx[lo:hi] = np.argmax(sims, axis=1)
    mnn = float((nn1_idx[nn1_idx] == np.arange(n)).mean())
    return {
        "random_pair_cosine": quantile_table(rnd),
        "nn1_top1": quantile_table(nn1),
        "nn_top10_quantiles": quantile_table(nn10.ravel()),
        "nn1_gt_0.9": float((nn1 > 0.9).mean()),
        "nn1_gt_0.7": float((nn1 > 0.7).mean()),
        "nn1_gt_0.5": float((nn1 > 0.5).mean()),
        "mutual_nn_rate_top1": mnn,
        "n": int(n),
    }


def spectrum_stats(m: np.ndarray) -> dict:
    g = m @ m.T if m.shape[0] <= m.shape[1] else m.T @ m
    eigs = np.linalg.eigvalsh(g)[::-1]
    eigs = np.clip(eigs, 0, None)
    return {
        "singular_values_sq_top20": eigs[:20].tolist(),
        "effective_rank_entropy": entropic_effective(eigs),
        "participation_ratio": participation_ratio(eigs),
        "top1_energy_share": float(eigs[0] / eigs.sum()),
        "top10_energy_share": float(eigs[:10].sum() / eigs.sum()),
        "top1pct_energy_share": float(
            eigs[:max(1, int(0.01 * eigs.size))].sum() / eigs.sum()),
        "rank": int((eigs > 1e-12 * eigs[0]).sum()),
    }


def analyze(which: str) -> dict:
    t0 = time.perf_counter()
    d = ASSET / f"layer_12_width_16k_l0_{which}"
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    params = load(d / "params.safetensors")
    w_enc = params["w_enc"]  # (640, 16384)
    w_dec = params["w_dec"]  # (16384, 640)
    b_enc = params["b_enc"]  # (16384,)
    b_dec = params["b_dec"]  # (640,)
    threshold = params["threshold"]  # (16384,)
    enc_norm, dec_norm, cos_align = spike_stats(w_enc, w_dec, threshold,
                                                b_enc, b_dec)

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        ra -= ra.mean()
        rb -= rb.mean()
        return float((ra * rb).sum() /
                     np.sqrt((ra * ra).sum() * (rb * rb).sum()))

    out = {
        "config": cfg,
        "shapes": {
            "w_enc": list(w_enc.shape), "w_dec": list(w_dec.shape),
            "b_enc": list(b_enc.shape), "b_dec": list(b_dec.shape),
            "threshold": list(threshold.shape),
        },
        "norms": {
            "encoder_direction": quantile_table(enc_norm),
            "decoder_direction": quantile_table(dec_norm),
            "b_enc_abs": quantile_table(np.abs(b_enc)),
            "b_dec": quantile_table(b_dec),
            "lorenz_top_share_decoder_norm": lorenz_top_share(dec_norm),
            "lorenz_top_share_encoder_norm": lorenz_top_share(enc_norm),
            "gini_decoder_norm": gini(dec_norm),
            "gini_encoder_norm": gini(enc_norm),
            "effective_number_decoder_norm": entropic_effective(dec_norm),
            "effective_number_encoder_norm": entropic_effective(enc_norm),
        },
        "threshold": {
            "quantiles": quantile_table(threshold),
            "fraction_gt_0": float((threshold > 0).mean()),
            "fraction_eq_0": float((threshold == 0).mean()),
            "gini": gini(threshold),
        },
        "relations": {
            "spearman_threshold_vs_encoder_norm": spearman(
                threshold, enc_norm),
            "spearman_threshold_vs_decoder_norm": spearman(
                threshold, dec_norm),
            "spearman_threshold_vs_b_enc": spearman(threshold, b_enc),
            "spearman_encoder_vs_decoder_norm": spearman(enc_norm, dec_norm),
            "encoder_decoder_cosine_align": quantile_table(cos_align),
        },
        "spectrum": {
            "encoder": spectrum_stats(w_enc),
            "decoder": spectrum_stats(w_dec),
        },
        "seconds": time.perf_counter() - t0,
    }
    # structure files are large; run NN analyses last (they are the slow part)
    out["decoder_similarity"] = cosine_chunk_stats(w_dec)
    out["encoder_similarity"] = cosine_chunk_stats(w_enc.T)
    out["seconds"] = time.perf_counter() - t0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", choices=["small", "medium", "compare"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.which in ("small", "medium"):
        res = analyze(args.which)
        OUT.joinpath(f"gemma_{args.which}.json").write_text(
            json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps({"which": args.which,
                          "seconds": res["seconds"],
                          "nn1_decoder": res["decoder_similarity"]["nn1_top1"],
                          "effective_rank_enc":
                              res["spectrum"]["encoder"][
                                  "effective_rank_entropy"]}, indent=2))
    else:
        # compare small vs medium on shared structural statistics
        comparison = {}
        for which in ("small", "medium"):
            p = OUT / f"gemma_{which}.json"
            if not p.exists():
                raise SystemExit(f"run --which {which} first")
            comparison[which] = json.loads(p.read_text(encoding="utf-8"))
        out = {"note": "normalized comparison; raw values live per-SAE files"}
        diff = {}
        for key in ("nn1_gt_0.9", "nn1_gt_0.7", "mutual_nn_rate_top1"):
            a = comparison["small"]["decoder_similarity"][key]
            b = comparison["medium"]["decoder_similarity"][key]
            diff[f"decoder_{key}"] = {"small": a, "medium": b, "ratio": b / a
                                      if a else None}
        for key, path in (
                ("gini_decoder_norm", ("norms", "gini_decoder_norm")),
                ("gini_encoder_norm", ("norms", "gini_encoder_norm")),
                ("thr_mean", ("threshold", "quantiles", "mean")),
                ("eff_rank_enc", ("spectrum", "encoder",
                                  "effective_rank_entropy")),
                ("eff_rank_dec", ("spectrum", "decoder",
                                  "effective_rank_entropy")),
                ("pr_dec", ("spectrum", "decoder", "participation_ratio")),
                ("enc_dec_cos_median", ("relations",
                                        "encoder_decoder_cosine_align",
                                        "p50"))):
            a = comparison["small"]
            b = comparison["medium"]
            for k in path:
                a = a[k]
                b = b[k]
            diff[key] = {"small": a, "medium": b,
                         "ratio": (b / a) if a else None}
        out["small_vs_medium"] = diff
        OUT.joinpath("gemma_comparison.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2)[:4000])


if __name__ == "__main__":
    main()
