"""P4 analysis: RoPE frequency-band structure + pair-preserving null model.

Reads pass-1 NPZ (per-pair mass/counts over all forward tokens and per-token
band statistics) and writes
  results/arm_a_science/frequency_bands.json
  results/arm_a_science/frequency_null.json

Band definition: RoPE pairs are (k, k+1); pair index i has angular frequency
omega_i = theta^(-2i/K) / (2*pi) with theta = 2^16, so pair 0 is the FASTEST
(shortest wavelength) and pair K/2-1 the slowest. Bands are contiguous pair
ranges; 8-band values are derived by summing adjacent 16-band values.

Null model (predeclared): randomly reassign the K/2 pairs to 16 bands with
the same band sizes (128 pairs each). This preserves each pair's two
coordinates (sine/cosine) and all per-pair statistics; it destroys only the
pair-index <-> band alignment. 5000 permutations, fixed seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import KEYS, RESULTS_DIR, RAW_DIR, save_json, band_slices  # noqa: E402

THETA = 2.0 ** 16
N_PERM = 5000
NULL_SEED = 20260916


def pair_frequencies(K: int) -> np.ndarray:
    i = np.arange(K // 2, dtype=np.float64)
    omega = 1.0 / (THETA ** (2.0 * i / K))
    return omega  # cycles per token (1/wavelength); pair 0 highest


def band_aggregate(values: np.ndarray, bands: int) -> np.ndarray:
    """values (..., K/2) -> (..., bands) sums of contiguous bands."""
    edges = band_slices(values.shape[-1] * 2, bands)  # coordinate slices
    out = np.empty(values.shape[:-1] + (bands,), dtype=np.float64)
    for bi, (lo, hi) in enumerate(edges):
        out[..., bi] = values[..., lo // 2:hi // 2].sum(-1)
    return out


def zscore_stats(obs: float, null: np.ndarray) -> dict:
    sd = float(null.std())
    return {
        "observed": float(obs),
        "null_mean": float(null.mean()),
        "null_sd": sd,
        "z": float((obs - null.mean()) / sd) if sd > 0 else None,
        "percentile": float((null < obs).mean()),
        "p_two_sided": float(2 * min((null <= obs).mean(),
                                     (null >= obs).mean())),
        "n_perm": int(null.size),
    }


def analyze(ckpt: str, raw: Path, n_perm: int = N_PERM) -> tuple[dict, dict]:
    p1 = np.load(raw / f"pass1_{ckpt}.npz")
    meta = json.loads((raw / f"meta_{ckpt}.json").read_text(encoding="utf-8"))
    L = p1["mass_sum_x"].shape[1]
    H = p1["mass_sum_x"].shape[2]
    K = p1["mass_sum_x"].shape[3]
    n_tokens = (p1["mass_sum_x"].shape[0] * meta["specs"][0]["rows"]
                * meta["model_config"]["T"])
    freq = pair_frequencies(K)
    rng = np.random.default_rng(NULL_SEED)
    perm = np.stack([rng.permutation(K // 2) for _ in range(n_perm)])

    observed = {"pairs_per_band_16": K // 2 // 16, "L": int(L), "H": int(H),
                "n_tokens": int(n_tokens)}
    nulls = {"statistics": {}, "note": "null = random pair->band assignment "
             "with equal band sizes; preserves pairs and per-pair values"}
    for key in ("x", "u"):
        pm = p1[f"pair_mass_sum_{key}"].sum(0)      # (L,H,K/2)
        pp = p1[f"pair_pos_count_{key}"].sum(0)     # (L,H,K/2)
        mem = None
        if f"tA_top64_idx_{key}" in p1.files:
            idx = p1[f"tA_top64_idx_{key}"]  # (B,L,S,H,64)
            Bt, Lt, St, Ht, _ = idx.shape
            pair_of = (idx.astype(np.int64) // 2)
            mem = np.empty((Lt, Ht, K // 2), dtype=np.float64)
            for l in range(Lt):
                for h in range(Ht):
                    counts = np.bincount(pair_of[:, l, :, h, :].ravel(),
                                         minlength=K // 2)
                    mem[l, h] = counts / (Bt * St * 64.0)
        key_out = {"levels": []}
        for level in range(L):
            heads = []
            for h in range(H):
                mass = pm[level, h]
                occup = pp[level, h] / float(n_tokens)
                entry = {"head": h}
                for nb in (8, 16):
                    bmass = band_aggregate(mass, nb)
                    bocc = band_aggregate(occup, nb)
                    bpos = band_aggregate(pp[level, h].astype(np.float64), nb)
                    entry[f"bands{nb}"] = {
                        "mass_share": (bmass / max(bmass.sum(), 1e-30)
                                       ).tolist(),
                        "occupancy": (bocc / (K // 2 // nb)).tolist(),
                        "mean_positive_mass": (bmass /
                                               np.maximum(bpos, 1e-30)
                                               ).tolist(),
                    }
                    if mem is not None:
                        bmem = band_aggregate(mem[level, h], nb)
                        entry[f"bands{nb}"]["top64_membership_share"] = \
                            (bmem / max(bmem.sum(), 1e-30)).tolist()
                    # frequency at band centers (cycles/token)
                    edges = band_slices(K, nb)
                    centers = [float(freq[(lo // 2 + hi // 2) // 2])
                               for lo, hi in edges]
                    entry[f"bands{nb}"]["band_center_frequency"] = centers
                heads.append(entry)
            key_out["levels"].append({"level": level, "heads": heads})
        observed[key] = key_out

        # ---- null model on aggregate pair statistics (all levels pooled) --
        mass_pool = pm.sum((0, 1))                    # (K/2,)
        occ_pool = pp.sum((0, 1)).astype(np.float64)  # (K/2,) activations
        mem_pool = mem.sum((0, 1)) if mem is not None else None
        stats = {}
        for nb in (8, 16):
            bmass = band_aggregate(mass_pool, nb)
            share = bmass / bmass.sum()
            bocc = band_aggregate(occ_pool.astype(np.float64), nb)
            occ_share = bocc / max(bocc.sum(), 1e-30)
            # trend: Spearman between band center frequency and share
            edges = band_slices(K, nb)
            centers = np.array([freq[(lo // 2 + hi // 2) // 2]
                                for lo, hi in edges])
            r_share = _spearman(centers, share)
            r_occ = _spearman(centers, occ_share)
            null_maxshare = np.empty(n_perm)
            null_r_share = np.empty(n_perm)
            null_r_occ = np.empty(n_perm)
            null_entropy = np.empty(n_perm)
            width = K // 2 // nb
            for pi in range(n_perm):
                p = perm[pi]
                permuted = np.empty_like(mass_pool)
                permuted[p] = mass_pool
                bs = band_aggregate(permuted, nb)
                sh = bs / bs.sum()
                null_maxshare[pi] = sh.max()
                null_entropy[pi] = _entropy(sh)
                null_r_share[pi] = _spearman(centers, sh)
                permuted_o = np.empty_like(occ_pool)
                permuted_o[p] = occ_pool
                osh = band_aggregate(permuted_o.astype(np.float64), nb)
                osh = osh / osh.sum()
                null_r_occ[pi] = _spearman(centers, osh)
            stats[f"bands{nb}"] = {
                "max_mass_share": zscore_stats(float(share.max()),
                                               null_maxshare),
                "share_entropy": zscore_stats(float(_entropy(share)),
                                              null_entropy),
                "spearman_freq_vs_mass_share": zscore_stats(
                    float(r_share), null_r_share),
                "spearman_freq_vs_occupancy_share": zscore_stats(
                    float(r_occ), null_r_occ),
                "observed_mass_share": share.tolist(),
                "observed_occupancy_share": occ_share.tolist(),
                "band_center_frequency": centers.tolist(),
                "band_center_wavelength_tokens": (1.0 / centers).tolist(),
            }
        # per-token band mass: split into slow/fast halves
        tb = p1[f"tA_band_mass_{key}"]  # (B,L,S,H,16)
        tot_tok = np.maximum(tb.sum(-1, keepdims=True), 1e-30)
        tok_share = tb / tot_tok
        pop_share16 = tb.sum(axis=(0, 1, 2, 3)) / max(tb.sum(), 1e-30)
        mean_share16 = tok_share.mean(axis=(0, 1, 2, 3))
        argmax16 = tb.argmax(-1)
        frac_argmax16 = [(argmax16 == b).mean() for b in range(16)]
        cv16 = tok_share.std(axis=(0, 1, 2, 3)) / np.maximum(
            mean_share16, 1e-30)
        # 8-band: sum adjacent band masses
        tb8 = tb.reshape(*tb.shape[:-1], 8, 2).sum(-1)
        tot8 = np.maximum(tb8.sum(-1, keepdims=True), 1e-30)
        share8 = tb8 / tot8
        pop8 = tb8.sum(axis=(0, 1, 2, 3)) / max(tb8.sum(), 1e-30)
        mean8 = share8.mean(axis=(0, 1, 2, 3))
        argmax8 = tb8.argmax(-1)
        stats["per_token_band_structure"] = {
            "bands16": {
                "population_mass_share": pop_share16.tolist(),
                "mean_per_token_share": mean_share16.tolist(),
                "cv_per_token_share": cv16.tolist(),
                "fraction_tokens_band_is_argmax": frac_argmax16,
            },
            "bands8": {
                "population_mass_share": pop8.tolist(),
                "mean_per_token_share": mean8.tolist(),
                "fraction_tokens_band_is_argmax": [
                    float((argmax8 == b).mean()) for b in range(8)],
            },
        }
        band_freqs16 = np.array(
            [freq[(lo // 2 + hi // 2) // 2] for lo, hi in band_slices(K, 16)])
        fast = np.where(band_freqs16 >= np.median(band_freqs16))[0]
        slow = np.where(band_freqs16 < np.median(band_freqs16))[0]
        tot = np.maximum(tb.sum(-1), 1e-30)
        stats["per_token_slow_fast"] = {
            "slow_band_indices": slow.tolist(),
            "fast_band_indices": fast.tolist(),
            "slow_mass_fraction_mean": float(
                (tb[..., slow].sum(-1) / tot).mean()),
            "fast_mass_fraction_mean": float(
                (tb[..., fast].sum(-1) / tot).mean()),
            "slow_occupancy_mean": float(
                p1[f"tA_band_occ_{key}"][..., slow].mean()),
            "fast_occupancy_mean": float(
                p1[f"tA_band_occ_{key}"][..., fast].mean()),
        }
        nulls["statistics"][key] = stats
    return observed, nulls


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() /
                 np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", nargs="+", default=["latest"])
    ap.add_argument("--raw", default=str(RAW_DIR))
    ap.add_argument("--perm", type=int, default=N_PERM)
    ap.add_argument("--out-bands", default=str(RESULTS_DIR / "frequency_bands.json"))
    ap.add_argument("--out-null", default=str(RESULTS_DIR / "frequency_null.json"))
    args = ap.parse_args()
    raw = Path(args.raw)
    bands = {"meta": {
        "evidence_class": "E1/E2 synthetic packed batches, one trajectory",
        "theta": THETA, "note": "pair 0 = fastest; band index increases "
                                "towards lower frequencies",
    }, "checkpoints": {}}
    nulls = {"checkpoints": {}}
    for ckpt in args.ckpts:
        try:
            obs, nul = analyze(ckpt, raw, args.perm)
            bands["checkpoints"][ckpt] = obs
            nulls["checkpoints"][ckpt] = nul
        except FileNotFoundError as e:
            print("skip", ckpt, e)
    save_json(Path(args.out_bands), bands)
    save_json(Path(args.out_null), nulls)
    print(json.dumps({"ok": True}, indent=2))


if __name__ == "__main__":
    main()
