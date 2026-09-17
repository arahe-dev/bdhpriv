"""Per-band permutation-null intervals for the 8/16-band frequency profiles.

Augments results/arm_a_science/frequency_null.json with a
"band_null_intervals" section:
  * pooled-over-levels/heads per-band 2.5/50/97.5 percentile null intervals;
  * per-(level,head) significance of band structure under the SAME
    pair-preserving random band assignment ensemble (5000 permutations).

A smooth frequency curve alone is not evidence; this quantifies it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import KEYS, RAW_DIR, RESULTS_DIR, band_slices  # noqa: E402

N_PERM = 5000
SEED = 20260916


def band_sums(v: np.ndarray, nb: int) -> np.ndarray:
    edges = band_slices(v.size * 2, nb)
    return np.array([v[lo // 2:hi // 2].sum() for lo, hi in edges])


def spearman_vec(ranks_c: np.ndarray, ref_c: np.ndarray) -> np.ndarray:
    """Spearman between each row of `ranks_c` values and centered ref."""
    return (ranks_c * ref_c).sum(-1) / np.sqrt(
        (ranks_c * ranks_c).sum(-1) * (ref_c * ref_c).sum())


def main():
    path = RESULTS_DIR / "frequency_null.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    for ckpt in data["checkpoints"]:
        p1 = np.load(RAW_DIR / f"pass1_{ckpt}.npz")
        K2 = p1["pair_mass_sum_x"].shape[-1]
        perm = np.stack([rng.permutation(K2) for _ in range(N_PERM)])
        entry = {"per_cell": {}}
        for key in KEYS:
            pm = p1[f"pair_mass_sum_{key}"].sum((0, 1, 2))
            pp = p1[f"pair_pos_count_{key}"].sum((0, 1, 2)).astype(np.float64)
            key_entry = {}
            for nb in (8, 16):
                obs_m = band_sums(pm, nb)
                obs_o = band_sums(pp, nb)
                obs_m_share = obs_m / obs_m.sum()
                obs_o_share = obs_o / obs_o.sum()
                null_m = np.empty((N_PERM, nb))
                null_o = np.empty((N_PERM, nb))
                for pi in range(N_PERM):
                    idx = perm[pi]
                    va = np.empty_like(pm)
                    va[idx] = pm
                    vb = np.empty_like(pp)
                    vb[idx] = pp
                    bm = band_sums(va, nb)
                    bo = band_sums(vb, nb)
                    null_m[pi] = bm / bm.sum()
                    null_o[pi] = bo / bo.sum()
                key_entry[f"bands{nb}"] = {
                    "observed_mass_share": obs_m_share.tolist(),
                    "mass_share_null_p2.5": np.quantile(null_m, 0.025,
                                                        axis=0).tolist(),
                    "mass_share_null_p50": np.median(null_m, axis=0).tolist(),
                    "mass_share_null_p97.5": np.quantile(null_m, 0.975,
                                                         axis=0).tolist(),
                    "observed_occupancy_share": obs_o_share.tolist(),
                    "occupancy_share_null_p2.5": np.quantile(null_o, 0.025,
                                                             axis=0).tolist(),
                    "occupancy_share_null_p97.5": np.quantile(null_o, 0.975,
                                                              axis=0).tolist(),
                    "n_perm": N_PERM,
                }

            # per-(level,head) significance under the same permutation set
            pmc = p1[f"pair_mass_sum_{key}"].sum(0)      # (L,H,K2)
            L_, H_, _ = pmc.shape
            width = K2 // 8
            obs_cell = pmc.reshape(L_, H_, 8, width).sum(-1)
            obs_share = obs_cell / obs_cell.sum(-1, keepdims=True)
            ref = np.arange(8) - 3.5
            ref_c = ref - ref.mean()
            rho_obs = spearman_vec(obs_share, ref_c)
            null_max = np.empty((N_PERM, L_, H_))
            null_rho = np.empty((N_PERM, L_, H_))
            for pi in range(N_PERM):
                idx = perm[pi]
                va = pmc[:, :, idx]
                bs = va.reshape(L_, H_, 8, width).sum(-1)
                sh = bs / bs.sum(-1, keepdims=True)
                null_max[pi] = sh.max(-1)
                ranks = np.argsort(np.argsort(sh, axis=-1), axis=-1)
                ranks = ranks.astype(np.float64)
                ranks -= ranks.mean(-1, keepdims=True)
                null_rho[pi] = spearman_vec(ranks, ref_c)
            cells = {}
            for l in range(L_):
                for h in range(H_):
                    nm = null_max[:, l, h]
                    nr = null_rho[:, l, h]
                    sd = nm.std()
                    cells[f"L{l}_H{h}"] = {
                        "observed_max_share": float(obs_share[l, h].max()),
                        "null_mean_max_share": float(nm.mean()),
                        "max_share_z": float(
                            (obs_share[l, h].max() - nm.mean()) / sd)
                        if sd > 0 else None,
                        "observed_band_mass_share": obs_share[l, h].tolist(),
                        "rho_freq_vs_mass_share_observed": float(rho_obs[l, h]),
                        "rho_null_mean": float(nr.mean()),
                        "rho_z": float(
                            (rho_obs[l, h] - nr.mean()) /
                            max(nr.std(), 1e-12)),
                    }
            entry["per_cell"][key] = cells
            entry[key] = key_entry
        data["checkpoints"][ckpt]["band_null_intervals"] = entry
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "checkpoints": list(
        data["checkpoints"])}, indent=2))


if __name__ == "__main__":
    main()
