"""Pass-1 and pass-2 activation capture for the Arm-A population analysis.

Pass 1 (population statistics):
  * per-batch/per-level/per-head coordinate mass sums and positive counts over
    ALL tokens of each forward batch (global ranking source);
  * per-batch RoPE-pair mass sums / positive counts, and band aggregates;
  * per-sampled-token (Tier A) totals, positive counts, local top-N mass
    ratios on the ladder, top-64 coordinate identity sets for x and u, and
    per-token band mass / pair occupancy.

Pass 2 (cross-fitted global / core-conditioned statistics):
  * masses captured by a FROZEN global top-N ranking built from the OTHER
    batch split (cross-fitting by batch parity);
  * core-conditioned mass fractions and residual (tail) top-N ratios for
    predeclared core definitions;
  * in-line population-stability metrics on Tier B tokens (Jaccard ladder,
    weighted Jaccard, RBO, Spearman, lagged support overlap).

No file below `opt/` is modified. All tensors are CPU float32.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from harness import BANDS, CHUNK, KEYS, LADDER, band_slices


def _pair_mass(v: torch.Tensor) -> torch.Tensor:
    """(S,H,K) -> (S,H,K/2) sums over RoPE coordinate pairs."""
    s, h, k = v.shape
    return v.reshape(s, h, k // 2, 2).sum(-1)


def _pair_any(v: torch.Tensor) -> torch.Tensor:
    s, h, k = v.shape
    return v.reshape(s, h, k // 2, 2).amax(-1) > 0


def _stack_batches(per_call: List[np.ndarray], n_batches: int, n_levels: int):
    """per_call is ordered (batch, level). Returns (B, L, S, ...)."""
    arr = np.stack(per_call, axis=0)
    return arr.reshape(n_batches, n_levels, *arr.shape[1:])


class Pass1Capture:
    """Accumulates pass-1 statistics across forward batches."""

    def __init__(self, cfg, n_batches: int, sA: List[int]):
        self.cfg = cfg
        self.n_batches = n_batches
        self.sA = sA
        self.L = cfg.L
        self.H = cfg.H
        self.K = cfg.K
        self.band_slices = band_slices(self.K, BANDS)
        self.batch_id = 0
        self.flatA = None

        self.mass_sum = {k: np.zeros((n_batches, self.L, self.H, self.K),
                                     dtype=np.float64) for k in KEYS}
        self.pos_count = {k: np.zeros((n_batches, self.L, self.H, self.K),
                                      dtype=np.int64) for k in KEYS}
        self.pair_mass_sum = {k: np.zeros(
            (n_batches, self.L, self.H, self.K // 2), dtype=np.float64)
            for k in KEYS}
        self.pair_pos_count = {k: np.zeros(
            (n_batches, self.L, self.H, self.K // 2), dtype=np.int64)
            for k in KEYS}
        self.band_mass_sum = {k: np.zeros(
            (n_batches, self.L, self.H, BANDS), dtype=np.float64)
            for k in KEYS}
        self.band_pair_pos = {k: np.zeros(
            (n_batches, self.L, self.H, BANDS), dtype=np.float64)
            for k in KEYS}

        self.total = {k: [] for k in KEYS}
        self.pos = {k: [] for k in KEYS}
        self.topn_ratio = {k: [] for k in KEYS}
        self.top64_idx = {k: [] for k in ("x", "u")}
        self.top64_val = {k: [] for k in ("x", "u")}
        self.band_mass = {k: [] for k in KEYS}
        self.band_occ = {k: [] for k in KEYS}

    def begin_batch(self, batch_id: int, flat_a: np.ndarray):
        self.batch_id = batch_id
        self.flatA = torch.as_tensor(flat_a)

    @torch.no_grad()
    def __call__(self, level: int, x, y, u, segment_start):
        # device-safe: all statistics are computed on CPU (no-op on CPU runs)
        x = x.detach().cpu()
        y = y.detach().cpu()
        u = u.detach().cpu()
        b, t, h, k = x.shape
        xf = x.reshape(b * t, h, k)
        yf = y.permute(0, 2, 1, 3).reshape(b * t, h, k)
        uf = u.reshape(b * t, h, k)
        views = {"x": xf, "y": yf, "u": uf}

        for key in KEYS:
            v_all = views[key]
            self.mass_sum[key][self.batch_id, level] += \
                v_all.sum(0, dtype=torch.float64).numpy()
            self.pos_count[key][self.batch_id, level] += \
                (v_all > 0).sum(0).numpy()
            pm = _pair_mass(v_all).sum(0, dtype=torch.float64).numpy()
            pp = _pair_any(v_all).sum(0).numpy().astype(np.float64)
            self.pair_mass_sum[key][self.batch_id, level] += pm
            self.pair_pos_count[key][self.batch_id, level] += \
                pp.astype(np.int64)
            for bi, (lo, hi) in enumerate(self.band_slices):
                self.band_mass_sum[key][self.batch_id, level, :, bi] += \
                    pm[:, lo:hi].sum(1)
                self.band_pair_pos[key][self.batch_id, level, :, bi] += \
                    pp[:, lo:hi].sum(1)

            v = views[key][self.flatA]
            v64 = v.double()
            total = v64.sum(-1)
            self.total[key].append(total.float().numpy())
            self.pos[key].append((v > 0).sum(-1).int().numpy())
            order = v64.sort(dim=-1, descending=True).values
            csum = order.cumsum(-1)
            denom = total.clamp_min(1e-30).unsqueeze(-1)
            ratios = np.empty((*v.shape[:2], len(LADDER)), dtype=np.float32)
            for i, n in enumerate(LADDER):
                ratios[..., i] = (csum[..., n - 1:n] / denom
                                  ).squeeze(-1).float().numpy()
            self.topn_ratio[key].append(ratios)

            pm_tok = _pair_mass(v)
            occ_tok = _pair_any(v)
            bm = np.empty((*v.shape[:2], BANDS), dtype=np.float32)
            bo = np.empty((*v.shape[:2], BANDS), dtype=np.float32)
            for bi, (lo, hi) in enumerate(self.band_slices):
                bm[..., bi] = pm_tok[..., lo:hi].sum(-1).float().numpy()
                bo[..., bi] = occ_tok[..., lo:hi].float().mean(-1).numpy()
            self.band_mass[key].append(bm)
            self.band_occ[key].append(bo)

            if key in ("x", "u"):
                tk = v.topk(64, dim=-1)
                self.top64_idx[key].append(tk.indices.numpy().astype(np.int16))
                self.top64_val[key].append(tk.values.numpy().astype(np.float16))

    def finalize(self) -> dict:
        out = {
            "n_batches": self.n_batches,
            "mass_sum": {k: v for k, v in self.mass_sum.items()},
            "pos_count": {k: v for k, v in self.pos_count.items()},
            "pair_mass_sum": self.pair_mass_sum,
            "pair_pos_count": self.pair_pos_count,
            "band_mass_sum": self.band_mass_sum,
            "band_pair_pos": self.band_pair_pos,
        }
        for k in KEYS:
            out[f"tA_total_{k}"] = _stack_batches(
                self.total[k], self.n_batches, self.L)
            out[f"tA_pos_{k}"] = _stack_batches(
                self.pos[k], self.n_batches, self.L)
            out[f"tA_topn_{k}"] = _stack_batches(
                self.topn_ratio[k], self.n_batches, self.L)
            out[f"tA_band_mass_{k}"] = _stack_batches(
                self.band_mass[k], self.n_batches, self.L)
            out[f"tA_band_occ_{k}"] = _stack_batches(
                self.band_occ[k], self.n_batches, self.L)
        for k in ("x", "u"):
            out[f"tA_top64_idx_{k}"] = _stack_batches(
                self.top64_idx[k], self.n_batches, self.L)
            out[f"tA_top64_val_{k}"] = _stack_batches(
                self.top64_val[k], self.n_batches, self.L)
        return out


class Pass2Capture:
    """Cross-fitted global-top-N, core/tail metrics, Tier-B stability."""

    def __init__(self, cfg, rankings, rankings_pooled, core_defs, tierB_pairs,
                 n_batches, detail=True, stability=True,
                 res_qs=(0.5, 0.8, 0.9)):
        self.cfg = cfg
        self.rankings = rankings
        self.rankings_pooled = rankings_pooled
        self.core_defs = core_defs
        self.tierB_pairs = tierB_pairs
        self.n_batches = n_batches
        self.detail = detail
        self.stability = stability
        self.res_qs = tuple(res_qs)
        self.batch_id = 0
        self.L = cfg.L
        self.H = cfg.H
        self.K = cfg.K

        self.g_topn_ratio = {k: [] for k in KEYS}
        self.g_total = {k: [] for k in KEYS}
        self.g_pos = {k: [] for k in KEYS}
        self.g_topn_pooled = {k: [] for k in KEYS}
        self.l_topn_pooled = {k: [] for k in KEYS}
        self.core_frac = {name: {k: [] for k in ("x", "u")}
                          for name in core_defs}
        self.core_pos = {name: {k: [] for k in ("x", "u")}
                         for name in core_defs}
        self.res_ratio = {name: {k: [] for k in ("x", "u")}
                          for name in core_defs}
        self.res_need = {name: {k: [] for k in ("x", "u")}
                         for name in core_defs}
        self.stab = [] if stability else None
        self.timers = {"global_gather": 0.0, "core": 0.0, "pooled": 0.0,
                       "stability": 0.0}

    def begin_batch(self, batch_id: int, flat_a: np.ndarray, flat_b: np.ndarray):
        self.batch_id = batch_id
        self.flatA = torch.as_tensor(flat_a)
        self.flatB = torch.as_tensor(flat_b)
        self.split = "A" if batch_id % 2 == 0 else "B"
        self.other = "B" if self.split == "A" else "A"

    @torch.no_grad()
    def __call__(self, level: int, x, y, u, segment_start):
        import time as _time
        # device-safe: all statistics are computed on CPU (no-op on CPU runs)
        x = x.detach().cpu()
        y = y.detach().cpu()
        u = u.detach().cpu()
        b, t, h, k = x.shape
        xf = x.reshape(b * t, h, k)
        yf = y.permute(0, 2, 1, 3).reshape(b * t, h, k)
        uf = u.reshape(b * t, h, k)
        views = {"x": xf, "y": yf, "u": uf}

        _t0 = _time.perf_counter()
        for key in KEYS:
            v = views[key][self.flatA]
            v64 = v.double()
            total = v64.sum(-1)
            self.g_total[key].append(total.float().numpy())
            self.g_pos[key].append((v > 0).sum(-1).int().numpy())
            rank = self.rankings[key][level][self.other]
            rank_t = torch.as_tensor(rank, dtype=torch.long)
            gv = torch.gather(v64, 2, rank_t.unsqueeze(0).expand(
                *v.shape[:2], -1))
            csum = gv.cumsum(2)
            denom = total.clamp_min(1e-30).unsqueeze(-1)
            ratios = np.empty((*v.shape[:2], len(LADDER)), dtype=np.float32)
            for i, n in enumerate(LADDER):
                ratios[..., i] = (csum[..., n - 1:n] / denom
                                  ).squeeze(-1).float().numpy()
            self.g_topn_ratio[key].append(ratios)
            del gv, csum
        self.timers["global_gather"] += _time.perf_counter() - _t0

        # pooled (all heads, N = H*K) local/global top-N per level
        _t0 = _time.perf_counter()
        for key in KEYS:
            vs = views[key][self.flatA]
            vp = vs.double().reshape(vs.shape[0], -1)
            total = vp.sum(-1)
            denom = total.clamp_min(1e-30).unsqueeze(-1)
            order = vp.sort(dim=-1, descending=True).values
            csum = order.cumsum(-1)
            ratios = np.empty((vp.shape[0], len(LADDER)), dtype=np.float32)
            for i, n in enumerate(LADDER):
                ratios[:, i] = (csum[:, n - 1:n] / denom
                                ).squeeze(-1).float().numpy()
            self.l_topn_pooled[key].append(ratios)
            del order, csum
            rank = self.rankings_pooled[key][level][self.other]
            rank_t = torch.as_tensor(rank, dtype=torch.long)
            gv = vp[:, rank_t]
            csum = gv.cumsum(-1)
            ratios = np.empty((vp.shape[0], len(LADDER)), dtype=np.float32)
            for i, n in enumerate(LADDER):
                ratios[:, i] = (csum[:, n - 1:n] / denom
                                ).squeeze(-1).float().numpy()
            self.g_topn_pooled[key].append(ratios)
            del vp, gv, csum
        self.timers["pooled"] += _time.perf_counter() - _t0

        _t0 = _time.perf_counter()
        if self.detail:
            for name in self.core_defs:
                for key in ("x", "u"):
                    mask = self.core_defs[name][key][level][self.other]
                    v = views[key][self.flatA].double()
                    mt = torch.as_tensor(mask)
                    core_vals = torch.where(mt.unsqueeze(0), v,
                                            torch.zeros_like(v))
                    total = v.sum(-1).clamp_min(1e-30)
                    self.core_frac[name][key].append(
                        (core_vals.sum(-1) / total).float().numpy())
                    self.core_pos[name][key].append(
                        (core_vals > 0).sum(-1).int().numpy())
                    res_vals = torch.where(mt.unsqueeze(0), torch.zeros_like(v),
                                           v)
                    res_total = res_vals.sum(-1).clamp_min(1e-30)
                    order = res_vals.sort(dim=-1, descending=True).values
                    csum = order.cumsum(-1)
                    ratios = np.empty((*v.shape[:2], len(LADDER)),
                                      dtype=np.float32)
                    for i, n in enumerate(LADDER):
                        ratios[..., i] = (csum[..., n - 1:n] /
                                          res_total.unsqueeze(-1)
                                          ).squeeze(-1).float().numpy()
                    self.res_ratio[name][key].append(ratios)
                    need = np.empty((*v.shape[:2], len(self.res_qs)),
                                    dtype=np.int32)
                    for qi, q in enumerate(self.res_qs):
                        cnt = (csum < q * res_total.unsqueeze(-1)).sum(-1) + 1
                        need[..., qi] = cnt.clamp_max(self.K).int().numpy()
                    self.res_need[name][key].append(need)
                    del core_vals, res_vals, order, csum
        self.timers["core"] += _time.perf_counter() - _t0

        _t0 = _time.perf_counter()
        if self.stability and self.batch_id < len(self.tierB_pairs):
            self._tierB_stability(level, views)
        self.timers["stability"] += _time.perf_counter() - _t0

    def _tierB_stability(self, level: int, views):
        pairs = self.tierB_pairs[self.batch_id]
        ar = np.arange(self.K)
        ladder = np.asarray(LADDER)
        depth = min(1024, self.K)
        rbo_w = (1 - 0.9) * (0.9 ** np.arange(depth))
        for key in ("x", "u"):
            v = views[key][self.flatB]
            S, H, K = v.shape
            vals = v.double().numpy()
            order = np.argsort(-vals, axis=-1)
            inv = np.empty_like(order)
            np.put_along_axis(inv, order,
                              np.broadcast_to(ar, order.shape), axis=-1)
            support = vals > 0
            for h in range(H):
                for cat, idx_pairs in pairs.items():
                    if not idx_pairs:
                        continue
                    P = len(idx_pairs)
                    t1 = np.asarray([p[0] for p in idx_pairs])
                    t2 = np.asarray([p[1] for p in idx_pairs])
                    o1 = order[t1, h]
                    rob = np.take_along_axis(inv[t2, h], o1, axis=1)
                    m = np.maximum(rob, ar[None, :])
                    occ = np.zeros((P, K + 1), dtype=np.int64)
                    rows = np.repeat(np.arange(P), K)
                    np.add.at(occ, (rows, m.ravel()), 1)
                    inter = np.cumsum(occ[:, :K], axis=1)
                    i_lad = inter[:, ladder - 1]
                    jac = i_lad / (2.0 * ladder - i_lad)
                    s1 = support[t1, h]
                    s2 = support[t2, h]
                    inter_s = np.logical_and(s1, s2).sum(1)
                    union_s = np.logical_or(s1, s2).sum(1)
                    supj = inter_s / np.maximum(union_s, 1)
                    supj[union_s == 0] = 0.0
                    r1 = inv[t1, h].astype(np.float64)
                    r2 = inv[t2, h].astype(np.float64)
                    r1 -= r1.mean(1, keepdims=True)
                    r2 -= r2.mean(1, keepdims=True)
                    den = np.sqrt((r1 * r1).sum(1) * (r2 * r2).sum(1))
                    rho = np.where(den > 0, (r1 * r2).sum(1) /
                                   np.maximum(den, 1e-30), 0.0)
                    rbo = (rbo_w[None, :] * (inter[:, :depth] /
                           np.arange(1, depth + 1)[None, :])).sum(1)
                    self.stab.append({
                        "key": key, "level": level, "head": h,
                        "category": cat,
                        "jaccard": jac.astype(np.float32),
                        "support_jaccard": supj.astype(np.float32),
                        "spearman": rho.astype(np.float32),
                        "rbo": rbo.astype(np.float32),
                    })

    def finalize(self) -> dict:
        out = {
            "g_topn_ratio": {k: _stack_batches(v, self.n_batches, self.L)
                             for k, v in self.g_topn_ratio.items()},
            "g_total": {k: _stack_batches(v, self.n_batches, self.L)
                        for k, v in self.g_total.items()},
            "g_pos": {k: _stack_batches(v, self.n_batches, self.L)
                      for k, v in self.g_pos.items()},
            "g_topn_pooled": {k: _stack_batches(v, self.n_batches, self.L)
                              for k, v in self.g_topn_pooled.items()},
            "l_topn_pooled": {k: _stack_batches(v, self.n_batches, self.L)
                              for k, v in self.l_topn_pooled.items()},
        }
        if self.detail:
            out["core"] = {}
            for name in self.core_defs:
                out["core"][name] = {}
                for key in ("x", "u"):
                    out["core"][name][f"frac_{key}"] = _stack_batches(
                        self.core_frac[name][key], self.n_batches, self.L)
                    out["core"][name][f"cpos_{key}"] = _stack_batches(
                        self.core_pos[name][key], self.n_batches, self.L)
                    out["core"][name][f"res_ratio_{key}"] = _stack_batches(
                        self.res_ratio[name][key], self.n_batches, self.L)
                    out["core"][name][f"res_need_{key}"] = _stack_batches(
                        self.res_need[name][key], self.n_batches, self.L)
        if self.stability:
            out["stability"] = self.stab
        out["timers"] = dict(self.timers)
        return out


def build_lag_pairs(layout, tierB_doc: np.ndarray, tierB_row: np.ndarray,
                    tierB_chunk: np.ndarray, tierB_local: np.ndarray,
                    lags=(1, 2, 4, 8, 16, 32, 63, 127), max_pairs: int = 48,
                    seed: int = 4242):
    """Vectorized Tier-B pair sampler: lag pairs within chunk+document and
    cross-row pairs (different documents). Returns {category: [(i,j), ...]}
    with indices into the Tier-B ordering."""
    sB = tierB_doc.shape[0]
    pairs = {}
    rng = np.random.default_rng(seed + layout.spec.seed)
    # group Tier-B tokens by (row, chunk, doc) preserving local order
    groups = {}
    for i in range(sB):
        groups.setdefault((int(tierB_row[i]), int(tierB_chunk[i]),
                           int(tierB_doc[i])), []).append(i)
    for lag in lags:
        cand = []
        for idxs in groups.values():
            if len(idxs) <= lag:
                continue
            for k in range(len(idxs) - lag):
                cand.append((idxs[k], idxs[k + lag]))
        if len(cand) > max_pairs:
            sel = np.sort(rng.choice(len(cand), size=max_pairs,
                                     replace=False))
            cand = [cand[k] for k in sel]
        pairs[f"lag_{lag}"] = cand
    cross = []
    seen = set()
    attempts = 0
    while len(cross) < max_pairs and attempts < 200000:
        attempts += 1
        i = int(rng.integers(0, sB))
        j = int(rng.integers(0, sB))
        if i == j or tierB_row[i] == tierB_row[j]:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        cross.append(key)
    pairs["cross_row"] = cross

    # long-range same-document pairs between the two sampled chunks, binned by
    # absolute token distance (natural-text repetition can exceed the 128-token
    # chunk), with the cross-document control already above.
    abs_pos = (layout.blocks[tierB_row, tierB_chunk] * CHUNK
               + tierB_local)
    long_bins = {"inter_128_256": (128, 256), "inter_256_512": (256, 512),
                 "inter_512_plus": (512, 10 ** 9)}
    buckets = {k: [] for k in long_bins}
    g0, g1 = {}, {}
    for i in range(sB):
        if tierB_chunk[i] == 0:
            g0.setdefault((int(tierB_row[i]), int(tierB_doc[i])), []).append(i)
        elif tierB_chunk[i] == 1:
            g1.setdefault((int(tierB_row[i]), int(tierB_doc[i])), []).append(i)
    for key, left in g0.items():
        right = g1.get(key)
        if right is None:
            continue
        for i in left:
            for j in right:
                d = int(abs_pos[j] - abs_pos[i])
                for name, (lo, hi) in long_bins.items():
                    if lo <= d < hi:
                        buckets[name].append((i, j))
                        break
    for name, cand in buckets.items():
        if len(cand) > max_pairs:
            sel = np.sort(rng.choice(len(cand), max_pairs, replace=False))
            cand = [cand[k] for k in sel]
        pairs[name] = cand
    return pairs
