"""Trained-checkpoint sparsity census metrics for the sparse/resonant campaign.

Used by opt/run_trained_sparsity_census.py. Accumulates, per level / head /
RoPE-frequency band / token-position bucket / document-position bucket:

  - exact zero and positive fractions for x, y, u = x*y
  - per-token active-count histograms (exact quantiles p10..p99)
  - sampled positive-magnitude quantiles
  - top 6.25/12.5/25/50% mass shares
  - token-to-token support Jaccard at lags 1..64 and level-to-level Jaccard
  - q nonzero inflation from RoPE pair completion, pair activity
  - pair-preserving block occupancy (16..256): zero blocks, >=25/50/75%
  - coordinate/pair utilization entropy, never-active fractions
  - per-band coefficient of variation
  - ideal scalar and block-granularity MAC-removability estimates

All estimates are structural opportunity, never measured speedups.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

LAGS = (1, 2, 4, 8, 16, 64)
TOP_FRACS = (0.0625, 0.125, 0.25, 0.5)
BLOCK_WIDTHS = (16, 32, 64, 128, 256)


class TrainedCensusAccumulator:
    def __init__(self, cfg, bands: int = 8,
                 block_widths: Tuple[int, ...] = BLOCK_WIDTHS,
                 token_buckets: int = 8, doc_buckets: int = 4,
                 lags: Tuple[int, ...] = LAGS,
                 sample_rows: int = 256, max_sample_rows: int = 2048,
                 magnitude_cap: int = 8192,
                 top_fracs: Tuple[float, ...] = TOP_FRACS):
        self.cfg = cfg
        self.K = cfg.K
        self.H = cfg.H
        self.bands = bands if (cfg.K // 2) % bands == 0 else 1
        self.block_widths = tuple(w for w in block_widths if w <= cfg.K
                                  and w % 2 == 0 and cfg.K % w == 0)
        self.token_buckets = token_buckets
        self.doc_buckets = doc_buckets
        self.lags = lags
        self.sample_rows = sample_rows
        self.max_sample_rows = max_sample_rows
        self.magnitude_cap = magnitude_cap
        self.top_fracs = top_fracs
        self._stats: Dict[int, List[dict]] = {}
        self.sampled: Dict[int, List[torch.Tensor]] = {}
        self.n_sample_rows = 0
        self._sample_idx: Optional[torch.Tensor] = None
        self._device: Optional[torch.device] = None

    def new_batch(self, rows: int, generator: torch.Generator):
        take = min(self.sample_rows, rows)
        self._sample_idx = torch.randperm(rows, generator=generator)[:take]

    def _new_head(self) -> dict:
        dev = self._device
        f = lambda *shape: torch.zeros(shape, dtype=torch.float64, device=dev)
        hist = {k: torch.zeros(self.K + 1, dtype=torch.float64, device=dev)
                for k in ("x", "y", "u")}
        return {
            "n": f(),
            "sample_n": f(),
            "active": {k: f() for k in ("x", "y", "u")},
            "hist": hist,
            "topmass": {k: f(len(self.top_fracs)) for k in ("x", "y", "u")},
            "pair_n": f(), "pair_active_x": f(), "pair_single_x": f(),
            "pair_active_q": f(), "pair_none_u": f(),
            "util_coord": {k: f(self.K) for k in ("x", "y", "u")},
            "util_pair": f(self.K // 2),
            "blocks": {
                w: {"n": f(), **{f"{stat}_{key}": f()
                                 for key in ("x", "u", "q")
                                 for stat in ("zero", "ge25", "ge50", "ge75")}}
                for w in self.block_widths
            },
            "bands": {k: f(self.bands) for k in ("x", "y", "u")},
            "band_n": f(),
            "token_bucket": {k: f(self.token_buckets) for k in ("x", "u")},
            "token_bucket_n": f(),
            "doc_bucket": {k: f(self.doc_buckets) for k in ("x", "u")},
            "doc_bucket_n": f(),
            "lag": {lag: {k: [f(), f()] for k in ("x", "y", "u")}
                    for lag in self.lags},
            "magn": {k: [] for k in ("x", "y", "u")},
        }

    def _head(self, level: int, head: int) -> dict:
        while len(self._stats) <= level:
            self._stats[len(self._stats)] = []
        while len(self._stats[level]) <= head:
            self._stats[level].append(self._new_head())
        return self._stats[level][head]

    @torch.no_grad()
    def add(self, level: int, x, y, u, start):
        if self._device is None:
            self._device = x.device
        y = y.permute(0, 2, 1, 3)
        b, t, h, k = x.shape
        dev = x.device
        xb, yb, ub = x > 0, y > 0, u > 0
        pairs = k // 2
        xp = xb.reshape(b, t, h, pairs, 2)
        pair_any = xp.any(-1)
        qb = pair_any.unsqueeze(-1).expand(-1, -1, -1, -1, 2).reshape(b, t, h, k)
        up = ub.reshape(b, t, h, pairs, 2)
        ar = torch.arange(t, device=dev).view(1, t).expand(b, t)
        new_doc = torch.zeros((b, t), dtype=torch.bool, device=dev)
        new_doc[:, 0] = True
        new_doc[:, 1:] = start[:, 1:] != start[:, :-1]
        start_pos = torch.where(new_doc, ar, torch.full_like(ar, t))
        suffix = torch.flip(torch.flip(start_pos, [1]).cummin(1).values, [1])
        next_start = torch.cat([suffix[:, 1:], torch.full((b, 1), t, device=dev)],
                               dim=1)
        doc_len = (next_start - start).clamp_min(1)
        rel = ar - start
        doc_b = (rel.double() / doc_len.double()
                 * self.doc_buckets).long().clamp_max(self.doc_buckets - 1)
        token_b = (ar * self.token_buckets // t).expand(b, t)
        same_pair = start[:, 1:] == start[:, :-1]
        sample = self._sample_idx.to(dev)

        for head in range(h):
            st = self._head(level, head)
            xh, yh, uh, qh = (t_[:, :, head] for t_ in (xb, yb, ub, qb))
            st["n"] += xh.numel()
            for key, mask in (("x", xh), ("y", yh), ("u", uh)):
                counts = mask.reshape(b * t, k).sum(-1)
                st["active"][key] += counts.sum()
                st["hist"][key] += torch.bincount(counts, minlength=k + 1)
                st["util_coord"][key] += mask.sum(dim=(0, 1))
            pa = pair_any[:, :, head]
            st["pair_n"] += pa.numel()
            st["pair_active_x"] += pa.sum()
            st["pair_single_x"] += (
                xp[:, :, head, :, 0] ^ xp[:, :, head, :, 1]).sum()
            st["pair_active_q"] += pa.sum()
            st["pair_none_u"] += (~up[:, :, head].any(-1)).sum()
            st["util_pair"] += pa.sum(dim=(0, 1))
            for lag in self.lags:
                if lag >= t:
                    continue
                same_lag = same_pair[:, lag - 1:]
                for key, mask in (("x", xh), ("y", yh), ("u", uh)):
                    inter = (mask[:, :-lag] & mask[:, lag:]).sum(-1).float()
                    union = (mask[:, :-lag] | mask[:, lag:]).sum(-1).float()
                    valid = same_lag & (union > 0)
                    st["lag"][lag][key][0] += (inter[valid] / union[valid]).sum()
                    st["lag"][lag][key][1] += valid.sum()
            band_pairs = pairs // self.bands
            for key, mask in (("x", xh), ("y", yh), ("u", uh)):
                occupied = mask.reshape(b, t, pairs, 2).any(-1)
                st["bands"][key] += occupied.reshape(
                    b, t, self.bands, band_pairs).sum(dim=(0, 1, 3))
            st["band_n"] += b * t
            for key, mask in (("x", xh), ("u", uh)):
                st["token_bucket"][key] += torch.bincount(
                    token_b.reshape(-1), weights=mask.sum(-1).double().reshape(-1),
                    minlength=self.token_buckets)
                st["doc_bucket"][key] += torch.bincount(
                    doc_b.reshape(-1), weights=mask.sum(-1).double().reshape(-1),
                    minlength=self.doc_buckets)
            st["token_bucket_n"] += b * t / self.token_buckets
            st["doc_bucket_n"] += b * t / self.doc_buckets
            for width in self.block_widths:
                for key, mask in (("x", xh), ("u", uh), ("q", qh)):
                    occ = mask.reshape(b * t, k // width, width).sum(-1)
                    n_blocks = occ.numel()
                    st["blocks"][width][f"n"] += n_blocks
                    st["blocks"][width][f"zero_{key}"] += (occ == 0).sum()
                    st["blocks"][width][f"ge25_{key}"] += (
                        occ >= 0.25 * width).sum()
                    st["blocks"][width][f"ge50_{key}"] += (
                        occ >= 0.5 * width).sum()
                    st["blocks"][width][f"ge75_{key}"] += (
                        occ >= 0.75 * width).sum()
            xs = x.reshape(b * t, h, k)[sample][:, head].float()
            ys = y.reshape(b * t, h, k)[sample][:, head].float()
            us = u.reshape(b * t, h, k)[sample][:, head].float()
            st["sample_n"] += xs.shape[0]
            for key, vals in (("x", xs), ("y", ys), ("u", us)):
                order = vals.sort(dim=-1, descending=True).values
                csum = order.cumsum(-1)
                total = csum[:, -1:].clamp_min(1e-30)
                for i, frac in enumerate(self.top_fracs):
                    idx = max(1, int(frac * k)) - 1
                    st["topmass"][key][i] += (
                        csum[:, idx] / total.squeeze(-1)).sum()
                positives = vals[vals > 0]
                if (positives.numel() and
                        sum(v.numel() for v in st["magn"][key])
                        < self.magnitude_cap):
                    st["magn"][key].append(
                        positives[::4].to("cpu", torch.float32))
        layer = self.sampled.setdefault(level, [])
        if sum(s.shape[0] for s in layer) < self.max_sample_rows:
            layer.append(
                xb.reshape(b * t, h, k)[sample].detach().to("cpu", torch.bool)
            )
        self.n_sample_rows += sample.numel()

    # -- finalize ----------------------------------------------------------

    def finalize(self) -> dict:
        levels = []
        for level in sorted(self._stats):
            heads = []
            for head, st in enumerate(self._stats[level]):
                heads.append(self._head_out(head, st))
            levels.append({"level": level, "heads": heads})
        global_out = self._global(levels)
        cross = self._cross_level()
        macs = self._mac_estimates(levels)
        decision = self._decision(global_out, macs)
        return {
            "levels": levels,
            "global": global_out,
            "cross_level_support_jaccard": cross,
            "mac_estimates": macs,
            "decision": decision,
            "sample_rows": self.n_sample_rows,
        }

    def _head_out(self, head: int, st: dict) -> dict:
        n = max(1.0, float(st["n"].item()))
        pairs = max(1.0, float(st["pair_n"].item()))
        sample_n = max(1.0, float(st["sample_n"].item()))
        band_pairs = (self.K // 2) // self.bands
        out = {"head": head}
        for key in ("x", "y", "u"):
            hist = st["hist"][key].double()
            total = hist.sum().clamp_min(1.0)
            cdf = hist.cumsum(0) / total
            q = {}
            for name, frac in (("p10", 0.10), ("p50", 0.50), ("p90", 0.90),
                               ("p99", 0.99)):
                q[name] = int(torch.searchsorted(
                    cdf, torch.tensor(frac, dtype=torch.float64)).item())
            out[key] = {
                "zero_fraction": 1.0 - float(st["active"][key].item()) / n,
                "positive_fraction": float(st["active"][key].item()) / n,
                "active_count": {
                    "mean": float(st["active"][key].item()) / n,
                    **q,
                },
                "top_mass_share": {
                    str(frac): float(st["topmass"][key][i].item()) / sample_n
                    for i, frac in enumerate(self.top_fracs)
                },
                "magnitude_quantiles_positive": self._quantiles(
                    st["magn"][key]),
            }
        out["q_nonzero_inflation"] = {
            "coord_active_fraction_x": float(st["active"]["x"].item()) / n,
            "coord_active_fraction_q": (
                2.0 * float(st["pair_active_q"].item()) / n
            ),
            "pair_active_fraction": (
                float(st["pair_active_q"].item()) / pairs
            ),
            "pair_single_active_fraction": (
                float(st["pair_single_x"].item()) / pairs
            ),
        }
        out["support_jaccard_token_to_token"] = {
            str(lag): {k: float(st["lag"][lag][k][0].item()) /
                           max(1.0, float(st["lag"][lag][k][1].item()))
                       for k in ("x", "y", "u")}
            for lag in self.lags
        }
        out["band_occupancy"] = {
            k: (st["bands"][k] / (max(1.0, float(st["band_n"].item()))
                                  * band_pairs)).tolist()
            for k in ("x", "y", "u")
        }
        out["token_position_active_fraction"] = {
            k: (st["token_bucket"][k] / (
                max(1.0, float(st["token_bucket_n"].item())) * self.K)).tolist()
            for k in ("x", "u")
        }
        out["document_position_active_fraction"] = {
            k: (st["doc_bucket"][k] / (
                max(1.0, float(st["doc_bucket_n"].item())) * self.K)).tolist()
            for k in ("x", "u")
        }
        out["block_occupancy"] = {}
        for width in self.block_widths:
            bst = st["blocks"][width]
            nb = max(1.0, float(bst["n"].item()))
            out["block_occupancy"][str(width)] = {
                key: {
                    "zero_fraction": float(bst[f"zero_{key}"].item()) / nb,
                    "ge25_fraction": float(bst[f"ge25_{key}"].item()) / nb,
                    "ge50_fraction": float(bst[f"ge50_{key}"].item()) / nb,
                    "ge75_fraction": float(bst[f"ge75_{key}"].item()) / nb,
                }
                for key in ("x", "u", "q")
            }
        out["utilization"] = self._utilization(st)
        return out

    def _utilization(self, st: dict) -> dict:
        out = {}
        for key in ("x", "y", "u"):
            p = st["util_coord"][key] / max(1.0, float(st["n"].item()))
            nz = (p > 0)
            ent = float(-(p * (p + 1e-30).log()).sum().item())
            out[f"coord_entropy_{key}"] = ent / math.log(self.K)
            out[f"coord_never_active_{key}"] = float((~nz).sum().item()) / self.K
        p = st["util_pair"] / max(1.0, float(st["n"].item()))
        out["pair_entropy_x"] = float(
            -(p * (p + 1e-30).log()).sum().item()) / math.log(self.K // 2)
        out["pair_never_active_x"] = float(
            (p <= 0).sum().item()) / (self.K // 2)
        bands = st["bands"]["x"] / max(1.0, float(st["band_n"].item()))
        mean = float(bands.mean().item())
        out["band_cv_x"] = (
            float(bands.std(unbiased=False).item()) / mean if mean > 0 else 0.0
        )
        return out

    def _global(self, levels: list) -> dict:
        out = {"levels": len(levels)}
        for key in ("x", "y", "u"):
            pos = 0.0
            n = 0.0
            for layer in levels:
                for head in layer["heads"]:
                    frac = head[key]["positive_fraction"]
                    pos += frac
                    n += 1.0
            out[f"{key}_positive_fraction"] = pos / max(1.0, n)
        pair_active = 0.0
        q_infl = 0.0
        freq_cv = 0.0
        heads = 0
        for layer in levels:
            for head in layer["heads"]:
                pair_active += head["q_nonzero_inflation"]["pair_active_fraction"]
                q_infl += head["q_nonzero_inflation"]["coord_active_fraction_q"]
                freq_cv += head["utilization"]["band_cv_x"]
                heads += 1
        out["pair_active_fraction"] = pair_active / max(1, heads)
        out["q_coord_active_fraction"] = q_infl / max(1, heads)
        out["band_cv_x_mean"] = freq_cv / max(1, heads)
        return out

    def _cross_level(self) -> dict:
        out = {}
        for level in sorted(self.sampled):
            if level + 1 not in self.sampled:
                continue
            a = torch.cat(self.sampled[level], dim=0).float()
            b_ = torch.cat(self.sampled[level + 1], dim=0).float()
            s = min(a.shape[0], b_.shape[0])
            a, b_ = a[:s], b_[:s]
            inter = (a * b_).sum(-1)
            union = ((a + b_) > 0).sum(-1)
            valid = union > 0
            out[str(level)] = float(
                (inter[valid] / union[valid]).mean().item())
        return out

    def _mac_estimates(self, levels: list) -> dict:
        d, k = self.cfg.D, self.K
        dense = {
            "dx": d * k,
            "dy": d * k,
            "u": k,
            "e": k * d,
            "attn_read": k * d,
            "attn_state": k * d,
        }
        total = sum(dense.values())
        rem_scalar = {name: 0.0 for name in ("dy", "u", "e", "attn_read",
                                             "attn_state")}
        rem_block = {name: {w: 0.0 for w in self.block_widths}
                     for name in rem_scalar}
        heads = 0
        for layer in levels:
            for head in layer["heads"]:
                t = 1.0
                x_pos = head["x"]["positive_fraction"]
                u_pos = head["u"]["positive_fraction"]
                q_act = head["q_nonzero_inflation"]["coord_active_fraction_q"]
                rem_scalar["dy"] += dense["dy"] * (1.0 - x_pos)
                rem_scalar["u"] += dense["u"] * (1.0 - u_pos)
                rem_scalar["e"] += dense["e"] * (1.0 - u_pos)
                rem_scalar["attn_read"] += dense["attn_read"] * (1.0 - q_act)
                rem_scalar["attn_state"] += dense["attn_state"] * (1.0 - q_act)
                for width, blocks in head["block_occupancy"].items():
                    zx = blocks["x"]["zero_fraction"]
                    zu = blocks["u"]["zero_fraction"]
                    zq = blocks["q"]["zero_fraction"]
                    rem_block["dy"][int(width)] += dense["dy"] * zx
                    rem_block["u"][int(width)] += dense["u"] * zu
                    rem_block["e"][int(width)] += dense["e"] * zu
                    rem_block["attn_read"][int(width)] += dense["attn_read"] * zq
                    rem_block["attn_state"][int(width)] += dense["attn_state"] * zq
                heads += 1
        scale = max(1, heads)
        scalar_fraction = sum(rem_scalar.values()) / (total * scale)
        block_fraction = {
            str(w): sum(rem_block[name][w] for name in rem_block)
            / (total * scale)
            for w in self.block_widths
        }
        best = max(block_fraction, key=block_fraction.get) \
            if block_fraction else None
        return {
            "dense_neuron_axis_mac_per_token": total,
            "ideal_scalar_removable_fraction": scalar_fraction,
            "block_removable_fraction": block_fraction,
            "best_block_width": best,
            "note": ("structural opportunity estimates only; not measured "
                     "speedups and not same-session benchmarks"),
        }

    def _decision(self, global_out: dict, macs: dict) -> dict:
        u_zero = 1.0 - global_out["u_positive_fraction"]
        x_zero = 1.0 - global_out["x_positive_fraction"]
        pair_zero = 1.0 - global_out["pair_active_fraction"]
        scalar = macs["ideal_scalar_removable_fraction"]
        best_block = macs["best_block_width"]
        best_block_frac = (macs["block_removable_fraction"].get(best_block, 0.0)
                           if best_block else 0.0)
        exact = scalar >= 0.30 and (u_zero >= 0.5 or x_zero >= 0.3)
        pair_sparse = pair_zero >= 0.30
        block_sparse = bool(best_block) and best_block_frac >= 0.30
        freq_cv = global_out.get("band_cv_x_mean", 0.0)
        freq_special = freq_cv >= 0.15
        moe = block_sparse or pair_sparse
        reasons = [
            f"u zero {u_zero:.3f}; x zero {x_zero:.3f}",
            f"q pair-zero {pair_zero:.3f}",
            f"ideal scalar removable MAC {scalar:.3f}",
            f"best block {best_block} removable {best_block_frac:.3f}",
        ]
        return {
            "exact_scalar_sparse_candidate": bool(exact),
            "pair_sparse_candidate": bool(pair_sparse),
            "block_sparse_candidate": bool(block_sparse),
            "best_block": best_block,
            "moe_structured_candidate": bool(moe),
            "frequency_specialization_candidate": bool(freq_special),
            "reason": "; ".join(reasons),
            "evidence_note": ("structural estimates, not speedups; decisions "
                              "require the 2.5B census before architecture "
                              "selection"),
        }

    @staticmethod
    def _quantiles(samples: List[torch.Tensor]) -> dict:
        if not samples:
            return {}
        vals = torch.cat(samples).double()
        if vals.numel() == 0:
            return {}
        qs = torch.tensor([0.5, 0.9, 0.99, 0.999, 1.0], dtype=torch.float64)
        out = torch.quantile(vals, qs).tolist()
        return {"p50": out[0], "p90": out[1], "p99": out[2], "p999": out[3],
                "max": out[4], "n": int(vals.numel())}
