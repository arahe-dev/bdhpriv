"""Census fork of Arm-A for the sparse/resonant autoresearch campaign.

CensusArmA is a strict fork of opt.model_opt.OptArmA instantiated with the
frozen production flags (dense coordinator, direct paper_y layout, cached
RoPE, branch-free packed update, zero-carry skip, no activation
checkpointing). It records per-level x = ReLU(Dx v), y = ReLU(LN(a) Dy) and
u = x*y before they are consumed; OptArmA itself is never modified.

SparsityAccumulator turns those captures into the Phase-1 metrics
(machine-readable JSON via opt/sparsity_census.py). The instrumentation
provably does not change outputs; opt/test_resonant_math.py checks that.

RoPE detail that the census must respect: RoPE mixes the two coordinates of
each pair, so an x coordinate that is zero can still produce a nonzero q
coordinate when its pair partner is active. q's support is the
pair-completion of x's support, which is why the accumulator reports
both coordinate-level and pair-level statistics.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from opt.model_opt import OptArmA


class CensusArmA(OptArmA):
    """Production Arm-A (dense coordinator) with activation capture."""

    def __init__(self, cfg, device, scan_block: int = 1024):
        super().__init__(
            cfg, device, scan_block=scan_block, use_checkpoint=False,
            coord="dense", single_scan="chunkwise",
            packed_update="branchfree", zero_carry=True,
            paper_layout="direct", cache_rope=True,
        )
        self.capture = None
        self._level_index = 0

    def begin_forward(self, capture):
        self.capture = capture
        self._level_index = 0

    def level(self, v, pos, segpos, full_mask, segment_start, single_doc=False,
              cs=None, sn=None):
        cfg = self.cfg
        x_bt = self.project_x_native(v)
        a = self.ln(self.attention_scan(x_bt, v, pos, segment_start,
                                        single_doc, cs, sn))
        ypre = F.relu(a @ self.decoder_y)
        prod = x_bt * ypre.permute(0, 2, 1, 3)
        if self.capture is not None:
            self.capture(self._level_index, x_bt, ypre, prod, segment_start)
        self._level_index += 1
        paper_y_flat = prod.reshape(v.shape[0], cfg.T, cfg.N)
        base = self.ln(paper_y_flat @ self.encoder)
        g = self.coordinator(v, segpos, full_mask)
        delta = self.writer(g * base)
        return self.ln(v + delta)


def _adjacent_sums(mask: torch.Tensor, start: torch.Tensor):
    """Mean per-pair Jaccard of supports between adjacent same-doc tokens.

    mask: (B,T,K) boolean supports, start: (B,T) document starts.
    Returns (sum of per-pair Jaccard, pair count).
    """
    same_pair = start[:, 1:] == start[:, :-1]
    inter = (mask[:, :-1] & mask[:, 1:]).sum(-1).float()
    union = (mask[:, :-1] | mask[:, 1:]).sum(-1).float()
    valid = same_pair & (union > 0)
    jaccard = inter[valid] / union[valid]
    return jaccard.sum(), valid.sum()


def _persistence_sums(mask: torch.Tensor, start: torch.Tensor):
    """Mean Jaccard between first and last token support of each document run."""
    b, t, k = mask.shape
    device = mask.device
    ar = torch.arange(t, device=device).view(1, t).expand(b, t)
    new = torch.zeros((b, t), dtype=torch.bool, device=device)
    new[:, 0] = True
    new[:, 1:] = start[:, 1:] != start[:, :-1]
    start_pos = torch.where(new, ar, torch.full_like(ar, t))
    suffix_min = torch.flip(
        torch.flip(start_pos, [1]).cummin(1).values, [1]
    )
    next_start = torch.cat(
        [suffix_min[:, 1:], torch.full((b, 1), t, device=device)], dim=1
    )
    last = next_start - 1
    run_len = last - ar + 1
    first_mask = torch.gather(
        mask, 1, ar[..., None].expand(-1, -1, k)
    )
    last_mask = torch.gather(
        mask, 1, last[..., None].expand(-1, -1, k)
    )
    inter = (first_mask & last_mask).sum(-1).float()
    union = (first_mask | last_mask).sum(-1).float()
    keep = new & (run_len >= 2) & (union > 0)
    jaccard = inter[keep] / union[keep]
    return jaccard.sum(), keep.sum()


class SparsityAccumulator:
    """Phase-1 metrics over x/y/u captures for every level and head."""

    def __init__(self, cfg, sample_rows: int = 256,
                 max_sample_rows: int = 1024,
                 block_widths: Optional[Tuple[int, ...]] = None,
                 bands: int = 8,
                 top_fracs: Tuple[float, ...] = (0.0625, 0.125, 0.25, 0.5),
                 quantile_stride: int = 8):
        self.cfg = cfg
        self.K = cfg.K
        self.H = cfg.H
        self.sample_rows = sample_rows
        self.max_sample_rows = max_sample_rows
        if block_widths is None:
            block_widths = tuple(
                w for w in (16, 32, 64, 128, 256)
                if w <= cfg.K and cfg.K % w == 0
            )
        self.block_widths = tuple(block_widths)
        self.bands = bands if bands >= 1 and (cfg.K // 2) % bands == 0 else 1
        self.top_fracs = top_fracs
        self.quantile_stride = quantile_stride
        self._stats: Dict[int, List[dict]] = {}
        self.sampled: Dict[str, Dict[int, List[torch.Tensor]]] = {
            "x": {}, "y": {}, "u": {},
        }
        self.coact_width = (
            128 if 128 in self.block_widths
            else (self.block_widths[-1] if self.block_widths else None)
        )
        self.n_sample_rows = 0
        self._sample_idx: Optional[torch.Tensor] = None
        self._device: Optional[torch.device] = None

    def new_batch(self, rows: int, generator: torch.Generator):
        take = min(self.sample_rows, rows)
        self._sample_idx = torch.randperm(rows, generator=generator)[:take]

    def _head(self, level: int, head: int) -> dict:
        while len(self._stats) <= level:
            self._stats[len(self._stats)] = []
        while len(self._stats[level]) <= head:
            self._stats[level].append(self._new_head_stats())
        return self._stats[level][head]

    def _new_head_stats(self) -> dict:
        device = self._device
        return {
            "n": torch.zeros((), dtype=torch.float64, device=device),
            "x_pos": torch.zeros((), dtype=torch.float64, device=device),
            "y_pos": torch.zeros((), dtype=torch.float64, device=device),
            "u_pos": torch.zeros((), dtype=torch.float64, device=device),
            "n_pairs": torch.zeros((), dtype=torch.float64, device=device),
            "x_pair_none": torch.zeros((), dtype=torch.float64, device=device),
            "x_pair_one": torch.zeros((), dtype=torch.float64, device=device),
            "u_pair_none": torch.zeros((), dtype=torch.float64, device=device),
            "adj": {
                k: [torch.zeros((), dtype=torch.float64, device=device),
                    torch.zeros((), dtype=torch.float64, device=device)]
                for k in ("x", "y", "u")
            },
            "pers": {
                k: [torch.zeros((), dtype=torch.float64, device=device),
                    torch.zeros((), dtype=torch.float64, device=device)]
                for k in ("x", "y", "u")
            },
            "bands": {
                k: torch.zeros(self.bands, dtype=torch.float64, device=device)
                for k in ("x", "y", "u")
            },
            "band_n": torch.zeros((), dtype=torch.float64, device=device),
            "blocks": {
                w: {
                    "x_any": torch.zeros((), dtype=torch.float64, device=device),
                    "x_all": torch.zeros((), dtype=torch.float64, device=device),
                    "u_any": torch.zeros((), dtype=torch.float64, device=device),
                    "u_all": torch.zeros((), dtype=torch.float64, device=device),
                    "n": torch.zeros((), dtype=torch.float64, device=device),
                }
                for w in self.block_widths
            },
            "topmass": {
                k: torch.zeros(len(self.top_fracs), dtype=torch.float64,
                               device=device)
                for k in ("x", "y", "u")
            },
            "topmass_n": torch.zeros((), dtype=torch.float64, device=device),
            "mass": {
                k: torch.zeros(self.K, dtype=torch.float64, device=device)
                for k in ("x", "y", "u")
            },
            "mass_n": torch.zeros((), dtype=torch.float64, device=device),
            "coact": (
                torch.zeros((self.H, self.K // self.coact_width,
                             self.K // self.coact_width),
                            dtype=torch.float64, device=device)
                if self.coact_width else None
            ),
            "quant": {k: [] for k in ("x", "y", "u")},
        }

    @torch.no_grad()
    def add(self, level: int, x: torch.Tensor, y: torch.Tensor,
            u: torch.Tensor, start: torch.Tensor):
        if self._device is None:
            self._device = x.device
        # Arm-A emits ypre as (B,H,T,K); x and u are (B,T,H,K).
        y = y.permute(0, 2, 1, 3)
        b, t, h, k = x.shape
        xb, yb, ub = x > 0, y > 0, u > 0
        sample = self._sample_idx.to(x.device)
        for head in range(h):
            st = self._head(level, head)
            xh, yh, uh = xb[:, :, head], yb[:, :, head], ub[:, :, head]
            st["n"] += xh.numel()
            st["x_pos"] += xh.sum()
            st["y_pos"] += yh.sum()
            st["u_pos"] += uh.sum()
            xp = xh.reshape(b, t, k // 2, 2)
            pair_any = xp.any(-1)
            st["n_pairs"] += pair_any.numel()
            st["x_pair_none"] += (~pair_any).sum()
            st["x_pair_one"] += (xp[..., 0] ^ xp[..., 1]).sum()
            st["u_pair_none"] += (~uh.reshape(b, t, k // 2, 2).any(-1)).sum()
            for key, mask in (("x", xh), ("y", yh), ("u", uh)):
                inter, union = _adjacent_sums(mask, start)
                st["adj"][key][0] += inter
                st["adj"][key][1] += union
                inter, union = _persistence_sums(mask, start)
                st["pers"][key][0] += inter
                st["pers"][key][1] += union
            band_pairs = k // 2
            per_band = band_pairs // self.bands
            for key, mask in (("x", xh), ("y", yh), ("u", uh)):
                paired = mask.reshape(b, t, band_pairs, 2).any(-1)
                occupied = paired.reshape(b, t, self.bands, per_band)
                st["bands"][key] += occupied.sum(dim=(0, 1, 3)) / per_band
            st["band_n"] += b * t
            for width in self.block_widths:
                xr = xh.reshape(b, t, k // width, width)
                ur = uh.reshape(b, t, k // width, width)
                st["blocks"][width]["x_any"] += xr.any(-1).sum()
                st["blocks"][width]["x_all"] += xr.all(-1).sum()
                st["blocks"][width]["u_any"] += ur.any(-1).sum()
                st["blocks"][width]["u_all"] += ur.all(-1).sum()
                st["blocks"][width]["n"] += xr[..., 0].numel()
            xs = x.reshape(b * t, h, k)[sample][:, head].float()
            ys = y.reshape(b * t, h, k)[sample][:, head].float()
            us = u.reshape(b * t, h, k)[sample][:, head].float()
            for key, vals in (("x", xs), ("y", ys), ("u", us)):
                order = vals.sort(dim=-1, descending=True).values
                csum = order.cumsum(-1)
                total = csum[:, -1:].clamp_min(1e-30)
                for i, frac in enumerate(self.top_fracs):
                    idx = max(1, int(frac * k)) - 1
                    st["topmass"][key][i] += (
                        csum[:, idx] / total.squeeze(-1)
                    ).sum()
                st["mass"][key] += vals.double().sum(0)
                positives = vals[vals > 0]
                if positives.numel():
                    st["quant"][key].append(
                        positives[::self.quantile_stride].to("cpu",
                                                             torch.float32)
                    )
            st["topmass_n"] += xs.shape[0]
            st["mass_n"] += xs.shape[0]
            if st["coact"] is not None:
                w = self.coact_width
                blocks = uh.reshape(b * t, k // w, w).any(-1)[sample]
                st["coact"][head] += torch.einsum(
                    "sb,sc->bc", blocks.double(), blocks.double()
                )
        for key, tensor in (("x", xb), ("y", yb), ("u", ub)):
            layer = self.sampled[key].setdefault(level, [])
            if sum(s.shape[0] for s in layer) < self.max_sample_rows:
                flat = tensor.reshape(b * t, h, k)[sample]
                layer.append(flat.detach().to("cpu", torch.bool))
        self.n_sample_rows += sample.numel()

    @torch.no_grad()
    def finalize(self) -> dict:
        layers = []
        for level in sorted(self._stats):
            heads = []
            for head, st in enumerate(self._stats[level]):
                n = float(st["n"].item())
                pairs = float(st["n_pairs"].item())
                head_out = {
                    "head": head,
                    "x": {
                        "zero_fraction": 1.0 - float(st["x_pos"].item()) / n,
                        "positive_fraction": float(st["x_pos"].item()) / n,
                        "pair_zero_fraction": float(st["x_pair_none"].item()) / pairs,
                        "pair_single_active_fraction": float(st["x_pair_one"].item()) / pairs,
                    },
                    "y": {
                        "zero_fraction": 1.0 - float(st["y_pos"].item()) / n,
                        "positive_fraction": float(st["y_pos"].item()) / n,
                    },
                    "u": {
                        "zero_fraction": 1.0 - float(st["u_pos"].item()) / n,
                        "positive_fraction": float(st["u_pos"].item()) / n,
                        "support_intersection_fraction": float(st["u_pos"].item()) / n,
                        "pair_zero_fraction": float(st["u_pair_none"].item()) / pairs,
                    },
                    "support_intersection_vs_x": (
                        float(st["u_pos"].item()) / max(1.0, float(st["x_pos"].item()))
                    ),
                    "adjacent_jaccard": {
                        k: float(st["adj"][k][0].item()) / max(1.0, float(st["adj"][k][1].item()))
                        for k in ("x", "y", "u")
                    },
                    "document_persistence_jaccard": {
                        k: float(st["pers"][k][0].item()) / max(1.0, float(st["pers"][k][1].item()))
                        for k in ("x", "y", "u")
                    },
                    "band_occupancy": {
                        k: (st["bands"][k] / st["band_n"].clamp_min(1)).tolist()
                        for k in ("x", "y", "u")
                    },
                    "block_occupancy": {
                        str(w): {
                            "x_any": float(st["blocks"][w]["x_any"].item()) / float(st["blocks"][w]["n"].item()),
                            "x_all": float(st["blocks"][w]["x_all"].item()) / float(st["blocks"][w]["n"].item()),
                            "u_any": float(st["blocks"][w]["u_any"].item()) / float(st["blocks"][w]["n"].item()),
                            "u_all": float(st["blocks"][w]["u_all"].item()) / float(st["blocks"][w]["n"].item()),
                        }
                        for w in self.block_widths
                    },
                    "top_mass_share": {
                        k: {
                            str(frac): float(st["topmass"][k][i].item()) /
                            max(1.0, float(st["topmass_n"].item()))
                            for i, frac in enumerate(self.top_fracs)
                        }
                        for k in ("x", "y", "u")
                    },
                    "quantiles_positive": {
                        k: self._quantiles(st["quant"][k]) for k in ("x", "y", "u")
                    },
                    "neuron_concentration": self._concentration(st["mass"]),
                    "coactivation_block": (
                        (st["coact"] / max(1, self.n_sample_rows)).tolist()
                        if st["coact"] is not None else None
                    ),
                    "quantile_sample_size": float(
                        sum(v.numel() for v in st["quant"]["u"])
                    ),
                }
                heads.append(head_out)
            layers.append({"level": level, "heads": heads})
        cross = {}
        for key in ("x", "y", "u"):
            for level in sorted(self.sampled[key]):
                if level + 1 not in self.sampled[key]:
                    continue
                a = torch.cat(self.sampled[key][level], dim=0).float()
                b = torch.cat(self.sampled[key][level + 1], dim=0).float()
                s = min(a.shape[0], b.shape[0])
                a, b = a[:s], b[:s]
                inter = (a * b).sum(-1)
                union = ((a + b) > 0).sum(-1)
                valid = union > 0
                j = (inter[valid] / union[valid]).mean().item()
                cross.setdefault(str(level), {})[key] = j
        return {
            "levels": layers,
            "cross_level_support_jaccard": cross,
            "sample_rows": self.n_sample_rows,
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
        return {
            "p50": out[0], "p90": out[1], "p99": out[2], "p999": out[3],
            "max": out[4], "n": int(vals.numel()),
        }

    @staticmethod
    def _concentration(mass: Dict[str, torch.Tensor]) -> dict:
        out = {}
        for key, vec in mass.items():
            total = vec.sum().clamp_min(1e-30)
            p = (vec / total)
            nz = int((vec > 0).sum().item())
            entropy = float(-(p * (p + 1e-30).log()).sum().item())
            denom = math.log(vec.numel()) if vec.numel() > 1 else 1.0
            order = p.sort(descending=True).values
            take1 = max(1, vec.numel() // 100)
            take5 = max(1, vec.numel() // 20)
            out[key] = {
                "entropy_normalized": entropy / denom,
                "max_share": float(order[0].item()),
                "top_1pct_share": float(order[:take1].sum().item()),
                "top_5pct_share": float(order[:take5].sum().item()),
                "active_neurons": nz,
            }
        return out
