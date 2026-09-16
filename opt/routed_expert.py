"""Routed sparse executor for the MoE/resonant campaign (E1 static expert loop).

Head-shared frequency-band routing (priority mode): one route decision per
token group selects the same experts for all H heads. Weights are stored
expert-major so one expert executes as a single dense D -> H*Ke GEMM:

  DxE[M, D, H*Ke], DyE[M, D, H*Ke], EE[M, H*Ke, D]

Routed semantics (semantics A):
  a_e   = scan over the expert's active tokens only (state resets on the
          original document identity carried by each gathered token)
  a_tot = sum_{e active at t} gate_e * a_e
  y_e   = ReLU(LN(a_tot) @ Dy_e)        (shared a_tot, Arm-A coupling)
  base  = LN(sum_{e active} gate_e * (x_e * y_e) @ E_e)

Route sources for DOE: fixed/constant experts, deterministic varying sets,
and contiguous frequency windows. Learned routing is a later iteration.
All execution shapes are static: capacity per expert is fixed and padded
slots are masked to zero.
"""

from __future__ import annotations

import math
from typing import List, NamedTuple, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from opt.model_ref import Coordinator, DenseWriter, rope_pair_freq
from opt.scan_attn import scan_chunkwise_bthk


class Route(NamedTuple):
    """Compacted static route: one stream per routed expert only."""
    active: Tuple[int, ...]
    sel_idx: torch.Tensor
    sel_mask: torch.Tensor
    gate: torch.Tensor


def _rope_flat(q, pos, freq):
    """RoPE for gathered tokens. q: (S,H,Ke), pos: (S,), freq: (Ke/2,)."""
    s, h, k = q.shape
    qp = q.reshape(s, h, k // 2, 2)
    p = pos.to(q.dtype) if pos.dtype != q.dtype else pos
    phase = p.unsqueeze(-1).unsqueeze(-1) * freq.to(q.dtype).view(1, 1, -1)
    phase = torch.remainder(phase, 1.0) * (2.0 * math.pi)
    cs = torch.cos(phase)
    sn = torch.sin(phase)
    qe, qo = qp[..., 0], qp[..., 1]
    return torch.stack((qe * cs - qo * sn, qo * cs + qe * sn),
                       dim=-1).reshape_as(q)


def fixed_route_sets(experts: int, top_r: int,
                     offset: int = 0) -> List[Tuple[int, ...]]:
    return [tuple((offset + k) % experts for k in range(top_r))]


def varying_route_sets(experts: int, top_r: int, step: int = None
                       ) -> List[Tuple[int, ...]]:
    step = step or top_r
    sets = []
    cursor = 0
    while len(sets) < experts:
        sets.append(tuple((cursor + k) % experts for k in range(top_r)))
        cursor += step
    return sets


def window_route_sets(experts: int, width: int, cyclic: bool = False
                      ) -> List[Tuple[int, ...]]:
    windows = []
    if cyclic:
        windows = [tuple((i + k) % experts for k in range(width))
                   for i in range(experts)]
    else:
        windows = [tuple(range(i, i + width))
                   for i in range(experts - width + 1)]
    return windows


def group_route_table(route_sets: Sequence[Sequence[int]],
                      groups_per_row: int) -> List[Tuple[int, ...]]:
    """Cycle the route sets over the row's token groups."""
    return [tuple(route_sets[g % len(route_sets)]) for g in range(groups_per_row)]


def build_route_tensors(group_table: Sequence[Sequence[int]], group: int,
                        batch: int, time: int, experts: int, device,
                        capacity_factor: float = 1.0) -> Route:
    """Static capacity packing over routed experts only.

    Only experts that receive at least one group get a stream; capacity is
    the max token count across those experts (times capacity_factor for the
    padding intervention), and unused slots are masked. Inactive experts are
    absent from the loop entirely (no zero-work GEMMs).
    """
    if capacity_factor < 1.0:
        raise ValueError("capacity_factor must be >= 1.0")
    token_positions = [[] for _ in range(experts)]
    groups_per_row = time // group
    for b in range(batch):
        for ga in range(groups_per_row):
            selected = group_table[ga % len(group_table)]
            base = b * time + ga * group
            for expert in selected:
                token_positions[expert].extend(range(base, base + group))
    active = tuple(e for e, positions in enumerate(token_positions)
                   if positions)
    if not active:
        raise ValueError("no route assignments")
    used = max(len(token_positions[e]) for e in active)
    total = int(math.ceil(used * capacity_factor))
    sel_idx = torch.zeros((len(active), total), dtype=torch.long)
    sel_mask = torch.zeros((len(active), total), dtype=torch.float32)
    for row, expert in enumerate(active):
        positions = token_positions[expert]
        count = len(positions)
        sel_idx[row, :count] = torch.tensor(positions, dtype=torch.long)
        sel_mask[row, :count] = 1.0
    return Route(active, sel_idx.to(device), sel_mask.to(device),
                 sel_mask.clone().to(device))


class RoutedExpertArmA(nn.Module):
    def __init__(self, cfg, device, experts: int = 8, expert_width: int = 512,
                 scan_block: int = 1024, gate_mode: str = "binary"):
        super().__init__()
        assert gate_mode in ("binary", "weighted")
        self.cfg = cfg
        self.M = int(experts)
        self.Ke = int(expert_width)
        assert self.Ke % 2 == 0
        self.stored_K = self.M * self.Ke
        self.scan_block = int(scan_block)
        self.gate_mode = gate_mode
        self.embedding = nn.Embedding(cfg.V, cfg.D)
        self.DxE = nn.Parameter(torch.empty(self.M, cfg.D,
                                            cfg.H * self.Ke))
        self.DyE = nn.Parameter(torch.empty(self.M, cfg.D,
                                            cfg.H * self.Ke))
        self.EE = nn.Parameter(torch.empty(self.M, cfg.H * self.Ke, cfg.D))
        self.readout = nn.Parameter(torch.empty(cfg.D, cfg.V))
        self.coordinator = Coordinator(cfg)
        self.writer = DenseWriter(cfg)
        self.ln = nn.LayerNorm(cfg.D, elementwise_affine=False, bias=False)
        base_freq = rope_pair_freq(cfg, device)
        self.register_buffer("rope_freq_bands",
                             base_freq.reshape(self.M, self.Ke // 2).clone())

    # -- conversion ---------------------------------------------------------

    def load_canonical(self, state_dict):
        h, d, m, ke = self.cfg.H, self.cfg.D, self.M, self.Ke
        dx = state_dict["decoder_x"].view(h, d, m, ke)
        dy = state_dict["decoder_y"].view(h, d, m, ke)
        ee = state_dict["encoder"].view(h, m, ke, d)
        with torch.no_grad():
            self.DxE.copy_(dx.permute(2, 1, 0, 3).reshape(m, d, h * ke))
            self.DyE.copy_(dy.permute(2, 1, 0, 3).reshape(m, d, h * ke))
            self.EE.copy_(ee.permute(1, 0, 2, 3).reshape(m, h * ke, d))
        rest = {k: v for k, v in state_dict.items()
                if k in self.state_dict()
                and self.state_dict()[k].shape == v.shape
                and not k.startswith(("DxE", "DyE", "EE"))}
        self.load_state_dict(rest, strict=False)

    # -- forward ------------------------------------------------------------

    def forward_route(self, idx, pos, segpos, full_mask, segment_start,
                      route: Route):
        cfg = self.cfg
        b, t = idx.shape
        v = self.ln(self.embedding(idx))
        for _ in range(cfg.L):
            v = self._level(v, pos, segpos, full_mask, segment_start, route)
        return v @ self.readout

    def _level(self, v, pos, segpos, full_mask, segment_start, route: Route):
        cfg = self.cfg
        b, t, d = v.shape
        h, ke = cfg.H, self.Ke
        vf = v.reshape(b * t, d)
        posf = pos.reshape(b * t)
        segf = segment_start.reshape(b * t)
        s = route.sel_idx.shape[1]
        a_flat = torch.zeros((b * t, h, d), dtype=v.dtype, device=v.device)
        x_kept = []
        for row, expert in enumerate(route.active):
            idx_e = route.sel_idx[row]
            mask = route.sel_mask[row]
            gate_e = route.gate[row]
            vg = vf[idx_e] * mask.unsqueeze(-1)
            x_e = F.relu(vg @ self.DxE[expert]).view(s, h, ke)
            x_kept.append(x_e * gate_e.view(-1, 1, 1))
            q_e = _rope_flat(x_e, posf[idx_e],
                             self.rope_freq_bands[expert])
            seg_g = segf[idx_e] * mask.long() + (~mask.bool()).long() * 10**9
            a_e = scan_chunkwise_bthk(
                q_e.permute(1, 0, 2).unsqueeze(0),
                vg.view(1, 1, s, d).expand(1, h, s, d), seg_g.view(1, s),
                block=self.scan_block, skip_zero_carry=True,
            )
            a_e = a_e.view(h, s, d).permute(1, 0, 2)
            a_flat.index_add_(0, idx_e, a_e * mask.view(-1, 1, 1))
        a_ln = self.ln(a_flat)
        base_flat = torch.zeros((b * t, d), dtype=v.dtype, device=v.device)
        for row, expert in enumerate(route.active):
            idx_e = route.sel_idx[row]
            mask = route.sel_mask[row]
            gate_e = route.gate[row]
            a_g = a_ln[idx_e] * mask.view(-1, 1, 1)
            dy = self.DyE[expert].view(d, h, ke)
            y_e = F.relu(torch.einsum("shd,dhk->shk", a_g, dy))
            u_e = x_kept[row] * y_e
            ee = self.EE[expert].view(h, ke, d)
            contrib = torch.einsum("shk,hkd->sd", u_e, ee)
            base_flat.index_add_(0, idx_e,
                                 contrib * mask.view(-1, 1) * gate_e.view(-1, 1))
        base = self.ln(base_flat.view(b, t, d))
        g = self.coordinator(v, segpos, full_mask)
        delta = self.writer(g * base)
        return self.ln(v + delta)

    def parameter_ledger(self, top_r: int) -> dict:
        stored = sum(p.numel() for p in self.parameters())
        return {
            "M": self.M,
            "Ke": self.Ke,
            "top_r": top_r,
            "stored_params": int(stored),
            "stored_K_per_head": self.stored_K,
            "active_K_per_head": top_r * self.Ke,
            "stored_N": self.cfg.H * self.stored_K,
            "active_N": self.cfg.H * top_r * self.Ke,
            "active_fraction": top_r / self.M,
        }
