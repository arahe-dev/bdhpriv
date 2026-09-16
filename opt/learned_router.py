"""Learned cyclic-window router for the frozen compact executor.

Architecture (frozen executor geometry):
  M=8, Ke=512, top2, head-shared, cyclic contiguous windows, G=128,
  once-forward route refresh, minimum exact capacity.

Router per token group:
  s = mean(LN(v_group))            (D,)
  logits = s @ Wr + br             (M,)
  p = softmax(logits)
  j = argmax(p)   -> physical window [j, (j+1) mod M]

Differentiation (explicit straight-through proxy, forward is exactly hard):
  h = one_hot(j)
  g_st = h + p - stop_gradient(p)
  gate value forward: exactly 1 for selected experts, exactly 0 elsewhere.
  gate gradient: flows through p (hence Wr/br) only; physical dispatch uses
  the hard window indices, so inactive experts never execute and receive no
  gradients.

Capacity is the expected stream size B*T*top_r/M with static shapes; if a
router over-selects an expert, the top-k packing drops the latest overflow
tokens deterministically (reported by route_overflow_count).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from opt.routed_expert import Route, RoutedExpertArmA


class WindowRouter(nn.Module):
    def __init__(self, cfg, experts: int, tau: float = 1.0):
        super().__init__()
        self.tau = float(tau)
        self.Wr = nn.Parameter(torch.empty(cfg.D, experts))
        self.br = nn.Parameter(torch.zeros(experts))
        with torch.no_grad():
            self.Wr.normal_(0.0, cfg.INIT_STD)

    def logits_for(self, v_ln, group, groups):
        b, t, d = v_ln.shape
        summary = v_ln.reshape(b, groups, group, d).mean(dim=2)
        return summary @ self.Wr + self.br


def straight_through(logits, tau: float = 1.0):
    """Return (hard_onehot, straight_through_gate) with exact-hard forward."""
    p = torch.softmax(logits / tau, dim=-1)
    idx = p.argmax(dim=-1, keepdim=True)
    h = torch.zeros_like(p).scatter_(-1, idx, 1.0)
    return h, h + p - p.detach(), idx.squeeze(-1)


def build_learned_route(model, v_ln, group: int, capacity: int,
                        logits=None, count_overflow: bool = False) -> Route:
    """Tensorized route build (static shapes, no host syncs in the hot path).

    Every expert gets a stream of exact `capacity` slots; slots are packed by
    descending (selected * BIG - index) so selected tokens keep time order.
    `count_overflow=True` adds host syncs and must only be used outside
    compiled forwards.
    """
    cfg = model.cfg
    m = model.M
    b, t, d = v_ln.shape
    groups = t // group
    if logits is None:
        logits = model.router.logits_for(v_ln, group, groups)
    h, g_st, idx = straight_through(logits, tau=model.router.tau)
    ar = torch.arange(m, device=v_ln.device)
    prev = (ar - 1) % m
    sel = ((idx[:, :, None] == ar[None, None, :])
           | (idx[:, :, None] == prev[None, None, :]))      # (B,groups,M)
    # Both experts of the chosen window carry the selected window's
    # straight-through value (forward exactly 1, gradient through p_j).
    window_gate = (g_st * h).sum(dim=-1)                    # (B,groups)
    gate_groups = window_gate.unsqueeze(-1) * sel
    flat = b * t
    arange = torch.arange(flat, dtype=torch.float32, device=v_ln.device)
    sel_idx = []
    sel_mask = []
    gate_rows = []
    overflow = 0
    for expert in range(m):
        mask_tok = sel[:, :, expert].unsqueeze(-1).expand(
            b, groups, group).reshape(flat)
        gate_tok = gate_groups[:, :, expert].unsqueeze(-1).expand(
            b, groups, group).reshape(flat)
        scores = mask_tok.float() * 1e9 - arange
        top = torch.topk(scores, capacity, dim=0).indices
        valid = mask_tok[top]
        gate = torch.where(valid, gate_tok[top],
                           torch.zeros_like(gate_tok[top]))
        sel_idx.append(top)
        sel_mask.append(valid)
        gate_rows.append(gate)
        if count_overflow:
            overflow += int(mask_tok.sum().item()) - int(valid.sum().item())
    return Route(tuple(range(m)),
                 torch.stack(sel_idx), torch.stack(sel_mask),
                 torch.stack(gate_rows)), overflow


class LearnedRoutedExpertArmA(RoutedExpertArmA):
    """Frozen compact executor + once-forward learned cyclic-window route."""

    def __init__(self, cfg, device, experts: int = 8, expert_width: int = 512,
                 scan_block: int = 1024, top_r: int = 2, route_group: int = 128):
        super().__init__(cfg, device, experts=experts,
                         expert_width=expert_width, scan_block=scan_block)
        self.top_r = int(top_r)
        self.route_group = int(route_group)
        self.router = WindowRouter(cfg, experts)
        self.capacity_override = None
        self.capacity_factor = 1.0
        self.last_route_overflow = 0

    def capacity_tokens(self, batch: int, time: int) -> int:
        if self.capacity_override is not None:
            return int(self.capacity_override)
        return max(1, math.ceil(
            self.capacity_factor * batch * time * self.top_r / self.M))

    def forward_learned(self, idx, pos, segpos, full_mask, segment_start):
        cfg = self.cfg
        v = self.ln(self.embedding(idx))
        capacity = self.capacity_tokens(idx.shape[0], idx.shape[1])
        route, _ = build_learned_route(self, v, self.route_group, capacity)
        for _ in range(cfg.L):
            v = self._level(v, pos, segpos, full_mask, segment_start, route)
        return v @ self.readout

    def parameter_ledger(self, top_r: int = None) -> dict:
        ledger = super().parameter_ledger(top_r if top_r else self.top_r)
        ledger["router_params"] = sum(p.numel() for p in self.router.parameters())
        ledger["route_group"] = self.route_group
        return ledger
