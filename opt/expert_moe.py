"""MoE / resonant Arm-A derivative family (local autoresearch campaign).

Frozen Arm-A (opt.model_opt.OptArmA) is never modified. This module forks it
into an expertized family with RoPE-pair-preserving experts:

  ExpertizedArmA:
    - partition each head's stored K coordinates into M contiguous experts
      (Ke coords each, Ke even; expert e owns RoPE pair band
      [e*Ke/2, (e+1)*Ke/2))
    - attention decomposes additively over the neuron axis:
        a_total[t] = sum_e a_e[t]
      because q_t^T q_s = sum_e q_{t,e}^T q_{s,e}
    - the shared D-dimensional a_total (LN applied after the sum) drives
      every active expert's Dy slice -- the Arm-A-faithful coupling
    - all-active (top_r = M) is numerically equivalent to Arm-A

  Routing (top_r < M) is a later iteration in this file; this stage is the
  exact expertized foundation (campaign experiment 01). Oscillator flags
  O1 (per-expert learned frequency scale beta, init 0) and O2 (per-expert
  band amplitude gamma, init 1) are present but default-off.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from opt.model_ref import Coordinator, DenseWriter, rope_pair_freq
from opt.scan_attn import scan_chunkwise_bthk


def _rope_bands(q, pos, freq_he):
    """RoPE with per-head band frequencies. q: (B,T,H,Ke), freq: (H,Ke/2)."""
    b, t, h, k = q.shape
    qp = q.reshape(b, t, h, k // 2, 2)
    phase = pos.float().unsqueeze(-1).unsqueeze(-1) * freq_he.view(1, 1, h, -1)
    phase = torch.remainder(phase, 1.0) * (2.0 * math.pi)
    cs = torch.cos(phase)
    sn = torch.sin(phase)
    qe, qo = qp[..., 0], qp[..., 1]
    return torch.stack((qe * cs - qo * sn, qo * cs + qe * sn),
                       dim=-1).reshape_as(q)


class ExpertizedArmA(nn.Module):
    def __init__(self, cfg, device, experts: int = 8, expert_width: int = None,
                 scan_block: int = 1024, learn_freq_scale: bool = False,
                 learn_band_amp: bool = False):
        super().__init__()
        self.cfg = cfg
        self.M = int(experts)
        self.Ke = int(expert_width if expert_width is not None
                      else cfg.K // self.M)
        assert self.Ke % 2 == 0, "expert width must preserve RoPE pairs"
        assert self.M * self.Ke == cfg.K or self.M * self.Ke % 2 == 0
        self.stored_K = self.M * self.Ke
        self.scan_block = int(scan_block)
        self.learn_freq_scale = bool(learn_freq_scale)
        self.learn_band_amp = bool(learn_band_amp)

        self.embedding = nn.Embedding(cfg.V, cfg.D)
        self.encoder = nn.Parameter(torch.empty(cfg.H * self.stored_K, cfg.D))
        self.decoder_x = nn.Parameter(torch.empty(cfg.H, cfg.D, self.stored_K))
        self.decoder_y = nn.Parameter(torch.empty(cfg.H, cfg.D, self.stored_K))
        self.readout = nn.Parameter(torch.empty(cfg.D, cfg.V))
        self.coordinator = Coordinator(cfg)
        self.writer = DenseWriter(cfg)
        self.ln = nn.LayerNorm(cfg.D, elementwise_affine=False, bias=False)
        base_freq = rope_pair_freq(cfg, device)
        self.register_buffer(
            "rope_freq_bands",
            base_freq.reshape(self.M, self.Ke // 2).clone(),
        )
        if self.learn_freq_scale:
            self.beta = nn.Parameter(torch.zeros(cfg.H, self.M))
        if self.learn_band_amp:
            self.gamma = nn.Parameter(torch.ones(cfg.H, self.M))

    # -- parameter slicing --------------------------------------------------

    def load_canonical(self, state_dict):
        """Load a canonical Arm-A state dict (same stored width only)."""
        own = self.state_dict()
        accepted = {
            key: value for key, value in state_dict.items()
            if key in own and own[key].shape == value.shape
        }
        missing = [key for key in own
                   if key not in accepted
                   and key not in ("rope_freq_bands", "beta", "gamma")]
        if missing:
            raise ValueError(f"cannot load canonical state: missing {missing}")
        self.load_state_dict(accepted, strict=False)

    def _dx(self, expert: int):
        return self.decoder_x.view(
            self.cfg.H, self.cfg.D, self.M, self.Ke)[:, :, expert]

    def _dy(self, expert: int):
        return self.decoder_y.view(
            self.cfg.H, self.cfg.D, self.M, self.Ke)[:, :, expert]

    def _e_rows(self, expert: int):
        return self.encoder.view(
            self.cfg.H, self.M, self.Ke, self.cfg.D)[:, expert]

    def _freq(self, expert: int):
        band = self.rope_freq_bands[expert].view(1, self.Ke // 2)
        if self.learn_freq_scale:
            return band * torch.exp(self.beta[:, expert]).view(-1, 1)
        return band.expand(self.cfg.H, -1)

    # -- forward ------------------------------------------------------------

    def _level(self, v, pos, segpos, full_mask, segment_start):
        cfg = self.cfg
        b, t, _ = v.shape
        vh = v.unsqueeze(1).expand(-1, cfg.H, -1, -1)
        x_experts = []
        a_pre = None
        for e in range(self.M):
            freq = self._freq(e)
            x_e = F.relu(torch.einsum("btd,hdk->bthk", v, self._dx(e)))
            if self.learn_band_amp:
                x_e = x_e * self.gamma[:, e].view(1, 1, -1, 1)
            x_experts.append(x_e)
            q_e = _rope_bands(x_e, pos, freq).permute(0, 2, 1, 3)
            a_e = scan_chunkwise_bthk(
                q_e, vh, segment_start, block=self.scan_block,
                skip_zero_carry=True)
            a_pre = a_e if a_pre is None else a_pre + a_e
        a = self.ln(a_pre)
        base = None
        for e in range(self.M):
            y_e = F.relu(torch.einsum("bhtd,hdk->bthk", a, self._dy(e)))
            u_e = x_experts[e] * y_e
            contribution = torch.einsum("bthk,hkd->btd", u_e, self._e_rows(e))
            base = contribution if base is None else base + contribution
        base = self.ln(base)
        g = self.coordinator(v, segpos, full_mask)
        delta = self.writer(g * base)
        return self.ln(v + delta)

    def forward_packed(self, idx, pos, segpos, full_mask, segment_start=None):
        cfg = self.cfg
        v = self.ln(self.embedding(idx))
        for _ in range(cfg.L):
            v = self._level(v, pos, segpos, full_mask, segment_start)
        return v @ self.readout

    # -- accounting ---------------------------------------------------------

    def parameter_ledger(self, top_r: int = None) -> dict:
        active = self.M if top_r is None else int(top_r)
        stored = sum(p.numel() for p in self.parameters())
        neuron_stored = self.cfg.H * self.stored_K
        neuron_active = self.cfg.H * active * self.Ke
        return {
            "M": self.M,
            "Ke": self.Ke,
            "top_r": active,
            "stored_params": int(stored),
            "stored_K_per_head": self.stored_K,
            "active_K_per_head": active * self.Ke,
            "stored_N": neuron_stored,
            "active_N": neuron_active,
            "active_fraction": active / self.M,
        }
