"""Optimized Arm-A candidates. Same parameters, same math, new execution.

Variant opt1 (this file): exact score-free chunked-scan attention +
segmented-prefix coordinator, optional activation checkpointing.
Parameter names/shapes are identical to opt.model_ref so canonical
inits load unchanged and B/C/D adapters see the same boundary.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from opt.model_ref import Coordinator, DenseWriter, rope_bthk
from opt.scan_attn import (
    CAUSAL_CACHE,
    scan_chunkwise_bthk,
    scan_chunkwise_where_bthk,
    scan_hybrid_bthk,
    scan_parallel_bthk,
    scan_static4_bthk,
)


def rope_phase(pos, freq):
    """RoPE cos/sin phase tables (identical values to rope_bthk internals)."""
    phase = pos.float().unsqueeze(-1).unsqueeze(-1) * freq.view(1, 1, 1, -1)
    phase = torch.remainder(phase, 1.0) * (2.0 * math.pi)
    return torch.cos(phase), torch.sin(phase)


def rope_bthk_pre(q, cs, sn):
    """RoPE with precomputed phase tables (exact same arithmetic)."""
    b, t, h, k = q.shape
    qp = q.reshape(b, t, h, k // 2, 2)
    qe, qo = qp[..., 0], qp[..., 1]
    cs = cs.to(q.dtype)
    sn = sn.to(q.dtype)
    return torch.stack((qe * cs - qo * sn, qo * cs + qe * sn), dim=-1).reshape_as(q)


def segment_start_from_full_mask(full_mask):
    """Recover per-token document-start columns from the canonical mask.

    same-doc is symmetric: same = full_mask | full_mask^T, plus the diagonal
    (a token is always in its own document; the strict-past mask excludes
    it). seg_start(t) is the first True entry of same[t]. Exact for any
    packed layout the canonical harness can express.
    """
    t = full_mask.shape[-1]
    eye = torch.eye(t, dtype=torch.bool, device=full_mask.device)
    same = full_mask | full_mask.transpose(-1, -2) | eye
    cols = torch.arange(t, device=same.device).view(1, 1, t)
    big = torch.full((), t, dtype=torch.long, device=same.device)
    first = torch.where(same, cols.expand_as(same), big).min(dim=-1).values
    return torch.where(first == t, cols.squeeze(0).squeeze(0).expand_as(first), first).to(torch.long)


def scan_chunked_bthk(qh, vh, segment_start, block=128):
    """Exact chunked scan. qh:[B,H,T,K] vh:[B,H,T,Dv] -> [B,H,T,Dv].

    Carries state S [B,H,K,Dv]; per-block terms from a short cumsum over
    on-the-fly outer products (per-block transient only, never T-global).
    Pure-torch prototype: bitwise-different summation order vs dense is
    possible; tolerance-gated, not bitwise-gated.
    """
    b, h, t, k = qh.shape
    dv = vh.shape[-1]
    q = qh.reshape(b, h, t, k)
    vv = vh
    state = torch.zeros((b, h, k, dv), dtype=q.dtype, device=q.device)
    outs = []
    for t0 in range(0, t, block):
        t1 = min(t0 + block, t)
        qb = q[:, :, t0:t1]          # [B,H,W,K]
        vb = vv[:, :, t0:t1]         # [B,H,W,Dv]
        # Block outer products (transient [B,H,W,K*Dv]).
        ab = (qb.unsqueeze(-1) * vb.unsqueeze(-2)).reshape(b, h, t1 - t0, k * dv)
        loc = torch.cumsum(ab, dim=2)
        loc_shift = torch.cat((torch.zeros_like(loc[:, :, :1]), loc[:, :, :-1]), dim=2)
        seg = segment_start[:, t0:t1]
        cont = seg < t0
        j = (seg - 1 - t0).clamp_min(0).view(b, 1, t1 - t0, 1).expand(b, h, t1 - t0, k * dv)
        lbase = loc.gather(2, j)
        at_t0 = (seg == t0).view(b, 1, t1 - t0, 1).expand_as(lbase)
        lbase = torch.where(at_t0, torch.zeros_like(lbase), lbase)
        s_t = torch.where(
            cont.view(b, 1, t1 - t0, 1).expand_as(loc_shift),
            state.reshape(b, h, 1, k * dv).expand_as(loc_shift) + loc_shift,
            loc_shift - lbase,
        ).reshape(b, h, t1 - t0, k, dv)
        outs.append(torch.einsum("bhwk,bhwkd->bhwd", qb, s_t))
        # S_{t+1} = S_t + A_t with reset at document starts.
        s_last = s_t[:, :, -1].reshape(b, h, k * dv) + ab[:, :, -1]
        if t1 < t:
            new_doc = segment_start[:, t1] != segment_start[:, t1 - 1]
            s_last = torch.where(
                new_doc.view(b, 1, 1).expand(b, h, k * dv),
                torch.zeros_like(s_last), s_last,
            )
        state = s_last.reshape(b, h, k, dv)
    return torch.cat(outs, dim=2)


def prefix_coordinator_value(v, segpos, segment_start, Wc, bc, single_doc=False):
    z = v @ Wc + bc
    b, t, w = z.shape
    # Last-dim scan on transposed view == cumsum(z, dim=1); keeps the scan
    # dimension contiguous for inductor (see scan_attn note).
    incl = z.transpose(1, 2).cumsum(-1).transpose(1, 2)
    before = torch.cat((torch.zeros_like(z[:, :1]), incl[:, :-1]), dim=1)
    if single_doc:
        prev = before
    else:
        idx = segment_start.clamp(0, t - 1).unsqueeze(-1).expand(b, t, w)
        prev = before - before.gather(1, idx)
    den = segpos.clamp_min(1).to(z.dtype).unsqueeze(-1)
    return prev / den - z


class PrefixCoordinator(Coordinator):
    """Same parameters as canonical Coordinator; exact prefix execution."""

    def forward(self, v, segpos, full_mask, segment_start=None, single_doc=False):
        if segment_start is None:
            segment_start = segment_start_from_full_mask(full_mask)
        c = prefix_coordinator_value(v, segpos, segment_start, self.Wc, self.bc,
                                     single_doc=single_doc)
        rho = torch.sigmoid(self.alpha)
        return 1.0 + rho.to(c.dtype) * torch.tanh(c)


class OptArmA(nn.Module):
    def __init__(self, cfg, device, scan_block=128, use_checkpoint=True,
                 sac_context_fn=None, coord="prefix", single_scan="parallel",
                 packed_update="branchfree", zero_carry=False,
                 paper_layout="flat", cache_rope=False):
        super().__init__()
        from opt.model_ref import ArmAConfig  # noqa: type-check import
        assert isinstance(cfg, ArmAConfig)
        assert coord in ("prefix", "dense")
        assert single_scan in ("parallel", "chunkwise", "hybrid", "static4")
        assert packed_update in ("branchfree", "where")
        assert paper_layout in ("flat", "direct")
        self.cfg = cfg
        self.scan_block = scan_block
        self.use_checkpoint = use_checkpoint
        self.sac_context_fn = sac_context_fn
        self.coord = coord
        self.single_scan = single_scan
        self.packed_update = packed_update
        self.zero_carry = zero_carry
        self.paper_layout = paper_layout
        self.cache_rope = cache_rope
        self.embedding = nn.Embedding(cfg.V, cfg.D)
        self.encoder = nn.Parameter(torch.empty(cfg.N, cfg.D))
        self.decoder_x = nn.Parameter(torch.empty(cfg.H, cfg.D, cfg.K))
        self.decoder_y = nn.Parameter(torch.empty(cfg.H, cfg.D, cfg.K))
        self.readout = nn.Parameter(torch.empty(cfg.D, cfg.V))
        if coord == "prefix":
            self.coordinator = PrefixCoordinator(cfg)
        else:
            from opt.model_ref import Coordinator as DenseCoordinator
            self.coordinator = DenseCoordinator(cfg)
        self.writer = DenseWriter(cfg)
        self.ln = nn.LayerNorm(cfg.D, elementwise_affine=False, bias=False)
        from opt.model_ref import rope_pair_freq
        self.rope_freq = rope_pair_freq(cfg, device)

    def project_x_native(self, v):
        cfg = self.cfg
        w_wide = self.decoder_x.permute(1, 0, 2).reshape(cfg.D, cfg.N)
        return F.relu((v.reshape(v.shape[0] * cfg.T, cfg.D) @ w_wide).reshape(
            v.shape[0], cfg.T, cfg.H, cfg.K))

    def attention_scan(self, x_bt, v, pos, segment_start, single_doc=False,
                       cs=None, sn=None):
        cfg = self.cfg
        if cs is not None:
            qh = rope_bthk_pre(x_bt, cs, sn).permute(0, 2, 1, 3)
        else:
            qh = rope_bthk(x_bt, pos, self.rope_freq).permute(0, 2, 1, 3)
        vh = v.unsqueeze(1).expand(-1, cfg.H, -1, -1)
        if single_doc:
            if self.single_scan == "parallel":
                return scan_parallel_bthk(qh, vh, block=self.scan_block)
            if self.single_scan == "hybrid":
                return scan_hybrid_bthk(qh, vh, block=self.scan_block)
            if self.single_scan == "static4":
                w = self.scan_block
                dev = qh.device
                mask = CAUSAL_CACHE.get((str(dev), w))
                if mask is None or mask.device != dev:
                    mask = torch.ones((w, w), dtype=torch.bool, device=dev).tril(diagonal=-1)
                    CAUSAL_CACHE[(str(dev), w)] = mask
                return scan_static4_bthk(qh, vh, mask)
            return self._chunkwise(qh, vh, segment_start, single_doc=True)
        return self._chunkwise(qh, vh, segment_start, single_doc=False)

    def _chunkwise(self, qh, vh, segment_start, single_doc):
        if self.packed_update == "where":
            return scan_chunkwise_where_bthk(
                qh, vh, segment_start, block=self.scan_block,
                single_doc=single_doc)
        return scan_chunkwise_bthk(qh, vh, segment_start,
                                   block=self.scan_block, single_doc=single_doc,
                                   skip_zero_carry=self.zero_carry)

    def level(self, v, pos, segpos, full_mask, segment_start, single_doc=False,
              cs=None, sn=None):
        x_bt = self.project_x_native(v)
        a = self.ln(self.attention_scan(x_bt, v, pos, segment_start, single_doc,
                                        cs, sn))
        ypre = F.relu(a @ self.decoder_y)
        if self.paper_layout == "direct":
            # Exact same elementwise product, written straight into the
            # [B,T,H,K] layout so the N-flatten is a view (no transpose copy).
            prod = x_bt * ypre.permute(0, 2, 1, 3)
            paper_y_flat = prod.reshape(v.shape[0], self.cfg.T, self.cfg.N)
        else:
            paper_y = x_bt.permute(0, 2, 1, 3) * ypre
            paper_y_flat = paper_y.transpose(1, 2).reshape(v.shape[0], self.cfg.T, self.cfg.N)
        base = self.ln(paper_y_flat @ self.encoder)
        if self.coord == "prefix":
            g = self.coordinator(v, segpos, full_mask, segment_start, single_doc)
        else:
            g = self.coordinator(v, segpos, full_mask)
        delta = self.writer(g * base)
        return self.ln(v + delta)

    def _forward_impl(self, idx, pos, segpos, full_mask, segment_start,
                      single_doc):
        v = self.ln(self.embedding(idx))
        cs = sn = None
        if self.cache_rope:
            # pos is identical across all levels; compute phase tables once.
            cs, sn = rope_phase(pos, self.rope_freq)
        for _ in range(self.cfg.L):
            def level_fn(vv):
                return self.level(vv, pos, segpos, full_mask, segment_start,
                                  single_doc, cs, sn)
            if self.use_checkpoint:
                v = checkpoint(level_fn, v, use_reentrant=False,
                               preserve_rng_state=False,
                               context_fn=self.sac_context_fn)
            else:
                v = level_fn(v)
        return v @ self.readout

    def forward_single_doc(self, idx, pos, segpos, full_mask,
                           segment_start=None):
        """Static production entry: clean same-document windows.

        No host sync, no .item()/.all(), no segment derivation: safe to
        torch.compile directly (this is what the G4 benchmark compiles).
        """
        return self._forward_impl(idx, pos, segpos, full_mask,
                                  segment_start, True)

    def forward_packed(self, idx, pos, segpos, full_mask, segment_start=None):
        """Static production entry: packed documents (no host syncs).

        segment_start must be supplied by the data pipeline; if absent it is
        derived from full_mask with pure tensor ops (no .item()).
        """
        if segment_start is None:
            segment_start = segment_start_from_full_mask(full_mask)
        return self._forward_impl(idx, pos, segpos, full_mask,
                                  segment_start, False)

    def forward(self, idx, pos, segpos, full_mask, segment_start=None,
                single_doc=None):
        # Eager dispatcher for tests/tools. Production compiled paths use
        # forward_single_doc / forward_packed directly.
        if segment_start is None:
            segment_start = segment_start_from_full_mask(full_mask)
        if single_doc is None:
            single_doc = bool((segment_start == 0).all())
        return self._forward_impl(idx, pos, segpos, full_mask, segment_start,
                                  single_doc)
