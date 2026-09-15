"""Local canonical Arm-A reference, extracted from the verified timing cell.

Source: reference/arm_a_timing_cell_colab.py (executable source of truth;
canonical model/init/RoPE/selective-checkpoint defs copied verbatim from the
pinned vendor source per the cell header). No semantic changes here: shapes
are parameterized (production values are the defaults) so reduced local
shapes can run on small GPUs. Anything that changes math lives in opt/
candidates, never in this file.

Provenance caveat (see context/canonical_source_provenance.md): the original
vendor/canonical file bytes are not in this repo; do not claim this file's
hash equals the pinned vendor SHA.
"""

import math
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)


@dataclass(frozen=True)
class ArmAConfig:
    T: int = 2048
    V: int = 8192
    D: int = 256
    N: int = 16384
    H: int = 4
    L: int = 8
    HIDDEN: int = 1040
    SEED: int = 1337
    INIT_STD: float = 0.02
    THETA: float = 2**16
    READ_BLOCK: int = 256
    PEAK_LR: float = 1e-3
    BETAS: tuple = (0.9, 0.95)
    EPS: float = 1e-8
    WEIGHT_DECAY: float = 0.1
    CLIP_NORM: float = 1.0

    @property
    def K(self):
        return self.N // self.H


def canonical_init(cfg, device="cpu"):
    g = torch.Generator(device="cpu")
    g.manual_seed(cfg.SEED)

    def rnd(shape):
        return torch.randn(shape, generator=g, dtype=torch.float32) * cfg.INIT_STD

    return {
        "embedding": rnd((cfg.V, cfg.D)),
        "encoder": rnd((cfg.N, cfg.D)),
        "decoder_x": rnd((cfg.H, cfg.D, cfg.K)),
        "decoder_y": rnd((cfg.H, cfg.D, cfg.K)),
        "readout": rnd((cfg.D, cfg.V)),
        "coord_Wc": rnd((cfg.D, cfg.D)),
        "coord_bc": torch.zeros(cfg.D, dtype=torch.float32),
        "coord_alpha": torch.zeros((), dtype=torch.float32),
        "writer_W1": rnd((cfg.D, cfg.HIDDEN)),
        "writer_W2": rnd((cfg.HIDDEN, cfg.D)),
    }


def rope_pair_freq(cfg, device):
    return (
        1.0
        / (cfg.THETA ** ((2.0 * torch.arange(cfg.K // 2, dtype=torch.float32, device=device)) / cfg.K))
        / (2.0 * math.pi)
    )


def rope_bthk(q, pos, freq):
    # q [B,T,H,K] — same pairwise rotation, different physical layout.
    b, t, h, k = q.shape
    qp = q.reshape(b, t, h, k // 2, 2)
    phase = pos.float().unsqueeze(-1).unsqueeze(-1) * freq.view(1, 1, 1, -1)
    phase = torch.remainder(phase, 1.0) * (2.0 * math.pi)
    cs = torch.cos(phase).to(q.dtype)
    sn = torch.sin(phase).to(q.dtype)
    qe, qo = qp[..., 0], qp[..., 1]
    return torch.stack((qe * cs - qo * sn, qo * cs + qe * sn), dim=-1).reshape_as(q)


class DenseWriter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(cfg.D, cfg.HIDDEN))
        self.W2 = nn.Parameter(torch.empty(cfg.HIDDEN, cfg.D))

    def forward(self, x):
        return F.relu(x @ self.W1) @ self.W2


class Coordinator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.Wc = nn.Parameter(torch.empty(cfg.D, cfg.D))
        self.bc = nn.Parameter(torch.zeros(cfg.D))
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, v, segpos, full_mask):
        z = v @ self.Wc + self.bc
        prev_sum = torch.bmm(full_mask.to(dtype=z.dtype), z)
        den = segpos.clamp_min(1).to(z.dtype).unsqueeze(-1)
        c = prev_sum / den - z
        rho = torch.sigmoid(self.alpha)
        return 1.0 + rho.to(c.dtype) * torch.tanh(c)


_BMM = torch.ops.aten.bmm.default
_MATMUL = torch.ops.aten.matmul.default


def make_sac_policy(cfg):
    def native_read_sac_policy(ctx, op, *args, **kwargs):
        # Save ONLY QK score blocks, exactly analogous to the accepted full-Gram SAC policy.
        if op in (_BMM, _MATMUL) and len(args) >= 2:
            a, b = args[0], args[1]
            if hasattr(a, "shape") and hasattr(b, "shape") and len(a.shape) >= 3 and len(b.shape) >= 3:
                try:
                    if (
                        int(a.shape[-1]) == cfg.K
                        and int(b.shape[-2]) == cfg.K
                        and int(a.shape[-2]) <= cfg.READ_BLOCK
                        and int(b.shape[-1]) <= cfg.T
                    ):
                        return CheckpointPolicy.MUST_SAVE
                except Exception:
                    pass
        return CheckpointPolicy.PREFER_RECOMPUTE

    return native_read_sac_policy


class NativeReadStage1ArmA(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.V, cfg.D)
        self.encoder = nn.Parameter(torch.empty(cfg.N, cfg.D))
        self.decoder_x = nn.Parameter(torch.empty(cfg.H, cfg.D, cfg.K))
        self.decoder_y = nn.Parameter(torch.empty(cfg.H, cfg.D, cfg.K))
        self.readout = nn.Parameter(torch.empty(cfg.D, cfg.V))
        self.coordinator = Coordinator(cfg)
        self.writer = DenseWriter(cfg)
        self.ln = nn.LayerNorm(cfg.D, elementwise_affine=False, bias=False)
        self.causal = torch.ones((cfg.T, cfg.T), device=device, dtype=torch.bool).tril(diagonal=-1)
        self.rope_freq = rope_pair_freq(cfg, device)
        self.sac_context_fn = partial(create_selective_checkpoint_contexts, make_sac_policy(cfg))

    def project_x_native(self, v):
        cfg = self.cfg
        w_wide = self.decoder_x.permute(1, 0, 2).reshape(cfg.D, cfg.N)
        return F.relu((v.reshape(v.shape[0] * cfg.T, cfg.D) @ w_wide).reshape(v.shape[0], cfg.T, cfg.H, cfg.K))

    def attention_native(self, x_bt, v, pos, full_mask):
        cfg = self.cfg
        q_bt = rope_bthk(x_bt, pos, self.rope_freq)
        qh = q_bt.permute(0, 2, 1, 3)
        vh = v.unsqueeze(1).expand(-1, cfg.H, -1, -1).contiguous()
        outs = []
        for s0 in range(0, cfg.T, cfg.READ_BLOCK):
            s1 = s0 + cfg.READ_BLOCK
            qb = qh[:, :, s0:s1, :]
            kp = qh[:, :, :s1, :]
            scores = qb @ kp.transpose(-1, -2)
            scores = scores.masked_fill(~full_mask[:, None, s0:s1, :s1], 0.0)
            outs.append(scores @ vh[:, :, :s1, :])
        return torch.cat(outs, dim=2)

    def level(self, v, pos, segpos, full_mask):
        # ---------------- EXACT BDH PAPER CORE ----------------
        x_bt = self.project_x_native(v)                       # physical [B,T,H,K]
        a = self.ln(self.attention_native(x_bt, v, pos, full_mask))
        ypre = F.relu(a @ self.decoder_y)                     # unchanged [B,H,T,K]

        # Only a view back to the baseline logical orientation; write path remains unchanged.
        paper_y = x_bt.permute(0, 2, 1, 3) * ypre
        paper_y_flat = paper_y.transpose(1, 2).reshape(v.shape[0], self.cfg.T, self.cfg.N)
        base = self.ln(paper_y_flat @ self.encoder)

        # ---------------- FROZEN STAGE-1 EXTENSION ------------
        g = self.coordinator(v, segpos, full_mask)
        delta = self.writer(g * base)
        return self.ln(v + delta)

    def forward(self, idx, pos, segpos, full_mask):
        v = self.ln(self.embedding(idx))
        for _ in range(self.cfg.L):
            def level_fn(vv):
                return self.level(vv, pos, segpos, full_mask)
            v = checkpoint(
                level_fn, v,
                use_reentrant=False,
                preserve_rng_state=False,
                context_fn=self.sac_context_fn,
            )
        return v @ self.readout

    @torch.no_grad()
    def forward_eval(self, idx, pos, segpos, full_mask):
        v = self.ln(self.embedding(idx))
        for _ in range(self.cfg.L):
            v = self.level(v, pos, segpos, full_mask)
        return v @ self.readout


def load_init(model, init, device):
    with torch.no_grad():
        model.embedding.weight.copy_(init["embedding"].to(device))
        model.encoder.copy_(init["encoder"].to(device))
        model.decoder_x.copy_(init["decoder_x"].to(device))
        model.decoder_y.copy_(init["decoder_y"].to(device))
        model.readout.copy_(init["readout"].to(device))
        model.coordinator.Wc.copy_(init["coord_Wc"].to(device))
        model.coordinator.bc.copy_(init["coord_bc"].to(device))
        model.coordinator.alpha.copy_(init["coord_alpha"].to(device))
        model.writer.W1.copy_(init["writer_W1"].to(device))
        model.writer.W2.copy_(init["writer_W2"].to(device))


def ce_sum(logits, target, valid, V):
    per = F.cross_entropy(logits.reshape(-1, V), target.reshape(-1), reduction="none")
    return per[valid.reshape(-1)].sum(dtype=torch.float32)


def make_optimizer(model, cfg, device_type):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": cfg.WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups, lr=cfg.PEAK_LR, betas=cfg.BETAS, eps=cfg.EPS,
        fused=(device_type == "cuda"),
    )


def synthetic_single_doc_batch(cfg, global_batch, device, seed=0):
    """Mimic canonical single-doc full windows: start=zeros, segpos=arange,
    pos=doc offsets, full_mask=CAUSAL broadcast, mid-doc valid (all True)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randint(0, cfg.V, (global_batch, cfg.T), generator=g)
    y = torch.randint(0, cfg.V, (global_batch, cfg.T), generator=g)
    pos = torch.arange(cfg.T).unsqueeze(0).expand(global_batch, -1).contiguous()
    segpos = pos.clone()
    valid = torch.ones((global_batch, cfg.T), dtype=torch.bool)
    causal = torch.ones((cfg.T, cfg.T), dtype=torch.bool).tril(diagonal=-1)
    full_mask = causal.unsqueeze(0).expand(global_batch, -1, -1).contiguous()
    return {
        "x": x.to(device), "y": y.to(device),
        "pos": pos.to(device, dtype=torch.int32),
        "segpos": segpos.to(device, dtype=torch.int32),
        "valid": valid.to(device),
        "full_mask": full_mask.to(device),
    }


def synthetic_packed_batch(cfg, global_batch, device, seed=0, mode="mixed"):
    """Packed-document windows with realistic layouts.

    Modes: "single" (one doc, == single-doc batch), "two" (two docs split
    at T//2), "four" (four docs at quarter boundaries), "mixed" (exponential
    doc lens, mean ~1429 like the frozen corpus), "heavy" (all docs len 128).
    pos/segpos are doc-local offsets; full_mask = same-doc & strict-past.
    """
    import random
    rng = random.Random(seed)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 999)
    t = cfg.T
    seg = torch.zeros((global_batch, t), dtype=torch.long)
    for i in range(global_batch):
        if mode == "single":
            continue
        elif mode == "two":
            cuts = [t // 2]
        elif mode == "four":
            cuts = [t // 4, t // 2, 3 * t // 4]
        elif mode == "heavy":
            cuts = list(range(128, t, 128))
        else:  # mixed: exponential doc lens, mean 1429
            cuts = []
            s = 0
            while True:
                ln = max(1, int(rng.expovariate(1.0 / 1429)))
                s += ln
                if s >= t:
                    break
                cuts.append(s)
        col = []
        s = 0
        for c in cuts + [t]:
            col += [s] * (c - s)
            s = c
        seg[i] = torch.tensor(col, dtype=torch.long)
    pos = (torch.arange(t).unsqueeze(0).expand(global_batch, -1) - seg).to(torch.int32)
    segpos = pos.clone()
    same = seg[:, :, None] == seg[:, None, :]
    strict = torch.ones((t, t), dtype=torch.bool).tril(diagonal=-1)
    full_mask = (same & strict.unsqueeze(0)).contiguous()
    x = torch.randint(0, cfg.V, (global_batch, t), generator=g)
    y = torch.randint(0, cfg.V, (global_batch, t), generator=g)
    valid = torch.ones((global_batch, t), dtype=torch.bool)
    return {
        "x": x.to(device), "y": y.to(device),
        "pos": pos.to(device, dtype=torch.int32),
        "segpos": segpos.to(device, dtype=torch.int32),
        "valid": valid.to(device),
        "full_mask": full_mask.to(device),
        "segment_start": seg.to(device, dtype=torch.long),
    }


def full_update(model, optimizer, cfg, batch, microbatch, device_type, compiled=None,
                assume_single_doc=False, entry_mode=None):
    """One canonical full update: forward + CE + backward (+accum), clip, step.

    entry_mode selects the production static entry point the harness calls
    (these are what torch.compile traces in production):
      "single" -> fwd(x, pos, segpos, full_mask, segment_start=zeros)
      "packed" -> fwd(x, pos, segpos, full_mask, batch["segment_start"])
      "canonical"/None -> fwd(x, pos, segpos, full_mask)  (ref/legacy)
    assume_single_doc is the legacy alias for entry_mode="single" and is
    kept for old call sites. SAFETY: refuses to run the single-doc bypass
    on a packed batch.
    """
    if assume_single_doc and entry_mode is None:
        entry_mode = "single"
    fwd = compiled if compiled is not None else model
    denom = int(batch["valid"].sum().item())
    assert denom > 0
    optimizer.zero_grad(set_to_none=True)
    gb = batch["x"].shape[0]
    use_amp = device_type == "cuda"
    for off in range(0, gb, microbatch):
        sl = slice(off, off + microbatch)
        x = batch["x"][sl]
        y = batch["y"][sl]
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                            cache_enabled=False, enabled=use_amp):
            if entry_mode == "single":
                seg = batch.get("segment_start", None)
                if seg is not None and bool((seg != 0).any()):
                    raise RuntimeError("single-doc entry set on a packed batch")
                seg = torch.zeros((x.shape[0], cfg.T), dtype=torch.long, device=x.device)
                logits = fwd(x, batch["pos"][sl], batch["segpos"][sl],
                             batch["full_mask"][sl], seg)
            elif entry_mode == "packed":
                seg = batch.get("segment_start", None)
                if seg is None:
                    raise RuntimeError("packed entry requires batch segment_start")
                logits = fwd(x, batch["pos"][sl], batch["segpos"][sl],
                             batch["full_mask"][sl], seg[sl])
            else:
                logits = fwd(x, batch["pos"][sl], batch["segpos"][sl], batch["full_mask"][sl])
            loss = ce_sum(logits, y, batch["valid"][sl], cfg.V) / denom
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_NORM)
    optimizer.step()
    return denom
