"""Memory-diet Arm-A: identical math, fewer saved activations.

Subclasses OptArmA (same params/init/contract). Two exact mechanisms:

1. diet_rope: RoPE wrapped in non-reentrant checkpoint. Saves x_bt (already
   retained for the paper path) instead of q_bt + RoPE intermediates, and --
   critically -- stops inductor from recomputing the 17.2 GF project_x GEMM
   in backward (its output x_bt is now saved by the checkpoint boundary).
   Expected: less memory AND less compute (double win).

2. diet_ypre_paper: Dy GEMM + relu + paper multiply + flat reshape fused into
   ONE checkpointed function of (a_ln, x_bt). Saves only a_ln (4 MB, already
   retained) and x_bt (already retained). Drops ypre, paper_y, paper_y_flat
   saves entirely. Paper is built directly contiguous [B,T,H,K] so the flat
   reshape is a view (no copy). Recompute in backward: Dy GEMM (already
   recomputed by inductor today) + relu + multiply.

Both use checkpoint without RNG (preserve_rng_state=False); autocast state
is recorded/re-applied by checkpoint. Exact: recomputation is bitwise the
same ops, gated to 1e-8 fp32 vs canonical.
"""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from opt.model_opt import OptArmA, segment_start_from_full_mask
from opt.model_ref import rope_bthk


class DietArmA(OptArmA):
    def __init__(self, *args, diet_rope=True, diet_ypre_paper=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.diet_rope = diet_rope
        self.diet_ypre_paper = diet_ypre_paper

    def _rope_q(self, x_bt, pos):
        # Returns roped queries in [B,H,T,K] layout.
        return rope_bthk(x_bt, pos, self.rope_freq).permute(0, 2, 1, 3)

    def _fused_ypre_paper(self, a, x_bt):
        # a: [B,H,T,Dv] post-LN attention out. Returns paper flat [B,T,N].
        # Paper built directly contiguous [B,T,H,K] so reshape is a view.
        ypre = F.relu(a @ self.decoder_y)  # [B,H,T,K], like canonical
        b, t = x_bt.shape[0], x_bt.shape[1]
        pb = (x_bt.permute(0, 2, 1, 3) * ypre).permute(0, 2, 1, 3)
        return pb.reshape(b, t, -1)

    def attention_scan_diet(self, x_bt, v, pos, segment_start, single_doc=False):
        from opt.scan_attn import scan_chunkwise_bthk, scan_hybrid_bthk, scan_parallel_bthk
        if self.diet_rope:
            qh = checkpoint(self._rope_q, x_bt, pos, use_reentrant=False,
                            preserve_rng_state=False)
        else:
            qh = self._rope_q(x_bt, pos)
        cfg = self.cfg
        vh = v.unsqueeze(1).expand(-1, cfg.H, -1, -1)
        if single_doc:
            if self.single_scan == "parallel":
                return scan_parallel_bthk(qh, vh, block=self.scan_block)
            if self.single_scan == "hybrid":
                return scan_hybrid_bthk(qh, vh, block=self.scan_block)
            return scan_chunkwise_bthk(qh, vh, segment_start, block=self.scan_block,
                                       single_doc=True)
        return scan_chunkwise_bthk(qh, vh, segment_start, block=self.scan_block,
                                   single_doc=False)

    def level(self, v, pos, segpos, full_mask, segment_start, single_doc=False):
        x_bt = self.project_x_native(v)
        a = self.ln(self.attention_scan_diet(x_bt, v, pos, segment_start, single_doc))
        if self.diet_ypre_paper:
            paper_y_flat = checkpoint(self._fused_ypre_paper, a, x_bt,
                                      use_reentrant=False, preserve_rng_state=False)
        else:
            ypre = F.relu(a @ self.decoder_y)
            paper_y = x_bt.permute(0, 2, 1, 3) * ypre
            paper_y_flat = paper_y.transpose(1, 2).reshape(
                v.shape[0], self.cfg.T, self.cfg.N)
        base = self.ln(paper_y_flat @ self.encoder)
        if self.coord == "prefix":
            g = self.coordinator(v, segpos, full_mask, segment_start, single_doc)
        else:
            g = self.coordinator(v, segpos, full_mask)
        delta = self.writer(g * base)
        return self.ln(v + delta)

    def forward(self, idx, pos, segpos, full_mask, segment_start=None,
                single_doc=None):
        # Optional explicit single_doc avoids the .item() graph break and the
        # [B,T,T] segment derivation inside the compiled region.
        if segment_start is None:
            segment_start = segment_start_from_full_mask(full_mask)
        if single_doc is None:
            single_doc = bool((segment_start == 0).all())
        v = self.ln(self.embedding(idx))
        for _ in range(self.cfg.L):
            def level_fn(vv):
                return self.level(vv, pos, segpos, full_mask, segment_start, single_doc)
            if self.use_checkpoint:
                v = checkpoint(level_fn, v, use_reentrant=False,
                               preserve_rng_state=False,
                               context_fn=self.sac_context_fn)
            else:
                v = level_fn(v)
        return v @ self.readout
