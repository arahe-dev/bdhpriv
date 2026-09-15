"""Correctness gate for DietArmA: identical math to canonical.

Covers diet_rope x diet_ypre_paper x ckpt x coord x scan x single-doc/packed.
fp32 CPU eager, tolerance-gated (1e-4 logits, 1e-3 grads).
"""

import sys

import torch

sys.path.insert(0, ".")

from functools import partial

from torch.utils.checkpoint import create_selective_checkpoint_contexts

from opt.model_diet import DietArmA
from opt.model_ref import (
    ArmAConfig,
    NativeReadStage1ArmA,
    canonical_init,
    load_init,
    make_sac_policy,
)
from opt.test_model_equiv import packed_batch, run_grads


def main():
    torch.manual_seed(0)
    cfg = ArmAConfig(T=12, V=48, D=12, N=48, H=2, L=2, HIDDEN=24)
    dev = torch.device("cpu")
    sac = partial(create_selective_checkpoint_contexts, make_sac_policy(cfg))
    worst = 0.0
    n = 0
    for seed in (0, 1):
        for single in (False, True):
            batch = packed_batch(cfg, 2, seed, single=single)
            ref = NativeReadStage1ArmA(cfg, dev)
            init = canonical_init(cfg)
            load_init(ref, init, dev)
            ref.train()
            r_logits, r_loss, r_grads = run_grads(ref, batch, cfg)
            for blk in (3, 12):
                for ckpt in (True, False):
                    for coord in ("prefix", "dense"):
                        for scan in ("chunkwise", "parallel"):
                            for dr, dy in ((True, True), (True, False), (False, True)):
                                opt = DietArmA(
                                    cfg, dev, scan_block=blk, use_checkpoint=ckpt,
                                    sac_context_fn=sac if ckpt else None,
                                    coord=coord, single_scan=scan,
                                    diet_rope=dr, diet_ypre_paper=dy)
                                load_init(opt, init, dev)
                                opt.train()
                                o_logits, o_loss, o_grads = run_grads(opt, batch, cfg)
                                d = float((o_logits - r_logits).abs().max())
                                dg = max(float((o_grads[k] - r_grads[k]).abs().max())
                                         for k in r_grads)
                                worst = max(worst, d, dg)
                                torch.testing.assert_close(o_logits, r_logits,
                                                           rtol=1e-4, atol=1e-4)
                                for k in r_grads:
                                    torch.testing.assert_close(o_grads[k], r_grads[k],
                                                               rtol=1e-3, atol=1e-3)
                                n += 1
    print({"status": "DIET_GATE_PASS", "configs": n, "worst": worst})


if __name__ == "__main__":
    main()
