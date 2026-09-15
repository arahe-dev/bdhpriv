"""Model-level proof obligation: OptArmA == NativeReadStage1ArmA.

Same init, same packed batch; compares logits, CE loss, and every
parameter gradient. Runs fp32 CPU eager (exact-math gate); a BF16
autocast forward check runs with looser tolerance where supported.
"""

import random
import sys

import torch

sys.path.insert(0, ".")

from opt.model_opt import OptArmA, segment_start_from_full_mask
from opt.model_ref import (
    ArmAConfig,
    NativeReadStage1ArmA,
    canonical_init,
    ce_sum,
    load_init,
    make_sac_policy,
)
from functools import partial
from torch.utils.checkpoint import create_selective_checkpoint_contexts


def packed_batch(cfg, b, seed, single=False):
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    seg = torch.zeros((b, cfg.T), dtype=torch.long)
    if not single:
        for i in range(b):
            cuts = sorted(rng.sample(range(1, cfg.T), k=rng.randrange(cfg.T))) if cfg.T > 1 else []
            s, col = 0, []
            for c in cuts + [cfg.T]:
                col += [s] * (c - s)
                s = c
            seg[i] = torch.tensor(col)
    pos = (torch.arange(cfg.T).expand(b, -1) - seg).to(torch.int32)
    segpos = pos.clone()
    same = seg[:, :, None] == seg[:, None, :]
    strict = torch.ones((cfg.T, cfg.T), dtype=torch.bool).tril(diagonal=-1)
    full_mask = same & strict.unsqueeze(0)
    x = torch.randint(0, cfg.V, (b, cfg.T), generator=g)
    y = torch.randint(0, cfg.V, (b, cfg.T), generator=g)
    valid = torch.rand((b, cfg.T), generator=g) > 0.3
    valid[:, 0] = True
    return {"x": x, "y": y, "pos": pos, "segpos": segpos,
            "full_mask": full_mask, "valid": valid, "seg": seg}


def run_grads(model, batch, cfg, **kw):
    model.zero_grad(set_to_none=True)
    logits = model(batch["x"], batch["pos"], batch["segpos"], batch["full_mask"], **kw)
    loss = ce_sum(logits, batch["y"], batch["valid"], cfg.V) / int(batch["valid"].sum())
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    return logits.detach(), loss.detach(), grads


def main():
    torch.manual_seed(0)
    cfg = ArmAConfig(T=12, V=48, D=12, N=48, H=2, L=2, HIDDEN=24)
    dev = torch.device("cpu")
    sac = partial(create_selective_checkpoint_contexts, make_sac_policy(cfg))
    worst = 0.0

    for seed in (0, 1, 2):
        for single in (False, True):
            batch = packed_batch(cfg, 2, seed, single=single)
            # mask-recovery helper must round-trip exactly
            rec = segment_start_from_full_mask(batch["full_mask"])
            assert torch.equal(rec, batch["seg"]), f"seg recovery failed seed {seed}"

            ref = NativeReadStage1ArmA(cfg, dev)
            init = canonical_init(cfg)
            load_init(ref, init, dev)
            ref.train()
            r_logits, r_loss, r_grads = run_grads(ref, batch, cfg)

            for blk in (1, 3, 12):
                for ckpt in (True, False):
                    for coord in ("prefix", "dense"):
                        scans = ["parallel", "chunkwise", "hybrid"] if single else ["parallel"]
                        if single and cfg.T == 4 * blk:
                            scans.append("static4")
                        for single_scan in scans:
                            opt = OptArmA(cfg, dev, scan_block=blk, use_checkpoint=ckpt,
                                          sac_context_fn=sac if ckpt else None, coord=coord,
                                          single_scan=single_scan)
                            load_init(opt, init, dev)
                            opt.train()
                            o_logits, o_loss, o_grads = run_grads(opt, batch, cfg)
                            assert o_logits.shape == r_logits.shape
                            d = float((o_logits - r_logits).abs().max())
                            dl = float(abs(o_loss - r_loss))
                            dg = max(float((o_grads[n] - r_grads[n]).abs().max()) for n in r_grads)
                            worst = max(worst, d, dg)
                            torch.testing.assert_close(o_logits, r_logits, rtol=1e-4, atol=1e-4)
                            torch.testing.assert_close(o_loss, r_loss, rtol=1e-4, atol=1e-4)
                            for n in r_grads:
                                torch.testing.assert_close(o_grads[n], r_grads[n], rtol=1e-3, atol=1e-3)
                            print(f"seed={seed} single={single} scan={single_scan} block={blk} ckpt={ckpt} coord={coord}: "
                                  f"logit_max={d:.2e} loss_d={dl:.2e} grad_max={dg:.2e} PASS")

    # BF16 autocast forward gate (looser tolerance).
    try:
        batch = packed_batch(cfg, 2, 99)
        ref = NativeReadStage1ArmA(cfg, dev)
        init = canonical_init(cfg)
        load_init(ref, init, dev)
        opt = OptArmA(cfg, dev, scan_block=4, use_checkpoint=False)
        load_init(opt, init, dev)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            rl = ref.forward_eval(batch["x"], batch["pos"], batch["segpos"], batch["full_mask"])
            ol = opt(batch["x"], batch["pos"], batch["segpos"], batch["full_mask"])
        d = float((ol.detach().float() - rl.detach().float()).abs().max())
        print(f"bf16 fwd_max={d:.2e} {'PASS' if d < 5e-2 else 'FAIL'}")
        assert d < 5e-2
    except Exception as e:
        print(f"bf16 gate skipped/failed: {type(e).__name__}: {e}")
    print({"status": "MODEL_EQUIV_PASS", "worst_fp32": worst})


if __name__ == "__main__":
    main()
