"""Compile smoke: does OptArmA survive inductor, and do compiled logits
match eager? Fast gate before any compiled benchmark. Usage:
python opt/compile_smoke.py [--shape T512_L2]  (runs in container/CUDA).
"""

import argparse
import sys
import time

import torch

sys.path.insert(0, ".")

from opt.model_opt import OptArmA
from opt.model_ref import (
    ArmAConfig,
    NativeReadStage1ArmA,
    canonical_init,
    load_init,
    make_optimizer,
    make_sac_policy,
    synthetic_single_doc_batch,
)
from functools import partial
from torch.utils.checkpoint import create_selective_checkpoint_contexts

SHAPES = {
    "T256_L2": (dict(T=256, L=2), 2),
    "T512_L8": (dict(T=512, L=8), 1),
    "T2048_L8": (dict(T=2048, L=8), 1),
}


def one_update(fwd, model, opt, cfg, batch, mb):
    opt.zero_grad(set_to_none=True)
    from opt.model_ref import ce_sum
    denom = int(batch["valid"].sum().item())
    for off in range(0, batch["x"].shape[0], mb):
        sl = slice(off, off + mb)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False):
            logits = fwd(batch["x"][sl], batch["pos"][sl], batch["segpos"][sl],
                         batch["full_mask"][sl])
            from opt.model_ref import ce_sum
            loss = ce_sum(logits, batch["y"][sl], batch["valid"][sl], cfg.V) / denom
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_NORM)
    opt.step()
    return logits.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="T256_L2", choices=sorted(SHAPES))
    ap.add_argument("--coord", default="prefix", choices=("prefix", "dense"))
    ap.add_argument("--scan", default="parallel", choices=("parallel", "chunkwise"))
    args = ap.parse_args()
    kw, gb = SHAPES[args.shape]
    cfg = ArmAConfig(**kw)
    dev = torch.device("cuda")
    sac = partial(create_selective_checkpoint_contexts, make_sac_policy(cfg))
    batch = synthetic_single_doc_batch(cfg, gb, dev)
    for name, mk in (
        ("ref", lambda: NativeReadStage1ArmA(cfg, dev)),
        ("opt2", lambda: OptArmA(cfg, dev, scan_block=256, use_checkpoint=False, coord=args.coord, single_scan=args.scan)),
    ):
        model = mk().to(dev)
        load_init(model, canonical_init(cfg), dev)
        model.train()
        opt = make_optimizer(model, cfg, "cuda")
        eager_logits = one_update(model, model, opt, cfg, batch, gb)
        try:
            cmodel = torch.compile(model, mode="default")
            t0 = time.perf_counter()
            torch.cuda.synchronize()
            comp_logits = one_update(cmodel, model, opt, cfg, batch, gb)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            d = float((comp_logits.float() - eager_logits.float()).abs().max())
            print({"variant": name, "compiled": "OK", "compile_plus_1upd_s": round(dt, 1),
                   "logit_max_vs_eager": d}, flush=True)
        except Exception as e:
            print({"variant": name, "compiled": "FAIL",
                   "error": f"{type(e).__name__}: {str(e)[:200]}"}, flush=True)


if __name__ == "__main__":
    main()
