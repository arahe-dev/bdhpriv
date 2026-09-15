"""E0: fwd-only vs full-update CUDA profiles for a variant.

Usage (container):
  python opt/e0_profile.py --variant opt3c_nockpt_b1024 --shape T2048_L8 [--compiled]
Writes results/e0_<variant>_<shape>_{fwd,full}.txt
"""

import argparse
import sys

import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, ".")

from opt.bench_matrix import ladder
from opt.model_opt import OptArmA
from opt.model_ref import (
    ArmAConfig,
    canonical_init,
    ce_sum,
    load_init,
    make_optimizer,
    synthetic_single_doc_batch,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--coord", default="dense", choices=("dense", "prefix"))
    ap.add_argument("--scan", default="chunkwise", choices=("chunkwise", "parallel"))
    ap.add_argument("--ckpt", action="store_true")
    ap.add_argument("--shape", default="T2048_L8")
    ap.add_argument("--compiled", action="store_true")
    args = ap.parse_args()
    shapes = {s: (kw, gb) for s, kw, gb in ladder(True)}
    kw, gb = shapes[args.shape]
    cfg = ArmAConfig(**kw)
    dev = torch.device("cuda")
    from opt.model_ref import make_sac_policy
    from functools import partial
    from torch.utils.checkpoint import create_selective_checkpoint_contexts
    sac = partial(create_selective_checkpoint_contexts, make_sac_policy(cfg))
    model = OptArmA(cfg, dev, scan_block=args.block, use_checkpoint=args.ckpt,
                    sac_context_fn=sac if args.ckpt else None,
                    coord=args.coord, single_scan=args.scan).to(dev)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    load_init(model, canonical_init(cfg), dev)
    model.train()
    opt = make_optimizer(model, cfg, "cuda")
    batch = synthetic_single_doc_batch(cfg, gb, dev)
    fwd = torch.compile(model, mode="default") if args.compiled else model
    mb = gb
    sl = slice(0, mb)

    def fwd_only():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False):
            logits = fwd(batch["x"][sl], batch["pos"][sl], batch["segpos"][sl],
                         batch["full_mask"][sl])
        return logits

    def full():
        from opt.model_ref import full_update
        full_update(model, opt, cfg, batch, mb, "cuda", compiled=fwd)

    # warmup (+compile)
    fwd_only()
    full()
    torch.cuda.synchronize()

    tag = f"opt3_scan{args.scan}_b{args.block}_{args.shape}{'_compiled' if args.compiled else ''}"
    for name, fn in (("fwd", fwd_only), ("full", full)):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True, with_stack=False) as prof:
            with record_function(name):
                fn()
        torch.cuda.synchronize()
        tbl = prof.key_averages(group_by_input_shape=True).table(
            sort_by="cuda_time_total", row_limit=40)
        path = f"results/e0_{tag}_{name}.txt"
        with open(path, "w") as f:
            f.write(f"tag={tag} shape={args.shape} compiled={args.compiled}\n")
            f.write(tbl)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
