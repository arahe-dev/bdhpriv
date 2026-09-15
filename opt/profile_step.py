"""Profiler: one full update under torch.profiler, top ops by CUDA time.
Usage (container): python opt/profile_step.py --variant opt3_nockpt_b256
Writes results/profile_<variant>.txt (top-30 CUDA-time table).
"""

import argparse
import sys

import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, ".")

from opt.bench_matrix import ladder, variants
from opt.model_ref import (
    ArmAConfig,
    canonical_init,
    load_init,
    make_optimizer,
    synthetic_single_doc_batch,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="opt3_nockpt_b256")
    ap.add_argument("--shape", default="T2048_L8")
    ap.add_argument("--compiled", action="store_true")
    args = ap.parse_args()
    shapes = {s: (kw, gb) for s, kw, gb in ladder(True)}
    kw, gb = shapes[args.shape]
    cfg = ArmAConfig(**kw)
    dev = torch.device("cuda")
    vm = dict(variants(cfg, dev))[args.variant]
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    model = vm().to(dev)
    load_init(model, canonical_init(cfg), dev)
    model.train()
    opt = make_optimizer(model, cfg, "cuda")
    batch = synthetic_single_doc_batch(cfg, gb, dev)
    fwd = torch.compile(model, mode="default") if args.compiled else model
    from opt.model_ref import full_update
    full_update(model, opt, cfg, batch, gb, "cuda", compiled=fwd)  # warmup (+compile)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, with_stack=False) as prof:
        with record_function("full_update"):
            full_update(model, opt, cfg, batch, gb, "cuda", compiled=fwd)
    tbl = prof.key_averages(group_by_input_shape=False).table(
        sort_by="cuda_time_total", row_limit=30)
    path = f"results/profile_{args.variant}{'_compiled' if args.compiled else ''}.txt"
    with open(path, "w") as f:
        f.write(f"variant={args.variant} shape={args.shape} compiled={args.compiled}\n")
        f.write(tbl)
    print(tbl[:3000])


if __name__ == "__main__":
    main()
