"""E0: saved-tensor memory census via saved_tensors_hooks.

Runs one forward+backward for a variant and records every tensor autograd
saves, with sizes. Answers: what actually dominates activation memory?

Usage (container): python opt/e0_memcensus.py --variant opt3c_nockpt_b1024 [--compiled]
Writes results/e0_memcensus_<variant>[_compiled].txt
"""

import argparse
import collections
import sys

import torch

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

    saved = collections.Counter()
    saved_bytes = collections.Counter()

    def pack_hook(t):
        key = (tuple(t.shape), str(t.dtype).replace("torch.", ""))
        saved[key] += 1
        saved_bytes[key] += t.numel() * t.element_size()
        return t

    def unpack_hook(t):
        return t

    opt.zero_grad(set_to_none=True)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False):
            logits = fwd(batch["x"][sl], batch["pos"][sl], batch["segpos"][sl],
                         batch["full_mask"][sl])
            loss = ce_sum(logits, batch["y"][sl], batch["valid"][sl], cfg.V)
            loss = loss / int(batch["valid"].sum().item())
        loss.backward()
    torch.cuda.synchronize()
    total = sum(saved_bytes.values())
    tag = f"opt3_scan{args.scan}_b{args.block}_{args.shape}{'_compiled' if args.compiled else ''}"
    lines = [f"tag={tag} shape={args.shape} compiled={args.compiled}",
             f"total saved bytes = {total/2**30:.3f} GiB over {sum(saved.values())} tensors", ""]
    lines.append(f"{'shape':38s} {'dtype':>8s} {'n':>4s} {'MiB':>10s}")
    for (shape, dt), n in sorted(saved_bytes.items(), key=lambda kv: -kv[1])[:40]:
        mib = saved_bytes[(shape, dt)] / 2**20
        lines.append(f"{str(shape):38s} {dt:>8s} {saved[(shape, dt)]:4d} {mib:10.1f}")
    out = "\n".join(lines)
    path = f"results/e0_memcensus_{tag}.txt"
    with open(path, "w") as f:
        f.write(out)
    print(out[:4000])


if __name__ == "__main__":
    main()
