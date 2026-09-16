"""B3 A/B: level-invariant gather caching, same-process, mb2 global update.

Three models in one process: dense Arm-A control, sparse with gather cache,
sparse without. Reports wall medians, kernels/update, copies.

  python opt/onehour_b3_cache.py --out results/350k_b3_cache.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opt.model_opt import OptArmA
from opt.model_ref import ArmAConfig, canonical_init, load_init, \
    synthetic_packed_batch
from opt.routed_expert import (
    RoutedExpertArmA,
    build_route_tensors,
    group_route_table,
    window_route_sets,
)

torch._dynamo.config.automatic_dynamic_shapes = False
GLOBAL = 64


def make_batch(cfg, nseq, device, seed):
    batch = synthetic_packed_batch(cfg, nseq, device, seed=seed, mode="mixed")
    g = torch.Generator(device="cpu").manual_seed(seed + 77)
    batch["y"] = torch.randint(0, cfg.V, batch["x"].shape, generator=g).to(
        device)
    batch["valid"] = torch.ones(batch["x"].shape, dtype=torch.bool,
                                device=device)
    return batch


def build_dense(cfg, device):
    model = OptArmA(cfg, device, scan_block=1024, use_checkpoint=False,
                    coord="dense", single_scan="chunkwise",
                    packed_update="branchfree", zero_carry=True,
                    paper_layout="direct", cache_rope=True).to(device)
    load_init(model, canonical_init(cfg), device)
    model.train()
    entry = torch.compile(model.forward_packed, mode="default")

    def fn(b):
        return entry(b["x"], b["pos"], b["segpos"], b["full_mask"],
                     b["segment_start"])
    return model, fn


def build_sparse(cfg, device, state, microbatch, cache):
    model = RoutedExpertArmA(cfg, device, experts=8, expert_width=512,
                             scan_block=1024).to(device)
    model.load_canonical(state)
    model.use_gather_cache = cache
    model.train()
    table = group_route_table(window_route_sets(8, 1, cyclic=True),
                              cfg.T // 128)
    route = build_route_tensors(table, 128, microbatch, cfg.T, 8, device)
    entry = torch.compile(model.forward_route, mode="default")

    def fn(b):
        return entry(b["x"], b["pos"], b["segpos"], b["full_mask"],
                     b["segment_start"], route)
    return model, fn


def measure(model, fn, cfg, device, microbatch, steps, warmups):
    acc = GLOBAL // microbatch
    micro = [make_batch(cfg, microbatch, device, 4000 + i)
             for i in range(acc)]
    denom = float(sum(int(b["valid"].sum().item()) for b in micro))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.PEAK_LR,
                            betas=cfg.BETAS, eps=cfg.EPS, fused=True)

    def update():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            cache_enabled=False):
            for batch in micro:
                logits = fn(batch)
                loss = F.cross_entropy(
                    logits.reshape(-1, cfg.V), batch["y"].reshape(-1),
                    reduction="none",
                )[batch["valid"].reshape(-1)].sum(
                    dtype=torch.float32) / denom
                loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_NORM)
        opt.step()

    for _ in range(warmups):
        update()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        update()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    tok = GLOBAL * cfg.T
    return {"median_ms": statistics.median(times), "p10_ms": times[0],
            "p90_ms": times[-1], "packed_tok_s": tok
            / (statistics.median(times) / 1000.0), "peak_mem_GiB": peak,
            "ms": times}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--out", default="results/350k_b3_cache.json")
    args = parser.parse_args()
    cfg = ArmAConfig()
    device = torch.device("cuda")

    dense, dense_fn = build_dense(cfg, device)
    dense_ms = measure(dense, dense_fn, cfg, device, 1, args.steps,
                       args.warmups)
    state = {k: v.detach().clone() for k, v in dense.state_dict().items()}
    del dense
    torch.cuda.empty_cache()

    records = {"dense_mb1": dense_ms}
    for label, cache in (("cache_on", True), ("cache_off", False)):
        model, fn = build_sparse(cfg, device, state, args.microbatch, cache)
        records[label] = measure(model, fn, cfg, device, args.microbatch,
                                 args.steps, args.warmups)
        del model
        torch.cuda.empty_cache()

    on = records["cache_on"]["median_ms"]
    off = records["cache_off"]["median_ms"]
    summary = {
        "microbatch": args.microbatch,
        "dense_mb1_ms": dense_ms["median_ms"],
        "cache_on_ms": on,
        "cache_off_ms": off,
        "improvement_pct": (off - on) / off * 100.0,
        "cache_on_tok_s": records["cache_on"]["packed_tok_s"],
        "cache_off_tok_s": records["cache_off"]["packed_tok_s"],
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str),
                   encoding="utf-8")
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in summary.items() if k != "records"},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
