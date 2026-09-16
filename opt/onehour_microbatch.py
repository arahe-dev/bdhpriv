"""Microbatch-geometry DOE at constant global batch (64 x 2048 tokens).

One process per (arm, microbatch). Measures a full optimizer update:
accumulate 64/B microbatches, one clip + step. Reports wall, kernel count
per update, peak memory.

  python opt/onehour_microbatch.py --arm sparse --microbatch 4 --out ...
  python opt/onehour_microbatch.py --arm dense --microbatch 2 --out ...
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
GLOBAL_SEQUENCES = 64


def make_batch(cfg, nseq, device, seed):
    batch = synthetic_packed_batch(cfg, nseq, device, seed=seed, mode="mixed")
    g = torch.Generator(device="cpu").manual_seed(seed + 77)
    batch["y"] = torch.randint(0, cfg.V, batch["x"].shape, generator=g).to(
        device)
    batch["valid"] = torch.ones(batch["x"].shape, dtype=torch.bool,
                                device=device)
    return batch


def build(cfg, device, arm, microbatch):
    baseline = OptArmA(
        cfg, device, scan_block=1024, use_checkpoint=False, coord="dense",
        single_scan="chunkwise", packed_update="branchfree", zero_carry=True,
        paper_layout="direct", cache_rope=True,
    ).to(device)
    load_init(baseline, canonical_init(cfg), device)
    baseline.train()
    if arm == "dense":
        entry = torch.compile(baseline.forward_packed, mode="default")

        def fn(b):
            return entry(b["x"], b["pos"], b["segpos"], b["full_mask"],
                         b["segment_start"])
        ledger = {"stored_params": sum(p.numel()
                                       for p in baseline.parameters())}
        return baseline, fn, ledger
    model = RoutedExpertArmA(cfg, device, experts=8, expert_width=512,
                             scan_block=1024).to(device)
    model.load_canonical(baseline.state_dict())
    model.train()
    table = group_route_table(window_route_sets(8, 1, cyclic=True),
                              cfg.T // 128)
    route = build_route_tensors(table, 128, microbatch, cfg.T, 8, device)
    entry = torch.compile(model.forward_route, mode="default")

    def fn(b):
        return entry(b["x"], b["pos"], b["segpos"], b["full_mask"],
                     b["segment_start"], route)
    return model, fn, model.parameter_ledger(1)


def update(model, fn, optimizer, cfg, micro_batches, denom):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                        cache_enabled=False):
        for batch in micro_batches:
            logits = fn(batch)
            loss = F.cross_entropy(
                logits.reshape(-1, cfg.V), batch["y"].reshape(-1),
                reduction="none",
            )[batch["valid"].reshape(-1)].sum(dtype=torch.float32) / denom
            loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_NORM)
    optimizer.step()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("dense", "sparse"), required=True)
    parser.add_argument("--microbatch", type=int, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = ArmAConfig()
    device = torch.device("cuda")
    acc = GLOBAL_SEQUENCES // args.microbatch
    micro_batches = [make_batch(cfg, args.microbatch, device, 1000 + i)
                     for i in range(acc)]
    denom = float(sum(int(b["valid"].sum().item()) for b in micro_batches))

    t0 = time.perf_counter()
    model, fn, ledger = build(cfg, device, args.arm, args.microbatch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.PEAK_LR,
                                  betas=cfg.BETAS, eps=cfg.EPS, fused=True)
    for _ in range(args.warmups):
        update(model, fn, optimizer, cfg, micro_batches, denom)
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - t0

    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(args.steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        update(model, fn, optimizer, cfg, micro_batches, denom)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    peak = torch.cuda.max_memory_allocated(device) / 2**30

    kernels = None
    try:
        schedule = torch.profiler.schedule(wait=0, warmup=0, active=1)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=schedule,
        ) as prof:
            update(model, fn, optimizer, cfg, micro_batches, denom)
            prof.step()
        kernels = sum(e.count for e in prof.key_averages()
                      if e.device_time_total > 0)
    except Exception as exc:  # noqa: BLE001
        kernels = f"error: {type(exc).__name__}"

    tokens = GLOBAL_SEQUENCES * cfg.T
    record = {
        "arm": args.arm,
        "microbatch": args.microbatch,
        "accumulation_steps": acc,
        "global_sequences": GLOBAL_SEQUENCES,
        "tokens_per_update": tokens,
        "median_ms": statistics.median(times),
        "p10_ms": times[0],
        "p90_ms": times[-1],
        "ms": times,
        "tok_per_s": tokens / (statistics.median(times) / 1000.0),
        "peak_mem_GiB": peak,
        "compile_s": compile_s,
        "kernels_per_update": kernels,
        "ledger": ledger,
        "device": torch.cuda.get_device_name(0),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=str),
                   encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items()
                      if k not in ("ms", "ledger")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
