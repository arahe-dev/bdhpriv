"""Reprofile the mb2 sparse champion with a corrected kernel census.

  python opt/onehour_reprofile.py --microbatch 2 --out results/350k_local_reprofile.json
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--out", default="results/350k_local_reprofile.json")
    args = parser.parse_args()

    cfg = ArmAConfig()
    device = torch.device("cuda")
    acc = GLOBAL_SEQUENCES // args.microbatch
    micro = [make_batch(cfg, args.microbatch, device, 2000 + i)
             for i in range(acc)]
    denom = float(sum(int(b["valid"].sum().item()) for b in micro))

    dense = OptArmA(
        cfg, device, scan_block=1024, use_checkpoint=False, coord="dense",
        single_scan="chunkwise", packed_update="branchfree", zero_carry=True,
        paper_layout="direct", cache_rope=True,
    ).to(device)
    load_init(dense, canonical_init(cfg), device)
    dense.train()
    model = RoutedExpertArmA(cfg, device, experts=8, expert_width=512,
                             scan_block=1024).to(device)
    model.load_canonical(dense.state_dict())
    model.train()
    table = group_route_table(window_route_sets(8, 1, cyclic=True),
                              cfg.T // 128)
    route = build_route_tensors(table, 128, args.microbatch, cfg.T, 8,
                                device)
    entry = torch.compile(model.forward_route, mode="default")

    def fn(b):
        return entry(b["x"], b["pos"], b["segpos"], b["full_mask"],
                     b["segment_start"], route)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.PEAK_LR,
                                  betas=cfg.BETAS, eps=cfg.EPS, fused=True)

    def update():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            cache_enabled=False):
            for batch in micro:
                logits = fn(batch)
                loss = F.cross_entropy(
                    logits.reshape(-1, cfg.V), batch["y"].reshape(-1),
                    reduction="none",
                )[batch["valid"].reshape(-1)].sum(dtype=torch.float32) / denom
                loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_NORM)
        optimizer.step()

    for _ in range(args.warmups):
        update()
    torch.cuda.synchronize()
    times = []
    for _ in range(args.steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        update()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    peak = torch.cuda.max_memory_allocated(device) / 2**30

    prof_records = {}
    try:
        schedule = torch.profiler.schedule(wait=1, warmup=0, active=1)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            schedule=schedule,
        ) as prof:
            for _ in range(2):
                update()
                prof.step()
        by_count = {}
        by_device = {}
        api_cpu_ms = 0.0
        for event in prof.key_averages():
            if event.device_time_total > 0:
                by_count[event.key] = by_count.get(event.key, 0) + event.count
                by_device[event.key] = (
                    by_device.get(event.key, 0.0)
                    + event.self_device_time_total / 1000.0)
            key = event.key
            if any(tag in key for tag in ("cudaLaunchKernel", "cudaMemcpy",
                                          "cudaMemset", "cudaStreamSync",
                                          "cudaEvent")):
                api_cpu_ms += event.self_cpu_time_total / 1000.0
        total_kernels = sum(by_count.values())
        top_by_count = sorted(by_count.items(), key=lambda x: -x[1])[:12]
        top_by_device = sorted(by_device.items(), key=lambda x: -x[1])[:12]
        prof_records = {
            "kernels_per_global_update": total_kernels,
            "launch_api_cpu_ms_per_update": api_cpu_ms,
            "top_kernels_by_count": [
                {"name": name[:90], "count": count}
                for name, count in top_by_count],
            "top_kernels_by_device_ms": [
                {"name": name[:90], "device_ms": round(ms, 2)}
                for name, ms in top_by_device],
            "note": "profiler overhead inflates wall; counts are valid",
        }
    except Exception as exc:  # noqa: BLE001
        prof_records = {"error": f"{type(exc).__name__}: {exc}"}

    tok = GLOBAL_SEQUENCES * cfg.T
    record = {
        "candidate": "M8 Ke512 top1 fixed cyclic window, compact executor",
        "microbatch": args.microbatch,
        "accumulation": acc,
        "global_tokens": tok,
        "median_ms": statistics.median(times),
        "p10_ms": times[0],
        "p90_ms": times[-1],
        "ms": times,
        "packed_tok_s": tok / (statistics.median(times) / 1000.0),
        "peak_mem_GiB": peak,
        "profiler": prof_records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=str),
                   encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "ms"},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
