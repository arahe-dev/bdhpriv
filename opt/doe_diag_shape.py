"""Diagnose the cap-256 vs cap-512 Inductor pathology (Round-1 anomaly).

For routed top1 M8/Ke512 G128 cyclic windows:
  - graph-break count via torch._dynamo.explain for cap 256 and cap 512
  - torch.profiler CUDA kernel totals for both shapes
  - eager baseline for cap 256 (is compiled-slow == eager-slow?)
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opt.bench_expert_moe import full_update, make_batch, make_forward
from opt.model_opt import OptArmA
from opt.model_ref import ArmAConfig, canonical_init, load_init
from opt.routed_expert import (
    RoutedExpertArmA,
    build_route_tensors,
    group_route_table,
    window_route_sets,
)

torch._dynamo.config.automatic_dynamic_shapes = False


def measure(fn, model, opt, cfg, batch, device, steps=4, warmups=2):
    for _ in range(warmups):
        full_update(model, fn, opt, cfg, batch, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        full_update(model, fn, opt, cfg, batch, device)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def main():
    cfg = ArmAConfig()
    device = torch.device("cuda")
    batch = make_batch(cfg, 1, device, "mixed", 1337)
    baseline = OptArmA(
        cfg, device, scan_block=1024, use_checkpoint=False, coord="dense",
        single_scan="chunkwise", packed_update="branchfree", zero_carry=True,
        paper_layout="direct", cache_rope=True,
    ).to(device)
    load_init(baseline, canonical_init(cfg), device)
    model = RoutedExpertArmA(cfg, device, experts=8, expert_width=512,
                             scan_block=1024).to(device)
    model.load_canonical(baseline.state_dict())
    model.train()

    sets = window_route_sets(8, 1, cyclic=True)
    table = group_route_table(sets, cfg.T // 128)

    report = {}
    for cap in (1.0, 2.0):
        route = build_route_tensors(table, 128, 1, cfg.T, 8, device,
                                    capacity_factor=cap)
        args = (batch["x"], batch["pos"], batch["segpos"], batch["full_mask"],
                batch["segment_start"], route)
        torch._dynamo.reset()
        exp = torch._dynamo.explain(model.forward_route)(*args)
        breaks = int(getattr(exp, "graph_break_count", -1))
        entry = torch.compile(model.forward_route, mode="default")
        fn = make_forward(entry, "routed", route)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.PEAK_LR,
                                betas=cfg.BETAS, eps=cfg.EPS, fused=True)
        compiled_times = measure(fn, model, opt, cfg, batch, device)
        prof_schedule = torch.profiler.schedule(wait=1, warmup=1, active=2)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=prof_schedule,
        ) as prof:
            for _ in range(4):
                full_update(model, fn, opt, cfg, batch, device)
                prof.step()
        key_averages = prof.key_averages()
        kernels = sorted(
            key_averages, key=lambda k: -k.self_device_time_total
        )[:8]
        report[f"cap{int(route.sel_idx.shape[1])}"] = {
            "capacity_tokens": int(route.sel_idx.shape[1]),
            "graph_breaks": breaks,
            "compiled_ms": compiled_times,
            "compiled_median_ms": statistics.median(compiled_times),
            "top_cuda_kernels": [
                {
                    "name": k.key[:90],
                    "self_cuda_ms": round(k.self_device_time_total / 1000.0,
                                          2),
                }
                for k in kernels
            ],
            "total_cuda_ms": round(sum(
                k.self_device_time_total for k in key_averages) / 1000.0, 2),
        }

    route256 = build_route_tensors(table, 128, 1, cfg.T, 8, device,
                                   capacity_factor=1.0)
    fn_eager = make_forward(model.forward_route, "routed", route256)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.PEAK_LR,
                            betas=cfg.BETAS, eps=cfg.EPS, fused=True)
    eager_times = measure(fn_eager, model, opt, cfg, batch, device)
    report["eager_cap256"] = {"ms": eager_times,
                              "median_ms": statistics.median(eager_times)}

    # baseline reference in the same session
    fn_b = make_forward(baseline.forward_packed, "expertized")
    opt_b = torch.optim.AdamW(baseline.parameters(), lr=cfg.PEAK_LR,
                              betas=cfg.BETAS, eps=cfg.EPS, fused=True)
    report["arm_a_baseline"] = {"median_ms": statistics.median(
        measure(fn_b, baseline, opt_b, cfg, batch, device))}

    print(json.dumps(report, indent=2))
    Path("results/doe_shape_diag.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
