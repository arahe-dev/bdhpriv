"""B2: attribute the ~36 ms non-kernel gap in the M8 top1 champion.

One process, same-session Arm-A control. Measures: wall vs device kernel
time, CPU-side launch/API time, allocator activity per update, isolated
CE forward+backward, isolated clip, and an in-graph-loss variant (same
mathematics, fewer framework boundaries).

Run normal and with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to A/B
the allocator hypothesis.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

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


def sync():
    torch.cuda.synchronize()


def timed(fn, steps=6, warmups=2):
    for _ in range(warmups):
        fn()
    sync()
    times = []
    for _ in range(steps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {"median_ms": statistics.median(times), "ms": times}


def main():
    cfg = ArmAConfig()
    device = torch.device("cuda")
    batch = make_batch(cfg, 1, device, "mixed", 1337)
    env_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")

    baseline = OptArmA(
        cfg, device, scan_block=1024, use_checkpoint=False, coord="dense",
        single_scan="chunkwise", packed_update="branchfree", zero_carry=True,
        paper_layout="direct", cache_rope=True,
    ).to(device)
    load_init(baseline, canonical_init(cfg), device)
    baseline.train()
    entry_b = torch.compile(baseline.forward_packed, mode="default")
    fn_b = make_forward(entry_b, "expertized")
    opt_b = torch.optim.AdamW(baseline.parameters(), lr=cfg.PEAK_LR,
                              betas=cfg.BETAS, eps=cfg.EPS, fused=True)
    baseline_ms = timed(lambda: full_update(baseline, fn_b, opt_b, cfg, batch,
                                            device))

    model = RoutedExpertArmA(cfg, device, experts=8, expert_width=512,
                             scan_block=1024).to(device)
    model.load_canonical(baseline.state_dict())
    model.train()
    table = group_route_table(window_route_sets(8, 1, cyclic=True),
                              cfg.T // 128)
    route = build_route_tensors(table, 128, 1, cfg.T, 8, device)
    entry = torch.compile(model.forward_route, mode="default")
    fn = make_forward(entry, "routed", route)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.PEAK_LR,
                            betas=cfg.BETAS, eps=cfg.EPS, fused=True)
    reference = timed(lambda: full_update(model, fn, opt, cfg, batch, device))

    stats_before = torch.cuda.memory_stats()
    reference2 = timed(lambda: full_update(model, fn, opt, cfg, batch, device),
                       steps=4, warmups=1)
    stats_after = torch.cuda.memory_stats()
    alloc_delta = {
        "num_device_alloc": stats_after["num_device_alloc"]
        - stats_before["num_device_alloc"],
        "num_alloc_retries": stats_after["num_alloc_retries"]
        - stats_before["num_alloc_retries"],
        "num_ooms": stats_after["num_ooms"] - stats_before["num_ooms"],
        "active_bytes_peak_GiB": stats_after["active_bytes.all.peak"] / 2**30,
        "reserved_GiB": stats_after["reserved_bytes.all.current"] / 2**30,
        "segments": stats_after["segment.all.current"],
    }

    prof_schedule = torch.profiler.schedule(wait=1, warmup=1, active=3)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        schedule=prof_schedule,
    ) as prof:
        for _ in range(5):
            full_update(model, fn, opt, cfg, batch, device)
            prof.step()

    device_kernels = 0
    device_ms = 0.0
    api_cpu_ms = 0.0
    top_cpu = []
    for event in prof.key_averages():
        if event.device_time_total > 0:
            device_kernels += event.count
            device_ms += event.self_device_time_total / 1000.0
        key = event.key
        if any(tag in key for tag in ("cudaLaunchKernel", "cudaMemcpy",
                                      "cudaMemset", "cudaStreamSync",
                                      "cudaDeviceSync", "cudaEvent")):
            api_cpu_ms += event.self_cpu_time_total / 1000.0
        if event.self_cpu_time_total > 0:
            top_cpu.append((event.self_cpu_time_total / 1000.0, key))
    top_cpu.sort(reverse=True)
    prof_report = {
        "profiled_steps": 5,
        "device_kernels_total": device_kernels,
        "device_kernels_per_update": device_kernels / 5.0,
        "device_ms_total": device_ms,
        "device_ms_per_update": device_ms / 5.0,
        "cuda_api_cpu_ms_per_update": api_cpu_ms / 5.0,
        "top_cpu_ops": [{"ms_per_update": round(ms / 5.0, 3),
                         "name": name[:80]} for ms, name in top_cpu[:10]],
    }

    with torch.no_grad():
        logits = fn(batch).detach()

    def ce_only():
        lg = logits.clone().requires_grad_(True)
        loss = F.cross_entropy(lg.reshape(-1, cfg.V), batch["y"].reshape(-1))
        loss.backward()
    ce_ms = timed(ce_only, steps=6, warmups=2)

    def clip_only():
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_NORM)
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    clip_ms = timed(clip_only, steps=6, warmups=2)

    def fused_step():
        model.zero_grad(set_to_none=True)
        lg = entry(batch["x"], batch["pos"], batch["segpos"],
                   batch["full_mask"], batch["segment_start"], route)
        loss = F.cross_entropy(lg.reshape(-1, cfg.V),
                               batch["y"].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_NORM)
        opt.step()
    fused_ms = timed(fused_step, steps=6, warmups=2)

    report = {
        "env_PYTORCH_CUDA_ALLOC_CONF": env_conf,
        "baseline_arm_a_ms": baseline_ms["median_ms"],
        "reference_wall_ms": reference["median_ms"],
        "reference_wall_ms_r2": reference2["median_ms"],
        "speedup_same_session": baseline_ms["median_ms"]
        / reference["median_ms"],
        "profiler": prof_report,
        "allocator_delta_over_4_updates": alloc_delta,
        "isolated": {
            "ce_fwd_bwd_ms": ce_ms["median_ms"],
            "clip_only_ms": clip_ms["median_ms"],
            "optimizer_only_ms_prior": 4.73,
        },
        "variant_fused_in_graph_boundary_ms": fused_ms["median_ms"],
        "variant_delta_ms": reference["median_ms"] - fused_ms["median_ms"],
        "budget_check": {
            "wall_ms": reference["median_ms"],
            "device_ms_per_update": prof_report["device_ms_per_update"],
            "non_kernel_gap_ms": reference["median_ms"]
            - prof_report["device_ms_per_update"],
            "target_one_hour_ms": baseline_ms["median_ms"] / 8.69,
        },
    }
    out = Path("results/onehour_gap_breakdown.json")
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
