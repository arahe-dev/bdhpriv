"""Single-config Round-1 measurement (one process, one compiled shape set).

Baseline Arm-A and the routed candidate are compiled and measured in the
same process so every config has its own same-session control. Run one
process per config to eliminate cross-shape compile interference (observed:
a single compiled object handling heterogeneous capacities silently degrades
some configs to ~eager speed).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
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


def telemetry():
    try:
        return subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=clocks.sm,temperature.gpu,power.draw",
             "--format=csv,noheader"], text=True).strip()
    except Exception:
        return None


def measure(model, fn, optimizer, cfg, batch, device, steps, warmups,
            pre_step=None):
    for _ in range(warmups):
        if pre_step is not None:
            pre_step()
        full_update(model, fn, optimizer, cfg, batch, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(steps):
        if pre_step is not None:
            pre_step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        full_update(model, fn, optimizer, cfg, batch, device)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    peak = (torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" else 0)
    return {
        "ms": times,
        "median_ms": statistics.median(times),
        "p10_ms": sorted(times)[max(0, int(0.1 * len(times)) - 1)],
        "p90_ms": sorted(times)[min(len(times) - 1, int(0.9 * len(times)))],
        "peak_mem_GiB": peak / 2**30,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-r", type=int, required=True)
    parser.add_argument("--G", type=int, required=True)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--expert-width", type=int, default=512)
    parser.add_argument("--cap", type=float, default=1.0)
    parser.add_argument("--compile-mode", default="default",
                        choices=("default", "reduce-overhead",
                                 "max-autotune-no-cudagraphs"))
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--mode", default="mixed")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--id", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = ArmAConfig()
    device = torch.device("cuda")
    batch = make_batch(cfg, args.batch, device, args.mode, args.seed)

    t0 = time.perf_counter()
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
    baseline_result = measure(baseline, fn_b, opt_b, cfg, batch, device,
                              args.steps, args.warmups)
    baseline_compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    model = RoutedExpertArmA(cfg, device, experts=args.experts,
                             expert_width=args.expert_width,
                             scan_block=1024).to(device)
    model.load_canonical(baseline.state_dict())
    model.train()
    entry = torch.compile(model.forward_route, mode=args.compile_mode)
    options = window_route_sets(args.experts, args.top_r, cyclic=True)
    table = group_route_table(options, cfg.T // args.G)
    pack_t0 = time.perf_counter()
    route = build_route_tensors(table, args.G, args.batch, cfg.T, model.M,
                                device, capacity_factor=args.cap)
    pack_ms = (time.perf_counter() - pack_t0) * 1000.0
    fn = make_forward(entry, "routed", route)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.PEAK_LR,
                            betas=cfg.BETAS, eps=cfg.EPS, fused=True)
    pre_step = None
    if args.compile_mode == "reduce-overhead":
        # Populate lazy mask caches eagerly, then mark graph boundaries so
        # cudagraph replay never overwrites tensors created in-graph.
        with torch.no_grad():
            model.forward_route(batch["x"], batch["pos"], batch["segpos"],
                                batch["full_mask"], batch["segment_start"],
                                route)
        torch.cuda.synchronize()
        pre_step = torch.compiler.cudagraph_mark_step_begin
    candidate_result = measure(model, fn, opt, cfg, batch, device,
                               args.steps, args.warmups, pre_step=pre_step)
    candidate_compile_s = time.perf_counter() - t0

    used = float(route.sel_mask.sum().item())
    slots = float(route.sel_mask.numel())
    record = {
        "id": args.id or f"top{args.top_r}_G{args.G}_cap{args.cap}",
        "top_r": args.top_r,
        "G": args.G,
        "capacity_factor": args.cap,
        "route": {
            "active_experts": list(route.active),
            "capacity_tokens": int(route.sel_idx.shape[1]),
            "padding_fraction": 1.0 - used / max(1.0, slots),
        },
        "pack_ms": pack_ms,
        "baseline": baseline_result,
        "candidate": candidate_result,
        "speedup_same_session": (baseline_result["median_ms"]
                                 / candidate_result["median_ms"]),
        "compile_seconds": {"baseline": baseline_compile_s,
                            "candidate": candidate_compile_s},
        "compile_mode": args.compile_mode,
        "peak_mem_GiB": candidate_result["peak_mem_GiB"],
        "telemetry": telemetry(),
        "device": torch.cuda.get_device_name(0),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k: record[k] for k in
                      ("id", "speedup_same_session", "route",
                       "compile_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
