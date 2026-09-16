"""Learned-router systems gate (one candidate per process + Arm-A control).

python opt/bench_learned.py --mode learned --out results/bench_learned.json
python opt/bench_learned.py --mode fixed   --out results/bench_fixed_ref.json
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
from opt.learned_router import LearnedRoutedExpertArmA, build_learned_route
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
            ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
             "--format=csv,noheader"], text=True).strip()
    except Exception:
        return None


def measure(model, fn, optimizer, cfg, batch, device, steps, warmups):
    for _ in range(warmups):
        full_update(model, fn, optimizer, cfg, batch, device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        full_update(model, fn, optimizer, cfg, batch, device)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    return {"ms": times, "median_ms": statistics.median(times),
            "p10_ms": sorted(times)[0], "p90_ms": sorted(times)[-1],
            "peak_mem_GiB": peak}


def graph_info(model, args):
    try:
        torch._dynamo.reset()
        exp = torch._dynamo.explain(model)(*args)
        info = {"graph_count": int(getattr(exp, "graph_count", -1)),
                "graph_break_count": int(getattr(exp, "graph_break_count", -1))}
        torch._dynamo.reset()
        return info
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("learned", "fixed"), required=True)
    parser.add_argument("--top-r", type=int, default=2)
    parser.add_argument("--G", type=int, default=128)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--router-init-scale", type=float, default=1.0)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = ArmAConfig()
    device = torch.device("cuda")
    batch = make_batch(cfg, 1, device, "mixed", 1337)

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

    if args.mode == "learned":
        model = LearnedRoutedExpertArmA(cfg, device, experts=8, expert_width=512,
                                        scan_block=1024, top_r=args.top_r,
                                        route_group=args.G).to(device)
        model.router.tau = args.tau
        with torch.no_grad():
            model.router.Wr.normal_(0.0, args.router_init_scale)
        model.capacity_factor = args.capacity_factor
        model.load_canonical(baseline.state_dict())
        model.train()
        t0 = time.perf_counter()
        with torch.no_grad():
            v = model.ln(model.embedding(batch["x"]))
            torch.cuda.synchronize()
            r0 = time.perf_counter()
            route_check, overflow = build_learned_route(
                model, v, args.G, model.capacity_tokens(1, cfg.T),
                count_overflow=True)
            torch.cuda.synchronize()
            router_ms = (time.perf_counter() - r0) * 1000.0
        entry = torch.compile(model.forward_learned, mode="default")
        args_tuple = (batch["x"], batch["pos"], batch["segpos"],
                      batch["full_mask"], batch["segment_start"])
        info = graph_info(model.forward_learned, args_tuple)
        fn = make_forward(entry, "expertized")
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.PEAK_LR,
                                betas=cfg.BETAS, eps=cfg.EPS, fused=True)
        result = measure(model, fn, opt, cfg, batch, device, args.steps,
                         args.warmups)
        candidate_compile_s = time.perf_counter() - t0
        idx = route_check.sel_mask.new_zeros(0)
        candidate = {
            "kind": "learned_cyclic_window",
            "top_r": args.top_r, "G": args.G, "tau": args.tau,
            "router_init_scale": args.router_init_scale,
            "capacity_factor": args.capacity_factor,
            "capacity_tokens": int(route_check.sel_idx.shape[1]),
            "capacity_required": model.capacity_tokens(1, cfg.T),
            "route_overflow_first_call": overflow,
            "router_build_ms_eager": router_ms,
            "graph_info": info,
            "ledger": model.parameter_ledger(),
        }
    else:
        model = RoutedExpertArmA(cfg, device, experts=8, expert_width=512,
                                 scan_block=1024).to(device)
        model.load_canonical(baseline.state_dict())
        model.train()
        t0 = time.perf_counter()
        sets = window_route_sets(8, args.top_r, cyclic=True)
        table = group_route_table(sets, cfg.T // args.G)
        route = build_route_tensors(table, args.G, 1, cfg.T, 8, device)
        entry = torch.compile(model.forward_route, mode="default")
        fn = make_forward(entry, "routed", route)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.PEAK_LR,
                                betas=cfg.BETAS, eps=cfg.EPS, fused=True)
        result = measure(model, fn, opt, cfg, batch, device, args.steps,
                         args.warmups)
        candidate_compile_s = time.perf_counter() - t0
        candidate = {
            "kind": "fixed_cyclic_window",
            "top_r": args.top_r, "G": args.G,
            "capacity_tokens": int(route.sel_idx.shape[1]),
            "route": {"active_experts": list(route.active)},
            "ledger": model.parameter_ledger(args.top_r),
        }

    record = {
        "mode": args.mode,
        "baseline": baseline_result,
        "baseline_compile_s": baseline_compile_s,
        "candidate": {**candidate, **result},
        "candidate_compile_s": candidate_compile_s,
        "speedup_same_session": (baseline_result["median_ms"]
                                 / result["median_ms"]),
        "telemetry": telemetry(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=str),
                   encoding="utf-8")
    print(json.dumps({
        "out": str(out), "mode": args.mode,
        "baseline_ms": baseline_result["median_ms"],
        "candidate_ms": result["median_ms"],
        "speedup": record["speedup_same_session"],
        "candidate": {k: v for k, v in candidate.items()
                      if k not in ("ledger",)},
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
