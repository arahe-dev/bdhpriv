"""Round 1 sparse-executor DOE (master directive Priority 1-3).

Compact response-surface batch for M8/Ke512 head-shared cyclic-window
execution: G sweep x active width, capacity-padding intervention, interleaved
Arm-A controls, repeated reference points.

Outputs results/doe_round1.json with per-run records, main effects, an OLS
response surface, and the residual-runtime decomposition
T(Kactive) = T_fixed + a*Kactive + T_dispatch.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))

from opt.bench_expert_moe import full_update, make_batch, make_forward
from opt.model_ref import ArmAConfig, canonical_init, load_init
from opt.model_opt import OptArmA
from opt.routed_expert import (
    RoutedExpertArmA,
    build_route_tensors,
    group_route_table,
    window_route_sets,
)

TINY = dict(T=16, V=32, D=8, N=16, H=2, L=2, HIDDEN=8)

# Keep every compiled shape static: with dynamic shapes enabled, the first
# new capacity shape switches the shared compiled function into dynamic
# codegen and silently degrades later configs (observed as cap=256 slower
# than cap=512). Recompile per shape instead.
try:
    torch._dynamo.config.automatic_dynamic_shapes = False
except Exception:
    pass


def telemetry():
    try:
        return subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw",
             "--format=csv,noheader"], text=True).strip()
    except Exception:
        return None


def measure(model, fn, optimizer, cfg, batch, device, steps, warmups):
    losses = []
    for _ in range(warmups):
        full_update(model, fn, optimizer, cfg, batch, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss, _ = full_update(model, fn, optimizer, cfg, batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
        losses.append(loss)
    peak = (torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" else 0)
    return {
        "ms": times,
        "median_ms": statistics.median(times),
        "p10_ms": sorted(times)[max(0, int(0.1 * len(times)) - 1)],
        "p90_ms": sorted(times)[min(len(times) - 1, int(0.9 * len(times)))],
        "loss_last": losses[-1],
        "peak_mem_GiB": peak / 2**30,
    }


def route_stats(route, time, top_r):
    used = float(route.sel_mask.sum().item())
    slots = float(route.sel_mask.numel())
    return {
        "active_experts": list(route.active),
        "capacity_tokens": int(route.sel_idx.shape[1]),
        "used_tokens": int(used),
        "padding_fraction": 1.0 - used / max(1.0, slots),
        "routed_token_slots": int(time * top_r),
    }


def build_candidate(cfg, device, baseline, top_r, scan_block, experts):
    model = RoutedExpertArmA(cfg, device, experts=experts,
                             expert_width=cfg.K // experts,
                             scan_block=scan_block)
    model = model.to(device)
    model.load_canonical(baseline.state_dict())
    model.train()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/doe_round1.json")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--mode", default="mixed")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--compile", dest="compile", action="store_true")
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.set_defaults(compile=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = ArmAConfig(**TINY) if args.tiny else ArmAConfig()
    scan_block = cfg.K if args.tiny else 1024
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    batch = make_batch(cfg, args.batch, device, args.mode, args.seed)

    baseline = OptArmA(
        cfg, device, scan_block=scan_block, use_checkpoint=False,
        coord="dense", single_scan="chunkwise", packed_update="branchfree",
        zero_carry=True, paper_layout="direct", cache_rope=True,
    ).to(device)
    load_init(baseline, canonical_init(cfg), device)
    baseline.train()
    opt_b = torch.optim.AdamW(baseline.parameters(), lr=cfg.PEAK_LR,
                              betas=cfg.BETAS, eps=cfg.EPS,
                              fused=device.type == "cuda")
    entry_b = (torch.compile(baseline.forward_packed, mode="default")
               if args.compile else baseline.forward_packed)
    fn_b = make_forward(entry_b, "expertized")

    top_rs = (1, 2) if args.tiny else (1, 2, 4)
    experts = 4 if args.tiny else 8
    candidates = {}
    for top_r in top_rs:
        model = build_candidate(cfg, device, baseline, top_r, scan_block,
                                experts)
        entry = (torch.compile(model.forward_route, mode="default")
                 if args.compile else model.forward_route)
        candidates[top_r] = (model, entry)

    if args.tiny:
        runs = [
            {"id": "top2_G4", "top_r": 2, "G": 4, "cap": 1.0},
            {"id": "top1_G4", "top_r": 1, "G": 4, "cap": 1.0},
            {"id": "top2_G4_cap1.5", "top_r": 2, "G": 4, "cap": 1.5},
        ]
        repeats = [{"id": "ctrl_arm_a", "kind": "baseline"}]
    else:
        runs = []
        for top_r in (2, 1, 4):
            for group in (64, 128, 256, 512):
                runs.append({"id": f"top{top_r}_G{group}", "top_r": top_r,
                             "G": group, "cap": 1.0})
        for factor in (1.25, 1.5, 2.0):
            runs.append({"id": f"top2_G128_cap{factor}", "top_r": 2,
                         "G": 128, "cap": factor})
        repeats = [
            {"id": "ctrl_arm_a", "kind": "baseline"},
            {"id": "rep_top2_G128_a", "top_r": 2, "G": 128, "cap": 1.0},
            {"id": "ctrl_arm_a", "kind": "baseline"},
            {"id": "rep_top2_G128_b", "top_r": 2, "G": 128, "cap": 1.0},
            {"id": "ctrl_arm_a", "kind": "baseline"},
            {"id": "rep_top2_G128_c", "top_r": 2, "G": 128, "cap": 1.0},
        ]

    schedule = []
    for run in runs:
        schedule.append(run)
    for run in repeats:
        schedule.append(run)
    random.Random(args.seed).shuffle(schedule)

    records = []
    for run in schedule:
        if run.get("kind") == "baseline":
            result = measure(baseline, fn_b, opt_b, cfg, batch, device,
                             args.steps, args.warmups)
            records.append({"id": run["id"], "kind": "baseline",
                            "telemetry": telemetry(), **result})
            continue
        model, entry = candidates[run["top_r"]]
        options = window_route_sets(8 if not args.tiny else 4, run["top_r"],
                                    cyclic=True)
        table = group_route_table(options, cfg.T // run["G"])
        t0 = time.perf_counter()
        route = build_route_tensors(table, run["G"], args.batch, cfg.T,
                                    model.M, device,
                                    capacity_factor=run["cap"])
        pack_ms = (time.perf_counter() - t0) * 1000.0
        fn = make_forward(entry, "routed", route)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.PEAK_LR,
                                      betas=cfg.BETAS, eps=cfg.EPS,
                                      fused=device.type == "cuda")
        result = measure(model, fn, optimizer, cfg, batch, device,
                         args.steps, args.warmups)
        records.append({
            "id": run["id"], "kind": "candidate", "top_r": run["top_r"],
            "G": run["G"], "capacity_factor": run["cap"],
            "pack_ms": pack_ms, "telemetry": telemetry(),
            "route": route_stats(route, cfg.T, run["top_r"]),
            **result,
        })

    baselines = [r["median_ms"] for r in records
                 if r.get("kind") == "baseline"]
    baseline_median = statistics.median(baselines) if baselines else None
    for record in records:
        if baseline_median:
            record["speedup_same_session"] = baseline_median / record["median_ms"]

    candidates_records = [r for r in records if r.get("kind") == "candidate"]
    fit = fit_response_surface(candidates_records)
    residual = residual_runtime(candidates_records)
    summary = {
        "out": args.out,
        "baseline_median_ms": baseline_median,
        "baseline_samples": baselines,
        "noise_ms": (statistics.pstdev(baselines) if len(baselines) > 1
                     else None),
        "runs": len(records),
        "best": min(candidates_records, key=lambda r: r["median_ms"])["id"]
        if candidates_records else None,
        "fit": fit,
        "residual_runtime_model": residual,
        "speedups": {r["id"]: round(r["speedup_same_session"], 3)
                     for r in candidates_records
                     if "speedup_same_session" in r},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "records": records},
                              indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return 0


def fit_response_surface(records):
    if len(records) < 6:
        return {"skipped": "too few records"}
    def feature(r):
        width = r["top_r"] * 512
        group = r["G"]
        return [1.0, width / 1024.0, group / 128.0, (group / 128.0) ** 2,
                r["route"]["capacity_tokens"] / 512.0,
                r["route"]["padding_fraction"],
                (width / 1024.0) * (group / 128.0)]
    x = np.array([feature(r) for r in records])
    y = np.array([r["median_ms"] for r in records])
    coeff, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coeff
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "terms": ["intercept", "active_K/1024", "G/128", "G2",
                  "capacity/512", "padding_fraction", "activeK:G"],
        "coefficients": coeff.tolist(),
        "r2": 1.0 - ss_res / max(1e-12, ss_tot),
        "n": len(records),
    }


def residual_runtime(records):
    base = [r for r in records if r["route"]["padding_fraction"] < 0.05]
    if len(base) < 3:
        return {"skipped": "need >=3 low-padding records"}
    k = np.array([r["top_r"] * 512 for r in base], dtype=np.float64)
    t = np.array([r["median_ms"] for r in base], dtype=np.float64)
    a = np.polyfit(k, t, 1)
    return {
        "records": [r["id"] for r in base],
        "Kactive": k.tolist(),
        "median_ms": t.tolist(),
        "T_fixed_ms": float(a[1]),
        "ms_per_Kactive_head": float(a[0]),
        "fit_r2": float(1 - ((t - np.polyval(a, k)) ** 2).sum()
                        / max(1e-12, ((t - t.mean()) ** 2).sum())),
        "note": ("T(Kactive) = T_fixed + a*Kactive; dispatch overhead is "
                 "folded into T_fixed for this linear model"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
