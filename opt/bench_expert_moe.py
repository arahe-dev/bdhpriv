"""Same-session full-update benchmark: Arm-A oracle vs MoE/resonant candidates.

Per the local directive: T2048/L8/B1, full forward + CE + backward + clip +
AdamW, BF16 autocast + FP32 params, optional torch.compile(default), 3+
warmups, 10+ measured steps, randomized A/B order, median + p10/p90, peak
allocated memory. No cross-session claims.

Smoke (plumbing only, not evidence): py -3.12 opt/bench_expert_moe.py --tiny
GPU run: py -3.12 opt/bench_expert_moe.py --M 8 --Ke 512 --out results/...
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opt.expert_moe import ExpertizedArmA
from opt.model_opt import OptArmA
from opt.model_ref import ArmAConfig, canonical_init, load_init, \
    synthetic_packed_batch
from opt.routed_expert import (
    RoutedExpertArmA,
    build_route_tensors,
    fixed_route_sets,
    group_route_table,
    varying_route_sets,
    window_route_sets,
)

TINY = dict(T=16, V=32, D=8, N=16, H=2, L=2, HIDDEN=8)


def make_batch(cfg, batch_size, device, mode, seed):
    batch = synthetic_packed_batch(cfg, batch_size, device, seed=seed,
                                   mode=mode)
    g = torch.Generator(device="cpu").manual_seed(seed + 77)
    batch["y"] = torch.randint(0, cfg.V, batch["x"].shape, generator=g).to(
        device)
    batch["valid"] = torch.ones(batch["x"].shape, dtype=torch.bool,
                                device=device)
    return batch


def build_models(cfg, device, experts, expert_width, scan_block,
                 oscillator_flags, arch="expertized"):
    baseline = OptArmA(
        cfg, device, scan_block=scan_block, use_checkpoint=False,
        coord="dense", single_scan="chunkwise", packed_update="branchfree",
        zero_carry=True, paper_layout="direct", cache_rope=True,
    )
    if arch == "routed":
        candidate = RoutedExpertArmA(
            cfg, device, experts=experts, expert_width=expert_width,
            scan_block=scan_block)
    else:
        candidate = ExpertizedArmA(
            cfg, device, experts=experts, expert_width=expert_width,
            scan_block=scan_block,
            learn_freq_scale=oscillator_flags["O1"],
            learn_band_amp=oscillator_flags["O2"],
        )
    baseline = baseline.to(device)
    candidate = candidate.to(device)
    load_init(baseline, canonical_init(cfg), device)
    candidate.load_canonical(baseline.state_dict())
    return baseline, candidate


def make_forward(entry, arch, route_tensors=None):
    if arch == "routed":
        def forward_fn(batch):
            return entry(batch["x"], batch["pos"], batch["segpos"],
                         batch["full_mask"], batch["segment_start"],
                         route_tensors)
    else:
        def forward_fn(batch):
            return entry(batch["x"], batch["pos"], batch["segpos"],
                         batch["full_mask"], batch["segment_start"])
    return forward_fn


def full_update(model, forward_fn, optimizer, cfg, batch, device):
    denom = int(batch["valid"].sum().item())
    optimizer.zero_grad(set_to_none=True)
    enabled = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        cache_enabled=False, enabled=enabled):
        logits = forward_fn(batch)
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.V), batch["y"].reshape(-1),
            reduction="none",
        )[batch["valid"].reshape(-1)].sum(dtype=torch.float32) / denom
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.CLIP_NORM)
    optimizer.step()
    return float(loss.detach()), float(grad_norm)


def measured_steps(model, forward_fn, optimizer, cfg, batch, device, steps,
                   warmups, order):
    for _ in range(warmups):
        full_update(model, forward_fn, optimizer, cfg, batch, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    losses = []
    for _ in range(steps):
        if order == "sync":
            torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        loss, _ = full_update(model, forward_fn, optimizer, cfg, batch,
                              device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
        losses.append(loss)
    peak = (torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" else 0)
    return {
        "ms": times,
        "losses": losses,
        "median_ms": statistics.median(times),
        "p10_ms": sorted(times)[max(0, int(0.1 * len(times)) - 1)],
        "p90_ms": sorted(times)[min(len(times) - 1, int(0.9 * len(times)))],
        "peak_mem_GiB": peak / 2**30,
    }


def graph_check(model, forward_name, call_args):
    try:
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        exp = torch._dynamo.explain(getattr(model, forward_name))(*call_args)
        out = {
            "graph_count": int(getattr(exp, "graph_count", -1)),
            "graph_break_count": int(getattr(exp, "graph_break_count", -1)),
            "break_reasons": [str(r)[:200]
                              for r in getattr(exp, "break_reasons", [])],
        }
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        return out
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--Ke", type=int, default=512)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--mode", default="mixed")
    parser.add_argument("--scan-block", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--compile", dest="compile", action="store_true")
    parser.add_argument("--no-compile", dest="compile",
                        action="store_false")
    parser.set_defaults(compile=True)
    parser.add_argument("--O1", action="store_true")
    parser.add_argument("--O2", action="store_true")
    parser.add_argument("--arch", choices=("expertized", "routed"),
                        default="expertized")
    parser.add_argument("--route",
                        choices=("fixed", "varying", "window", "window_cyclic"),
                        default="fixed")
    parser.add_argument("--top-r", type=int, default=2)
    parser.add_argument("--G", type=int, default=128)
    parser.add_argument("--route-offset", type=int, default=0)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--graph-check", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = ArmAConfig(**TINY) if args.tiny else ArmAConfig()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.tiny:
        args.scan_block = cfg.K
        if args.Ke == 512:
            args.Ke = cfg.K // args.M if cfg.K % args.M == 0 else 2
        if args.G == 128:
            args.G = max(2, cfg.T // 4)
    batch = make_batch(cfg, args.batch, device, args.mode, args.seed)

    flags = {"O1": args.O1, "O2": args.O2}
    baseline, candidate = build_models(cfg, device, args.M, args.Ke,
                                       args.scan_block, flags, arch=args.arch)
    baseline.train()
    candidate.train()
    opt_b = torch.optim.AdamW(baseline.parameters(), lr=cfg.PEAK_LR,
                              betas=cfg.BETAS, eps=cfg.EPS,
                              fused=device.type == "cuda")
    opt_c = torch.optim.AdamW(candidate.parameters(), lr=cfg.PEAK_LR,
                              betas=cfg.BETAS, eps=cfg.EPS,
                              fused=device.type == "cuda")
    route_tensors = None
    route_info = None
    if args.arch == "routed":
        assert cfg.T % args.G == 0, "--G must divide T"
        groups_per_row = cfg.T // args.G
        if args.route == "fixed":
            sets = fixed_route_sets(args.M, args.top_r, args.route_offset)
        elif args.route == "varying":
            sets = varying_route_sets(args.M, args.top_r)
        elif args.route == "window_cyclic":
            sets = window_route_sets(args.M, args.top_r, cyclic=True)
        else:
            sets = window_route_sets(args.M, args.top_r)
        table = group_route_table(sets, groups_per_row)
        route_tensors = build_route_tensors(table, args.G, args.batch,
                                            cfg.T, args.M, device)
        route_info = {
            "route": args.route,
            "top_r": args.top_r,
            "G": args.G,
            "sets": [list(s) for s in sets],
            "capacity_tokens": int(route_tensors.sel_idx.shape[1]),
            "active_experts": list(route_tensors.active),
        }
    entry_b = torch.compile(baseline.forward_packed, mode="default") \
        if args.compile else baseline.forward_packed
    candidate_fn = (candidate.forward_route if args.arch == "routed"
                    else candidate.forward_packed)
    entry_c = torch.compile(candidate_fn, mode="default") \
        if args.compile else candidate_fn
    fn_b = make_forward(entry_b, "expertized")
    fn_c = make_forward(entry_c, args.arch, route_tensors)

    compile_seconds = {}
    if args.compile:
        t0 = time.perf_counter()
        full_update(baseline, fn_b, opt_b, cfg, batch, device)
        compile_seconds["baseline"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        full_update(candidate, fn_c, opt_c, cfg, batch, device)
        compile_seconds["candidate"] = time.perf_counter() - t0

    order = ["baseline", "candidate"]
    random.Random(args.seed).shuffle(order)
    results = {}
    for name in order:
        model, fn, opt = (
            (baseline, fn_b, opt_b) if name == "baseline"
            else (candidate, fn_c, opt_c)
        )
        results[name] = measured_steps(
            model, fn, opt, cfg, batch, device, args.steps, args.warmups,
            "sync",
        )
    speedup = results["baseline"]["median_ms"] / results["candidate"]["median_ms"]
    tokens = args.batch * cfg.T
    report = {
        "meta": {
            "tiny_diagnostic": bool(args.tiny),
            "device": str(device),
            "gpu": (torch.cuda.get_device_name(0)
                    if device.type == "cuda" else None),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "compile": args.compile,
            "batch": args.batch,
            "T": cfg.T,
            "L": cfg.L,
            "mode": args.mode,
            "steps": args.steps,
            "warmups": args.warmups,
            "arch": args.arch,
        },
        "candidate": {
            "M": args.M,
            "Ke": args.Ke,
            "oscillator_flags": flags,
            "route_info": route_info,
            "ledger": candidate.parameter_ledger(
                args.top_r if args.arch == "routed" else args.M),
        },
        "baseline": results["baseline"],
        "candidate_results": results["candidate"],
        "speedup_same_session": speedup,
        "tok_s": {
            "baseline": tokens / (results["baseline"]["median_ms"] / 1000.0),
            "candidate": tokens / (results["candidate"]["median_ms"] / 1000.0),
        },
        "compile_seconds": compile_seconds,
    }
    if args.graph_check:
        base_args = [batch["x"], batch["pos"], batch["segpos"],
                     batch["full_mask"], batch["segment_start"]]
        report["graph_breaks"] = {
            "baseline": graph_check(baseline, "forward_packed", base_args),
            "candidate": graph_check(
                candidate,
                "forward_route" if args.arch == "routed" else "forward_packed",
                base_args + [route_tensors] if args.arch == "routed"
                else base_args),
        }
    out = Path(args.out) if args.out else (
        Path("results") / f"bench_expert_moe_M{args.M}_Ke{args.Ke}"
                          f"_g{args.mode}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str),
                   encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "median_ms": {
            "baseline": results["baseline"]["median_ms"],
            "candidate": results["candidate"]["median_ms"],
        },
        "speedup_same_session": speedup,
        "tok_s": report["tok_s"],
        "candidate": report["candidate"],
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
