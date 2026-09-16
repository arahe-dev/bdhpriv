"""Pass B: fixed-floor autopsy of the M8 top1 champion (one process).

Measures the full update, forward/backward/optimizer split, L-level skeleton,
coordinator and writer ablations (diagnostic, semantics-changing), and a
profiler kernel breakdown. Writes results/onehour_floor_breakdown.json.

Diagnostic ablations are labeled and must never be promoted as-is.
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
    return {"median_ms": statistics.median(times), "p10_ms": times[0],
            "p90_ms": times[-1], "ms": times}


def build_candidate(cfg, device, baseline):
    model = RoutedExpertArmA(cfg, device, experts=8, expert_width=512,
                             scan_block=1024).to(device)
    model.load_canonical(baseline.state_dict())
    model.train()
    table = group_route_table(window_route_sets(8, 1, cyclic=True),
                              cfg.T // 128)
    route = build_route_tensors(table, 128, 1, cfg.T, 8, device)
    return model, route


def optimizer_only(model, cfg, device):
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)

    def step():
        model.zero_grad(set_to_none=False)
        for p in model.parameters():
            p.grad.normal_(0, 1e-3)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_NORM)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.PEAK_LR,
                                      betas=cfg.BETAS, eps=cfg.EPS, fused=True)
        optimizer.step()
    return step


def profiler_breakdown(fn):
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule,
    ) as prof:
        for _ in range(5):
            fn()
            prof.step()
    totals = {}
    kernel_count = 0
    for event in prof.key_averages():
        ms = event.self_device_time_total / 1000.0
        kernel_count += event.count
        name = event.key
        if "gemm" in name or "cutlass" in name:
            cat = "gemm"
        elif "optimizer" in name or "multi_tensor" in name:
            cat = "optimizer_elemwise"
        elif "layer_norm" in name or "native_layer_norm" in name:
            cat = "layernorm"
        elif "index" in name or "gather" in name or "scatter" in name:
            cat = "gather_scatter"
        elif "copy" in name or "cat" in name or "stack" in name:
            cat = "copies"
        elif "elementwise" in name or "triton_poi" in name or "triton_per" in name:
            cat = "pointwise"
        else:
            cat = "other"
        totals[cat] = totals.get(cat, 0.0) + ms
    return {"category_cuda_ms": totals, "kernel_count": kernel_count,
            "total_cuda_ms": sum(totals.values())}


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
    baseline.train()
    entry_b = torch.compile(baseline.forward_packed, mode="default")
    fn_b = make_forward(entry_b, "expertized")
    opt_b = torch.optim.AdamW(baseline.parameters(), lr=cfg.PEAK_LR,
                              betas=cfg.BETAS, eps=cfg.EPS, fused=True)
    baseline_ms = timed(lambda: full_update(baseline, fn_b, opt_b, cfg, batch,
                                            device))

    model, route = build_candidate(cfg, device, baseline)
    entry = torch.compile(model.forward_route, mode="default")
    fn = make_forward(entry, "routed", route)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.PEAK_LR,
                            betas=cfg.BETAS, eps=cfg.EPS, fused=True)
    candidate_ms = timed(lambda: full_update(model, fn, opt, cfg, batch,
                                             device))

    def loss_only_forward():
        with torch.no_grad():
            logits = fn(batch)
            torch.nn.functional.cross_entropy(
                logits.reshape(-1, cfg.V), batch["y"].reshape(-1))
    forward_ms = timed(loss_only_forward)

    def fwd_bwd():
        model.zero_grad(set_to_none=True)
        logits = fn(batch)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.V), batch["y"].reshape(-1))
        loss.backward()
    fwd_bwd_ms = timed(fwd_bwd)
    backward_only_ms = {"median_ms": fwd_bwd_ms["median_ms"]
                        - forward_ms["median_ms"]}
    optimizer_ms = timed(optimizer_only(model, cfg, device))

    skeleton = RoutedExpertArmA(cfg, device, experts=8, expert_width=512,
                                scan_block=1024).to(device)
    skeleton.load_canonical(baseline.state_dict())
    skeleton.train()
    skeleton._level = lambda v, *a, **k: v
    entry_s = torch.compile(skeleton.forward_route, mode="default")
    fn_s = make_forward(entry_s, "routed", route)
    opt_s = torch.optim.AdamW(skeleton.parameters(), lr=cfg.PEAK_LR,
                              betas=cfg.BETAS, eps=cfg.EPS, fused=True)
    skeleton_ms = timed(lambda: full_update(skeleton, fn_s, opt_s, cfg, batch,
                                            device))

    no_coord = RoutedExpertArmA(cfg, device, experts=8, expert_width=512,
                                scan_block=1024).to(device)
    no_coord.load_canonical(baseline.state_dict())
    no_coord.train()
    no_coord.coordinator.forward = (
        lambda v, segpos, full_mask: torch.ones_like(v))
    entry_c = torch.compile(no_coord.forward_route, mode="default")
    fn_c = make_forward(entry_c, "routed", route)
    opt_c = torch.optim.AdamW(no_coord.parameters(), lr=cfg.PEAK_LR,
                              betas=cfg.BETAS, eps=cfg.EPS, fused=True)
    no_coord_ms = timed(lambda: full_update(no_coord, fn_c, opt_c, cfg, batch,
                                            device))

    no_writer = RoutedExpertArmA(cfg, device, experts=8, expert_width=512,
                                 scan_block=1024).to(device)
    no_writer.load_canonical(baseline.state_dict())
    no_writer.train()
    no_writer.writer.forward = lambda x: torch.zeros_like(x)
    entry_w = torch.compile(no_writer.forward_route, mode="default")
    fn_w = make_forward(entry_w, "routed", route)
    opt_w = torch.optim.AdamW(no_writer.parameters(), lr=cfg.PEAK_LR,
                              betas=cfg.BETAS, eps=cfg.EPS, fused=True)
    no_writer_ms = timed(lambda: full_update(no_writer, fn_w, opt_w, cfg,
                                            batch, device))

    prof = profiler_breakdown(
        lambda: full_update(model, fn, opt, cfg, batch, device))

    cand = candidate_ms["median_ms"]
    per_level = (cand - skeleton_ms["median_ms"]) / 8.0
    breakdown = {
        "protocol": "one process; same-session Arm-A control; ablations are "
                    "diagnostic (semantics-changing) and never promoted",
        "baseline_arm_a_ms": baseline_ms["median_ms"],
        "candidate_full_ms": cand,
        "speedup_same_session": baseline_ms["median_ms"] / cand,
        "components": {
            "forward_loss_ms": {
                "estimated_ms": forward_ms["median_ms"],
                "fraction_of_total": forward_ms["median_ms"] / cand,
                "method": "no_grad compiled forward + CE",
                "confidence": "medium",
                "semantics_preserving_optimization": "possible",
            },
            "backward_ms": {
                "estimated_ms": backward_only_ms["median_ms"],
                "fraction_of_total": backward_only_ms["median_ms"] / cand,
                "method": "fwd+bwd minus forward-only",
                "confidence": "medium",
                "semantics_preserving_optimization": "possible",
            },
            "optimizer_step_ms": {
                "estimated_ms": optimizer_ms["median_ms"],
                "fraction_of_total": optimizer_ms["median_ms"] / cand,
                "method": "zero_grad + clip + fused AdamW on 17.4M params "
                          "(stale grads regenerated per rep)",
                "confidence": "high",
                "semantics_preserving_optimization": "maybe (param traversal)",
            },
            "levels_total_ms": {
                "estimated_ms": cand - skeleton_ms["median_ms"],
                "fraction_of_total": (cand - skeleton_ms["median_ms"]) / cand,
                "method": "candidate minus L-level-skeleton variant",
                "confidence": "medium",
                "semantics_preserving_optimization": "possible",
            },
            "per_level_ms": {
                "estimated_ms": per_level,
                "fraction_of_total": per_level / cand,
                "method": "levels_total / 8",
                "confidence": "low",
                "semantics_preserving_optimization": "possible",
            },
            "skeleton_ms": {
                "estimated_ms": skeleton_ms["median_ms"],
                "fraction_of_total": skeleton_ms["median_ms"] / cand,
                "method": "identity levels + embedding/readout/CE/optimizer",
                "confidence": "high",
                "semantics_preserving_optimization": "n/a diagnostic",
            },
            "coordinator_ablation_ms": {
                "estimated_ms": cand - no_coord_ms["median_ms"],
                "fraction_of_total": (cand - no_coord_ms["median_ms"]) / cand,
                "method": "coordinator returns ones (diagnostic)",
                "confidence": "high",
                "semantics_preserving_optimization": "systems only",
            },
            "writer_ablation_ms": {
                "estimated_ms": cand - no_writer_ms["median_ms"],
                "fraction_of_total": (cand - no_writer_ms["median_ms"]) / cand,
                "method": "writer returns zeros (diagnostic)",
                "confidence": "high",
                "semantics_preserving_optimization": "systems only",
            },
        },
        "profiler": prof,
        "unexplained_ms": cand - (
            forward_ms["median_ms"] + backward_only_ms["median_ms"]
            + optimizer_ms["median_ms"]),
    }
    out = Path("results/onehour_floor_breakdown.json")
    out.write_text(json.dumps(breakdown, indent=2, default=str),
                   encoding="utf-8")
    print(json.dumps({k: v for k, v in breakdown.items()
                      if k != "components"}, indent=2, default=str))
    print("components:")
    for name, comp in breakdown["components"].items():
        print("  %-26s %7.2f ms  %5.1f%%  (%s)" % (
            name, comp["estimated_ms"], 100 * comp["fraction_of_total"],
            comp["confidence"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
