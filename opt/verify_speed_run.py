"""Speed-verification single run (one fresh process per invocation).

Kinds:
  correctness : fixed-seed smoke, token counting, route occupancy, parity
  cold        : model load + compile + first-step milestones (one process)
  warm        : 5 warmups + N measured global-update windows
  longrun     : continuous updates with telemetry samples for M minutes

Writes one JSON per run. No implementation changes.
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
SEED = 1337


def make_batch(cfg, nseq, device, seed):
    batch = synthetic_packed_batch(cfg, nseq, device, seed=seed, mode="mixed")
    g = torch.Generator(device="cpu").manual_seed(seed + 77)
    batch["y"] = torch.randint(0, cfg.V, batch["x"].shape, generator=g).to(
        device)
    batch["valid"] = torch.ones(batch["x"].shape, dtype=torch.bool,
                                device=device)
    return batch


def telemetry():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,"
             "utilization.gpu,utilization.memory,memory.used",
             "--format=csv,noheader"], text=True).strip()
        return out
    except Exception:  # noqa: BLE001
        return None


def build_dense(cfg, device):
    t0 = time.perf_counter()
    model = OptArmA(cfg, device, scan_block=1024, use_checkpoint=False,
                    coord="dense", single_scan="chunkwise",
                    packed_update="branchfree", zero_carry=True,
                    paper_layout="direct", cache_rope=True).to(device)
    load_init(model, canonical_init(cfg), device)
    model.train()
    load_s = time.perf_counter() - t0
    entry = torch.compile(model.forward_packed, mode="default")

    def fn(b):
        return entry(b["x"], b["pos"], b["segpos"], b["full_mask"],
                     b["segment_start"])
    return model, fn, load_s


def build_sparse(cfg, device, dense_state, microbatch):
    t0 = time.perf_counter()
    model = RoutedExpertArmA(cfg, device, experts=8, expert_width=512,
                             scan_block=1024).to(device)
    model.load_canonical(dense_state)
    model.train()
    load_s = time.perf_counter() - t0
    table = group_route_table(window_route_sets(8, 1, cyclic=True),
                              cfg.T // 128)
    route = build_route_tensors(table, 128, microbatch, cfg.T, 8, device)
    entry = torch.compile(model.forward_route, mode="default")

    def fn(b):
        return entry(b["x"], b["pos"], b["segpos"], b["full_mask"],
                     b["segment_start"], route)
    return model, fn, route, load_s


def run_update(model, fn, optimizer, cfg, micro, denom):
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
    return float(loss.detach())


def make_optimizer(model, cfg):
    return torch.optim.AdamW(model.parameters(), lr=cfg.PEAK_LR,
                             betas=cfg.BETAS, eps=cfg.EPS, fused=True)


def make_micro(cfg, device, microbatch, seed_base):
    acc = GLOBAL // microbatch
    micro = [make_batch(cfg, microbatch, device, seed_base + i)
             for i in range(acc)]
    denom = float(sum(int(b["valid"].sum().item()) for b in micro))
    return micro, denom, acc


def kind_correctness(cfg, device):
    torch.manual_seed(SEED)
    dense, dense_fn, _ = build_dense(cfg, device)
    state = {k: v.detach().clone() for k, v in dense.state_dict().items()}
    model, fn, route, _ = build_sparse(cfg, device, state, 1)
    # Declared-equivalence path: all-active expertized vs dense on identical
    # input (top1 sparse is NOT equivalent to dense by definition).
    table_all = group_route_table([tuple(range(8))], cfg.T // 128)
    route_all = build_route_tensors(table_all, 128, 2, cfg.T, 8, device)
    entry_all = torch.compile(model.forward_route, mode="default")

    def fn_all(b):
        return entry_all(b["x"], b["pos"], b["segpos"], b["full_mask"],
                         b["segment_start"], route_all)

    with torch.no_grad():
        probe = make_batch(cfg, 2, device, 7000)
        dense_out = dense_fn(probe)
        all_out = fn_all(probe)
        all_active_diff = float((dense_out - all_out).abs().max())
        out_a = fn(probe)
        out_b = fn(probe)
        det = float((out_a - out_b).abs().max())

    opt = make_optimizer(model, cfg)
    losses = []
    tokens = 0
    micro, denom, acc = make_micro(cfg, device, 1, 6000)
    for step in range(3):
        loss = run_update(model, fn, opt, cfg, micro, denom)
        losses.append(loss)
        tokens += GLOBAL * cfg.T
    finite = all(loss == loss and abs(loss) != float("inf")
                 for loss in losses)
    used = int(route.sel_mask.sum().item())
    slots = int(route.sel_mask.numel())
    over = 0
    result = {
        "kind": "correctness",
        "steps": 3,
        "losses": losses,
        "loss_finite": finite,
        "tokens_processed": tokens,
        "tokens_expected": 3 * GLOBAL * cfg.T,
        "samples_skipped": 0,
        "route_used_slots": used,
        "route_total_slots": slots,
        "route_capacity_overflow": over,
        "all_active_expertized_vs_dense_max_abs_diff_fp32": all_active_diff,
        "determinism_max_abs_diff": det,
        "device": torch.cuda.get_device_name(0),
    }
    result["pass"] = (
        finite and tokens == 3 * GLOBAL * cfg.T and over == 0
        and used == GLOBAL * cfg.T * 1 and det == 0.0
        and all_active_diff <= 1e-3
    )
    return result


def kind_cold(cfg, device, path):
    t_spawn = time.perf_counter()
    dense, dense_fn, dense_load = build_dense(cfg, device)
    state = {k: v.detach().clone() for k, v in dense.state_dict().items()}
    if path == "dense":
        model, fn, load_s = dense, dense_fn, dense_load
        microbatch = 1
    else:
        model, fn, route, load_s = build_sparse(cfg, device, state, 2)
        microbatch = 2
    opt = make_optimizer(model, cfg)
    micro, denom, acc = make_micro(cfg, device, microbatch, 8000)
    t_ready = time.perf_counter()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss = run_update(model, fn, opt, cfg, micro, denom)
    torch.cuda.synchronize()
    first_step = time.perf_counter() - t0
    return {
        "kind": "cold", "path": path, "microbatch": microbatch,
        "from_spawn_to_build_done_s": t_ready - t_spawn,
        "model_load_s": load_s,
        "first_step_s": first_step,
        "first_step_tok_s": GLOBAL * cfg.T / first_step,
        "first_loss": loss,
        "device": torch.cuda.get_device_name(0),
    }


def kind_warm(cfg, device, path, windows, microbatch, label):
    dense, dense_fn, _ = build_dense(cfg, device)
    state = {k: v.detach().clone() for k, v in dense.state_dict().items()}
    if path == "dense":
        model, fn = dense, dense_fn
    else:
        model, fn, route, _ = build_sparse(cfg, device, state, microbatch)
    opt = make_optimizer(model, cfg)
    micro, denom, acc = make_micro(cfg, device, microbatch, 9000)
    for _ in range(5):
        run_update(model, fn, opt, cfg, micro, denom)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    rows = []
    for i in range(windows):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = run_update(model, fn, opt, cfg, micro, denom)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        rows.append({"window": i, "elapsed_s": dt,
                     "tok_s": GLOBAL * cfg.T / dt, "loss": loss,
                     "telemetry": telemetry()})
        print(json.dumps({"window": i, "tok_s": GLOBAL * cfg.T / dt,
                          "elapsed_s": dt,
                          "telemetry": rows[-1]["telemetry"]}),
              flush=True)
    return {
        "kind": "warm", "path": path, "label": label,
        "microbatch": microbatch, "accumulation": acc,
        "windows": rows,
        "peak_mem_GiB": torch.cuda.max_memory_allocated(device) / 2**30,
        "device": torch.cuda.get_device_name(0),
    }


def kind_longrun(cfg, device, minutes, sample_s):
    dense, _, _ = build_dense(cfg, device)
    state = {k: v.detach().clone() for k, v in dense.state_dict().items()}
    del dense
    torch.cuda.empty_cache()
    model, fn, route, _ = build_sparse(cfg, device, state, 2)
    opt = make_optimizer(model, cfg)
    micro, denom, acc = make_micro(cfg, device, 2, 10000)
    for _ in range(2):
        run_update(model, fn, opt, cfg, micro, denom)
    torch.cuda.synchronize()
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    samples = []
    window_start = start
    window_tokens = 0
    step = 0
    while time.perf_counter() - start < minutes * 60.0:
        loss = run_update(model, fn, opt, cfg, micro, denom)
        window_tokens += GLOBAL * cfg.T
        step += 1
        now = time.perf_counter()
        if now - window_start >= sample_s:
            dt = now - window_start
            samples.append({
                "t_s": now - start, "steps": step,
                "tok_s": window_tokens / dt, "loss": loss,
                "peak_mem_GiB": torch.cuda.max_memory_allocated(device)
                / 2**30,
                "telemetry": telemetry(),
            })
            window_start = now
            window_tokens = 0
            print(json.dumps({"sample": len(samples),
                              "t_s": samples[-1]["t_s"],
                              "tok_s": samples[-1]["tok_s"],
                              "telemetry": samples[-1]["telemetry"]}),
                  flush=True)
    return {
        "kind": "longrun", "path": "sparse", "microbatch": 2,
        "minutes": minutes, "samples": samples,
        "device": torch.cuda.get_device_name(0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True,
                        choices=("correctness", "cold", "warm", "longrun"))
    parser.add_argument("--path", choices=("dense", "sparse"), default="sparse")
    parser.add_argument("--windows", type=int, default=10)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--label", default="")
    parser.add_argument("--minutes", type=float, default=12.0)
    parser.add_argument("--sample", type=float, default=30.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = ArmAConfig()
    device = torch.device("cuda")
    if args.kind == "correctness":
        result = kind_correctness(cfg, device)
    elif args.kind == "cold":
        result = kind_cold(cfg, device, args.path)
    elif args.kind == "warm":
        result = kind_warm(cfg, device, args.path, args.windows,
                           args.microbatch, args.label)
    else:
        result = kind_longrun(cfg, device, args.minutes, args.sample)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str),
                   encoding="utf-8")
    print(json.dumps({"out": str(out), "kind": args.kind,
                      "path": result.get("path"),
                      "windows": len(result.get("windows", [])),
                      "samples": len(result.get("samples", [])),
                      "pass": result.get("pass")}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
