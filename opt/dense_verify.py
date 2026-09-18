"""Dense BDH verification run (one fresh process per invocation).

Dense only. No sparse paths anywhere in this file.

Kinds:
  correctness : control-vs-reference forward/backward/param-update,
                determinism, finiteness, document boundary, B=1/B=2,
                cross-row non-interference
  cold        : isolated compiler caches, compile + first-step timing
  warm        : 5 warmups + N measured full updates (131072 tokens)
  comparator  : reference implementation full updates (limited sample)
  longrun     : continuous dense control updates with telemetry
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
from opt.model_ref import ArmAConfig, NativeReadStage1ArmA, canonical_init, \
    load_init, synthetic_packed_batch

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
        return subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,"
             "utilization.gpu,utilization.memory,memory.used",
             "--format=csv,noheader"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def build_control(cfg, device, compile_forward=True):
    model = OptArmA(cfg, device, scan_block=1024, use_checkpoint=False,
                    coord="dense", single_scan="chunkwise",
                    packed_update="branchfree", zero_carry=True,
                    paper_layout="direct", cache_rope=True).to(device)
    load_init(model, canonical_init(cfg), device)
    model.train()
    if not compile_forward:
        return model, model.forward_packed
    entry = torch.compile(model.forward_packed, mode="default")

    def fn(b):
        return entry(b["x"], b["pos"], b["segpos"], b["full_mask"],
                     b["segment_start"])
    return model, fn


def build_reference(cfg, device):
    model = NativeReadStage1ArmA(cfg, device).to(device)
    load_init(model, canonical_init(cfg), device)
    model.train()
    return model


def make_optimizer(model, cfg):
    return torch.optim.AdamW(model.parameters(), lr=cfg.PEAK_LR,
                             betas=cfg.BETAS, eps=cfg.EPS, fused=True)


def make_micro(cfg, device, microbatch, seed_base):
    acc = GLOBAL // microbatch
    micro = [make_batch(cfg, microbatch, device, seed_base + i)
             for i in range(acc)]
    denom = float(sum(int(b["valid"].sum().item()) for b in micro))
    return micro, denom, acc


def run_update(model, forward_fn, optimizer, cfg, micro, denom):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                        cache_enabled=False):
        for batch in micro:
            logits = forward_fn(batch)
            loss = F.cross_entropy(
                logits.reshape(-1, cfg.V), batch["y"].reshape(-1),
                reduction="none",
            )[batch["valid"].reshape(-1)].sum(dtype=torch.float32) / denom
            loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_NORM)
    optimizer.step()
    return float(loss.detach())


# ---------------------------------------------------------------------------
# correctness
# ---------------------------------------------------------------------------

def _grads(model, cfg, batch, reference_style):
    model.zero_grad(set_to_none=True)
    if reference_style:
        logits = model.forward(batch["x"], batch["pos"], batch["segpos"],
                               batch["full_mask"])
    else:
        logits = model.forward_packed(batch["x"], batch["pos"],
                                      batch["segpos"], batch["full_mask"],
                                      batch["segment_start"])
    loss = F.cross_entropy(
        logits.reshape(-1, cfg.V), batch["y"].reshape(-1),
        reduction="none",
    )[batch["valid"].reshape(-1)].sum(dtype=torch.float32) / float(
        batch["valid"].sum())
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    return logits.detach(), float(loss.detach()), grads


def kind_correctness(cfg, device):
    result = {"kind": "correctness", "checks": {}}
    small = ArmAConfig(T=64, V=cfg.V, D=cfg.D, N=cfg.N, H=cfg.H, L=2,
                       HIDDEN=cfg.HIDDEN, SEED=cfg.SEED, THETA=cfg.THETA,
                       READ_BLOCK=64)
    torch.manual_seed(SEED)
    batch = make_batch(small, 2, device, 11000)

    control = OptArmA(small, device, scan_block=64, use_checkpoint=False,
                      coord="dense", single_scan="chunkwise",
                      packed_update="branchfree", zero_carry=True,
                      paper_layout="direct", cache_rope=True).to(device)
    reference = NativeReadStage1ArmA(small, device).to(device)
    init = canonical_init(small)
    load_init(control, init, device)
    load_init(reference, init, device)
    control.train()
    reference.train()
    c_logits, c_loss, c_grads = _grads(control, small, batch, False)
    r_logits, r_loss, r_grads = _grads(reference, small, batch, True)
    result["checks"]["forward_max_abs_diff_small"] = float(
        (c_logits - r_logits).abs().max())
    result["checks"]["loss_abs_diff_small"] = abs(c_loss - r_loss)
    result["checks"]["grad_max_abs_diff_small"] = max(
        float((c_grads[n] - r_grads[n]).abs().max()) for n in r_grads)
    result["checks"]["loss_finite"] = bool(
        c_loss == c_loss and abs(c_loss) != float("inf"))
    result["checks"]["all_grads_finite"] = all(
        bool(torch.isfinite(g).all()) for g in c_grads.values())

    opt_c = make_optimizer(control, small)
    opt_r = make_optimizer(reference, small)
    opt_c.step()
    opt_r.step()
    update_diff = max(
        float((p_c.detach() - p_r.detach()).abs().max())
        for (n, p_c), (_, p_r) in zip(control.named_parameters(),
                                      reference.named_parameters()))
    result["checks"]["param_update_max_abs_diff_small"] = update_diff

    # production-shape forward parity, B=1 and B=2 (fp32, no grad)
    prod = ArmAConfig()
    prod_control = OptArmA(prod, device, scan_block=1024,
                           use_checkpoint=False, coord="dense",
                           single_scan="chunkwise",
                           packed_update="branchfree", zero_carry=True,
                           paper_layout="direct",
                           cache_rope=True).to(device)
    prod_reference = NativeReadStage1ArmA(prod, device).to(device)
    init_p = canonical_init(prod)
    load_init(prod_control, init_p, device)
    load_init(prod_reference, init_p, device)
    prod_control.eval()
    prod_reference.eval()
    for batch_size in (1, 2):
        probe = make_batch(prod, batch_size, device, 12000 + batch_size)
        with torch.no_grad():
            out_c = prod_control.forward_packed(
                probe["x"], probe["pos"], probe["segpos"], probe["full_mask"],
                probe["segment_start"])
            out_r = prod_reference.forward_eval(
                probe["x"], probe["pos"], probe["segpos"], probe["full_mask"])
            diff = float((out_c - out_r).abs().max())
            again = prod_control.forward_packed(
                probe["x"], probe["pos"], probe["segpos"], probe["full_mask"],
                probe["segment_start"])
            det = float((out_c - again).abs().max())
        result["checks"][f"prod_forward_max_abs_diff_B{batch_size}"] = diff
        result["checks"][f"determinism_B{batch_size}"] = det
        del probe
        torch.cuda.empty_cache()

    # cross-row non-interference, production shape, B=2, fp32 no grad
    probe = make_batch(prod, 2, device, 13000)
    with torch.no_grad():
        full_c = prod_control.forward_packed(
            probe["x"], probe["pos"], probe["segpos"], probe["full_mask"],
            probe["segment_start"])
        row0_c = prod_control.forward_packed(
            probe["x"][:1], probe["pos"][:1], probe["segpos"][:1],
            probe["full_mask"][:1], probe["segment_start"][:1])
        full_r = prod_reference.forward_eval(
            probe["x"], probe["pos"], probe["segpos"], probe["full_mask"])
        row0_r = prod_reference.forward_eval(
            probe["x"][:1], probe["pos"][:1], probe["segpos"][:1],
            probe["full_mask"][:1])
        changed = {k: (v.clone() if torch.is_tensor(v) else v)
                   for k, v in probe.items()}
        changed["x"][1] = (changed["x"][1] + 1) % prod.V
        changed["y"][1] = (changed["y"][1] + 1) % prod.V
        full_c_mod = prod_control.forward_packed(
            changed["x"], changed["pos"], changed["segpos"],
            changed["full_mask"], changed["segment_start"])
    result["checks"]["cross_row_row0_control_diff"] = float(
        (full_c[0] - row0_c[0]).abs().max())
    result["checks"]["cross_row_row0_reference_diff"] = float(
        (full_r[0] - row0_r[0]).abs().max())
    result["checks"]["cross_row_modified_row0_control_diff"] = float(
        (full_c_mod[0] - full_c[0]).abs().max())
    result["checks"]["cross_row_modified_row1_changed"] = float(
        (full_c_mod[1] - full_c[1]).abs().max())

    gates = {
        "small_forward": result["checks"][
            "forward_max_abs_diff_small"] < 1e-4,
        "small_grads": result["checks"]["grad_max_abs_diff_small"] < 1e-3,
        "param_update": result["checks"][
            "param_update_max_abs_diff_small"] < 1e-3,
        "prod_b1": result["checks"][
            "prod_forward_max_abs_diff_B1"] < 1e-3,
        "prod_b2": result["checks"][
            "prod_forward_max_abs_diff_B2"] < 1e-3,
        "determinism": (result["checks"]["determinism_B1"] == 0.0
                        and result["checks"]["determinism_B2"] == 0.0),
        "cross_row": (result["checks"]["cross_row_row0_control_diff"] < 1e-4
                      and result["checks"][
                          "cross_row_row0_reference_diff"] < 1e-4
                      and result["checks"][
                          "cross_row_modified_row0_control_diff"] == 0.0
                      and result["checks"][
                          "cross_row_modified_row1_changed"] > 1e-2),
        "finite": (result["checks"]["loss_finite"]
                   and result["checks"]["all_grads_finite"]),
    }
    result["gates"] = gates
    result["pass"] = all(gates.values())
    return result


# ---------------------------------------------------------------------------
# benchmarks
# ---------------------------------------------------------------------------

def kind_cold(cfg, device):
    t0 = time.perf_counter()
    model, fn = build_control(cfg, device, compile_forward=True)
    build_s = time.perf_counter() - t0
    opt = make_optimizer(model, cfg)
    micro, denom, acc = make_micro(cfg, device, 1, 14000)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    loss = run_update(model, fn, opt, cfg, micro, denom)
    torch.cuda.synchronize()
    first_step = time.perf_counter() - t1
    return {"kind": "cold", "build_s": build_s,
            "first_step_s": first_step,
            "first_step_tok_s": GLOBAL * cfg.T / first_step,
            "first_loss": loss, "telemetry": telemetry(),
            "device": torch.cuda.get_device_name(0)}


def kind_warm(cfg, device, microbatch, windows):
    model, fn = build_control(cfg, device, compile_forward=True)
    opt = make_optimizer(model, cfg)
    micro, denom, acc = make_micro(cfg, device, microbatch, 15000)
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
        print(json.dumps(rows[-1]), flush=True)
    return {"kind": "warm", "microbatch": microbatch, "accumulation": acc,
            "windows": rows,
            "peak_mem_GiB": torch.cuda.max_memory_allocated(device) / 2**30,
            "device": torch.cuda.get_device_name(0)}


def kind_comparator(cfg, device, updates):
    reference = build_reference(cfg, device)
    opt = make_optimizer(reference, cfg)
    micro, denom, acc = make_micro(cfg, device, 1, 16000)

    def fn(b):
        return reference.forward(b["x"], b["pos"], b["segpos"],
                                 b["full_mask"])

    rows = []
    for i in range(updates):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = run_update(reference, fn, opt, cfg, micro, denom)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        rows.append({"update": i, "elapsed_s": dt,
                     "tok_s": GLOBAL * cfg.T / dt, "loss": loss,
                     "telemetry": telemetry()})
        print(json.dumps(rows[-1]), flush=True)
    return {"kind": "comparator", "implementation":
            "opt/model_ref.py NativeReadStage1ArmA (checkpointed)",
            "microbatch": 1, "accumulation": acc, "updates": rows,
            "peak_mem_GiB": torch.cuda.max_memory_allocated(device) / 2**30,
            "device": torch.cuda.get_device_name(0)}


def kind_longrun(cfg, device, minutes, sample_s):
    model, fn = build_control(cfg, device, compile_forward=True)
    opt = make_optimizer(model, cfg)
    micro, denom, acc = make_micro(cfg, device, 1, 17000)
    for _ in range(2):
        run_update(model, fn, opt, cfg, micro, denom)
    torch.cuda.synchronize()
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    samples = []
    window_start = start
    window_tokens = 0
    steps = 0
    while time.perf_counter() - start < minutes * 60.0:
        loss = run_update(model, fn, opt, cfg, micro, denom)
        window_tokens += GLOBAL * cfg.T
        steps += 1
        now = time.perf_counter()
        if now - window_start >= sample_s:
            dt = now - window_start
            samples.append({"t_s": now - start, "steps": steps,
                            "tok_s": window_tokens / dt, "loss": loss,
                            "peak_mem_GiB":
                                torch.cuda.max_memory_allocated(device)
                                / 2**30,
                            "telemetry": telemetry()})
            window_start = now
            window_tokens = 0
            print(json.dumps(samples[-1]), flush=True)
    return {"kind": "longrun", "microbatch": 1, "minutes": minutes,
            "samples": samples, "device": torch.cuda.get_device_name(0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True,
                        choices=("correctness", "cold", "warm", "comparator",
                                 "longrun"))
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--minutes", type=float, default=12.0)
    parser.add_argument("--sample", type=float, default=30.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = ArmAConfig()
    device = torch.device("cuda")
    if args.kind == "correctness":
        result = kind_correctness(cfg, device)
    elif args.kind == "cold":
        result = kind_cold(cfg, device)
    elif args.kind == "warm":
        result = kind_warm(cfg, device, args.microbatch, args.windows)
    elif args.kind == "comparator":
        result = kind_comparator(cfg, device, args.updates)
    else:
        result = kind_longrun(cfg, device, args.minutes, args.sample)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str),
                   encoding="utf-8")
    print(json.dumps({"out": str(out), "kind": args.kind,
                      "pass": result.get("pass")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
