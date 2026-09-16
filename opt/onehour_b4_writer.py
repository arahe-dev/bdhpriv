"""B4: writer-share ablation on the mb2 champion (diagnostic, one process).

Diagnostic only (writer zeroed changes semantics); never promoted.
"""

from __future__ import annotations

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
GLOBAL = 64


def make_batch(cfg, nseq, device, seed):
    batch = synthetic_packed_batch(cfg, nseq, device, seed=seed, mode="mixed")
    g = torch.Generator(device="cpu").manual_seed(seed + 77)
    batch["y"] = torch.randint(0, cfg.V, batch["x"].shape, generator=g).to(
        device)
    batch["valid"] = torch.ones(batch["x"].shape, dtype=torch.bool,
                                device=device)
    return batch


def build_dense(cfg, device):
    model = OptArmA(cfg, device, scan_block=1024, use_checkpoint=False,
                    coord="dense", single_scan="chunkwise",
                    packed_update="branchfree", zero_carry=True,
                    paper_layout="direct", cache_rope=True).to(device)
    load_init(model, canonical_init(cfg), device)
    model.train()
    entry = torch.compile(model.forward_packed, mode="default")

    def fn(b):
        return entry(b["x"], b["pos"], b["segpos"], b["full_mask"],
                     b["segment_start"])
    return model, fn


def build_sparse(cfg, device, state, microbatch, zero_writer=False):
    model = RoutedExpertArmA(cfg, device, experts=8, expert_width=512,
                             scan_block=1024).to(device)
    model.load_canonical(state)
    model.train()
    if zero_writer:
        model.writer.forward = lambda x: torch.zeros_like(x)
    table = group_route_table(window_route_sets(8, 1, cyclic=True),
                              cfg.T // 128)
    route = build_route_tensors(table, 128, microbatch, cfg.T, 8, device)
    entry = torch.compile(model.forward_route, mode="default")

    def fn(b):
        return entry(b["x"], b["pos"], b["segpos"], b["full_mask"],
                     b["segment_start"], route)
    return model, fn


def measure(model, fn, cfg, device, microbatch, steps=3, warmups=1):
    acc = GLOBAL // microbatch
    micro = [make_batch(cfg, microbatch, device, 5000 + i)
             for i in range(acc)]
    denom = float(sum(int(b["valid"].sum().item()) for b in micro))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.PEAK_LR,
                            betas=cfg.BETAS, eps=cfg.EPS, fused=True)

    def update():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            cache_enabled=False):
            for batch in micro:
                logits = fn(batch)
                loss = F.cross_entropy(
                    logits.reshape(-1, cfg.V), batch["y"].reshape(-1),
                    reduction="none",
                )[batch["valid"].reshape(-1)].sum(
                    dtype=torch.float32) / denom
                loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_NORM)
        opt.step()

    for _ in range(warmups):
        update()
    torch.cuda.synchronize()
    times = []
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        update()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    tok = GLOBAL * cfg.T
    return {"median_ms": statistics.median(times), "ms": times,
            "packed_tok_s": tok / (statistics.median(times) / 1000.0)}


def main():
    cfg = ArmAConfig()
    device = torch.device("cuda")
    dense, dense_fn = build_dense(cfg, device)
    dense_ms = measure(dense, dense_fn, cfg, device, 1)
    state = {k: v.detach().clone() for k, v in dense.state_dict().items()}
    del dense
    torch.cuda.empty_cache()

    normal, normal_fn = build_sparse(cfg, device, state, 2)
    normal_ms = measure(normal, normal_fn, cfg, device, 2)
    del normal
    torch.cuda.empty_cache()
    zeroed, zeroed_fn = build_sparse(cfg, device, state, 2,
                                     zero_writer=True)
    zeroed_ms = measure(zeroed, zeroed_fn, cfg, device, 2)

    share = (normal_ms["median_ms"] - zeroed_ms["median_ms"]) / \
        normal_ms["median_ms"]
    report = {
        "diagnostic_note": "writer zeroed changes semantics; never promoted",
        "dense_mb1_ms": dense_ms["median_ms"],
        "sparse_mb2_normal_ms": normal_ms["median_ms"],
        "sparse_mb2_writer_zeroed_ms": zeroed_ms["median_ms"],
        "writer_share_of_mb2_wall": share,
        "normal_tok_s": normal_ms["packed_tok_s"],
        "zeroed_tok_s": zeroed_ms["packed_tok_s"],
    }
    out = Path("results/350k_b4_writer.json")
    out.write_text(json.dumps(report, indent=2, default=str),
                   encoding="utf-8")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                      for k, v in report.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
