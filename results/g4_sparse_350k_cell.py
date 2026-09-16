# ICLR — G4 SPARSE 350K TRANSFER CELL (single cell, G4 Colab)
#
# Purpose: measure the sparse one-hour candidate on the production G4 and
# return machine-readable throughput + same-session dense reference.
#
# Requirements:
#   - fresh G4 runtime (RTX PRO 6000 Blackwell, torch 2.11.0+cu128)
#   - this repo available locally; set REPO_PATH below (Drive or clone)
#   - runs the sparse executor from opt/ unchanged
#
# Outputs (stdout):
#   G4_DENSE_TOK_S_<geom>=...
#   G4_SPARSE_TOK_S_<geom>=...
#   G4_SAME_SESSION_SPEEDUP_<geom>=...
#   G4_BEST_GEOMETRY=...
#   G4_RESULT_JSON=<path>
#
# No 2.5B training is launched here. This is a benchmark + correctness cell.

import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------- config
REPO_PATH = "/content/drive/Shareddrives/ICLR PHASE BDH/phase_bdh/iclr-oc"
OUT_JSON = "/content/drive/Shareddrives/ICLR PHASE BDH/phase_bdh/runs/arm_a_sparse_350k/g4_sparse_350k.json"
GEOMETRIES = [8, 16, 32, 64]          # microbatch sequences; 64 = 1 accumulation step
DENSE_GEOMETRIES = [16]               # production B16x4 baseline geometry
WARMUPS = 1
STEPS = 3
GLOBAL_SEQUENCES = 64

if not Path(REPO_PATH).is_dir():
    raise SystemExit(f"REPO_PATH not found: {REPO_PATH}")
sys.path.insert(0, REPO_PATH)

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

# ---------------------------------------------------------------- gates
if not torch.cuda.is_available():
    raise SystemExit("CUDA required")
GPU = torch.cuda.get_device_name(0)
CAP = torch.cuda.get_device_capability(0)
if "RTX PRO 6000 Blackwell" not in GPU or CAP != (12, 0):
    raise SystemExit(f"unexpected GPU: {GPU} {CAP}")
if torch.__version__ != "2.11.0+cu128" or torch.version.cuda != "12.8":
    raise SystemExit(f"unexpected runtime: torch={torch.__version__} "
                     f"cuda={torch.version.cuda}")
print(f"ENV_OK gpu={GPU} cap={CAP} torch={torch.__version__}")


def make_batch(cfg, nseq, device, seed):
    batch = synthetic_packed_batch(cfg, nseq, device, seed=seed,
                                   mode="mixed")
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


def build_sparse(cfg, device, dense_state, microbatch):
    model = RoutedExpertArmA(cfg, device, experts=8, expert_width=512,
                             scan_block=1024).to(device)
    model.load_canonical(dense_state)
    model.train()
    table = group_route_table(window_route_sets(8, 1, cyclic=True),
                              cfg.T // 128)
    route = build_route_tensors(table, 128, microbatch, cfg.T, 8, device)
    entry = torch.compile(model.forward_route, mode="default")

    def fn(b):
        return entry(b["x"], b["pos"], b["segpos"], b["full_mask"],
                     b["segment_start"], route)
    return model, fn, route


def update(model, fn, optimizer, cfg, micro, denom):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                        cache_enabled=False):
        for batch in micro:
            logits = fn(batch)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, cfg.V), batch["y"].reshape(-1),
                reduction="none",
            )[batch["valid"].reshape(-1)].sum(dtype=torch.float32) / denom
            loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_NORM)
    optimizer.step()


def bench(name, model, fn, optimizer, cfg, device, microbatch):
    acc = GLOBAL_SEQUENCES // microbatch
    micro = [make_batch(cfg, microbatch, device, 3000 + i)
             for i in range(acc)]
    denom = float(sum(int(b["valid"].sum().item()) for b in micro))
    try:
        for _ in range(WARMUPS):
            update(model, fn, optimizer, cfg, micro, denom)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(STEPS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            update(model, fn, optimizer, cfg, micro, denom)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        valid = sum(int(b["valid"].sum().item()) for b in micro)
        tok = GLOBAL_SEQUENCES * cfg.T
        med = statistics.median(times)
        record = {
            "name": name, "microbatch": microbatch, "accum": acc,
            "median_ms": med, "p10_ms": times[0], "p90_ms": times[-1],
            "packed_tok_s": tok / (med / 1000.0),
            "valid_pairs_s": valid / (med / 1000.0),
            "peak_mem_GiB": torch.cuda.max_memory_allocated(device) / 2**30,
            "status": "ok",
        }
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        record = {"name": name, "microbatch": microbatch,
                  "status": "OOM"}
    except Exception as exc:  # noqa: BLE001
        record = {"name": name, "microbatch": microbatch,
                  "status": f"error: {type(exc).__name__}: {str(exc)[:200]}"}
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                      for k, v in record.items()}), flush=True)
    return record


def correctness_gates(cfg, device):
    results = {}
    dense, dense_fn = build_dense(cfg, device)
    sparse, sparse_fn, route = build_sparse(cfg, device, dense.state_dict(), 1)
    probe = make_batch(cfg, 1, device, 42)
    with torch.no_grad():
        a = dense_fn(probe)
        b1 = sparse_fn(probe)
        b2 = sparse_fn(probe)
    results["fixed_route_determinism"] = float((b1 - b2).abs().max()) == 0.0
    results["sparse_vs_dense_fp32_max_abs_diff"] = float(
        (a - b1).abs().max())
    results["graph_breaks"] = None
    try:
        torch._dynamo.reset()
        exp = torch._dynamo.explain(sparse.forward_route)(
            probe["x"], probe["pos"], probe["segpos"], probe["full_mask"],
            probe["segment_start"], route)
        results["graph_count"] = int(getattr(exp, "graph_count", -1))
        results["graph_breaks"] = int(getattr(exp, "graph_break_count", -1))
        torch._dynamo.reset()
    except Exception as exc:  # noqa: BLE001
        results["graph_breaks"] = f"error: {type(exc).__name__}"
    del dense, sparse
    torch.cuda.empty_cache()
    print("CORRECTNESS " + json.dumps(results), flush=True)
    return results


def main():
    cfg = ArmAConfig()
    device = torch.device("cuda")
    results = {"env": {"gpu": GPU, "capability": list(CAP),
                       "torch": torch.__version__,
                       "cuda": torch.version.cuda},
               "correctness": correctness_gates(cfg, device),
               "benchmarks": []}

    dense, dense_fn = build_dense(cfg, device)
    for geom in DENSE_GEOMETRIES:
        opt = torch.optim.AdamW(dense.parameters(), lr=cfg.PEAK_LR,
                                betas=cfg.BETAS, eps=cfg.EPS, fused=True)
        rec = bench("dense", dense, dense_fn, opt, cfg, device, geom)
        results["benchmarks"].append(rec)
        print(f"G4_DENSE_TOK_S_B{geom}="
              f"{rec.get('packed_tok_s', 'NA')}", flush=True)
    dense_state = {k: v.detach().clone()
                   for k, v in dense.state_dict().items()}
    del dense
    torch.cuda.empty_cache()

    for geom in GEOMETRIES:
        sparse, sparse_fn, _ = build_sparse(cfg, device, dense_state, geom)
        opt = torch.optim.AdamW(sparse.parameters(), lr=cfg.PEAK_LR,
                                betas=cfg.BETAS, eps=cfg.EPS, fused=True)
        rec = bench("sparse", sparse, sparse_fn, opt, cfg, device, geom)
        results["benchmarks"].append(rec)
        print(f"G4_SPARSE_TOK_S_B{geom}="
              f"{rec.get('packed_tok_s', 'NA')}", flush=True)
        del sparse
        torch.cuda.empty_cache()

    dense_rec = next((r for r in results["benchmarks"]
                      if r["name"] == "dense" and r["status"] == "ok"), None)
    sparse_ok = [r for r in results["benchmarks"]
                 if r["name"] == "sparse" and r["status"] == "ok"]
    best = max(sparse_ok, key=lambda r: r["packed_tok_s"]) if sparse_ok \
        else None
    if best and dense_rec:
        speedup = best["packed_tok_s"] / dense_rec["packed_tok_s"]
        results["summary"] = {
            "best_sparse": best, "dense": dense_rec, "speedup": speedup,
            "projected_2p5b_hours": 2.5e9 / best["packed_tok_s"] / 3600.0,
        }
        print(f"G4_SAME_SESSION_SPEEDUP_B{best['microbatch']}={speedup:.3f}")
        print(f"G4_BEST_GEOMETRY=B{best['microbatch']}")
        print(f"G4_PROJECTED_2P5B_HOURS={2.5e9/best['packed_tok_s']/3600.0:.2f}")
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_JSON).write_text(json.dumps(results, indent=2, default=str),
                              encoding="utf-8")
    print("G4_RESULT_JSON=" + OUT_JSON)


main()
