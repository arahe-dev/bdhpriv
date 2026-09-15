"""Shared full-step benchmark harness (methodology mirrors the canonical cell).

One timed step = H2D already resident (synthetic batch on device) + forward
+ CE + backward (+accumulation) + grad clip + AdamW step. Warmups, CUDA
sync around timed updates, repeated samples, median/p10/p90. Never
forward-only. tok/s = input tokens / median step time.
"""

import gc
import time

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from opt.model_ref import (
    full_update,
    make_optimizer,
    synthetic_packed_batch,
    synthetic_single_doc_batch,
)


def bench_variant(make_model, cfg, device, global_batch, microbatch,
                  seed=0, warmups=3, samples=10, compiled=False,
                  compile_mode="default", packed="single", assume_single_doc=False):
    torch.manual_seed(cfg.SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.SEED)
    model = make_model().to(device)
    from opt.model_ref import canonical_init, load_init
    load_init(model, canonical_init(cfg), device)
    model.train()
    opt = make_optimizer(model, cfg, device.type)
    if packed == "single":
        batch = synthetic_single_doc_batch(cfg, global_batch, device, seed=seed)
    else:
        batch = synthetic_packed_batch(cfg, global_batch, device, seed=seed, mode=packed)

    # Production path: compile the static entry point (no host syncs, no
    # .item()/.all()); the canonical ref has no static entries and uses its
    # module forward.
    if packed == "single" and hasattr(model, "forward_single_doc"):
        entry = model.forward_single_doc
        entry_mode = "single"
    elif packed != "single" and hasattr(model, "forward_packed"):
        entry = model.forward_packed
        entry_mode = "packed"
    else:
        entry = model
        entry_mode = "canonical"
    if assume_single_doc and entry_mode == "packed":
        raise RuntimeError("assume_single_doc requested for a packed batch")
    cmodel = torch.compile(entry, mode=compile_mode) if compiled else entry
    for w in range(warmups):
        full_update(model, opt, cfg, batch, microbatch, device.type,
                    compiled=cmodel, entry_mode=entry_mode)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for s in range(samples):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        full_update(model, opt, cfg, batch, microbatch, device.type,
                    compiled=cmodel, entry_mode=entry_mode)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    out = {
        "median_ms": float(np.percentile(times, 50)),
        "p10_ms": float(np.percentile(times, 10)),
        "p90_ms": float(np.percentile(times, 90)),
        "times_ms": [float(x) for x in times],
        "input_tok_s": float(global_batch * cfg.T / (np.percentile(times, 50) / 1000.0)),
    }
    if device.type == "cuda":
        out["peak_alloc_GiB"] = float(torch.cuda.max_memory_allocated(device) / 2**30)
        out["peak_reserved_GiB"] = float(torch.cuda.max_memory_reserved(device) / 2**30)
    del model, opt, batch
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    # Harness self-check on CPU (timing not meaningful; validates plumbing).
    from opt.model_ref import ArmAConfig, NativeReadStage1ArmA
    cfg = ArmAConfig(T=16, V=64, D=16, N=64, H=2, L=1, HIDDEN=32)
    dev = torch.device("cpu")
    r = bench_variant(lambda: NativeReadStage1ArmA(cfg, dev), cfg, dev, 2, 2,
                      warmups=1, samples=2)
    print({"harness": "OK", "median_ms": r["median_ms"], "input_tok_s": r["input_tok_s"]})
