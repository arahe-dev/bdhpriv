"""Akasha V0.5 baseline inference profiling harness (measurement only).

Modes:
  FULL_RECOMPUTE_EAGER   source-faithful prefix recomputation per token
  RECURRENT_EAGER        verified token-major recurrence (batched engine)
  RECURRENT_COMPILED     same recurrence under torch.compile

No Triton/CUDA kernels, no DeltaLog, no manual fusion. This module only
measures. It writes the ``results/akasha/v05_*.json`` artifacts.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from akasha.bench.correctness import canonical_init
from akasha.bench.recurrent_batched import (
    CompiledDecodeModule,
    batched_step,
    create_batched_state,
)
from akasha.config import REPO_SOURCE_COMMIT, REPO_SOURCE_COMMIT_FULL, TRAINER_SHA256
from akasha.models.arma.config import production_config
from akasha.models.arma.ops import weights_fingerprint
from akasha.models.arma.reference_full import full_forward

PRECISION_MODES = ("P0_FP32", "P1_BF16_AUTOCAST_FP32_STATE")
BATCH_SIZES = (1, 2, 4, 8, 16, 32)
PREFILL_LENGTHS = (32, 128, 512, 1024, 2048)
DECODE_PROMPT = 32
DECODE_WARMUP = 10
DECODE_STEPS = 50
CROSSOVER_LENGTHS = (32, 128, 512, 1024, 2048)


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("V0.5 profiling requires CUDA")
    return torch.device("cuda")


def _autocast_ctx(precision: str):
    if precision == "P1_BF16_AUTOCAST_FP32_STATE":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _sync() -> None:
    torch.cuda.synchronize()


def _sm_clock_mhz():
    try:
        return int(torch.cuda.clock_rate())
    except Exception:  # noqa: BLE001
        return None


def _clock_snapshot() -> Dict[str, Any]:
    return {"sm_clock_mhz": _sm_clock_mhz()}


def _event_device_us(entry) -> float:
    value = getattr(entry, "device_time_total", None)
    if value is None:
        value = getattr(entry, "cuda_time_total", 0.0)
    return float(value)


def _percentiles(times_ms: List[float]) -> Dict[str, float]:
    ordered = sorted(times_ms)
    n = len(ordered)
    return {
        "median_ms": statistics.median(ordered),
        "p10_ms": ordered[max(0, int(0.10 * (n - 1)))],
        "p90_ms": ordered[min(n - 1, int(0.90 * (n - 1)))],
        "mean_ms": statistics.fmean(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples": n,
    }


def _gpu_env() -> Dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    env = {
        "gpu": props.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_bytes": int(props.total_memory),
        "total_memory_gib": props.total_memory / 2**30,
        "sm_count": int(props.multi_processor_count),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "v0_commit": REPO_SOURCE_COMMIT,
        "v0_commit_full": REPO_SOURCE_COMMIT_FULL,
        "trainer_sha256": TRAINER_SHA256,
        "note": (
            "Local laptop RTX 4060 (sm_89, 8 GiB). Not the production G4 "
            "RTX PRO 6000 Blackwell Server Edition; do not conflate."
        ),
    }
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        env["nvidia_smi"] = out.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        env["nvidia_smi"] = f"unavailable: {type(exc).__name__}"
    return env


def _measure_events(
    loop, warmup: int, steps: int, min_warmup_seconds: float = 2.0
) -> List[float]:
    """Warm up for at least ``min_warmup_seconds`` of real work.

    The local laptop RTX 4060 downclocks to as low as ~210-825 MHz on
    short bursts and only boosts (~2.6 GHz) under sustained load, so a
    fixed warmup count is not enough for stable measurements.
    """
    with torch.inference_mode():
        started = time.perf_counter()
        iterations = 0
        while iterations < warmup or (time.perf_counter() - started) < min_warmup_seconds:
            loop()
            iterations += 1
        _sync()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        for index in range(steps):
            starts[index].record()
            loop()
            ends[index].record()
        _sync()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _new_generator(seed: int = 20260916) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _prompt(batch: int, length: int, cfg, device) -> torch.Tensor:
    gen = _new_generator(1000 + batch + length)
    return torch.randint(0, cfg.V, (batch, length), generator=gen).to(device)


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


def _decode_full_recompute(
    weights, cfg, precision: str, batch: int, device,
    prompt_len: int, warmup: int, steps: int,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "mode": "FULL_RECOMPUTE_EAGER",
        "precision": precision,
        "batch": batch,
        "prompt_len": prompt_len,
    }
    try:
        prefix = _prompt(batch, prompt_len, cfg, device)
        positions = torch.arange(prompt_len, device=device).unsqueeze(0).expand(batch, -1).contiguous()
        segments = torch.zeros_like(prefix)
        with torch.inference_mode(), _autocast_ctx(precision):
            for _ in range(warmup):
                logits = full_forward(
                    weights, cfg, prefix, positions=positions,
                    segment_ids=segments, scan_block=None,
                ).logits[:, -1]
                next_tokens = torch.argmax(logits, dim=-1)
                prefix = torch.cat([prefix, next_tokens.unsqueeze(1)], dim=1)
                positions = torch.cat([positions, positions[:, -1:] + 1], dim=1)
                segments = torch.zeros_like(prefix)
            _sync()
            torch.cuda.reset_peak_memory_stats()
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
            for index in range(steps):
                starts[index].record()
                logits = full_forward(
                    weights, cfg, prefix, positions=positions,
                    segment_ids=segments, scan_block=None,
                ).logits[:, -1]
                next_tokens = torch.argmax(logits, dim=-1)
                prefix = torch.cat([prefix, next_tokens.unsqueeze(1)], dim=1)
                positions = torch.cat([positions, positions[:, -1:] + 1], dim=1)
                segments = torch.zeros_like(prefix)
                ends[index].record()
            _sync()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        result.update(_percentiles(times))
        result["tokens_per_s"] = batch / (result["median_ms"] / 1000.0)
        result["single_session_ms"] = result["median_ms"]
        result["final_context"] = int(prefix.shape[1])
        result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        result["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    except torch.cuda.OutOfMemoryError as exc:  # noqa: PERF203
        result["status"] = "OOM"
        result["error"] = str(exc)[:200]
        torch.cuda.empty_cache()
    return result


def _decode_recurrent_eager(
    weights, cfg, precision: str, batch: int, device,
    prompt_len: int, warmup: int, steps: int,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "mode": "RECURRENT_EAGER",
        "precision": precision,
        "batch": batch,
        "prompt_len": prompt_len,
    }
    state = create_batched_state(weights, cfg, batch, device=device)
    prompt = _prompt(batch, prompt_len, cfg, device)
    counters = {"steps": 0}
    with torch.inference_mode(), _autocast_ctx(precision):
        tokens = prompt[:, 0]
        for index in range(prompt_len):
            logits = batched_step(weights, cfg, state, prompt[:, index])
        tokens = torch.argmax(logits, dim=-1)

        def loop():
            nonlocal tokens
            logits = batched_step(weights, cfg, state, tokens)
            tokens = torch.argmax(logits, dim=-1)
            counters["steps"] += 1

        torch.cuda.reset_peak_memory_stats()
        result["clock_before"] = _clock_snapshot()
        times = _measure_events(loop, warmup, steps)
        result["clock_after"] = _clock_snapshot()
    result.update(_percentiles(times))
    result["tokens_per_s"] = batch / (result["median_ms"] / 1000.0)
    result["single_session_ms"] = result["median_ms"] / batch
    result["single_session_tokens_per_s"] = 1000.0 / result["single_session_ms"]
    result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    result["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    return result


def _decode_recurrent_compiled(
    weights, cfg, precision: str, batch: int, device,
    prompt_len: int, warmup: int, steps: int, layout: str = "buffers",
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "mode": "RECURRENT_COMPILED",
        "precision": precision,
        "batch": batch,
        "prompt_len": prompt_len,
        "layout": layout,
    }
    module = CompiledDecodeModule(weights, cfg, batch, layout=layout, device=device)
    prompt = _prompt(batch, prompt_len, cfg, device)
    with torch.inference_mode(), _autocast_ctx(precision):
        logits = None
        for index in range(prompt_len):
            logits = module.decode(prompt[:, index])
        tokens = torch.argmax(logits, dim=-1)

        # equivalence gate before timing
        torch._dynamo.reset()
        compiled = torch.compile(module.decode, mode="default")
        t0 = time.perf_counter()
        compiled_out = compiled(tokens)
        _sync()
        result["compile_seconds"] = time.perf_counter() - t0
        eager_out = None
        # compare on a fresh module at the same state
        torch.cuda.synchronize()
        verify_module = CompiledDecodeModule(
            weights, cfg, batch, layout=layout, device=device
        )
        for index in range(prompt_len):
            verify_module.decode(prompt[:, index])
        eager_out = verify_module.decode(tokens)
        result["compiled_vs_eager_max_abs"] = float(
            (compiled_out - eager_out).abs().max().item()
        )
        del verify_module
        torch.cuda.empty_cache()

        def loop():
            nonlocal tokens
            logits = compiled(tokens)
            tokens = torch.argmax(logits, dim=-1)

        torch.cuda.reset_peak_memory_stats()
        result["clock_before"] = _clock_snapshot()
        times = _measure_events(loop, warmup, steps)
        result["clock_after"] = _clock_snapshot()
    result.update(_percentiles(times))
    result["tokens_per_s"] = batch / (result["median_ms"] / 1000.0)
    result["single_session_ms"] = result["median_ms"] / batch
    result["single_session_tokens_per_s"] = 1000.0 / result["single_session_ms"]
    result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    result["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    result["graph_breaks"] = _graph_break_count(module)
    return result


def _graph_break_count(module) -> Optional[int]:
    try:
        fresh = CompiledDecodeModule(
            module.arm_weights, module.cfg, int(module.C.shape[0]),
            layout=module.layout, device=module.C.device,
        )
        tokens = torch.randint(
            0, module.cfg.V, (module.C.shape[0],), device=module.C.device
        )
        torch._dynamo.reset()
        explained = torch._dynamo.explain(fresh.decode)(tokens)
        return int(getattr(explained, "graph_break_count", -1))
    except Exception as exc:  # noqa: BLE001
        return None


def run_decode(out_dir: Path, device, warmup: int, steps: int,
               batches=BATCH_SIZES, modes=("FULL_RECOMPUTE_EAGER",
                                           "RECURRENT_EAGER",
                                           "RECURRENT_COMPILED"),
               precisions=PRECISION_MODES) -> Dict[str, Any]:
    cfg = production_config()
    weights = canonical_init(cfg).to(device)
    payload: Dict[str, Any] = {
        "format": "akasha_v05_decode_v1",
        "env": _gpu_env(),
        "protocol": {
            "prompt_len": DECODE_PROMPT,
            "warmup_steps": warmup,
            "measured_steps": steps,
            "sampling": "argmax per step (included in step timing)",
            "modes": list(modes),
            "precisions": list(precisions),
            "weights_fingerprint_synthetic": weights_fingerprint(weights),
        },
        "results": [],
    }
    for precision in precisions:
        for mode in modes:
            for batch in batches:
                print(f"[decode] {mode} {precision} B={batch}", flush=True)
                if mode == "FULL_RECOMPUTE_EAGER":
                    entry = _decode_full_recompute(
                        weights, cfg, precision, batch, device,
                        DECODE_PROMPT, warmup, steps,
                    )
                elif mode == "RECURRENT_EAGER":
                    entry = _decode_recurrent_eager(
                        weights, cfg, precision, batch, device,
                        DECODE_PROMPT, warmup, steps,
                    )
                else:
                    entry = _decode_recurrent_compiled(
                        weights, cfg, precision, batch, device,
                        DECODE_PROMPT, warmup, steps,
                    )
                entry["status"] = entry.get("status", "OK")
                print(
                    f"    -> {entry['status']} median={entry.get('median_ms', float('nan')):.3f} ms "
                    f"tok/s={entry.get('tokens_per_s', float('nan')):.1f}",
                    flush=True,
                )
                payload["results"].append(entry)
                (out_dir / "v05_decode.json").write_text(
                    json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
                )
                torch.cuda.empty_cache()
                gc.collect()
    return payload


# ---------------------------------------------------------------------------
# components
# ---------------------------------------------------------------------------


def _time_op(fn, steps: int = 30, min_warmup_seconds: float = 1.5) -> Dict[str, Any]:
    with torch.inference_mode():
        started = time.perf_counter()
        while time.perf_counter() - started < min_warmup_seconds:
            fn()
        _sync()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        for index in range(steps):
            starts[index].record()
            fn()
            ends[index].record()
        _sync()
    times = [a.elapsed_time(b) for a, b in zip(starts, ends)]
    return {
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "clock_mhz": _sm_clock_mhz(),
    }


def _component_microbench(weights, cfg, batch: int, device) -> Dict[str, Any]:
    """Isolated equivalent microbenchmarks at production shapes.

    Structural ablations are not used; every entry times the exact op from the
    verified step, in isolation, with CUDA events and sustained warmup.
    """
    from akasha.models.arma.ops import (
        apply_rope,
        cached_pair_freq,
        layer_norm,
        rope_phase,
        wide_projection_matrix,
    )

    S = torch.zeros(batch, cfg.L, cfg.H, cfg.K, cfg.D, device=device)
    C = torch.zeros(batch, cfg.L, cfg.D, device=device)
    v = torch.randn(batch, cfg.D, device=device)
    x = torch.randn(batch, cfg.H, cfg.K, device=device)
    q = torch.randn(batch, cfg.H, cfg.K, device=device)
    a = torch.randn(batch, cfg.H, cfg.D, device=device)
    u = torch.randn(batch, cfg.H, cfg.K, device=device)
    base = torch.randn(batch, cfg.D, device=device)
    den = torch.ones(batch, 1, device=device)
    tokens = torch.randint(0, cfg.V, (batch,), device=device)
    position = torch.zeros(batch, dtype=torch.long, device=device)
    w_wide = wide_projection_matrix(weights, cfg)
    freq = cached_pair_freq(cfg, device)
    cos, sin = rope_phase(position.unsqueeze(1), freq)
    cos = cos.reshape(batch, 1, -1)
    sin = sin.reshape(batch, 1, -1)
    rho = torch.sigmoid(weights.coord_alpha)

    def op_embed_ln():
        return layer_norm(weights.embedding[tokens])

    def op_wide_x():
        return torch.relu(torch.matmul(v, w_wide))

    def op_rope():
        return apply_rope(x, cos, sin)

    def op_state_read():
        for level in range(cfg.L):
            torch.matmul(q.unsqueeze(-2), S[:, level]).squeeze(-2)

    def op_state_write():
        for level in range(cfg.L):
            for head in range(cfg.H):
                S[:, level, head].add_(q[:, head].unsqueeze(-1) * v.unsqueeze(1))

    def op_attn_ln_y():
        a_ln = layer_norm(a)
        return torch.relu(torch.einsum("bhd,hdk->bhk", a_ln, weights.decoder_y))

    def op_collapse():
        flat = (x * u).reshape(batch, cfg.N)
        return layer_norm(flat @ weights.encoder)

    def op_coordinator():
        z = v @ weights.coord_Wc + weights.coord_bc
        c = C[:, 0] / den - z
        g = 1.0 + rho.to(v.dtype) * torch.tanh(c)
        C[:, 0].add_(z)
        return g

    def op_writer():
        return torch.relu((base) @ weights.writer_W1) @ weights.writer_W2

    def op_ln_residual():
        return layer_norm(v + base)

    def op_readout():
        return v @ weights.readout

    def op_sample():
        return torch.argmax(v @ weights.readout, dim=-1)

    ops = {
        "embed_ln": op_embed_ln,
        "wide_x_projection": op_wide_x,
        "rope": op_rope,
        "state_read_all_levels": op_state_read,
        "state_write_all_levels": op_state_write,
        "attn_ln_y": op_attn_ln_y,
        "neuronal_collapse": op_collapse,
        "coordinator_gate": op_coordinator,
        "writer": op_writer,
        "ln_residual": op_ln_residual,
        "readout": op_readout,
        "sampling_argmax": op_sample,
    }
    results = {}
    for name, fn in ops.items():
        results[name] = _time_op(fn)
    total = sum(entry["median_ms"] for entry in results.values())
    for entry in results.values():
        entry["share_of_isolated_sum"] = entry["median_ms"] / total if total else None
    return {"batch": batch, "ops": results, "isolated_sum_ms": total}


def run_components(out_dir: Path, device, steps: int = 20) -> Dict[str, Any]:
    from collections import defaultdict

    from torch.profiler import ProfilerActivity, profile

    cfg = production_config()
    weights = canonical_init(cfg).to(device)
    payload: Dict[str, Any] = {
        "format": "akasha_v05_components_v1",
        "env": _gpu_env(),
        "protocol": {
            "batch": 1,
            "prompt_len": DECODE_PROMPT,
            "profiled_steps": steps,
            "method": (
                "torch.profiler ranges (eager, deduplicated raw events), kernel "
                "tables (compiled) and isolated equivalent microbenchmarks; no ablations"
            ),
            "overhead_note": (
                "profiled absolute times are inflated by CUPTI; isolated "
                "microbenchmarks use CUDA events and are the absolute reference"
            ),
        },
    }
    payload["clock_start"] = _clock_snapshot()

    state = create_batched_state(weights, cfg, 1, device=device)
    prompt = _prompt(1, DECODE_PROMPT, cfg, device)
    token = prompt[:, 0]
    with torch.inference_mode():
        for index in range(DECODE_PROMPT):
            batched_step(weights, cfg, state, prompt[:, index])
        token = torch.argmax(batched_step(weights, cfg, state, token), dim=-1)
        started = time.perf_counter()
        while time.perf_counter() - started < 2.0:
            token = torch.argmax(batched_step(weights, cfg, state, token), dim=-1)
        _sync()
        payload["clock_before_ranges"] = _clock_snapshot()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(steps):
                token = torch.argmax(
                    batched_step(weights, cfg, state, token, ranges=True), dim=-1
                )
            _sync()
    phase_us = defaultdict(float)
    phase_calls = defaultdict(int)
    for event in prof.events():
        if event.name.startswith("phase:") and int(event.scope) == 0:
            name = event.name.split(":", 1)[1]
            phase_us[name] += float(event.device_time)
            phase_calls[name] += 1
    ranges = [
        {
            "name": name,
            "cuda_time_total_us": total,
            "cuda_time_avg_us": total / steps,
            "calls": phase_calls[name],
            "calls_per_step": phase_calls[name] / steps,
        }
        for name, total in phase_us.items()
    ]
    ranges.sort(key=lambda item: -item["cuda_time_total_us"])
    total_cuda_us = sum(item["cuda_time_total_us"] for item in ranges)
    for item in ranges:
        item["share_of_profiled"] = (
            item["cuda_time_total_us"] / total_cuda_us if total_cuda_us else 0.0
        )
    payload["eager_ranges"] = {
        "steps": steps,
        "total_profiled_cuda_us": total_cuda_us,
        "per_step_cuda_us": total_cuda_us / steps,
        "phases": ranges,
        "clock_after": _clock_snapshot(),
    }

    module = CompiledDecodeModule(weights, cfg, 1, layout="buffers", device=device)
    with torch.inference_mode():
        for index in range(DECODE_PROMPT):
            module.decode(prompt[:, index])
        torch._dynamo.reset()
        compiled = torch.compile(module.decode, mode="default")
        compiled(prompt[:, -1])
        tokens = torch.argmax(module.decode(token), dim=-1)
        started = time.perf_counter()
        while time.perf_counter() - started < 2.0:
            tokens = torch.argmax(compiled(tokens), dim=-1)
        _sync()
        payload["clock_before_compiled_kernels"] = _clock_snapshot()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof2:
            for _ in range(steps):
                tokens = torch.argmax(compiled(tokens), dim=-1)
            _sync()
    kernels = defaultdict(lambda: [0.0, 0])
    for event in prof2.events():
        if event.name.startswith("phase:"):
            continue
        if int(event.scope) != 0:
            continue
        if float(event.device_time) <= 0:
            continue
        if "cudaLaunch" in event.name or "Occupancy" in event.name or "Memset" in event.name:
            continue
        if "cudaDeviceSynchronize" in event.name or "cuLaunch" in event.name:
            continue
        if event.name.startswith("aten::") or "Compiled" in event.name:
            continue
        if event.name.startswith("Torch-Compiled") or event.name.startswith("## "):
            continue
        kernels[event.name][0] += float(event.device_time)
        kernels[event.name][1] += 1
    kernel_rows = [
        {
            "kernel": name,
            "cuda_time_total_us": values[0],
            "cuda_time_avg_us": values[0] / steps,
            "calls_per_step": values[1] / steps,
        }
        for name, values in kernels.items()
    ]
    kernel_rows.sort(key=lambda item: -item["cuda_time_total_us"])
    payload["compiled_kernels"] = {
        "steps": steps,
        "kernel_time_per_step_us": sum(
            item["cuda_time_total_us"] for item in kernel_rows
        ) / steps,
        "kernel_calls_per_step": sum(item["calls_per_step"] for item in kernel_rows),
        "top_kernels": kernel_rows[:20],
        "clock_after": _clock_snapshot(),
        "note": (
            "kernel-level attribution only; record_function ranges inside the "
            "inductor graph are not reliably preserved"
        ),
    }

    payload["component_microbench"] = {
        "method": (
            "isolated wall-clock (CUDA events) timing of the exact ops from the "
            "verified step at production shapes; no structural ablations"
        ),
        "batch": {
            str(batch): _component_microbench(weights, cfg, batch, device)
            for batch in (1, 8)
        },
    }

    layout_rows = []
    for layout in ("stacked", "grouped", "buffers"):
        for batch in (1, 8):
            print(f"[components] layout={layout} B={batch}", flush=True)
            entry = _measure_layout(weights, cfg, layout, batch, device)
            layout_rows.append(entry)
    payload["layout_experiment"] = {
        "method": (
            "same mathematics, three torch.compile state layouts; compile time "
            "and steady-state ms/step measured on the local GPU"
        ),
        "results": layout_rows,
    }
    payload["clock_end"] = _clock_snapshot()
    (out_dir / "v05_components.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def _measure_layout(weights, cfg, layout: str, batch: int, device,
                    warmup: int = 5, steps: int = 20,
                    passes: int = 2) -> Dict[str, Any]:
    """Two passes; the min across passes resists clock drift.

    The local laptop GPU downclocks between bursts and can drop below 1 GHz
    after heavy neighbors; repeated passes with sustained warmup make the
    per-layout comparison as fair as unlocked clocks allow.
    """
    module = CompiledDecodeModule(weights, cfg, batch, layout=layout, device=device)
    tokens = _prompt(batch, 1, cfg, device)[:, 0]
    entry: Dict[str, Any] = {"layout": layout, "batch": batch, "passes": []}
    try:
        with torch.inference_mode():
            torch._dynamo.reset()
            compiled = torch.compile(module.decode, mode="default")
            t0 = time.perf_counter()
            compiled(tokens)
            _sync()
            entry["compile_seconds"] = time.perf_counter() - t0
            best = None
            for _ in range(passes):
                started = time.perf_counter()
                iterations = 0
                while iterations < warmup or (time.perf_counter() - started) < 2.0:
                    compiled(tokens)
                    iterations += 1
                _sync()
                clock_before = _clock_snapshot()
                starts = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
                ends = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
                for index in range(steps):
                    starts[index].record()
                    compiled(tokens)
                    ends[index].record()
                _sync()
                times = [a.elapsed_time(b) for a, b in zip(starts, ends)]
                pass_entry = {
                    "median_ms": statistics.median(times),
                    "min_ms": min(times),
                    "clock_before": clock_before,
                    "clock_after": _clock_snapshot(),
                }
                entry["passes"].append(pass_entry)
                if best is None or pass_entry["min_ms"] < best["min_ms"]:
                    best = pass_entry
            entry["ms_per_step"] = best["median_ms"]
            entry["min_ms_per_step"] = best["min_ms"]
            entry["best_pass_clock"] = best["clock_before"]
            entry["status"] = "OK"
    except torch.cuda.OutOfMemoryError as exc:  # noqa: PERF203
        entry["status"] = "OOM"
        entry["error"] = str(exc)[:200]
        torch.cuda.empty_cache()
    return entry


# ---------------------------------------------------------------------------
# state microbenchmark and traffic
# ---------------------------------------------------------------------------


def _copy_bandwidth(device, bytes_per_tensor: int = 512 * 2**20) -> Dict[str, float]:
    src = torch.empty(bytes_per_tensor // 4, dtype=torch.float32, device=device)
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    _sync()
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        dst.copy_(src)
    _sync()
    elapsed = time.perf_counter() - t0
    moved = 2 * bytes_per_tensor * reps
    return {"gbs": moved / elapsed / 1e9, "reps": reps}


def run_state_microbench(out_dir: Path, device, steps: int = 30) -> Dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    cfg = production_config()
    payload: Dict[str, Any] = {
        "format": "akasha_v05_state_microbench_v1",
        "env": _gpu_env(),
        "shapes": {"L": cfg.L, "H": cfg.H, "K": cfg.K, "D": cfg.D},
        "logical_bytes_per_token": {
            "state_read": cfg.L * cfg.H * cfg.K * cfg.D * 4,
            "state_write": cfg.L * cfg.H * cfg.K * cfg.D * 4,
            "coordinator_read_write": 2 * cfg.L * cfg.D * 4,
            "total": 2 * cfg.L * cfg.H * cfg.K * cfg.D * 4 + 2 * cfg.L * cfg.D * 4,
        },
        "device_copy_bandwidth": _copy_bandwidth(device),
        "variants": [],
    }
    for batch in (1, 4, 16):
        S = torch.zeros(batch, cfg.L, cfg.H, cfg.K, cfg.D, device=device)
        q = torch.randn(batch, cfg.H, cfg.K, device=device)
        v = torch.randn(batch, cfg.D, device=device)
        state_bytes = S.numel() * 4
        logical = 2 * state_bytes

        def ref_separate():
            for level in range(cfg.L):
                for head in range(cfg.H):
                    torch.matmul(q[:, head].unsqueeze(-2), S[:, level, head]).squeeze(-2)
                    S[:, level, head].add_(
                        q[:, head].unsqueeze(-1) * v.unsqueeze(1)
                    )

        def batched_read_add():
            for level in range(cfg.L):
                torch.matmul(q.unsqueeze(-2), S[:, level]).squeeze(-2)
                S[:, level].add_(q.unsqueeze(-1) * v.unsqueeze(1).unsqueeze(1))

        def addr_b1():
            for level in range(cfg.L):
                for head in range(cfg.H):
                    torch.matmul(q[0, head], S[0, level, head])
                    S[0, level, head].addr_(q[0, head], v[0])

        variants = [
            ("per_head_read_add", ref_separate),
            ("batched_level_read_add", batched_read_add),
        ]
        if batch == 1:
            variants.append(("addr_reference_b1", addr_b1))

        for name, fn in variants:
            with torch.inference_mode():
                started = time.perf_counter()
                while time.perf_counter() - started < 1.5:
                    fn()
                _sync()
                starts = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
                ends = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
                for index in range(steps):
                    starts[index].record()
                    fn()
                    ends[index].record()
                _sync()
                wall_ms = statistics.median(
                    s.elapsed_time(e) for s, e in zip(starts, ends)
                )
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    for _ in range(3):
                        fn()
                    _sync()
            entry = {
                "variant": name,
                "batch": batch,
                "status": "OK",
                "state_mib": state_bytes / 2**20,
                "logical_gib_per_iter": logical / 2**30,
                "ms_per_iter": wall_ms,
                "effective_logical_gbs": logical / (wall_ms / 1000.0) / 1e9,
                "kernel_calls_per_iter": sum(e.count for e in prof.key_averages()) / 3,
                "kernel_count_note": "kernel calls counted under profiler (3 iterations)",
                "clock_mhz": _sm_clock_mhz(),
            }
            payload["variants"].append(entry)
            print(
                f"[state] B={batch} {name}: {entry['ms_per_iter']:.3f} ms "
                f"{entry['effective_logical_gbs']:.1f} GB/s",
                flush=True,
            )
        del S
        torch.cuda.empty_cache()
    (out_dir / "v05_state_microbench.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def run_state_traffic(out_dir: Path, device, steps: int = 20) -> Dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    cfg = production_config()
    weights = canonical_init(cfg).to(device)
    state = create_batched_state(weights, cfg, 1, device=device)
    tokens = _prompt(1, 1, cfg, device)[:, 0]
    logical = 2 * cfg.L * cfg.H * cfg.K * cfg.D * 4 + 2 * cfg.L * cfg.D * 4
    with torch.inference_mode():
        for _ in range(5):
            tokens = torch.argmax(batched_step(weights, cfg, state, tokens), dim=-1)
        _sync()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(steps):
                tokens = torch.argmax(
                    batched_step(weights, cfg, state, tokens, ranges=True), dim=-1
                )
            _sync()
    phase_totals: Dict[str, float] = {}
    kernel_us = 0.0
    for entry in prof.key_averages():
        if entry.key.startswith("phase:"):
            phase_totals[entry.key.split(":", 1)[1]] = _event_device_us(entry)
        elif _event_device_us(entry) > 0:
            kernel_us += _event_device_us(entry)
    phase_sum_us = sum(phase_totals.values())
    state_us = phase_totals.get("state_read", 0.0) + phase_totals.get("state_write", 0.0)
    payload = {
        "format": "akasha_v05_state_traffic_v1",
        "env": _gpu_env(),
        "protocol": {
            "batch": 1,
            "steps": steps,
            "ranges": "state_read + state_write",
            "overhead_note": (
                "profiled absolute times can be inflated by CUPTI; shares and "
                "phase sums are the signal, wall-clock decode comes from v05_decode"
            ),
        },
        "logical_bytes_per_token": logical,
        "measured": {
            "state_path_us_per_token": state_us / steps,
            "state_read_us_per_token": phase_totals.get("state_read", 0.0) / steps,
            "state_write_us_per_token": phase_totals.get("state_write", 0.0) / steps,
            "phase_sum_us_per_token": phase_sum_us / steps,
            "kernel_sum_us_per_token": kernel_us / steps,
            "state_effective_logical_gbs": (logical / (state_us / steps / 1e6) / 1e9) if state_us > 0 else None,
            "state_share_of_phase_sum": (state_us / phase_sum_us) if phase_sum_us else None,
        },
        "analytical": {
            "fp32_minimum_bytes_per_token": 2 * 8 * 4 * 4096 * 256 * 4,
            "hardware_counters": "unavailable (Windows; no Nsight Compute access)",
        },
    }
    (out_dir / "v05_state_traffic.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


# ---------------------------------------------------------------------------
# prefill
# ---------------------------------------------------------------------------


def run_prefill(out_dir: Path, device, lengths=PREFILL_LENGTHS) -> Dict[str, Any]:
    cfg = production_config()
    weights = canonical_init(cfg).to(device)
    payload: Dict[str, Any] = {
        "format": "akasha_v05_prefill_v1",
        "env": _gpu_env(),
        "results": [],
    }
    payload["clock_start"] = _clock_snapshot()
    for length in lengths:
        ids = _prompt(1, length, cfg, device)[:, 0]
        positions = torch.arange(length, device=device)
        segments = torch.zeros(length, dtype=torch.long, device=device)
        entry: Dict[str, Any] = {"length": length}
        for method, scan_block in (("full_dense", None), ("chunkwise_1024", 1024)):
            try:
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    full_forward(
                        weights, cfg, ids, positions=positions,
                        segment_ids=segments, scan_block=scan_block,
                    )
                    _sync()
                    t0 = time.perf_counter()
                    full_forward(
                        weights, cfg, ids, positions=positions,
                        segment_ids=segments, scan_block=scan_block,
                    )
                    _sync()
                    elapsed = time.perf_counter() - t0
                entry[method] = {
                    "seconds": elapsed,
                    "tokens_per_s": length / elapsed,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                }
            except torch.cuda.OutOfMemoryError as exc:  # noqa: PERF203
                entry[method] = {"status": "OOM", "error": str(exc)[:200]}
                torch.cuda.empty_cache()
        state = create_batched_state(weights, cfg, 1, device=device)
        with torch.inference_mode():
            started = time.perf_counter()
            while time.perf_counter() - started < 1.0:
                batched_step(weights, cfg, state, ids[:1])
            _sync()
            entry["clock_before_recurrent"] = _clock_snapshot()
            t0 = time.perf_counter()
            for index in range(length):
                batched_step(weights, cfg, state, ids[index : index + 1])
            _sync()
            elapsed = time.perf_counter() - t0
        entry["recurrent_sequential"] = {
            "seconds": elapsed,
            "tokens_per_s": length / elapsed,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "clock_after": _clock_snapshot(),
        }
        print(
            f"[prefill] len={length}: "
            f"dense={entry.get('full_dense', {}).get('tokens_per_s', float('nan')):.0f} tok/s "
            f"chunked={entry.get('chunkwise_1024', {}).get('tokens_per_s', float('nan')):.0f} tok/s "
            f"recurrent={entry['recurrent_sequential']['tokens_per_s']:.0f} tok/s",
            flush=True,
        )
        payload["results"].append(entry)
        torch.cuda.empty_cache()
        gc.collect()
    (out_dir / "v05_prefill.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


# ---------------------------------------------------------------------------
# memory scaling
# ---------------------------------------------------------------------------


def run_memory(out_dir: Path, device, batches=BATCH_SIZES) -> Dict[str, Any]:
    cfg = production_config()
    weights = canonical_init(cfg).to(device)
    weights_bytes = sum(t.numel() * t.element_size() for t in weights.tensors().values())
    session_bytes = cfg.L * cfg.H * cfg.K * cfg.D * 4 + cfg.L * cfg.D * 4
    payload: Dict[str, Any] = {
        "format": "akasha_v05_memory_v1",
        "env": _gpu_env(),
        "model_weights_bytes": weights_bytes,
        "per_session_state_bytes": session_bytes,
        "per_session_state_mib": session_bytes / 2**20,
        "sessions": [],
    }
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    for batch in batches:
        torch.cuda.empty_cache()
        gc.collect()
        entry: Dict[str, Any] = {"active_sessions": batch}
        try:
            torch.cuda.reset_peak_memory_stats()
            state = create_batched_state(weights, cfg, batch, device=device)
            tokens = _prompt(batch, 1, cfg, device)[:, 0]
            with torch.inference_mode():
                batched_step(weights, cfg, state, tokens)
            _sync()
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            entry.update(
                {
                    "status": "OK",
                    "allocated_bytes": int(allocated),
                    "reserved_bytes": int(reserved),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    "state_plus_weights_expected_bytes": int(
                        weights_bytes + batch * session_bytes
                    ),
                    "allocated_minus_baseline_bytes": int(
                        allocated - baseline_allocated
                    ),
                    "deviation_from_expected_bytes": int(
                        allocated - baseline_allocated - batch * session_bytes
                    ),
                    "deviation_per_session_bytes": int(
                        (allocated - baseline_allocated - batch * session_bytes) / batch
                    ),
                }
            )
            del state
        except torch.cuda.OutOfMemoryError as exc:  # noqa: PERF203
            entry["status"] = "OOM"
            entry["error"] = str(exc)[:200]
            torch.cuda.empty_cache()
        print(
            f"[memory] sessions={batch}: {entry['status']} "
            f"alloc={entry.get('allocated_bytes', 0)/2**20:.1f} MiB "
            f"peak_reserved={entry.get('peak_reserved_bytes', 0)/2**20:.1f} MiB",
            flush=True,
        )
        payload["sessions"].append(entry)
        torch.cuda.empty_cache()
        gc.collect()
    ok_sessions = [e["active_sessions"] for e in payload["sessions"] if e["status"] == "OK"]
    payload["max_ok_sessions"] = max(ok_sessions) if ok_sessions else 0
    params = torch.cuda.get_device_properties(0)
    payload["max_local_concurrency_observed"] = payload["max_ok_sessions"]
    payload["memory_headroom_at_max_bytes"] = (
        int(params.total_memory) - max(
            e["peak_reserved_bytes"] for e in payload["sessions"] if e["status"] == "OK"
        )
        if ok_sessions
        else None
    )
    (out_dir / "v05_memory.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


# ---------------------------------------------------------------------------
# recompute crossover
# ---------------------------------------------------------------------------


def run_crossover(out_dir: Path, device, lengths=CROSSOVER_LENGTHS,
                  generate: int = 8) -> Dict[str, Any]:
    cfg = production_config()
    weights = canonical_init(cfg).to(device)
    payload: Dict[str, Any] = {
        "format": "akasha_v05_recompute_crossover_v1",
        "env": _gpu_env(),
        "protocol": {
            "batch": 1,
            "generated_tokens_measured": generate,
            "recurrent_prefill": "sequential batched_step over the prompt",
            "full_recompute": "dense full_forward on the current prefix per generated token",
        },
        "points": [],
    }
    for length in lengths:
        ids = _prompt(1, length, cfg, device)[:, 0]
        entry: Dict[str, Any] = {"prefill_tokens": length}

        state = create_batched_state(weights, cfg, 1, device=device)
        with torch.inference_mode():
            for index in range(length):
                batched_step(weights, cfg, state, ids[index:index + 1])
            _sync()
            token = torch.argmax(
                batched_step(weights, cfg, state, ids[-1:]), dim=-1
            )
            entry["clock_before"] = _clock_snapshot()
            times = _measure_events(
                lambda: batched_step(weights, cfg, state, token),
                warmup=2,
                steps=generate,
                min_warmup_seconds=1.5,
            )
            entry["clock_after"] = _clock_snapshot()
        entry["recurrent"] = {
            "prefill_ms": None,
            "decode_ms_per_token": statistics.median(times),
        }
        state_full = create_batched_state(weights, cfg, 1, device=device)
        with torch.inference_mode():
            t0 = time.perf_counter()
            for index in range(length):
                batched_step(weights, cfg, state_full, ids[index:index + 1])
            _sync()
            entry["recurrent"]["prefill_ms"] = (time.perf_counter() - t0) * 1000
        del state, state_full
        torch.cuda.empty_cache()

        prefix = ids.unsqueeze(0)
        positions = torch.arange(length, device=device).unsqueeze(0)
        segments = torch.zeros_like(prefix)
        with torch.inference_mode():
            times = []
            for step in range(generate + 2):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                logits = full_forward(
                    weights, cfg, prefix, positions=positions,
                    segment_ids=segments, scan_block=None,
                ).logits[:, -1]
                token = torch.argmax(logits, dim=-1)
                prefix = torch.cat([prefix, token.unsqueeze(1)], dim=1)
                positions = torch.cat([positions, positions[:, -1:] + 1], dim=1)
                segments = torch.zeros_like(prefix)
                end.record()
                _sync()
                times.append(start.elapsed_time(end))
        measured = times[2:]
        entry["full_recompute"] = {
            "decode_ms_per_token": statistics.median(measured),
            "all_ms": times,
            "final_context": int(prefix.shape[1]),
        }
        entry["recurrent_wins"] = (
            entry["recurrent"]["decode_ms_per_token"]
            < entry["full_recompute"]["decode_ms_per_token"]
        )
        print(
            f"[crossover] P={length}: recurrent={entry['recurrent']['decode_ms_per_token']:.2f} ms "
            f"recompute={entry['full_recompute']['decode_ms_per_token']:.2f} ms "
            f"recurrent_wins={entry['recurrent_wins']}",
            flush=True,
        )
        payload["points"].append(entry)
        torch.cuda.empty_cache()
        gc.collect()
    winners = [p["prefill_tokens"] for p in payload["points"] if p["recurrent_wins"]]
    payload["recurrent_crossover_prefill_tokens"] = min(winners) if winners else None
    payload["recompute_wins_below"] = [
        p["prefill_tokens"] for p in payload["points"] if not p["recurrent_wins"]
    ]
    (out_dir / "v05_recompute_crossover.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/akasha")
    parser.add_argument(
        "--sections",
        default="decode,components,state,prefill,memory,crossover",
    )
    parser.add_argument("--steps", type=int, default=DECODE_STEPS)
    parser.add_argument("--warmup", type=int, default=DECODE_WARMUP)
    parser.add_argument("--batches", default="1,2,4,8,16,32")
    parser.add_argument(
        "--modes",
        default=",".join(
            ["FULL_RECOMPUTE_EAGER", "RECURRENT_EAGER", "RECURRENT_COMPILED"]
        ),
    )
    parser.add_argument("--precisions", default=",".join(PRECISION_MODES))
    args = parser.parse_args(argv)

    device = _device()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sections = {s.strip() for s in args.sections.split(",") if s.strip()}
    batches = tuple(int(x) for x in args.batches.split(",") if x.strip())
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    precisions = tuple(p.strip() for p in args.precisions.split(",") if p.strip())

    print(json.dumps(_gpu_env(), indent=2), flush=True)
    if "decode" in sections:
        run_decode(
            out_dir,
            device,
            args.warmup,
            args.steps,
            batches,
            modes=modes,
            precisions=precisions,
        )
    if "components" in sections:
        run_components(out_dir, device)
    if "state" in sections:
        run_state_microbench(out_dir, device)
        run_state_traffic(out_dir, device)
    if "prefill" in sections:
        run_prefill(out_dir, device)
    if "memory" in sections:
        run_memory(out_dir, device, batches)
    if "crossover" in sections:
        run_crossover(out_dir, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())

