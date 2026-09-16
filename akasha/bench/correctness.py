"""Akasha V0 correctness bench.

Produces the frozen V0 artifacts:

    results/akasha/v0_correctness.json
    results/akasha/v0_manifest.json
    results/akasha/v0_memory_contract.json
    results/akasha/v0_blockers.json

The bench uses source-compatible synthetic weights (the trainer's
``canonical_init`` at seed 1337); it does not require the trained checkpoint
or the tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from akasha.checkpoint.loader import load_state, save_state
from akasha.config import (
    FP32_ATOL,
    FP32_RTOL,
    FP64_ATOL,
    REPO_SOURCE_COMMIT,
    REPO_SOURCE_COMMIT_FULL,
    REQUIRED_V0_LENGTHS,
    TOKENIZER_EXPECTED_SHA256,
    TOKENIZER_IDENTITY,
    TRAINER_PATH,
    TRAINER_SHA256,
)
from akasha.models.arma.config import ArmAConfig, production_config, tiny_config
from akasha.models.arma.manifest import manifest_dict, manifest_fingerprint
from akasha.models.arma.ops import (
    ArmAWeights,
    weights_fingerprint,
)
from akasha.models.arma.reference_full import full_forward
from akasha.models.arma.reference_recurrent import (
    create_state,
    logits_from_state,
    prefill_all_logits,
    prefill_tokens,
    step,
)
from akasha.models.arma.state import ContextPolicy, state_element_counts
from akasha.tokenizer.adapter import ArmATokenizerAdapter

HARDWARE = {
    "gpu": "RTX PRO 6000 Blackwell Server Edition",
    "memory_gb": 96,
    "bandwidth_gbs": 1597,
    "note": "Server Edition bandwidth; not the 1792 GB/s Workstation Edition",
}


def canonical_init(cfg: ArmAConfig) -> ArmAWeights:
    """Exact replica of the trainer's ``canonical_init`` (RNG order included)."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(cfg.SEED)

    def rnd(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32) * cfg.INIT_STD

    return ArmAWeights(
        embedding=rnd((cfg.V, cfg.D)),
        encoder=rnd((cfg.N, cfg.D)),
        decoder_x=rnd((cfg.H, cfg.D, cfg.K)),
        decoder_y=rnd((cfg.H, cfg.D, cfg.K)),
        readout=rnd((cfg.D, cfg.V)),
        coord_Wc=rnd((cfg.D, cfg.D)),
        coord_bc=torch.zeros(cfg.D, dtype=torch.float32),
        coord_alpha=torch.zeros((), dtype=torch.float32),
        writer_W1=rnd((cfg.D, cfg.HIDDEN)),
        writer_W2=rnd((cfg.HIDDEN, cfg.D)),
    )


def _segments(t: int, cuts: Sequence[int]) -> torch.Tensor:
    seg = torch.zeros(t, dtype=torch.long)
    for index, cut in enumerate(cuts):
        seg[cut:] = index + 1
    return seg


def attention_algebra(cfg: ArmAConfig, dtype=torch.float64, seed: int = 1) -> Dict[str, Any]:
    gen = torch.Generator().manual_seed(seed)
    t = cfg.T
    q = torch.randn(t, cfg.H, cfg.K, generator=gen, dtype=dtype)
    v = torch.randn(t, cfg.D, generator=gen, dtype=dtype)
    seg = _segments(t, [max(1, t // 3), max(2, 2 * t // 3)])

    dense = torch.zeros(t, cfg.H, cfg.D, dtype=dtype)
    state = torch.zeros(cfg.H, cfg.K, cfg.D, dtype=dtype)
    prev = None
    for i in range(t):
        s = int(seg[i])
        if prev is not None and s != prev:
            state = torch.zeros_like(state)
        prev = s
        dense[i] = torch.einsum("hk,hkd->hd", q[i], state)
        state = state + torch.einsum("hk,d->hkd", q[i], v[i])

    reference = torch.zeros(t, cfg.H, cfg.D, dtype=dtype)
    for i in range(t):
        for j in range(i):
            if int(seg[j]) == int(seg[i]):
                score = torch.einsum("hk,hk->h", q[i], q[j])
                reference[i] = reference[i] + score.unsqueeze(-1) * v[j]
    diff = (dense - reference).abs().max().item()
    return {
        "name": "recurrent_read_then_write_vs_strict_past_dense_attention",
        "max_abs_error": diff,
        "pass": diff <= FP64_ATOL,
    }


def coordinator_algebra(cfg: ArmAConfig, dtype=torch.float64, seed: int = 2) -> Dict[str, Any]:
    gen = torch.Generator().manual_seed(seed)
    t = cfg.T
    z = torch.randn(t, cfg.D, generator=gen, dtype=dtype)
    seg = _segments(t, [max(1, t // 4), max(2, t // 2), max(3, 3 * t // 4)])

    dense = torch.zeros(t, cfg.D, dtype=dtype)
    for i in range(t):
        prev = torch.zeros(cfg.D, dtype=dtype)
        count = 0
        for j in range(i):
            if int(seg[j]) == int(seg[i]):
                prev = prev + z[j]
                count += 1
        dense[i] = prev / max(count, 1) - z[i]

    recurrent = torch.zeros(t, cfg.D, dtype=dtype)
    acc = torch.zeros(cfg.D, dtype=dtype)
    count = 0
    prev_seg = None
    for i in range(t):
        s = int(seg[i])
        if prev_seg is not None and s != prev_seg:
            acc = torch.zeros_like(acc)
            count = 0
        prev_seg = s
        recurrent[i] = acc / max(count, 1) - z[i]
        acc = acc + z[i]
        count += 1
    diff = (dense - recurrent).abs().max().item()
    return {
        "name": "recurrent_coordinator_prefix_vs_dense_masked_mean",
        "max_abs_error": diff,
        "pass": diff <= FP64_ATOL,
    }


def token_major_gates(cfg: Optional[ArmAConfig] = None, dtype=torch.float64) -> Dict[str, Any]:
    cfg = cfg or tiny_config()
    weights = canonical_init(cfg).to(dtype=dtype)
    gen = torch.Generator().manual_seed(3)
    t = cfg.T
    ids = torch.randint(0, cfg.V, (t,), generator=gen)
    seg = _segments(t, [max(1, t // 3), max(2, 2 * t // 3)])
    positions = torch.arange(t)

    full = full_forward(
        weights, cfg, ids, positions=positions, segment_ids=seg, collect_debug=True
    )
    state = create_state(weights, cfg)
    rec_logits, rec_hidden = prefill_all_logits(
        weights, cfg, state, ids, positions=positions, segment_ids=seg,
        return_hidden=True,
    )
    logit_err = (full.logits[0] - rec_logits).abs().max().item()
    hidden_err = (full.debug[cfg.L - 1]["v"][0] - rec_hidden).abs().max().item()
    chunked = full_forward(
        weights, cfg, ids, positions=positions, segment_ids=seg, scan_block=4
    )
    chunk_err = (full.logits[0] - chunked.logits[0]).abs().max().item()
    return {
        "name": "token_major_vs_level_major",
        "config": cfg.to_dict(),
        "logits_max_abs_error": logit_err,
        "hidden_max_abs_error": hidden_err,
        "chunked_scan_max_abs_error": chunk_err,
        "pass": max(logit_err, hidden_err, chunk_err) <= FP64_ATOL,
    }


def parity_case(
    weights: ArmAWeights,
    cfg: ArmAConfig,
    ids: torch.Tensor,
    positions: torch.Tensor,
    segment_ids: torch.Tensor,
    label: str,
    scan_block: Optional[int] = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    full = full_forward(
        weights,
        cfg,
        ids,
        positions=positions,
        segment_ids=segment_ids,
        scan_block=scan_block,
    )
    t_full = time.perf_counter() - t0
    t0 = time.perf_counter()
    state = create_state(weights, cfg)
    rec = prefill_all_logits(
        weights, cfg, state, ids, positions=positions, segment_ids=segment_ids
    )
    t_rec = time.perf_counter() - t0

    a = full.logits[0].float()
    b = rec.float()
    diff = (a - b).abs()
    tolerance = FP32_ATOL + FP32_RTOL * a.abs()
    rel = diff / a.abs().clamp_min(1e-3)
    argmax_agree = (a.argmax(dim=-1) == b.argmax(dim=-1)).float().mean().item()
    case = {
        "label": label,
        "length": int(ids.numel()),
        "segments": int(segment_ids.unique().numel()),
        "max_abs_error": diff.max().item(),
        "mean_abs_error": diff.mean().item(),
        "max_rel_error_floor_1e-3": rel.max().item(),
        "argmax_agreement": argmax_agree,
        "tolerance_violations": int((diff > tolerance).sum().item()),
        "allclose_atol_rtol": bool((diff <= tolerance).all().item()),
        "full_seconds": t_full,
        "recurrent_seconds": t_rec,
    }
    return case


def pattern_case(
    weights: ArmAWeights,
    cfg: ArmAConfig,
    length: int,
    pattern: str,
    scan_block: Optional[int] = None,
) -> Dict[str, Any]:
    gen = torch.Generator().manual_seed(1000 + length)
    if pattern == "random":
        ids = torch.randint(0, cfg.V, (length,), generator=gen)
        seg = torch.zeros(length, dtype=torch.long)
    elif pattern == "repeated":
        ids = (torch.arange(length) % 3).to(torch.long)
        seg = torch.zeros(length, dtype=torch.long)
    elif pattern == "segments":
        ids = torch.randint(0, cfg.V, (length,), generator=gen)
        cuts = sorted({max(1, length // 4), max(2, length // 2), max(3, 3 * length // 4)})
        seg = _segments(length, cuts)
    else:
        raise ValueError(f"unknown pattern {pattern!r}")
    positions = torch.arange(length, dtype=torch.long)
    return parity_case(
        weights, cfg, ids, positions, seg, f"{pattern}_len{length}",
        scan_block=scan_block,
    )


def fp32_parity(
    lengths: Sequence[int] = REQUIRED_V0_LENGTHS,
    patterns: Sequence[str] = ("random", "repeated", "segments"),
    cfg: Optional[ArmAConfig] = None,
    scan_block: Optional[int] = None,
    progress: bool = True,
) -> Dict[str, Any]:
    cfg = cfg or production_config()
    weights = canonical_init(cfg)
    cases: List[Dict[str, Any]] = []
    for length in lengths:
        for pattern in patterns:
            case = pattern_case(weights, cfg, int(length), pattern, scan_block=scan_block)
            cases.append(case)
            if progress:
                print(
                    f"[parity] len={case['length']:>5} {pattern:<9} "
                    f"max_abs={case['max_abs_error']:.3e} "
                    f"argmax={case['argmax_agreement']:.4f} "
                    f"violations={case['tolerance_violations']}",
                    flush=True,
                )
    max_abs = max(c["max_abs_error"] for c in cases)
    cases_2048 = [c for c in cases if c["length"] == 2048]
    by_length = {}
    for case in cases:
        by_length.setdefault(str(case["length"]), []).append(
            {
                "pattern": case["label"].split("_")[0],
                "max_abs_error": case["max_abs_error"],
                "mean_abs_error": case["mean_abs_error"],
                "argmax_agreement": case["argmax_agreement"],
                "tolerance_violations": case["tolerance_violations"],
            }
        )
    return {
        "config": cfg.to_dict(),
        "weights_fingerprint": weights_fingerprint(weights),
        "scan_block": scan_block,
        "atol": FP32_ATOL,
        "rtol": FP32_RTOL,
        "cases": cases,
        "by_length": by_length,
        "max_abs_error": max_abs,
        "min_argmax_agreement": min(c["argmax_agreement"] for c in cases),
        "total_tolerance_violations": sum(c["tolerance_violations"] for c in cases),
        "allclose_all_cases": all(c["allclose_atol_rtol"] for c in cases),
        "length_2048_pass": bool(cases_2048)
        and all(c["allclose_atol_rtol"] for c in cases_2048),
    }


def memory_contract() -> Dict[str, Any]:
    cfg = production_config()
    counts = state_element_counts(cfg)
    bytes_per_element = 4
    s_bytes = counts["S"] * bytes_per_element
    c_bytes = counts["C"] * bytes_per_element
    total = s_bytes + c_bytes
    mib = total / 2**20
    traffic = 2 * total
    ceiling = HARDWARE["bandwidth_gbs"] * 1e9 / traffic
    return {
        "config": cfg.to_dict(),
        "elements": counts,
        "bytes_fp32": {"S": s_bytes, "C": c_bytes, "total": total},
        "mib": {
            "S": s_bytes / 2**20,
            "C": c_bytes / 2**20,
            "total": mib,
        },
        "assertions": {
            "S_elements": counts["S"] == 8 * 4 * 4096 * 256,
            "C_elements": counts["C"] == 8 * 256,
            "total_mib": abs(mib - 128.0078125) < 1e-9,
        },
        "hardware": HARDWARE,
        "analytical": {
            "state_traffic_bytes_per_token": traffic,
            "state_only_bandwidth_ceiling_token_steps_per_s": ceiling,
            "note": (
                "Analytical absolute minimum for eager fused FP32 recurrence "
                "(one read + one write of 128 MiB per token). NOT a benchmark "
                "result and NOT a throughput target."
            ),
        },
    }


def blockers_report(
    tokenizer_status: str = "BLOCKED_ARTIFACT_NOT_LOCAL",
    checkpoint_status: str = "BLOCKED",
) -> Dict[str, Any]:
    return {
        "TOKENIZER_STATUS": tokenizer_status,
        "TRAINED_CHECKPOINT_STATUS": checkpoint_status,
        "blockers": [
            {
                "id": "tokenizer_artifact",
                "status": tokenizer_status,
                "description": (
                    f"exact training tokenizer {TOKENIZER_IDENTITY} "
                    f"(sha256 {TOKENIZER_EXPECTED_SHA256}) is not present in the "
                    "repository; text prompt CLI stays disabled until the real "
                    "artifact is supplied and hashed"
                ),
                "validation": (
                    "py -3.12 -m akasha.checkpoint.validate_real_checkpoint --help; "
                    "tokenizer: akasha/tokenizer/adapter.py status()"
                ),
            },
            {
                "id": "trained_checkpoint",
                "status": checkpoint_status,
                "description": (
                    "final trained Arm-A checkpoint is not local; converter and "
                    "loader are validated against source-compatible generated "
                    "state_dicts only"
                ),
                "validation": (
                    "py -3.12 -m akasha.checkpoint.validate_real_checkpoint "
                    "--checkpoint <latest.pt> --out-dir results/akasha/real_package"
                ),
            },
        ],
    }


def run_gate_checks(parity: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compute the AKASHA_V0_REFERENCE_PASS gate flags."""
    cfg = tiny_config()
    weights = canonical_init(cfg)
    gates: Dict[str, Any] = {}

    attention = attention_algebra(cfg)
    coordinator = coordinator_algebra(cfg)
    schedule = token_major_gates(cfg)
    gates["float64_algebra"] = bool(attention["pass"])
    gates["coordinator_recurrence"] = bool(coordinator["pass"])
    gates["token_major_schedule"] = bool(schedule["pass"])

    state = create_state(weights, cfg)
    prefill_tokens(weights, cfg, state, [1, 2, 3, 4, 5])
    state.begin_segment(position=77)
    fresh = create_state(weights, cfg)
    fresh.position = 77
    reset_ok = (
        state.segment_count == 0
        and torch.equal(state.S, fresh.S)
        and torch.equal(state.C, fresh.C)
        and torch.equal(
            step(weights, cfg, state, 6), step(weights, cfg, fresh, 6)
        )
    )
    window_cfg = ArmAConfig(T=4, V=32, D=8, N=32, H=2, L=2, HIDDEN=12)
    window_weights = canonical_init(window_cfg)
    window_state = create_state(window_weights, window_cfg)
    prefill_tokens(
        window_weights, window_cfg, window_state, [1, 2, 3, 4],
        positions=[10, 11, 12, 13],
    )
    step(window_weights, window_cfg, window_state, 5)
    window_fresh = create_state(window_weights, window_cfg)
    window_fresh.position = 14
    step(window_weights, window_cfg, window_fresh, 5)
    reset_ok = reset_ok and torch.equal(window_state.S, window_fresh.S)
    gates["segment_reset"] = bool(reset_ok)

    state = create_state(weights, cfg)
    prefill_tokens(weights, cfg, state, [3, 1, 4, 1, 5, 9, 2])
    clone = state.clone()
    step(weights, cfg, state, 8)
    step(weights, cfg, clone, 8)
    replay = create_state(weights, cfg)
    prefill_tokens(weights, cfg, replay, [3, 1, 4, 1, 5, 9, 2])
    step(weights, cfg, replay, 8)
    gates["clone"] = bool(
        torch.equal(clone.S, replay.S)
        and torch.equal(state.S, replay.S)
        and torch.equal(clone.last_hidden, replay.last_hidden)
    )

    with tempfile.TemporaryDirectory(prefix="akasha_gate_") as tmp:
        state = create_state(weights, cfg)
        state.model_fingerprint = "gate"
        prefill_tokens(weights, cfg, state, [2, 7, 1, 8, 2, 8])
        save_state(Path(tmp) / "s", state)
        loaded, _ = load_state(Path(tmp) / "s", expected_fingerprint="gate")
        gates["serialization"] = bool(
            torch.equal(loaded.S, state.S)
            and torch.equal(loaded.C, state.C)
            and torch.equal(loaded.last_hidden, state.last_hidden)
            and loaded.position == state.position
            and loaded.segment_count == state.segment_count
            and loaded.context_policy == state.context_policy
            and loaded.model_fingerprint == "gate"
        )

    state = create_state(weights, cfg)
    logits = prefill_tokens(weights, cfg, state, [1, 2, 3]) 
    next_id = int(torch.argmax(logits).item())
    after = step(weights, cfg, state, next_id)
    expected = prefill_all_logits(
        weights, cfg, create_state(weights, cfg), [1, 2, 3, next_id]
    )[-1]
    gates["prefill_next_token_semantics"] = bool(torch.equal(after, expected))

    trainer_file = Path(__file__).resolve().parents[2] / TRAINER_PATH
    digest = hashlib.sha256(trainer_file.read_bytes()).hexdigest()
    gates["source_config_recorded"] = bool(digest == TRAINER_SHA256)

    tokenizer = ArmATokenizerAdapter()
    tokenizer_status = tokenizer.status().value
    gates["tokenizer_semantics_not_guessed"] = bool(
        tokenizer_status != "READY" or tokenizer.artifact_path is not None
    )

    if parity is not None:
        gates["fp32_full_vs_recurrent"] = bool(parity["allclose_all_cases"])
        gates["length_2048"] = bool(parity.get("length_2048_pass", False))
    else:
        gates["fp32_full_vs_recurrent"] = False
        gates["length_2048"] = False

    pass_keys = [
        "float64_algebra",
        "coordinator_recurrence",
        "token_major_schedule",
        "fp32_full_vs_recurrent",
        "length_2048",
        "segment_reset",
        "clone",
        "serialization",
        "prefill_next_token_semantics",
        "source_config_recorded",
        "tokenizer_semantics_not_guessed",
    ]
    gates["AKASHA_V0_REFERENCE_PASS"] = all(gates[key] for key in pass_keys)
    gates["tokenizer_status"] = tokenizer_status
    gates["fp64_errors"] = {
        "attention": attention["max_abs_error"],
        "coordinator": coordinator["max_abs_error"],
        "token_major_logits": schedule["logits_max_abs_error"],
        "chunked_scan": schedule["chunked_scan_max_abs_error"],
    }
    return gates


def apply_gate_checks(out_dir: str | Path) -> Dict[str, Any]:
    """Load the written correctness artifact, add gate flags, rewrite it."""
    out = Path(out_dir)
    payload = json.loads((out / "v0_correctness.json").read_text(encoding="utf-8"))
    gates = run_gate_checks(parity=payload.get("fp32_parity"))
    payload["gates"] = gates
    payload["AKASHA_V0_REFERENCE_PASS"] = gates["AKASHA_V0_REFERENCE_PASS"]
    (out / "v0_correctness.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return gates


def provisional_correctness(
    algebra: List[Dict[str, Any]],
    parity: Dict[str, Any],
) -> Dict[str, Any]:
    fp64_max = max(
        entry["max_abs_error"]
        for entry in algebra
        if "max_abs_error" in entry
    )
    return {
        "format": "akasha_v0_correctness_v1",
        "source_commit": REPO_SOURCE_COMMIT,
        "source_commit_full": REPO_SOURCE_COMMIT_FULL,
        "trainer_sha256": TRAINER_SHA256,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "fp64_errors": {
            "max_abs_error": fp64_max,
            "atol": FP64_ATOL,
            "pass": fp64_max <= FP64_ATOL,
        },
        "fp32_parity": parity,
    }


def run_all(
    out_dir: str | Path,
    lengths: Sequence[int] = REQUIRED_V0_LENGTHS,
    patterns: Sequence[str] = ("random", "repeated", "segments"),
    skip_fp64: bool = False,
    scan_block: Optional[int] = None,
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    algebra = []
    if not skip_fp64:
        algebra = [
            attention_algebra(tiny_config()),
            coordinator_algebra(tiny_config()),
            token_major_gates(tiny_config()),
        ]
    parity = fp32_parity(lengths=lengths, patterns=patterns, scan_block=scan_block)
    gates = run_gate_checks(parity=parity)
    correctness = provisional_correctness(algebra, parity)
    correctness["gates"] = gates
    correctness["AKASHA_V0_REFERENCE_PASS"] = gates["AKASHA_V0_REFERENCE_PASS"]

    manifest = manifest_dict(production_config())
    manifest["weights_fingerprint_synthetic"] = parity["weights_fingerprint"]
    manifest["manifest_fingerprint"] = manifest_fingerprint()
    manifest["torch"] = torch.__version__

    memory = memory_contract()
    blockers = blockers_report(
        tokenizer_status=gates["tokenizer_status"],
        checkpoint_status="BLOCKED",
    )

    (out / "v0_correctness.json").write_text(
        json.dumps(correctness, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out / "v0_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out / "v0_memory_contract.json").write_text(
        json.dumps(memory, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out / "v0_blockers.json").write_text(
        json.dumps(blockers, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "correctness": correctness,
        "manifest": manifest,
        "memory": memory,
        "blockers": blockers,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/akasha")
    parser.add_argument(
        "--lengths",
        default=",".join(str(x) for x in REQUIRED_V0_LENGTHS),
        help="comma-separated lengths",
    )
    parser.add_argument("--patterns", default="random,repeated,segments")
    parser.add_argument("--scan-block", type=int, default=None)
    parser.add_argument("--skip-fp64", action="store_true")
    args = parser.parse_args(argv)
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    patterns = [p for p in args.patterns.split(",") if p.strip()]
    result = run_all(
        args.out_dir,
        lengths=lengths,
        patterns=patterns,
        skip_fp64=args.skip_fp64,
        scan_block=args.scan_block,
    )
    parity = result["correctness"]["fp32_parity"]
    print(
        json.dumps(
            {
                "max_abs_error": parity["max_abs_error"],
                "min_argmax_agreement": parity["min_argmax_agreement"],
                "total_tolerance_violations": parity["total_tolerance_violations"],
                "length_2048_pass": parity["length_2048_pass"],
                "allclose_all_cases": parity["allclose_all_cases"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if parity["allclose_all_cases"] else 2


if __name__ == "__main__":
    sys.exit(main())
