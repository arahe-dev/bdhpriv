"""G4 hero-claim validation harness (Arm-A sparse systems claims).

ONE reusable implementation behind the single copy-paste Colab cell
(`campaigns/G4_HERO_CLAIM_CELL.md`). It prepares, runs and verdicts the G4
validation of the two currently testable hero claims:

  HERO 1  sparse Arm-A top1 (M8/Ke512, fixed cyclic window, G=128, exact
          capacity, compiled production stack) reaches >=350k packed tok/s
          at global batch 64 x T2048 on G4.
  HERO 2  learned top2 hard routing (M8/Ke512, learned cyclic window,
          straight-through proxy, inactive experts skipped) retains a large
          speedup vs dense while adding negligible overhead vs fixed top2.

Arm-B / Arm-C are not defined anywhere in this repository
(`campaigns/ARM_BC_SEMANTICS.md`) and are reported as
BLOCKED_BY_MISSING_CANONICAL_SPEC. Nothing here invents them.

Modes
-----
  plan      print the run plan and pre-registered thresholds (no GPU)
  env       environment gate only (writes environment JSON)
  gates     correctness gates: CPU tiny fp64/fp32 oracles + GPU
            production-shape equivalence, graph, determinism, boundary,
            inactive-expert audit, and the frozen-corpus contract probe
  bench1    ONE (arm, microbatch) full-update benchmark in THIS process;
            runall spawns one fresh process per config (the established
            anti-compile-contamination protocol)
  runall    orchestrate: gates -> randomized adaptive geometry DOE ->
            Hero-2 matched triple -> drift control -> OOM census -> report
  report    aggregate gates + bench JSONs into the final verdict table and
            machine-readable `g4_hero_claim_results.json`
  selftest  CPU/tiny local validation of this harness (no GPU, no corpus)

Every measured claim is same-session: dense reference, sparse top1,
fixed top2 and learned top2 all run in one G4 session over the same frozen
corpus sequence range, one fresh process and one fresh compile per config.

Evidence labels used in outputs: MEASURED (direct G4), INFERRED (2.5B
runtime projection), SPECULATIVE (anything about training/quality).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

T = 2048
VOCAB = 8192
GLOBAL_SEQUENCES = 64
ROUTE_GROUP = 128
MICROBATCH_ACCUM = {4: 16, 8: 8, 16: 4, 32: 2, 64: 1}
PROBE_MICROBATCHES = (8, 16, 32)
DENSE_PRODUCTION_MICROBATCH = 16
DENSE_OOM_PROBE_MICROBATCH = 32
HERO2_MICROBATCH = 16
DEFAULT_STEPS = 10
DEFAULT_WARMUPS = 3
DENSE_ANCHOR = {
    "packed_tok_s": 69183.0,
    "ms_per_update": 1894.56,
    "peak_allocated_GiB": 62.55,
    "source": "results/G4_CONFIRM.md (certified packed champion, frozen corpus)",
}
DENSE_ANCHOR_TOLERANCE = 0.10

THRESHOLDS = {
    "hero1_tok_s_pass": 350_000.0,
    "hero1_tok_s_hold": 245_000.0,
    "hero1_tok_s_preferred": 380_000.0,
    "hero1_tok_s_stretch": 400_000.0,
    "hero2_overhead_pass": 0.05,
    "hero2_overhead_hold": 0.15,
    "hero2_speedup_pass": 2.0,
    "hero2_speedup_hold": 1.5,
    "dense_anchor_rel_tol": DENSE_ANCHOR_TOLERANCE,
}

CORPUS_DEFAULT = Path(
    "/content/drive/Shareddrives/ICLR PHASE BDH/"
    "phase_bdh/corpus/stage2/frozen_5b_v1"
)

ARM_B_STATUS = "BLOCKED_BY_MISSING_CANONICAL_SPEC"
ARM_C_STATUS = "BLOCKED_BY_MISSING_CANONICAL_SPEC"

SEMANTIC_STATUS = {
    "dense": "canonical Arm-A executable source (opt3c_all production stack)",
    "sparse_top1": (
        "declared sparse Arm-A executor (M8/Ke512 top1 fixed cyclic window); "
        "executor exact relative to the declared sparse architecture; "
        "all-active expertized execution recovers dense Arm-A"
    ),
    "fixed_top2": (
        "declared sparse Arm-A executor (M8/Ke512 top2 fixed cyclic window); "
        "executor exact relative to the declared sparse architecture"
    ),
    "learned_top2": (
        "declared sparse Arm-A architecture change (M8/Ke512 learned cyclic "
        "window top2, straight-through proxy, forward exactly hard); "
        "not dense Arm-A; overflow drops reported"
    ),
}


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def jdump(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


def sha256_file(path: Path) -> str:
    """SHA-256 of text content with CRLF normalized to LF.

    The harness and its fingerprint contract must verify identically on
    Linux (Colab) and Windows checkouts of the same git blob.
    """
    data = Path(path).read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: Path, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(jdump(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# repo / contract / environment gates
# ---------------------------------------------------------------------------


def git_head(repo: Path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def git_is_ancestor(repo: Path, commit: str):
    try:
        subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                        commit, "HEAD"], check=True, capture_output=True)
        return True
    except Exception:
        return False


def verify_repo(repo: Path, contract_path):
    report = {
        "repo": str(repo),
        "git_head": git_head(repo),
        "contract_path": str(contract_path) if contract_path else None,
        "contract_found": False,
        "files": {},
        "missing": [],
        "mismatches": [],
        "harness_match": None,
    }
    contract = None
    if contract_path is not None and Path(contract_path).is_file():
        contract = load_json(Path(contract_path))
        report["contract_found"] = True
        report["contract_repo_commit"] = contract.get("repo_commit")
        report["contract_is_ancestor_of_head"] = (
            git_is_ancestor(repo, contract["repo_commit"])
            if contract.get("repo_commit") else None
        )
        for rel, want in (contract.get("fingerprints") or {}).items():
            path = repo / rel
            if not path.is_file():
                report["missing"].append(rel)
                continue
            got = sha256_file(path)
            ok = got == want
            report["files"][rel] = {"sha256": got, "match": ok}
            if not ok:
                report["mismatches"].append(rel)
        harness_want = ((contract.get("harness") or {}).get("sha256")
                        or contract.get("harness_sha256"))
        if harness_want:
            harness_path = repo / "scripts" / "g4_hero_claim_validation.py"
            got = sha256_file(harness_path) if harness_path.is_file() else None
            report["harness_match"] = got == harness_want
            report["harness_sha256"] = got
    report["pass"] = bool(
        report["contract_found"]
        and not report["missing"]
        and not report["mismatches"]
        and report.get("harness_match") is not False
    )
    return report, contract


def environment_report():
    import platform

    import torch

    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "bf16_supported": bool(torch.cuda.is_bf16_supported())
        if torch.cuda.is_available() else False,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        report.update({
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "vram_total_GiB": round(total / 2**30, 2),
            "vram_free_GiB": round(free / 2**30, 2),
            "sm_count": props.multi_processor_count,
        })
        report["expected_gpu"] = "RTX PRO 6000 Blackwell (sm_120)"
        report["expected_torch"] = "2.11.0+cu128"
        report["expected_cuda"] = "12.8"
        name_ok = "RTX PRO 6000 Blackwell" in report["gpu"]
        cap_ok = torch.cuda.get_device_capability(0) == (12, 0)
        report["ENVIRONMENT_PASS"] = bool(
            name_ok and cap_ok and report["bf16_supported"]
            and torch.__version__ == "2.11.0+cu128"
            and report["cuda"] == "12.8"
        )
        report["checks"] = {
            "gpu_name_expected": name_ok,
            "capability_sm120": cap_ok,
            "torch_version_expected": torch.__version__ == "2.11.0+cu128",
            "cuda_version_expected": report["cuda"] == "12.8",
        }
    else:
        report["ENVIRONMENT_PASS"] = False
        report["error"] = "CUDA unavailable"
    return report


# ---------------------------------------------------------------------------
# torch / model helpers
# ---------------------------------------------------------------------------


def configure_torch():
    import torch

    try:
        torch._dynamo.config.automatic_dynamic_shapes = False
    except Exception:
        pass
    return torch


def production_cfg():
    from opt.model_ref import ArmAConfig

    cfg = ArmAConfig()
    assert cfg.T == T and cfg.V == VOCAB
    return cfg


def build_dense(torch, cfg, device):
    from opt.model_opt import OptArmA
    from opt.model_ref import canonical_init, load_init

    model = OptArmA(
        cfg, device, scan_block=1024, use_checkpoint=False, coord="dense",
        single_scan="chunkwise", packed_update="branchfree", zero_carry=True,
        paper_layout="direct", cache_rope=True,
    ).to(device)
    load_init(model, canonical_init(cfg), device)
    model.train()
    return model


def dense_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def make_candidate(torch, arm, cfg, device, microbatch, source_state,
                   route_group=ROUTE_GROUP, experts=8, expert_width=512):
    """Build one candidate. Returns (model, route, info)."""
    from opt.routed_expert import (
        RoutedExpertArmA,
        build_route_tensors,
        group_route_table,
        window_route_sets,
    )

    if arm in ("sparse_top1", "fixed_top2"):
        top_r = 1 if arm == "sparse_top1" else 2
        model = RoutedExpertArmA(cfg, device, experts=experts,
                                 expert_width=expert_width,
                                 scan_block=1024).to(device)
        model.load_canonical(source_state)
        model.train()
        table = group_route_table(
            window_route_sets(experts, top_r, cyclic=True),
            cfg.T // route_group)
        route = build_route_tensors(table, route_group, microbatch, cfg.T,
                                    experts, device)
        used = float(route.sel_mask.sum().item())
        slots = float(route.sel_mask.numel())
        info = {
            "kind": f"fixed_cyclic_window_top{top_r}",
            "top_r": top_r,
            "route": "fixed",
            "route_group": route_group,
            "capacity_tokens_per_expert": int(route.sel_idx.shape[1]),
            "padding_fraction": 1.0 - used / max(1.0, slots),
            "active_experts": list(route.active),
            "overflow_tokens": 0,
            "ledger": model.parameter_ledger(top_r),
        }
        return model, route, info

    if arm == "learned_top2":
        from opt.learned_router import LearnedRoutedExpertArmA

        model = LearnedRoutedExpertArmA(
            cfg, device, experts=experts, expert_width=expert_width,
            scan_block=1024, top_r=2, route_group=route_group).to(device)
        with torch.no_grad():
            model.router.Wr.normal_(0.0, 1.0)  # bench_learned v2 protocol
        model.capacity_factor = 1.25          # bench_learned v2 protocol
        model.load_canonical(source_state)
        model.train()
        info = {
            "kind": "learned_cyclic_window_top2",
            "top_r": 2,
            "route": "learned",
            "route_group": route_group,
            "router_init_scale": 1.0,
            "capacity_factor": 1.25,
            "capacity_tokens_per_expert": int(
                model.capacity_tokens(microbatch, cfg.T)),
            "overflow_tokens": None,
            "ledger": model.parameter_ledger(),
        }
        return model, None, info

    raise ValueError(f"unknown arm: {arm}")


def static_entry(torch, arm, model):
    if arm == "learned_top2":
        return torch.compile(model.forward_learned, mode="default")
    if arm in ("sparse_top1", "fixed_top2"):
        return torch.compile(model.forward_route, mode="default")
    return torch.compile(model.forward_packed, mode="default")


def make_fwd(arm, entry, route):
    if arm in ("sparse_top1", "fixed_top2"):
        def fwd(b):
            return entry(b["x"], b["pos"], b["segpos"], b["full_mask"],
                         b["start"], route)
    else:
        def fwd(b):
            return entry(b["x"], b["pos"], b["segpos"], b["full_mask"],
                         b["start"])
    return fwd


def forward_loss(torch, arm, entry, route, batch, vocab=VOCAB):
    if arm in ("sparse_top1", "fixed_top2"):
        logits = entry(batch["x"], batch["pos"], batch["segpos"],
                       batch["full_mask"], batch["start"], route)
    else:
        logits = entry(batch["x"], batch["pos"], batch["segpos"],
                       batch["full_mask"], batch["start"])
    per = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab), batch["y"].reshape(-1), reduction="none")
    denom = int(batch["valid"].sum().item())
    return logits, per[batch["valid"].reshape(-1)].sum(
        dtype=torch.float32) / max(1, denom)


def packed_batch_from_starts(torch, cfg, device, starts_rows, seed=0):
    """Build a packed batch with explicit document starts (int64 seconds)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    b = len(starts_rows)
    t = cfg.T
    start = torch.tensor(starts_rows, dtype=torch.long, device=device)
    pos = (torch.arange(t, device=device).unsqueeze(0).expand(b, -1)
           - start).to(torch.int32)
    segpos = pos.clone()
    input_valid = start < t
    causal = torch.ones((t, t), dtype=torch.bool, device=device).tril(-1)
    same = start[:, :, None] == start[:, None, :]
    full_mask = (same & input_valid[:, :, None] & input_valid[:, None, :]
                 & causal.unsqueeze(0))
    x = torch.randint(0, cfg.V, (b, t), generator=gen).to(device)
    y = torch.randint(0, cfg.V, (b, t), generator=gen).to(device)
    return {
        "x": x, "y": y, "pos": pos, "segpos": segpos,
        "valid": input_valid.clone(), "full_mask": full_mask,
        "segment_start": start, "start": start,
    }


# ---------------------------------------------------------------------------
# frozen corpus
# ---------------------------------------------------------------------------


def corpus_open(repo: Path, root: Path, fast_verify: bool = True):
    sys.path.insert(0, str(repo))
    import training.arm_a_2p5b_trainer as trainer

    corpus = trainer.FrozenPackedCorpus(
        Path(root), trainer.PROD_CFG, verify_files=True,
        fast_verify=fast_verify)
    info = {
        "root": str(root),
        "corpus_id": corpus.contract.corpus_id,
        "sequences": corpus.total_sequences,
        "context_length": corpus.cfg.T,
        "fast_verify": fast_verify,
        "skipped": (
            "per-artifact SHA-256 of the 25x3 shard files (sizes, manifest "
            "digests and artifact_hashes.json digest were verified)"
            if fast_verify else "nothing"
        ),
    }
    return trainer, corpus, info


def corpus_probe(torch, trainer, corpus, start_sequence: int):
    """Fail-closed probe: one real packed update (64 rows) + contract stats."""
    batch = None
    cursor = None
    for cursor, batch in corpus.stream_batches(start_sequence, GLOBAL_SEQUENCES):
        break
    if batch is None:
        raise RuntimeError("corpus produced no batches")
    x = batch["x"]
    if tuple(x.shape) != (GLOBAL_SEQUENCES, T):
        raise RuntimeError(f"unexpected packed batch shape: {tuple(x.shape)}")
    valid = batch["valid"]
    start = batch["start"].long()
    input_valid = batch["input_valid"]
    n_pad = int((~input_valid).sum().item())
    causal_pairs = T * (T - 1) // 2 * GLOBAL_SEQUENCES
    same = (start[:, :, None] == start[:, None, :])
    strict = torch.ones((T, T), dtype=torch.bool).tril(-1)
    allowed = (same & input_valid[:, :, None] & input_valid[:, None, :]
               & strict.unsqueeze(0))
    same_doc_pairs = int(allowed.sum().item())
    doc_boundaries = int(
        ((start[:, 1:] != start[:, :-1])
         & input_valid[:, 1:] & input_valid[:, :-1]).sum().item())
    probe = {
        "start_sequence": int(cursor),
        "sequence_range": [int(cursor), int(cursor) + GLOBAL_SEQUENCES - 1],
        "input_tokens_per_update": GLOBAL_SEQUENCES * T,
        "valid_pairs_per_update": int(valid.sum().item()),
        "padding_positions_masked": n_pad,
        "document_boundaries": doc_boundaries,
        "strict_past_pairs": causal_pairs,
        "same_doc_pairs": same_doc_pairs,
        "cross_document_pairs_masked": causal_pairs - same_doc_pairs,
        "x_token_range": [int(x.min().item()), int(x.max().item())],
        "passed": bool(
            int(valid.sum().item()) > 0 and n_pad < GLOBAL_SEQUENCES * T
            and int(x.max().item()) < VOCAB and doc_boundaries > 0),
    }
    return probe, batch


# ---------------------------------------------------------------------------
# update glue (identical for every arm)
# ---------------------------------------------------------------------------


def prepare_gpu_batch(torch, slice_cpu, device, causal=None):
    x = slice_cpu["x"].to(device, dtype=torch.long, non_blocking=True)
    y = slice_cpu["y"].to(device, dtype=torch.long, non_blocking=True)
    pos = slice_cpu["pos"].to(device, dtype=torch.int32, non_blocking=True)
    segpos = slice_cpu["segpos"].to(device, dtype=torch.int32,
                                    non_blocking=True)
    valid = slice_cpu["valid"].to(device, dtype=torch.bool,
                                  non_blocking=True)
    start = slice_cpu["start"].to(device, dtype=torch.long,
                                  non_blocking=True)
    input_valid = slice_cpu["input_valid"].to(device, dtype=torch.bool,
                                              non_blocking=True)
    t = int(start.shape[-1])
    if causal is None or causal.shape[-1] != t:
        causal = torch.ones((t, t), dtype=torch.bool, device=device).tril(-1)
    full_mask = ((start[:, :, None] == start[:, None, :])
                 & input_valid[:, :, None] & input_valid[:, None, :]
                 & causal.unsqueeze(0))
    return {"x": x, "y": y, "pos": pos, "segpos": segpos, "valid": valid,
            "start": start, "full_mask": full_mask}


def lr_for_update(update_1based, cfg, global_tokens=GLOBAL_SEQUENCES * T,
                  warmup_tokens=10_000_000):
    return cfg.PEAK_LR * min(update_1based * global_tokens / warmup_tokens,
                             1.0)


def run_update(torch, model, fwd, optimizer, cfg, micro_cpu, device, denom,
               lr):
    """Full optimizer update: H2D + mask + fwd + CE + bwd + clip + step.

    Identical for dense and sparse arms. Mirrors
    `opt.model_ref.full_update` (entry_mode='packed') semantics, with the
    mask rebuilt on device from document starts (certified pipeline).
    """
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    enabled = device.type == "cuda"
    for slice_cpu in micro_cpu:
        batch = prepare_gpu_batch(torch, slice_cpu, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            cache_enabled=False, enabled=enabled):
            logits = fwd(batch)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), batch["y"].reshape(-1),
                reduction="none",
            )[batch["valid"].reshape(-1)].sum(
                dtype=torch.float32) / denom
        loss.backward()
        loss_total += float(loss.detach())
        del batch, logits, loss
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_NORM)
    optimizer.step()
    return loss_total


def summarise_times(times):
    ordered = sorted(times)
    return {
        "ms": [float(t) for t in times],
        "median_ms": float(statistics.median(times)),
        "p10_ms": float(ordered[max(0, int(0.1 * len(ordered)) - 1)]),
        "p90_ms": float(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]),
    }


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


class GateLog:
    def __init__(self):
        self.gates = []

    def add(self, name, passed, detail=None, hard=True, skipped=False):
        status = "SKIP" if skipped else ("PASS" if passed else "FAIL")
        gate = {"name": name, "status": status, "hard": bool(hard),
                "detail": detail or {}}
        self.gates.append(gate)
        print("GATE " + json.dumps(gate, sort_keys=True, default=str),
              flush=True)
        return gate

    def payload(self, extra=None):
        hard_failed = [g["name"] for g in self.gates
                       if g["hard"] and g["status"] == "FAIL"]
        payload = {
            "generated_utc": now_utc(),
            "gates": self.gates,
            "hard_failures": hard_failed,
            "GATES_PASS": not hard_failed,
        }
        if extra:
            payload.update(extra)
        return payload


def gate_cpu_tiny(log: GateLog):
    sys.path.insert(0, str(REPO))
    import torch  # noqa: F401

    from opt import test_learned_router as tlr
    from opt import test_routed_expert as tre

    checks = [
        ("cpu_tiny_all_active_equivalence",
         tre.test_all_active_equivalence, 1e-5),
        ("cpu_tiny_sparse_fp64_oracle",
         tre.test_sparse_semantics_oracle, 1e-8),
        ("cpu_tiny_document_boundary_no_leak",
         tre.test_document_boundary_no_leak, 0.0),
        ("cpu_tiny_route_builders", tre.test_route_builders, 0.0),
        ("cpu_tiny_learned_R0_constant_route", tlr.test_r0_constant, 1e-8),
        ("cpu_tiny_learned_R1_all_windows", tlr.test_r1_all_windows, 0.0),
        ("cpu_tiny_learned_R2R3_mixed_groups", tlr.test_r23_mixed_groups,
         1e-8),
        ("cpu_tiny_learned_R4_determinism", tlr.test_r4_determinism, 0.0),
        ("cpu_tiny_learned_R5_grad_routing", tlr.test_r5_grad_routing, 0.0),
        ("cpu_tiny_learned_R6_straight_through",
         tlr.test_r6_straight_through, 1e-12),
    ]
    for name, fn, tol in checks:
        try:
            detail = fn()
            log.add(name, True, {"result": detail, "tolerance": tol})
        except Exception as exc:  # noqa: BLE001
            log.add(name, False,
                    {"error": f"{type(exc).__name__}: {str(exc)[:400]}"})


def gate_cpu_tiny_unselected_expert_zero_grad(log: GateLog):
    """Fixed executor: experts absent from the route get exactly zero grads."""
    sys.path.insert(0, str(REPO))
    import torch
    import torch.nn.functional as F

    from opt.model_opt import OptArmA
    from opt.model_ref import ArmAConfig, canonical_init, load_init
    from opt.routed_expert import RoutedExpertArmA, build_route_tensors

    tiny = dict(T=16, V=32, D=8, N=16, H=2, L=2, HIDDEN=8)
    cfg = ArmAConfig(**tiny)
    device = torch.device("cpu")
    dense = OptArmA(cfg, device, scan_block=cfg.K, use_checkpoint=False,
                    coord="dense", single_scan="chunkwise",
                    packed_update="branchfree", zero_carry=True,
                    paper_layout="direct", cache_rope=True)
    load_init(dense, canonical_init(cfg), device)
    model = RoutedExpertArmA(cfg, device, experts=4,
                             expert_width=cfg.K // 4, scan_block=cfg.K)
    model.load_canonical(dense.state_dict())
    model.train()
    group = 4
    table = [(0,), (1,), (2,)]  # expert 3 never routed
    route = build_route_tensors(table, group, 1, cfg.T, 4, device)
    assert route.active == (0, 1, 2)
    gen = torch.Generator().manual_seed(11)
    x = torch.randint(0, cfg.V, (1, cfg.T), generator=gen)
    y = torch.randint(0, cfg.V, (1, cfg.T), generator=gen)
    start = torch.zeros((1, cfg.T), dtype=torch.long)
    pos = torch.arange(cfg.T).unsqueeze(0).to(torch.int32)
    segpos = pos.clone()
    strict = torch.ones((cfg.T, cfg.T), dtype=torch.bool).tril(-1)
    full_mask = strict.unsqueeze(0)
    model.zero_grad(set_to_none=True)
    logits = model.forward_route(x, pos, segpos, full_mask, start, route)
    F.cross_entropy(logits.reshape(-1, cfg.V), y.reshape(-1)).backward()
    inactive = 3
    grad_norms = {
        "DxE": float(model.DxE.grad[inactive].norm()),
        "DyE": float(model.DyE.grad[inactive].norm()),
        "EE": float(model.EE.grad[inactive].norm()),
    }
    active_norm = float(model.DxE.grad[0].norm())
    passed = all(v == 0.0 for v in grad_norms.values()) and active_norm > 0
    log.add("cpu_tiny_unselected_expert_zero_grad", passed,
            {"inactive_expert": inactive, "grad_norms": grad_norms,
             "active_expert0_DxE_grad_norm": active_norm})


def gate_corpus(log: GateLog, repo: Path, corpus_root: Path, start_sequence):
    import torch

    try:
        trainer, corpus, info = corpus_open(repo, corpus_root, fast_verify=True)
        probe, _ = corpus_probe(torch, trainer, corpus, start_sequence)
        log.add("corpus_contract_probe", bool(probe["passed"]),
                {"corpus": info, "probe": probe})
    except Exception as exc:  # noqa: BLE001
        log.add("corpus_contract_probe", False,
                {"error": f"{type(exc).__name__}: {str(exc)[:600]}",
                 "corpus_root": str(corpus_root)})


def _mapped_grads(torch, model, arm):
    """Gradient dict under canonical parameter names for both executors."""
    if arm == "dense":
        return {name: p.grad.detach().float().clone()
                for name, p in model.named_parameters()
                if p.grad is not None}
    mapping = {}
    m, h, d, ke = model.M, model.cfg.H, model.cfg.D, model.Ke
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if name == "DxE":
            mapping["decoder_x"] = (
                p.grad.reshape(m, d, h, ke).permute(2, 1, 0, 3)
                .reshape(h, d, m * ke).float().clone())
        elif name == "DyE":
            mapping["decoder_y"] = (
                p.grad.reshape(m, d, h, ke).permute(2, 1, 0, 3)
                .reshape(h, d, m * ke).float().clone())
        elif name == "EE":
            mapping["encoder"] = (
                p.grad.reshape(m, h, ke, d).permute(1, 0, 2, 3)
                .reshape(model.cfg.N, d).float().clone())
        elif name.startswith("router."):
            continue
        else:
            mapping[name] = p.grad.detach().float().clone()
    return mapping


def gate_gpu_production_equivalence(log: GateLog):
    """All-active expertized routed executor vs dense Arm-A at T=2048 on GPU."""
    import torch

    configure_torch()
    cfg = production_cfg()
    device = torch.device("cuda")
    layouts = [
        [0] * 1031 + [1031] * (T - 1031),       # boundary inside a block
        [0] * 2048,                              # single document
        [0] * 512 + [512] * 1536,                # boundary on a block edge
    ]
    starts = torch.tensor(layouts, dtype=torch.long, device=device)
    starts[0, 1550:] = T + 1                    # masked padding tail
    batch = packed_batch_from_starts(torch, cfg, device, starts.tolist(),
                                     seed=7)
    from opt.routed_expert import (
        RoutedExpertArmA,
        build_route_tensors,
    )

    dense = build_dense(torch, cfg, device)
    routed = RoutedExpertArmA(cfg, device, experts=8, expert_width=512,
                              scan_block=1024).to(device)
    routed.load_canonical(dense.state_dict())
    routed.train()
    all_active = build_route_tensors([(0, 1, 2, 3, 4, 5, 6, 7)], ROUTE_GROUP,
                                     3, T, 8, device)

    for bf16 in (False, True):
        name = f"gpu_prod_all_active_equiv_{'bf16' if bf16 else 'fp32'}_eager"
        worst = {"logit": 0.0, "loss": 0.0, "grad": 0.0}
        try:
            results = {}
            for tag, model, arm, route in (
                ("dense", dense, "dense", None),
                ("routed", routed, "sparse", all_active),
            ):
                model.zero_grad(set_to_none=True)
                enabled = bool(bf16)
                with torch.autocast(device_type="cuda",
                                    dtype=torch.bfloat16,
                                    cache_enabled=False, enabled=enabled):
                    if arm == "sparse":
                        logits = model.forward_route(
                            batch["x"], batch["pos"], batch["segpos"],
                            batch["full_mask"], batch["start"], route)
                    else:
                        logits = model.forward_packed(
                            batch["x"], batch["pos"], batch["segpos"],
                            batch["full_mask"], batch["start"])
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, cfg.V), batch["y"].reshape(-1),
                    )
                loss.backward()
                results[tag] = (logits.detach().float().clone(),
                                float(loss.detach()),
                                _mapped_grads(torch, model, arm))
            worst["logit"] = float(
                (results["routed"][0] - results["dense"][0]).abs().max())
            worst["loss"] = abs(results["routed"][1] - results["dense"][1])
            for key, grad in results["routed"][2].items():
                ref = results["dense"][2][key]
                worst["grad"] = max(worst["grad"],
                                    float((grad - ref).abs().max()))
            if bf16:
                ok = worst["logit"] <= 5e-2 and worst["loss"] <= 5e-2 \
                    and worst["grad"] <= 5e-2
            else:
                ok = worst["logit"] <= 1e-3 and worst["loss"] <= 1e-5 \
                    and worst["grad"] <= 1e-3
            log.add(name, ok, {"worst": worst, "layout_rows": len(layouts),
                               "T": T, "note": "all-active route recovery"})
        except Exception as exc:  # noqa: BLE001
            log.add(name, False,
                    {"error": f"{type(exc).__name__}: {str(exc)[:400]}"})
    del dense, routed
    gc.collect()
    torch.cuda.empty_cache()


def gate_gpu_graph_breaks(log: GateLog):
    import torch

    configure_torch()
    cfg = production_cfg()
    device = torch.device("cuda")
    batch = packed_batch_from_starts(
        torch, cfg, device, [[0] * 1031 + [1031] * (T - 1031)], seed=11)

    from opt.routed_expert import (
        RoutedExpertArmA,
        build_route_tensors,
    )

    dense = build_dense(torch, cfg, device)
    source_state = dense_state(dense)
    arms = {"dense": ("forward_packed", dense, None)}
    sparse = RoutedExpertArmA(cfg, device, experts=8, expert_width=512,
                              scan_block=1024).to(device)
    sparse.load_canonical(source_state)
    sparse.train()
    route = build_route_tensors([(0,)], ROUTE_GROUP, 1, T, 8, device)
    arms["sparse_top1"] = ("forward_route", sparse, route)
    fixed_model, fixed_route, _ = make_candidate(torch, "fixed_top2", cfg,
                                                 device, 1, source_state)
    arms["fixed_top2"] = ("forward_route", fixed_model, fixed_route)
    learned_model, _, _ = make_candidate(torch, "learned_top2", cfg, device,
                                         1, source_state)
    arms["learned_top2"] = ("forward_learned", learned_model, None)

    for name, (entry_name, model, route) in arms.items():
        try:
            torch._dynamo.reset()
            entry = getattr(model, entry_name)
            if route is not None:
                exp = torch._dynamo.explain(entry)(
                    batch["x"], batch["pos"], batch["segpos"],
                    batch["full_mask"], batch["start"], route)
            else:
                exp = torch._dynamo.explain(entry)(
                    batch["x"], batch["pos"], batch["segpos"],
                    batch["full_mask"], batch["start"])
            breaks = int(getattr(exp, "graph_break_count", -1))
            log.add(f"gpu_graph_breaks_{name}", breaks == 0,
                    {"graph_count": int(getattr(exp, "graph_count", -1)),
                     "graph_break_count": breaks,
                     "break_reasons": [str(r)[:200]
                                       for r in getattr(exp,
                                                        "break_reasons",
                                                        [])],
                     "shape": f"B1 T{T}"})
            torch._dynamo.reset()
        except Exception as exc:  # noqa: BLE001
            log.add(f"gpu_graph_breaks_{name}", False,
                    {"error": f"{type(exc).__name__}: {str(exc)[:400]}"})
    for model in (dense, sparse, fixed_model, learned_model):
        del model
    gc.collect()
    torch.cuda.empty_cache()


def gate_gpu_determinism_and_boundary(log: GateLog):
    import torch

    configure_torch()
    cfg = production_cfg()
    device = torch.device("cuda")
    source = build_dense(torch, cfg, device)
    source_state = dense_state(source)
    del source
    torch.cuda.empty_cache()

    from opt.routed_expert import (
        RoutedExpertArmA,
        build_route_tensors,
    )

    layout_a = [[0] * 1031 + [1031] * (T - 1031)]
    layout_b = [[0] * 1031 + [1031] * (T - 1031)]
    batch_a = packed_batch_from_starts(torch, cfg, device, layout_a, seed=21)
    batch_b = packed_batch_from_starts(torch, cfg, device, layout_b, seed=22)

    pairs = []
    dense = build_dense(torch, cfg, device)
    pairs.append(("dense", dense, None, "forward_packed"))
    sparse = RoutedExpertArmA(cfg, device, experts=8, expert_width=512,
                              scan_block=1024).to(device)
    sparse.load_canonical(source_state)
    sparse.train()
    route = build_route_tensors([(0,), (1,), (2,), (3,), (4,), (5,), (6,),
                                 (7,)], ROUTE_GROUP, 1, T, 8, device)
    pairs.append(("sparse_top1", sparse, route, "forward_route"))
    learned_model, _, _ = make_candidate(torch, "learned_top2", cfg, device,
                                         1, source_state)
    pairs.append(("learned_top2", learned_model, None, "forward_learned"))

    for name, model, route, entry_name in pairs:
        entry = getattr(model, entry_name)
        try:
            with torch.no_grad():
                def call(b):
                    if route is not None:
                        return entry(b["x"], b["pos"], b["segpos"],
                                     b["full_mask"], b["start"], route)
                    return entry(b["x"], b["pos"], b["segpos"],
                                 b["full_mask"], b["start"])
                out1 = call(batch_a).float()
                out2 = call(batch_a).float()
                repeat = float((out1 - out2).abs().max())
                other = call(batch_b).float()
                doc_a_diff = float((out1[:, :1031] - other[:, :1031])
                                   .abs().max())
                doc_b_diff = float((out1[:, 1031:] - other[:, 1031:])
                                   .abs().max())
            log.add(f"gpu_determinism_{name}", repeat == 0.0,
                    {"repeat_forward_max_abs_diff": repeat})
            log.add(f"gpu_boundary_no_leak_{name}",
                    doc_a_diff <= 1e-6 and doc_b_diff > 0.0,
                    {"doc_a_max_abs_diff_when_doc_b_changed": doc_a_diff,
                     "doc_b_max_abs_diff_when_doc_b_changed": doc_b_diff,
                     "boundary_column": 1031})
        except Exception as exc:  # noqa: BLE001
            log.add(f"gpu_packed_boundary_{name}", False,
                    {"error": f"{type(exc).__name__}: {str(exc)[:400]}"})
    for model in (dense, sparse, learned_model):
        del model
    gc.collect()
    torch.cuda.empty_cache()


def run_gates(repo: Path, corpus_root: Path, start_sequence: int, outdir: Path):
    log = GateLog()
    env = environment_report()
    log.add("environment", env.get("ENVIRONMENT_PASS", False), env)
    repo_report, _ = verify_repo(repo, repo / "results" /
                                 "g4_hero_claim_contract.json")
    log.add("repo_fingerprints", bool(repo_report["pass"]), repo_report)
    gate_cpu_tiny(log)
    gate_cpu_tiny_unselected_expert_zero_grad(log)
    import torch

    if torch.cuda.is_available():
        gate_gpu_production_equivalence(log)
        gate_gpu_graph_breaks(log)
        gate_gpu_determinism_and_boundary(log)
    else:
        log.add("gpu_gates", False, {"error": "CUDA unavailable"})
    if corpus_root is not None:
        gate_corpus(log, repo, corpus_root, start_sequence)
    payload = log.payload()
    save_json(outdir / "gates.json", payload)
    print("GATES_PASS=" + str(payload["GATES_PASS"]).lower(), flush=True)
    print("HARD_FAILURES=" + json.dumps(payload["hard_failures"]), flush=True)
    return payload


# ---------------------------------------------------------------------------
# bench1
# ---------------------------------------------------------------------------


def arm_from_table(table):
    from opt.routed_expert import fixed_route_sets, window_route_sets

    for name, sets in (
        ("sparse_top1_window", window_route_sets(8, 1, cyclic=True)),
        ("fixed_top2_window", window_route_sets(8, 2, cyclic=True)),
        ("fixed_top2", fixed_route_sets(8, 2)),
    ):
        if sets == table:
            return name
    return "custom"


def check_learned_overflow(torch, model, device, sample_cpu):
    """Eager route build with overflow counting (outside the timed path)."""
    from opt.learned_router import build_learned_route

    slice_cpu = {k: v[:min(2, v.shape[0])] for k, v in sample_cpu.items()}
    batch = prepare_gpu_batch(torch, slice_cpu, device)
    with torch.no_grad():
        v = model.ln(model.embedding(batch["x"]))
        route, overflow = build_learned_route(
            model, v, model.route_group,
            model.capacity_tokens(int(v.shape[0]), int(v.shape[1])),
            count_overflow=True)
    per_expert = route.sel_mask.sum(dim=1).tolist()
    return {
        "capacity_tokens_per_expert": int(route.sel_idx.shape[1]),
        "overflow_tokens_sampled": int(overflow),
        "selected_tokens_per_expert": [int(v) for v in per_expert],
        "active_experts": list(route.active),
    }


def run_bench1(args):
    import torch

    configure_torch()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    cfg = production_cfg()

    result = {
        "generated_utc": now_utc(),
        "arm": args.arm,
        "microbatch": int(args.microbatch),
        "tag": args.tag,
        "accumulation_steps": MICROBATCH_ACCUM.get(int(args.microbatch),
                                                   GLOBAL_SEQUENCES //
                                                   int(args.microbatch)),
        "global_sequences": GLOBAL_SEQUENCES,
        "T": T,
        "tokens_per_update": GLOBAL_SEQUENCES * T,
        "steps": int(args.steps),
        "warmups": int(args.warmups),
        "start_sequence": int(args.start_sequence),
        "status": "RUNNING",
    }
    try:
        import torch as _torch  # noqa: F401
        from opt.model_ref import synthetic_packed_batch

        repo_report, contract = verify_repo(
            repo, Path(args.contract) if args.contract else
            repo / "results" / "g4_hero_claim_contract.json")
        result["repo"] = repo_report
        if args.require_fingerprints and not repo_report["pass"]:
            raise RuntimeError(
                f"repo fingerprint gate failed: "
                f"mismatches={repo_report['mismatches']} "
                f"missing={repo_report['missing']} "
                f"harness_match={repo_report.get('harness_match')}")

        # ---- data ---------------------------------------------------------
        corpus_meta = None
        if args.corpus_root:
            trainer, corpus, cinfo = corpus_open(repo, Path(args.corpus_root),
                                                 fast_verify=True)
            batches = []
            cursors = []
            for cursor, batch in corpus.stream_batches(
                    int(args.start_sequence), GLOBAL_SEQUENCES):
                batches.append(batch)
                cursors.append(int(cursor))
                if len(batches) == int(args.warmups) + int(args.steps):
                    break
            corpus_meta = {
                "info": cinfo,
                "batch_start_sequences": cursors,
                "sequence_range": [cursors[0], cursors[-1] + GLOBAL_SEQUENCES
                                   - 1],
                "unique_batches": len(batches),
                "replayed": False,
                "mode": "frozen_packed_corpus",
            }
        else:
            batches = []
            for i in range(int(args.warmups) + int(args.steps)):
                batch = synthetic_packed_batch(cfg, GLOBAL_SEQUENCES, device,
                                               seed=1000 + i, mode="mixed")
                gen = torch.Generator(device="cpu").manual_seed(2000 + i)
                batch["y"] = torch.randint(
                    0, cfg.V, batch["x"].shape, generator=gen).to(device)
                batch["valid"] = torch.ones(batch["x"].shape,
                                            dtype=torch.bool, device=device)
                batch["input_valid"] = batch["valid"].clone()
                batch["start"] = batch["segment_start"]
                batches.append(batch)
            corpus_meta = {"mode": "synthetic_debug_fallback",
                           "production_scale": False,
                           "unique_batches": len(batches)}
        result["corpus"] = corpus_meta

        # ---- models -------------------------------------------------------
        source = build_dense(torch, cfg, device)
        source_state = dense_state(source)
        ledger = {"stored_params": int(sum(p.numel()
                                           for p in source.parameters()))}
        if args.arm == "dense":
            model = source
            route = None
            info = {"kind": "opt3c_all_dense", "top_r": None}
            result["semantic_status"] = SEMANTIC_STATUS["dense"]
        else:
            del source
            torch.cuda.empty_cache()
            model, route, info = make_candidate(
                torch, args.arm, cfg, device, int(args.microbatch),
                source_state)
            result["semantic_status"] = SEMANTIC_STATUS[args.arm]
            ledger = info["ledger"]
        result["candidate_info"] = info
        result["ledger"] = ledger
        result["parameter_count"] = int(sum(p.numel() for p in
                                            model.parameters()))

        # overflow audit for the learned router (outside timing)
        if args.arm == "learned_top2" and batches:
            first = {k: v for k, v in batches[0].items()
                     if k in ("x", "y", "pos", "segpos", "valid", "start",
                              "input_valid")}
            info["overflow_audit"] = check_learned_overflow(
                torch, model, device, first)

        # ---- compile ------------------------------------------------------
        t0 = time.perf_counter()
        entry = static_entry(torch, args.arm, model)
        fwd = make_fwd(args.arm, entry, route)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.PEAK_LR,
                                      betas=cfg.BETAS, eps=cfg.EPS,
                                      fused=True)

        # per-update CPU slices + denominators (precomputed, no replay)
        update_slices = []
        update_denoms = []
        update_valid_pairs = []
        for batch in batches:
            denom = int(batch["valid"].sum().item())
            update_denoms.append(denom)
            update_valid_pairs.append(denom)
            slices = []
            for off in range(0, GLOBAL_SEQUENCES, int(args.microbatch)):
                sl = slice(off, off + int(args.microbatch))
                slices.append({k: v[sl] if torch.is_tensor(v) else v
                               for k, v in batch.items()})
            update_slices.append(slices)
        result["valid_pairs_per_update"] = update_valid_pairs
        result["valid_pairs_median"] = float(
            statistics.median(update_valid_pairs))

        def one_update(index):
            return run_update(torch, model, fwd, optimizer, cfg,
                              update_slices[index], device,
                              update_denoms[index],
                              lr_for_update(index + 1, cfg))

        for i in range(int(args.warmups)):
            torch.cuda.synchronize()
            t0_upd = time.perf_counter()
            one_update(i)
            torch.cuda.synchronize()
            result.setdefault("warmup_ms", []).append(
                (time.perf_counter() - t0_upd) * 1000.0)
        torch.cuda.synchronize()
        result["compile_seconds"] = time.perf_counter() - t0

        torch.cuda.reset_peak_memory_stats(device)
        times = []
        losses = []
        for j in range(int(args.steps)):
            index = int(args.warmups) + j
            torch.cuda.synchronize()
            t0_upd = time.perf_counter()
            losses.append(one_update(index))
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0_upd) * 1000.0)

        tokens = GLOBAL_SEQUENCES * T
        median_ms = float(statistics.median(times))
        valid_med = float(statistics.median(update_valid_pairs))
        result.update(summarise_times(times))
        result.update({
            "status": "ok",
            "losses": losses,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "loss_finite": all(math.isfinite(v) for v in losses),
            "packed_tok_s": tokens / (median_ms / 1000.0),
            "valid_pair_tok_s": valid_med / (median_ms / 1000.0),
            "peak_allocated_GiB": torch.cuda.max_memory_allocated(device)
            / 2**30,
            "peak_reserved_GiB": torch.cuda.max_memory_reserved(device)
            / 2**30,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "lr_schedule": "certified warmup (peak 1e-3, 10M tokens)",
        })
        if not result["loss_finite"]:
            result["status"] = "nonfinite_loss"
    except torch.cuda.OutOfMemoryError as exc:  # noqa: PERF203
        free, total = torch.cuda.mem_get_info(0)
        result.update({
            "status": "OOM",
            "error": str(exc)[:400],
            "vram_free_GiB": free / 2**30,
            "vram_total_GiB": total / 2**30,
        })
    except Exception as exc:  # noqa: BLE001
        result.update({"status": "error",
                       "error": f"{type(exc).__name__}: {str(exc)[:800]}"})
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    out = Path(args.out) if args.out else (
        outdir / f"bench_{args.arm}_mb{args.microbatch}.json")
    save_json(out, result)
    print("BENCH_RESULT " + json.dumps(
        {k: v for k, v in result.items()
         if k in ("arm", "microbatch", "status", "median_ms", "packed_tok_s",
                  "valid_pair_tok_s", "peak_allocated_GiB",
                  "compile_seconds", "error")}, default=str), flush=True)
    return result


# ---------------------------------------------------------------------------
# runall orchestration
# ---------------------------------------------------------------------------


def spawn(mode, outdir, extra, repo):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--mode", mode,
           "--repo", str(repo), "--outdir", str(outdir)] + extra
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print("SPAWN " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=env, check=False).returncode


def bench_args(arm, microbatch, args, tag):
    extra = ["--arm", arm, "--microbatch", str(microbatch),
             "--tag", tag,
             "--steps", str(args.steps), "--warmups", str(args.warmups),
             "--start-sequence", str(args.start_sequence),
             "--out", str(Path(args.outdir) / f"bench_{arm}_mb{microbatch}"
                                             f"_{tag}.json")]
    if args.corpus_root:
        extra += ["--corpus-root", str(args.corpus_root)]
    else:
        extra += ["--allow-synthetic"]
    if args.contract:
        extra += ["--contract", str(args.contract)]
    if not args.require_fingerprints:
        extra += ["--no-require-fingerprints"]
    return extra


def load_bench(outdir: Path):
    records = []
    for path in sorted(Path(outdir).glob("bench_*.json")):
        try:
            records.append(load_json(path))
        except Exception:
            continue
    return records


def runall(args):
    repo = Path(args.repo).resolve()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plan = {
        "generated_utc": now_utc(),
        "probe_microbatches": list(PROBE_MICROBATCHES),
        "dense_production_microbatch": DENSE_PRODUCTION_MICROBATCH,
        "hero2_microbatch": HERO2_MICROBATCH,
        "steps": args.steps,
        "warmups": args.warmups,
        "corpus_root": str(args.corpus_root) if args.corpus_root else None,
        "start_sequence": args.start_sequence,
        "one_process_per_config": True,
        "rng_seed": args.seed,
    }

    gates_rc = spawn("gates", outdir, [
        "--start-sequence", str(args.start_sequence),
    ] + (["--corpus-root", str(args.corpus_root)]
         if args.corpus_root else []), repo)
    gates = load_json(outdir / "gates.json") if (
        outdir / "gates.json").is_file() else {"GATES_PASS": False,
                                               "hard_failures": ["missing"]}
    plan["gates_returncode"] = gates_rc
    if not gates.get("GATES_PASS"):
        report(args)
        print("GATES_FAILED_BENCHMARKS_SKIPPED=true", flush=True)
        return 1

    configs = [("dense", DENSE_PRODUCTION_MICROBATCH)]
    configs += [("sparse_top1", mb) for mb in PROBE_MICROBATCHES]
    rng = random.Random(args.seed)
    rng.shuffle(configs)
    plan["probe_order"] = [f"{a}_mb{m}" for a, m in configs]
    plan["adaptive"] = []
    save_json(outdir / "plan.json", plan)

    for arm, mb in configs:
        spawn("bench1", outdir, bench_args(arm, mb, args, "probe"), repo)

    def ok(arm):
        return [r for r in load_bench(outdir)
                if r.get("arm") == arm and r.get("status") == "ok"]

    sparse_ok = ok("sparse_top1")
    if sparse_ok:
        best = min(sparse_ok, key=lambda r: r["median_ms"])
        best_mb = int(best["microbatch"])
        if best_mb == 32:
            plan["adaptive"].append("sparse_top1_mb64 (larger B improving)")
            spawn("bench1", outdir,
                  bench_args("sparse_top1", 64, args, "adaptive"), repo)
        elif best_mb == 8:
            plan["adaptive"].append("sparse_top1_mb4 (smaller B improving)")
            spawn("bench1", outdir,
                  bench_args("sparse_top1", 4, args, "adaptive"), repo)
        else:
            plan["adaptive"].append(
                "none (B16 optimum within probe set)")

    for arm in ("fixed_top2", "learned_top2"):
        spawn("bench1", outdir, bench_args(arm, HERO2_MICROBATCH, args,
                                           "hero2"), repo)

    spawn("bench1", outdir, bench_args("dense", DENSE_PRODUCTION_MICROBATCH,
                                       args, "drift_control"), repo)

    if args.dense_oom_probe:
        spawn("bench1", outdir,
              bench_args("dense", DENSE_OOM_PROBE_MICROBATCH, args,
                         "oom_census"), repo)

    save_json(outdir / "plan.json", plan)
    report(args)
    return 0


# ---------------------------------------------------------------------------
# report / verdict
# ---------------------------------------------------------------------------


def _row(record):
    info = record.get("candidate_info", {}) or {}
    return {
        "candidate": info.get("kind", record.get("arm")),
        "arm": record.get("arm"),
        "tag": record.get("tag"),
        "semantic_status": record.get("semantic_status"),
        "microbatch": record.get("microbatch"),
        "accumulation": record.get("accumulation_steps"),
        "status": record.get("status"),
        "ms_per_update": record.get("median_ms"),
        "p10_ms": record.get("p10_ms"),
        "p90_ms": record.get("p90_ms"),
        "packed_tok_s": record.get("packed_tok_s"),
        "valid_pair_tok_s": record.get("valid_pair_tok_s"),
        "peak_allocated_GiB": record.get("peak_allocated_GiB"),
        "peak_reserved_GiB": record.get("peak_reserved_GiB"),
        "compile_seconds": record.get("compile_seconds"),
        "capacity_tokens_per_expert": info.get("capacity_tokens_per_expert"),
        "padding_fraction": info.get("padding_fraction"),
        "overflow_tokens": info.get("overflow_tokens"),
        "graph_breaks": record.get("graph_breaks"),
    }


def build_report(outdir: Path, contract, repo: Path):
    outdir = Path(outdir)
    gates = load_json(outdir / "gates.json") if (
        outdir / "gates.json").is_file() else {
        "GATES_PASS": False, "hard_failures": ["gates.json missing"],
        "gates": []}
    records = load_bench(outdir)
    rows = [_row(r) for r in records if r.get("status") != "RUNNING"]
    gate_graph = {}
    for g in gates.get("gates", []):
        name = g.get("name", "")
        if name.startswith("gpu_graph_breaks_"):
            gate_graph[name.replace("gpu_graph_breaks_", "")] = \
                (g.get("detail") or {}).get("graph_break_count")
    for row in rows:
        if gate_graph.get(row["arm"]) is not None:
            row["graph_breaks"] = gate_graph[row["arm"]]
    dense_probe = next((r for r in records if r.get("arm") == "dense"
                        and r.get("tag") == "probe"), None)
    dense_drift = next((r for r in records if r.get("arm") == "dense"
                        and r.get("tag") == "drift_control"), None)
    session_drift = None
    session_stable = None
    if (dense_probe and dense_drift
            and dense_probe.get("status") == "ok"
            and dense_drift.get("status") == "ok"):
        session_drift = float(dense_drift["median_ms"]
                              / dense_probe["median_ms"] - 1.0)
        session_stable = abs(session_drift) <= 0.05

    def ok(arm):
        return [r for r in records
                if r.get("arm") == arm and r.get("status") == "ok"]

    dense_all = ok("dense")
    sparse_all = ok("sparse_top1")
    fixed_all = ok("fixed_top2")
    learned_all = ok("learned_top2")

    dense_b16 = next((r for r in dense_all
                      if int(r["microbatch"]) == DENSE_PRODUCTION_MICROBATCH
                      and r.get("tag") == "probe"), None)
    if dense_b16 is None:
        dense_b16 = next((r for r in dense_all
                          if int(r["microbatch"]) == DENSE_PRODUCTION_MICROBATCH),
                         None)
    dense_best = min(dense_all, key=lambda r: r["median_ms"]) \
        if dense_all else None
    sparse_best = min(sparse_all, key=lambda r: r["median_ms"]) \
        if sparse_all else None
    sparse_b16 = next((r for r in sparse_all
                       if int(r["microbatch"]) == DENSE_PRODUCTION_MICROBATCH
                       and r.get("tag") == "probe"), None)
    if sparse_b16 is None:
        sparse_b16 = next((r for r in sparse_all
                           if int(r["microbatch"])
                           == DENSE_PRODUCTION_MICROBATCH), None)
    fixed_b16 = next((r for r in fixed_all
                      if int(r["microbatch"]) == HERO2_MICROBATCH), None)
    learned_b16 = next((r for r in learned_all
                        if int(r["microbatch"]) == HERO2_MICROBATCH), None)

    anchor_ok = None
    if dense_b16:
        rel = (abs(dense_b16["packed_tok_s"] - DENSE_ANCHOR["packed_tok_s"])
               / DENSE_ANCHOR["packed_tok_s"])
        anchor_ok = rel <= THRESHOLDS["dense_anchor_rel_tol"]

    hero1 = {"claim": ("sparse Arm-A top1 (M8/Ke512 fixed cyclic window, "
                       "exact capacity, compiled) >= 350k packed tok/s at "
                       "global batch 64x2048 on G4"),
             "measured": {}, "status": "FAIL"}
    if sparse_best:
        tok = float(sparse_best["packed_tok_s"])
        status = ("PASS" if tok >= THRESHOLDS["hero1_tok_s_pass"]
                  else "HOLD" if tok >= THRESHOLDS["hero1_tok_s_hold"]
                  else "FAIL")
        hero1.update({
            "status": status,
            "measured": {
                "packed_tok_s": tok,
                "microbatch": int(sparse_best["microbatch"]),
                "ms_per_update": float(sparse_best["median_ms"]),
                "valid_pair_tok_s": float(sparse_best["valid_pair_tok_s"]),
                "peak_allocated_GiB": float(sparse_best["peak_allocated_GiB"]),
            },
            "thresholds": {k: v for k, v in THRESHOLDS.items()
                           if k.startswith("hero1")},
            "speedup_vs_dense_best_feasible": (
                float(dense_best["median_ms"] / sparse_best["median_ms"])
                if dense_best else None),
            "speedup_vs_dense_matched_b16": (
                float(dense_b16["median_ms"] / sparse_b16["median_ms"])
                if dense_b16 and sparse_b16 else None),
        })

    hero2 = {"claim": ("learned top2 routing retains >=2x dense speedup with "
                       "<=5% overhead vs fixed top2 at matched geometry"),
             "measured": {}, "status": "FAIL"}
    if learned_b16 and fixed_b16 and dense_b16:
        overhead = float(learned_b16["median_ms"]
                         / fixed_b16["median_ms"] - 1.0)
        speedup = float(dense_b16["median_ms"] / learned_b16["median_ms"])
        status = ("PASS"
                  if overhead <= THRESHOLDS["hero2_overhead_pass"]
                  and speedup >= THRESHOLDS["hero2_speedup_pass"]
                  else "HOLD"
                  if overhead <= THRESHOLDS["hero2_overhead_hold"]
                  and speedup >= THRESHOLDS["hero2_speedup_hold"]
                  else "FAIL")
        hero2.update({
            "status": status,
            "measured": {
                "learned_top2_ms": float(learned_b16["median_ms"]),
                "fixed_top2_ms": float(fixed_b16["median_ms"]),
                "dense_ms": float(dense_b16["median_ms"]),
                "router_overhead": overhead,
                "learned_speedup_vs_dense": speedup,
                "learned_packed_tok_s": float(learned_b16["packed_tok_s"]),
                "overflow_tokens_sampled": learned_b16.get(
                    "candidate_info", {}).get("overflow_audit", {}).get(
                        "overflow_tokens_sampled"),
                "graph_breaks": (
                    next((g["detail"].get("graph_break_count")
                          for g in gates.get("gates", [])
                          if g["name"] == "gpu_graph_breaks_learned_top2"),
                         None)),
            },
            "thresholds": {k: v for k, v in THRESHOLDS.items()
                           if k.startswith("hero2")},
        })

    best_record = None
    for record in sparse_all + learned_all:
        if record.get("status") == "ok" and (
                best_record is None
                or record["median_ms"] < best_record["median_ms"]):
            best_record = record
    projected = None
    if best_record:
        projected = 2.5e9 / float(best_record["packed_tok_s"]) / 3600.0

    comparisons = {
        "matched_geometry_B16x4": {
            "dense_ms": dense_b16["median_ms"] if dense_b16 else None,
            "sparse_top1_ms": sparse_b16["median_ms"] if sparse_b16 else None,
            "fixed_top2_ms": fixed_b16["median_ms"] if fixed_b16 else None,
            "learned_top2_ms": (learned_b16["median_ms"]
                                if learned_b16 else None),
            "note": "same microbatch geometry for executor comparison",
        },
        "best_feasible": {
            "dense": ({"microbatch": dense_best["microbatch"],
                       "ms_per_update": dense_best["median_ms"],
                       "packed_tok_s": dense_best["packed_tok_s"]}
                      if dense_best else None),
            "sparse": ({"microbatch": sparse_best["microbatch"],
                        "ms_per_update": sparse_best["median_ms"],
                        "packed_tok_s": sparse_best["packed_tok_s"]}
                       if sparse_best else None),
            "speedup": (float(dense_best["median_ms"]
                              / sparse_best["median_ms"])
                        if dense_best and sparse_best else None),
            "note": "actual training throughput comparison",
        },
        "dense_anchor": {
            "expected": DENSE_ANCHOR,
            "measured": ({"packed_tok_s": dense_b16["packed_tok_s"],
                          "ms_per_update": dense_b16["median_ms"]}
                         if dense_b16 else None),
            "compatible": anchor_ok,
        },
        "session_stability": {
            "dense_b16_probe_ms": (dense_probe["median_ms"]
                                   if dense_probe else None),
            "dense_b16_drift_control_ms": (dense_drift["median_ms"]
                                           if dense_drift else None),
            "relative_drift": session_drift,
            "stable_within_5pct": session_stable,
        },
    }

    hard_pass = bool(gates.get("GATES_PASS"))
    hero_pass = hero1["status"] == "PASS" or hero2["status"] == "PASS"
    payload = {
        "generated_utc": now_utc(),
        "contract_id": (contract or {}).get("contract_id"),
        "repo": str(repo),
        "git_head": git_head(repo),
        "gates_pass": hard_pass,
        "hard_failures": gates.get("hard_failures"),
        "environment": next((g["detail"] for g in gates.get("gates", [])
                             if g["name"] == "environment"), None),
        "rows": rows,
        "comparisons": comparisons,
        "hero_claims": {"HERO_1": hero1, "HERO_2": hero2},
        "arm_b_status": ARM_B_STATUS,
        "arm_c_status": ARM_C_STATUS,
        "best_g4_candidate": (best_record.get("candidate_info", {}).get(
            "kind") if best_record else None),
        "best_packed_tok_s": (float(best_record["packed_tok_s"])
                              if best_record else None),
        "best_microbatch": (int(best_record["microbatch"])
                            if best_record else None),
        "projected_2p5b_hours": projected,
        "projected_2p5b_label": "INFERRED (from measured full-update "
                                "throughput; no 2.5B run executed)",
        "session_stable": session_stable,
        "final_g4_validation_pass": bool(
            hard_pass and hero_pass and session_stable is not False),
    }
    save_json(outdir / "g4_hero_claim_results.json", payload)
    print_report(payload)
    return payload


def print_report(payload):
    print("\n" + "=" * 78)
    print("G4 HERO CLAIM VALIDATION")
    print("=" * 78)
    header = ("candidate", "semantic", "micro", "ms/upd", "packed_tok_s",
              "valid_tok_s", "peak_GiB", "breaks", "overflow", "status")
    print("{:<26} {:<10} {:<7} {:>9} {:>12} {:>12} {:>9} {:>7} {:>9} "
          "{:<8}".format(*header))
    for row in payload["rows"]:
        semantic = "exact_dense" if row["arm"] == "dense" else (
            "declared_sparse" if row["arm"] != "learned_top2"
            else "arch_change")
        overflow = row.get("overflow_tokens")
        print("{:<26} {:<10} {:<7} {:>9} {:>12} {:>12} {:>9} {:>7} {:>9} "
              "{:<8}".format(
                  str(row["candidate"])[:26], semantic,
                  f"B{row['microbatch']}x{row['accumulation']}",
                  f"{row['ms_per_update']:.1f}"
                  if row["ms_per_update"] else "NA",
                  f"{row['packed_tok_s']:.0f}"
                  if row["packed_tok_s"] else "NA",
                  f"{row['valid_pair_tok_s']:.0f}"
                  if row["valid_pair_tok_s"] else "NA",
                  f"{row['peak_allocated_GiB']:.2f}"
                  if row["peak_allocated_GiB"] else "NA",
                  (str(row["graph_breaks"])
                   if row.get("graph_breaks") is not None else "-"),
                  str(overflow if overflow is not None else "-"),
                  str(row["status"])))
    hero1 = payload["hero_claims"]["HERO_1"]
    hero2 = payload["hero_claims"]["HERO_2"]
    print("-" * 78)
    legend = {}
    for row in payload["rows"]:
        if row["semantic_status"]:
            legend.setdefault(str(row["candidate"]), row["semantic_status"])
    for candidate, semantic in legend.items():
        print(f"  semantic[{candidate}] = {semantic}")
    print("-" * 78)
    print(f"HERO_1_STATUS = {hero1['status']}   measured="
          f"{hero1['measured'].get('packed_tok_s')} packed tok/s")
    print(f"HERO_2_STATUS = {hero2['status']}   overhead="
          f"{hero2['measured'].get('router_overhead')} speedup="
          f"{hero2['measured'].get('learned_speedup_vs_dense')}")
    print(f"ARM_B_STATUS   = {payload['arm_b_status']}")
    print(f"ARM_C_STATUS   = {payload['arm_c_status']}")
    print(f"BEST_G4_CANDIDATE = {payload['best_g4_candidate']} "
          f"(microbatch {payload['best_microbatch']})")
    print(f"BEST_PACKED_TOK_S = {payload['best_packed_tok_s']}")
    print(f"PROJECTED_2P5B_HOURS = {payload['projected_2p5b_hours']}  "
          f"[{payload['projected_2p5b_label']}]")
    stability = payload.get("comparisons", {}).get("session_stability", {})
    print(f"SESSION_STABLE = {payload.get('session_stable')} "
          f"(dense B16 drift {stability.get('relative_drift')})")
    print(f"FINAL_G4_VALIDATION_PASS = "
          f"{str(payload['final_g4_validation_pass']).lower()}")
    print("=" * 78, flush=True)


def report(args):
    repo = Path(args.repo).resolve()
    outdir = Path(args.outdir)
    contract_path = Path(args.contract) if args.contract else (
        repo / "results" / "g4_hero_claim_contract.json")
    contract = load_json(contract_path) if contract_path.is_file() else None
    return build_report(outdir, contract, repo)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def plan(args):
    plan_payload = {
        "generated_utc": now_utc(),
        "repo": str(Path(args.repo).resolve()),
        "git_head": git_head(Path(args.repo).resolve()),
        "hero_claims": {
            "HERO_1": {
                "claim": ("sparse Arm-A top1 M8/Ke512 fixed cyclic window, "
                          "exact capacity, compiled, >=350k packed tok/s at "
                          "global 64x2048 on G4"),
                "thresholds": {k: v for k, v in THRESHOLDS.items()
                               if k.startswith("hero1")},
                "local_evidence": ("results/doe_round1_sweep.json 3.18-3.67x, "
                                   "results/onehour_microbatch.json 4.22x "
                                   "best-feasible, results/onehour_active_"
                                   "width.json"),
            },
            "HERO_2": {
                "claim": ("learned top2 hard routing retains >=2x dense "
                          "speedup with <=5% overhead vs fixed top2 at "
                          "matched geometry"),
                "thresholds": {k: v for k, v in THRESHOLDS.items()
                               if k.startswith("hero2")},
                "local_evidence": ("results/bench_learned_v2.json 2.38x vs "
                                   "results/bench_fixed_ref.json 2.40x"),
            },
        },
        "arm_status": {"arm_b": ARM_B_STATUS, "arm_c": ARM_C_STATUS,
                       "gate_document": "campaigns/ARM_BC_SEMANTICS.md"},
        "protocol": {
            "global_sequences": GLOBAL_SEQUENCES,
            "T": T,
            "tokens_per_update": GLOBAL_SEQUENCES * T,
            "probe_microbatches": list(PROBE_MICROBATCHES),
            "adaptive": "B64 if B32 best; B4 if B8 best",
            "dense_production_geometry": f"B{DENSE_PRODUCTION_MICROBATCH}x"
                                         f"{64 // DENSE_PRODUCTION_MICROBATCH}",
            "hero2_matched_geometry": f"B{HERO2_MICROBATCH}x"
                                      f"{64 // HERO2_MICROBATCH}",
            "warmups": DEFAULT_WARMUPS,
            "measured_updates": DEFAULT_STEPS,
            "one_process_per_config": True,
            "randomized_order": True,
            "same_session_dense_control": True,
            "dense_oom_census": f"B{DENSE_OOM_PROBE_MICROBATCH} attempt",
            "corpus": str(CORPUS_DEFAULT),
            "corpus_verification": "manifest + artifact digest, fast_verify "
                                   "(skips per-shard SHA-256; sizes checked)",
            "no_replay": True,
        },
        "dense_anchor": DENSE_ANCHOR,
        "semantic_status": SEMANTIC_STATUS,
    }
    print(jdump(plan_payload))
    return plan_payload


# ---------------------------------------------------------------------------
# selftest (CPU/tiny, local)
# ---------------------------------------------------------------------------


def selftest(args):
    import torch

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    checks = {}

    from opt.learned_router import (
        LearnedRoutedExpertArmA,
        build_learned_route,
        straight_through,
    )
    from opt.model_opt import OptArmA
    from opt.model_ref import (
        ArmAConfig,
        canonical_init,
        full_update as ref_full_update,
        load_init,
        synthetic_packed_batch,
    )
    from opt.routed_expert import (
        RoutedExpertArmA,
        build_route_tensors,
        group_route_table,
        window_route_sets,
    )

    tiny = dict(T=16, V=32, D=8, N=16, H=2, L=2, HIDDEN=8)
    cfg = ArmAConfig(**tiny)
    device = torch.device("cpu")
    checks["imports_and_cfg"] = True

    # construction of all three arms at tiny shape
    dense = OptArmA(cfg, device, scan_block=cfg.K, use_checkpoint=False,
                    coord="dense", single_scan="chunkwise",
                    packed_update="branchfree", zero_carry=True,
                    paper_layout="direct", cache_rope=True)
    load_init(dense, canonical_init(cfg), device)
    state = {k: v.detach().clone() for k, v in dense.state_dict().items()}
    routed = RoutedExpertArmA(cfg, device, experts=4,
                              expert_width=cfg.K // 4, scan_block=cfg.K)
    routed.load_canonical(state)
    learned = LearnedRoutedExpertArmA(cfg, device, experts=4,
                                      expert_width=cfg.K // 4,
                                      scan_block=cfg.K, top_r=2, route_group=4)
    learned.capacity_override = cfg.T
    learned.load_canonical(state)
    checks["model_construction"] = {
        "dense_params": int(sum(p.numel() for p in dense.parameters())),
        "routed_params": int(sum(p.numel() for p in routed.parameters())),
        "learned_params": int(sum(p.numel() for p in learned.parameters())),
        "ledger": learned.parameter_ledger(),
    }
    checks["parameter_identity"] = (
        checks["model_construction"]["routed_params"]
        == checks["model_construction"]["dense_params"])

    # tiny forward/backward for each arm + learned router overflow audit
    batch = synthetic_packed_batch(cfg, 2, device, seed=5, mode="mixed")
    batch["start"] = batch["segment_start"]
    batch["input_valid"] = batch["start"] < cfg.T
    route = build_route_tensors(
        group_route_table(window_route_sets(4, 1, cyclic=True),
                          cfg.T // 4), 4, 2, cfg.T, 4, device)
    for tag, model, fn, extra in (
        ("dense", dense, dense.forward_packed, None),
        ("routed", routed, routed.forward_route, route),
    ):
        model.train()
        model.zero_grad(set_to_none=True)
        if extra is not None:
            logits = fn(batch["x"], batch["pos"], batch["segpos"],
                        batch["full_mask"], batch["start"], extra)
        else:
            logits = fn(batch["x"], batch["pos"], batch["segpos"],
                        batch["full_mask"], batch["start"])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.V), batch["y"].reshape(-1))
        loss.backward()
        finite = all(bool(torch.isfinite(p.grad).all())
                     for p in model.parameters() if p.grad is not None)
        checks[f"tiny_{tag}_fwd_bwd"] = {"loss": float(loss.detach()),
                                         "grads_finite": finite}
        assert finite
    learned.train()
    with torch.no_grad():
        v = learned.ln(learned.embedding(batch["x"]))
        r, overflow = build_learned_route(learned, v, 4,
                                          learned.capacity_override,
                                          count_overflow=True)
    learned.zero_grad(set_to_none=True)
    logits = learned.forward_learned(batch["x"], batch["pos"],
                                     batch["segpos"], batch["full_mask"],
                                     batch["start"])
    torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.V), batch["y"].reshape(-1)).backward()
    checks["learned_route"] = {
        "overflow": int(overflow),
        "capacity": int(r.sel_idx.shape[1]),
        "router_grad_norm": float(learned.router.Wr.grad.norm()),
    }
    assert float(learned.router.Wr.grad.norm()) > 0

    # straight-through unit check
    lg = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    h, g_st, idx = straight_through(lg)
    checks["straight_through"] = {
        "forward_value_equals_hard": torch.equal(g_st.detach(), h),
    }
    g_st.sum().backward()
    checks["straight_through"]["grad_finite"] = bool(
        torch.isfinite(lg.grad).all())

    # update glue == model_ref.full_update on tiny dense, bitwise params
    def fresh_dense():
        model = OptArmA(cfg, device, scan_block=cfg.K, use_checkpoint=False,
                        coord="dense", single_scan="chunkwise",
                        packed_update="branchfree", zero_carry=True,
                        paper_layout="direct", cache_rope=True)
        load_init(model, canonical_init(cfg), device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.PEAK_LR,
                                betas=cfg.BETAS, eps=cfg.EPS)
        return model, opt

    model_a, opt_a = fresh_dense()
    slices = []
    for off in range(0, 2, 1):
        slices.append({k: v[off:off + 1] for k, v in batch.items()
                       if torch.is_tensor(v)})
    denom = int(batch["valid"].sum().item())

    def entry(b):
        return model_a(b["x"], b["pos"], b["segpos"], b["full_mask"],
                       b["start"])

    run_update(torch, model_a, entry, opt_a, cfg, slices, device, denom,
               lr=cfg.PEAK_LR)
    model_b, opt_b = fresh_dense()
    ref_full_update(model_b, opt_b, cfg, batch, microbatch=1,
                    device_type="cpu", compiled=None, entry_mode="packed")
    diff = max(float((p1.detach() - p2.detach()).abs().max())
               for p1, p2 in zip(model_a.parameters(), model_b.parameters()))
    checks["update_glue_vs_model_ref_full_update"] = {
        "max_param_abs_diff": diff, "tolerance": 1e-6}
    assert diff <= 1e-6, diff

    # harness arm wiring (make_candidate / make_fwd / prepare_gpu_batch)
    harness_arms = {}
    harness_slice = {k: (v[:2] if torch.is_tensor(v) and v.shape[0] >= 2
                         else v) for k, v in batch.items()}
    for tag in ("sparse_top1", "fixed_top2", "learned_top2"):
        model_i, route_i, info_i = make_candidate(
            torch, tag, cfg, device, 2, state, route_group=4, experts=4,
            expert_width=cfg.K // 4)
        entry_i = (model_i.forward_learned if tag == "learned_top2"
                   else model_i.forward_route)
        fwd_i = make_fwd(tag, entry_i, route_i)
        gpu_batch = prepare_gpu_batch(torch, harness_slice, device)
        with torch.no_grad():
            out_i = fwd_i(gpu_batch)
        harness_arms[tag] = {
            "output_shape": list(out_i.shape),
            "finite": bool(torch.isfinite(out_i).all()),
            "kind": info_i["kind"],
            "capacity": info_i["capacity_tokens_per_expert"],
        }
        if tag == "learned_top2":
            harness_arms[tag]["overflow_audit"] = check_learned_overflow(
                torch, model_i, device, harness_slice)
        del model_i, out_i
    checks["harness_arm_wiring"] = harness_arms
    assert all(v["finite"] for v in harness_arms.values())

    # report aggregation on fabricated records
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        save_json(tmpdir / "gates.json", {"GATES_PASS": True,
                                          "hard_failures": [], "gates": []})
        fabricate = [
            ("dense", 16, 1900.0, "exact_dense"),
            ("sparse_top1", 16, 360.0, "x"),
            ("fixed_top2", 16, 700.0, "x"),
            ("learned_top2", 16, 710.0, "x"),
        ]
        for arm, mb, ms, kind in fabricate:
            tokens = GLOBAL_SEQUENCES * T
            save_json(tmpdir / f"bench_{arm}_mb{mb}.json", {
                "arm": arm, "microbatch": mb, "accumulation_steps":
                MICROBATCH_ACCUM[mb], "status": "ok", "median_ms": ms,
                "p10_ms": ms, "p90_ms": ms,
                "packed_tok_s": tokens / (ms / 1000.0),
                "valid_pair_tok_s": tokens / (ms / 1000.0),
                "peak_allocated_GiB": 10.0, "compile_seconds": 1.0,
                "semantic_status": kind,
                "candidate_info": {"kind": kind, "overflow_tokens": 0},
            })
        payload = build_report(tmpdir, {"contract_id": "selftest"}, repo)
        checks["report_logic"] = {
            "hero1": payload["hero_claims"]["HERO_1"]["status"],
            "hero2": payload["hero_claims"]["HERO_2"]["status"],
            "best": payload["best_g4_candidate"],
            "final_pass": payload["final_g4_validation_pass"],
            "arm_b": payload["arm_b_status"],
        }
        assert payload["hero_claims"]["HERO_1"]["status"] == "PASS"
        assert payload["hero_claims"]["HERO_2"]["status"] == "PASS"
        assert payload["final_g4_validation_pass"] is True

    print("SELFTEST " + jdump(checks))
    print("SELFTEST_PASS=true", flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("plan", "env", "gates", "bench1", "runall",
                                 "report", "selftest"))
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--outdir",
                        default=str(REPO / "results" / "g4_hero_run"))
    parser.add_argument("--contract", default=None)
    parser.add_argument("--corpus-root", default=str(CORPUS_DEFAULT))
    parser.add_argument("--start-sequence", type=int, default=0)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--arm", choices=("dense", "sparse_top1",
                                          "fixed_top2", "learned_top2"))
    parser.add_argument("--tag", default="probe")
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--out", default=None)
    parser.add_argument("--dense-oom-probe", dest="dense_oom_probe",
                        action="store_true")
    parser.add_argument("--no-dense-oom-probe", dest="dense_oom_probe",
                        action="store_false")
    parser.set_defaults(dense_oom_probe=True)
    parser.add_argument("--require-fingerprints", dest="require_fingerprints",
                        action="store_true")
    parser.add_argument("--no-require-fingerprints",
                        dest="require_fingerprints", action="store_false")
    parser.set_defaults(require_fingerprints=True)
    parser.add_argument("--allow-synthetic", action="store_true",
                        help="bench1 debug fallback when corpus is absent")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.corpus_root:
        args.corpus_root = None
    if args.mode == "plan":
        plan(args)
        return 0
    if args.mode == "env":
        rep = environment_report()
        print(jdump(rep))
        print("ENVIRONMENT_PASS=" + str(rep["ENVIRONMENT_PASS"]).lower())
        return 0 if rep["ENVIRONMENT_PASS"] else 1
    if args.mode == "gates":
        run_gates(Path(args.repo).resolve(), args.corpus_root,
                  args.start_sequence, Path(args.outdir))
        return 0
    if args.mode == "bench1":
        if not args.arm or not args.microbatch:
            raise SystemExit("bench1 requires --arm and --microbatch")
        result = run_bench1(args)
        return 0 if result.get("status") == "ok" else 1
    if args.mode == "runall":
        return runall(args)
    if args.mode == "report":
        report(args)
        return 0
    if args.mode == "selftest":
        return selftest(args)
    raise SystemExit(f"unhandled mode {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
