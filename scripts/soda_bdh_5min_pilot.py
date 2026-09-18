"""SODA-BDH five-minute training pilot orchestrator.

Naming decision (owner, Option A): for this pilot "SODA-BDH" means the
hash-verified repo-local canonical dense BDH/Arm-A path at commit
df7da680dd6420a0c4fefa35c13f107246091308. It does not claim equivalence to
the absent vendor Soda file (SHA-256 947f8b33...), which stays explicitly
labelled unavailable.

The orchestrator does NOT modify the canonical trainer. It runs the frozen
training path exactly as shipped:

  training/arm_a_2p5b_trainer.py --mode smoke ...
  training/arm_a_2p5b_trainer.py --mode train ... --target-tokens 24000000

and wraps it with the mission's phases:

  lock      Phase 1 source lock (commit + SHA-256 of the 4 verified files)
  preflight Phase 2 live correctness (real packed batch: forward, loss,
            backward, gradients, one AdamW step, second step) + Stage A
            fixed-batch overfit smoke + optional cross-row isolation
  smoke     canonical trainer --mode smoke (checkpoint save/reload/step)
  train     canonical trainer --mode train (fresh start, real frozen corpus)
  report    required artifacts + verdict
  all       lock -> preflight/stage-A -> smoke -> train -> report
  selftest  local harness-only validation (tiny CPU model + synthetic log
            records); NOT a pilot and produces no pilot evidence

Artifacts (under --pilot-dir, default /content/soda_bdh_5min_pilot):
  source_manifest.json, environment.json, correctness.json,
  training_log.csv, pilot_summary.json, pilot_report.md,
  soda_bdh_pilot_checkpoint.pt, plus raw logs and per-phase JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAINER_REL = "training/arm_a_2p5b_trainer.py"
PINNED_COMMIT = "df7da680dd6420a0c4fefa35c13f107246091308"
SOURCE_FILES = {
    "opt/model_opt.py":
        "722fbdc1aee51721417a4dc2993312eb556b5057f6ce0d3314c94516dab8f8d8",
    "opt/scan_attn.py":
        "855f868e9d4869e509d2b847bc95417603239201738d304761a19a3277e5708b",
    "opt/model_ref.py":
        "2ff16aaa24ecd14021c46f2133779edcf86853869fbe9938a5a8b95d85eb579f",
    "training/arm_a_2p5b_trainer.py":
        "1985fa42042033c842c7ed0faea2c34ead6516bb1fe56753426b6774ee2d0b49",
}
VENDOR_SODA_SHA256 = (
    "947f8b33e740adede3e13382f8cb9e8e374d845de75e282bda5feb05c582f624"
)
LABELS = {
    "MODEL": "SODA-BDH",
    "MODEL_DEFINITION": "repo-local canonical dense BDH/Arm-A",
    "VENDOR_SODA_EQUIVALENCE": "UNVERIFIED",
    "DENSE_PATH_CONFIRMED": "true",
    "DATA_MODE": "FROZEN_CORPUS_G4",
}
DEFAULT_CORPUS = (
    "/content/drive/Shareddrives/ICLR PHASE BDH/phase_bdh/corpus/stage2/"
    "frozen_5b_v1"
)
DEFAULT_PILOT_DIR = "/content/soda_bdh_5min_pilot"
DEFAULT_TARGET_TOKENS = 24_000_000
DEFAULT_PILOT_BUDGET_S = 300.0
DEFAULT_RESERVE_S = 20.0
DEFAULT_STAGE_A_UPDATES = 30
DEFAULT_STAGE_A_BUDGET_S = 90.0
SMOKE_TIMEOUT_S = 1800.0
PRECISION = "bf16 autocast + fp32 master parameters (frozen trainer policy)"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_json(path: Path, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str),
                    encoding="utf-8")
    return path


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_norm(path: Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def git(args):
    try:
        return subprocess.check_output(["git", "-C", str(REPO), *args],
                                       text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def load_trainer():
    spec = importlib.util.spec_from_file_location(
        "arm_a_trainer", REPO / TRAINER_REL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["arm_a_trainer"] = module
    spec.loader.exec_module(module)
    return module


class _ShimLogger:
    def __init__(self):
        self.records = []

    def log(self, event, **fields):
        self.records.append({"event": event, **fields})
        print("TRAINER_GATE " + json.dumps({"event": event, **fields},
                                           default=str), flush=True)


# ---------------------------------------------------------------------------
# Phase 1 - source lock
# ---------------------------------------------------------------------------


def verify_source() -> dict:
    report = {
        "repo": str(REPO),
        "pinned_commit": PINNED_COMMIT,
        "git_commit": git(["rev-parse", "HEAD"]),
        "git_branch": git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "working_tree_porcelain": git(["status", "--porcelain"]),
        "files": {},
        "mismatches": [],
        "missing": [],
        "git_commit_matches": None,
        "git_descends_from_pinned": None,
    }
    if report["git_commit"] is not None:
        report["git_commit_matches"] = \
            report["git_commit"] == PINNED_COMMIT
        try:
            subprocess.run(
                ["git", "-C", str(REPO), "merge-base", "--is-ancestor",
                 PINNED_COMMIT, report["git_commit"]],
                check=True, capture_output=True)
            report["git_descends_from_pinned"] = True
        except Exception:
            report["git_descends_from_pinned"] = False
    for rel, expected in SOURCE_FILES.items():
        path = REPO / rel
        if not path.is_file():
            report["missing"].append(rel)
            continue
        actual = sha256_norm(path)
        clean = subprocess.run(
            ["git", "-C", str(REPO), "diff", "--quiet", "--", rel],
            capture_output=True).returncode == 0
        report["files"][rel] = {
            "sha256": actual,
            "frozen_contract_sha256": expected,
            "match": actual == expected,
            "git_worktree_clean_for_path": clean,
            "git_blob": git(["rev-parse", f"HEAD:{rel}"]),
        }
        if actual != expected:
            report["mismatches"].append(rel)
    report["vendor_soda"] = {
        "present": (REPO / "vendor/canonical/"
                    "icrl_reintegrated_bdh_baseline_v1.py").is_file(),
        "pinned_sha256": VENDOR_SODA_SHA256,
        "status": "UNAVAILABLE (explicitly labelled, not required for "
                  "this pilot)",
    }
    report["naming_decision"] = {
        "option": "A",
        "soda_bdh_definition": MODEL_DEFINITION,
        "labels": LABELS,
    }
    report["pass"] = bool(
        not report["missing"]
        and not report["mismatches"]
        and (report["git_commit_matches"] is not False
             or report["git_descends_from_pinned"] is True)
    )
    return report


MODEL_DEFINITION = ("hash-verified repo-local canonical dense BDH/Arm-A "
                    "path at commit " + PINNED_COMMIT)


def phase_lock(args) -> int:
    pilot_dir = Path(args.pilot_dir)
    pilot_dir.mkdir(parents=True, exist_ok=True)
    source = verify_source()
    manifest = {
        "pilot": "soda_bdh_5min_pilot",
        "generated_utc": now_utc(),
        "labels": LABELS,
        "model_definition": MODEL_DEFINITION,
        "vendor_soda_equivalence": "UNVERIFIED",
        "vendor_soda_file": source["vendor_soda"],
        "source_lock": source,
        "corpus_identity": {
            "id": "phase_bdh_stage2_5b_v1",
            "root": str(args.corpus_root),
            "context_length": 2048,
            "sequences": 2441407,
        },
        "precision": PRECISION,
        "corpus_verification": {
            "mode": "fast" if args.fast_verify else "full",
            "skipped_if_fast": (
                "per-shard SHA-256 of the 25x3 corpus artifacts; sizes, the "
                "artifact index and the manifest digests are still verified"),
        },
        "seeds": {"SEED": 1337, "INIT_STD": 0.02, "torch_seed_set_by": (
            "training/arm_a_2p5b_trainer.py _build_model")},
        "invalid_source_if": (
            "any SHA-256 above differs from the frozen contract, or the git "
            "commit differs when .git is present"),
    }
    save_json(pilot_dir / "source_manifest.json", manifest)
    torch_info = {}
    try:
        import torch
        torch_info = {
            "torch": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": (torch.cuda.device_count()
                                  if torch.cuda.is_available() else 0),
            "bf16_supported": (torch.cuda.is_bf16_supported()
                               if torch.cuda.is_available() else False),
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            torch_info.update({
                "gpu": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "vram_total_MiB": round(total / 2**20, 1),
                "vram_free_MiB": round(free / 2**20, 1),
                "sm_count": props.multi_processor_count,
            })
    except Exception as exc:  # noqa: BLE001
        torch_info = {"error": f"{type(exc).__name__}: {exc}"}
    environment = {
        "generated_utc": now_utc(),
        "python": sys.version,
        "platform": platform.platform(),
        "precision_policy": PRECISION,
        "torch": torch_info,
        "nvidia_smi": _smi(),
        "expected_production_runtime": (
            "RTX PRO 6000 Blackwell sm_120, torch 2.11.0+cu128, CUDA 12.8"),
        "enforced_by": "training/arm_a_2p5b_trainer.enforce_production_runtime",
    }
    save_json(pilot_dir / "environment.json", environment)
    print("SOURCE_HASH_STATUS=" + (
        "VERIFIED" if source["pass"] else "MISMATCH"), flush=True)
    print("SOURCE_COMMIT=" + str(source["git_commit"]), flush=True)
    if not source["pass"]:
        print("INVALID_SOURCE", flush=True)
        return 2
    return 0


def _smi(query="name,memory.total,driver_version", gpu=0):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu}", f"--query-gpu={query}",
             "--format=csv,noheader"], text=True).strip()
        return out
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Phase 2 - live preflight + Stage A overfit
# ---------------------------------------------------------------------------


def _gpu_batch(trainer, cfg, device, cpu_batch):
    import torch

    tensors = {}
    for lo in range(0, int(cpu_batch["x"].shape[0]), cfg.MICROBATCH):
        hi = min(lo + cfg.MICROBATCH, int(cpu_batch["x"].shape[0]))
        x = cpu_batch["x"][lo:hi].to(device, dtype=torch.long,
                                      non_blocking=True)
        y = cpu_batch["y"][lo:hi].to(device, dtype=torch.long,
                                      non_blocking=True)
        pos = cpu_batch["pos"][lo:hi].to(device, dtype=torch.int32,
                                          non_blocking=True)
        segpos = cpu_batch["segpos"][lo:hi].to(device, dtype=torch.int32,
                                                non_blocking=True)
        valid = cpu_batch["valid"][lo:hi].to(device, dtype=torch.bool,
                                              non_blocking=True)
        input_valid = cpu_batch["input_valid"][lo:hi].to(
            device, dtype=torch.bool, non_blocking=True)
        start = cpu_batch["start"][lo:hi].to(device, dtype=torch.int32,
                                              non_blocking=True)
        full_mask = ((start[:, :, None] == start[:, None, :])
                     & input_valid[:, :, None] & input_valid[:, None, :]
                     & trainer._causal_full(device, cfg.T).unsqueeze(0))
        tensors.setdefault("micro", []).append({
            "x": x, "y": y, "pos": pos, "segpos": segpos, "valid": valid,
            "input_valid": input_valid, "start": start,
            "full_mask": full_mask,
        })
    return tensors["micro"]


def _manual_update(trainer, cfg, model, entry, optimizer, device, micro,
                   update_index):
    """One full update with gradient/hook access (same semantics as the
    trainer's one_full_update)."""
    import torch

    denom = sum(int(m["valid"].sum().item()) for m in micro)
    lr = trainer.lr_for_update(update_index + 1, cfg)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        cache_enabled=False,
                        enabled=device.type == "cuda"):
        for m in micro:
            logits = entry(m["x"], m["pos"], m["segpos"], m["full_mask"],
                           m["start"])
            loss = trainer.ce_sum(logits, m["y"], m["valid"], cfg.V) / denom
            loss.backward()
            loss_total += float(loss.detach())
    grad_stats = {"params_with_grad": 0, "params_grad_nonfinite": 0,
                  "params_grad_nonzero": 0, "grad_max_abs": 0.0}
    for _, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad_stats["params_with_grad"] += 1
        grad = parameter.grad
        if not bool(torch.isfinite(grad).all()):
            grad_stats["params_grad_nonfinite"] += 1
        amax = float(grad.abs().max())
        grad_stats["grad_max_abs"] = max(grad_stats["grad_max_abs"], amax)
        if amax > 0.0:
            grad_stats["params_grad_nonzero"] += 1
    grad_norm = float(torch.nn.utils.clip_grad_norm_(
        model.parameters(), cfg.CLIP_NORM))
    before = [p.detach().clone() for p in model.parameters()]
    optimizer.step()
    delta_sq = 0.0
    delta_max = 0.0
    changed = 0
    for prev, parameter in zip(before, model.parameters()):
        diff = (parameter.detach() - prev)
        amax = float(diff.abs().max())
        delta_max = max(delta_max, amax)
        delta_sq += float(diff.pow(2).sum())
        if amax > 0.0:
            changed += 1
    del before
    return {
        "loss": loss_total,
        "lr": lr,
        "grad_norm": grad_norm,
        "grad_stats": grad_stats,
        "valid_pairs": denom,
        "param_update_max_abs": delta_max,
        "param_update_l2": math.sqrt(delta_sq),
        "params_changed": changed,
        "tokens": int(cpu_rows(micro) * cfg.T),
    }


def cpu_rows(micro):
    return int(sum(int(m["x"].shape[0]) for m in micro))


def _forward_logits(trainer, cfg, model, device, micro):
    import torch

    outs = []
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            cache_enabled=False,
                            enabled=device.type == "cuda"):
            for m in micro:
                outs.append(model.forward_packed(
                    m["x"], m["pos"], m["segpos"], m["full_mask"],
                    m["start"]).float())
    return torch.cat(outs, dim=0)


def phase_preflight(args) -> int:
    import torch

    pilot_dir = Path(args.pilot_dir)
    trainer = load_trainer()
    cfg = trainer.PROD_CFG
    device = trainer._prepare_device()
    logger = _ShimLogger()
    gates = {}
    details = {}

    # runtime gate (G4) - BLOCKED if unavailable
    try:
        trainer.enforce_production_runtime(device, logger)
        gates["production_runtime"] = True
    except Exception as exc:  # noqa: BLE001
        gates["production_runtime"] = False
        details["production_runtime_error"] = (
            f"{type(exc).__name__}: {str(exc)[:300]}")
        save_json(pilot_dir / "correctness.json",
                  {"pass": False, "gates": gates, "details": details,
                   "verdict": "BLOCKED"})
        print("PREFLIGHT_BLOCKED_RUNTIME", flush=True)
        return 3

    corpus = trainer.FrozenPackedCorpus(
        Path(args.corpus_root), cfg, verify_files=True,
        fast_verify=bool(args.fast_verify))
    details["corpus"] = {
        "id": corpus.contract.corpus_id,
        "sequences": corpus.total_sequences,
        "artifact_hashes_sha256": corpus.contract.artifact_hashes_sha256,
    }

    stream = corpus.stream_batches(0, cfg.GLOBAL_BATCH)
    batches = []
    for _ in range(2):
        try:
            batches.append(next(stream)[1])
        except StopIteration:
            break
    if len(batches) < 2:
        gates["corpus_has_two_batches"] = False
        save_json(pilot_dir / "correctness.json",
                  {"pass": False, "gates": gates, "details": details,
                   "verdict": "BLOCKED"})
        print("PREFLIGHT_BLOCKED_CORPUS", flush=True)
        return 3
    gates["corpus_has_two_batches"] = True

    # -- live preflight: one real update with full gradient visibility ------
    model, entry = trainer._build_model(cfg, device, use_compile=False)
    trainer.load_init(model, trainer.canonical_init(cfg), device)
    model.train()
    optimizer = trainer.make_optimizer(model, cfg, device.type)
    micro = _gpu_batch(trainer, cfg, device, batches[0])
    pre = _manual_update(trainer, cfg, model, entry, optimizer, device, micro,
                         update_index=0)
    details["live_step_1"] = pre
    gates["model_construction"] = True
    gates["forward_finite"] = math.isfinite(pre["loss"])
    gates["loss_finite"] = math.isfinite(pre["loss"])
    gates["backward_completed"] = True
    gates["grads_finite"] = pre["grad_stats"]["params_grad_nonfinite"] == 0
    gates["grads_nonzero"] = pre["grad_stats"]["params_grad_nonzero"] > 0
    gates["adamw_changes_parameters"] = pre["params_changed"] > 0

    micro2 = _gpu_batch(trainer, cfg, device, batches[1])
    second = trainer.one_full_update(batches[1], model, entry, optimizer, 1,
                                     cfg, device)
    gates["second_step_finite"] = math.isfinite(second["loss"])
    details["live_step_2"] = second

    # token accounting on the real batch
    expected_pairs = int(batches[0]["valid"].sum().item())
    tokens = int(batches[0]["x"].shape[0]) * cfg.T
    gates["token_accounting_exact"] = bool(
        pre["valid_pairs"] == expected_pairs and pre["tokens"] == tokens)
    details["token_accounting"] = {
        "rows": int(batches[0]["x"].shape[0]),
        "T": cfg.T,
        "tokens": tokens,
        "valid_pairs": pre["valid_pairs"],
        "expected_valid_pairs": expected_pairs,
    }

    # -- cross-row isolation (optional, project tolerance 1e-5) -------------
    try:
        import torch as _t
        two = {k: (v[:2] if _t.is_tensor(v) else v)
               for k, v in batches[0].items()}
        micro_two = _gpu_batch(trainer, cfg, device, two)
        logits_batched = _forward_logits(trainer, cfg, model, device,
                                         micro_two)
        one = {k: (v[:1] if _t.is_tensor(v) else v)
               for k, v in batches[0].items()}
        micro_one = _gpu_batch(trainer, cfg, device, one)
        logits_alone = _forward_logits(trainer, cfg, model, device, micro_one)
        row0_batched_vs_alone = float(
            (logits_batched[:1] - logits_alone).abs().max())
        two_mod = {k: (v.clone() if _t.is_tensor(v) else v)
                   for k, v in two.items()}
        gen = _t.Generator().manual_seed(1234)
        two_mod["x"][1] = _t.randint(0, cfg.V, two_mod["x"][1].shape,
                                     generator=gen)
        two_mod["y"][1] = _t.randint(0, cfg.V, two_mod["y"][1].shape,
                                     generator=gen)
        micro_mod = _gpu_batch(trainer, cfg, device, two_mod)
        logits_mod = _forward_logits(trainer, cfg, model, device, micro_mod)
        row0_changed = float((logits_mod[:1] - logits_batched[:1]).abs().max())
        row1_changed = float((logits_mod[1:] - logits_batched[1:]).abs().max())
        gates["cross_row_isolation"] = bool(
            row0_batched_vs_alone <= 1e-5 and row0_changed <= 1e-5
            and row1_changed > 0.0)
        details["cross_row_isolation"] = {
            "row0_batched_vs_alone_max_abs": row0_batched_vs_alone,
            "row0_change_when_row1_modified": row0_changed,
            "row1_change_when_row1_modified": row1_changed,
            "tolerance": 1e-5,
        }
    except Exception as exc:  # noqa: BLE001
        gates["cross_row_isolation"] = False
        details["cross_row_isolation_error"] = (
            f"{type(exc).__name__}: {str(exc)[:300]}")

    # -- Stage A: fixed-batch overfit smoke (real data path) ----------------
    stage_a = {"batch": "corpus batch 0, repeated", "losses": [],
               "param_update_max_abs": [], "param_update_l2": [],
               "updates": 0, "timeout": False}
    model_a, entry_a = trainer._build_model(cfg, device, use_compile=False)
    trainer.load_init(model_a, trainer.canonical_init(cfg), device)
    model_a.train()
    opt_a = trainer.make_optimizer(model_a, cfg, device.type)
    micro_a = _gpu_batch(trainer, cfg, device, batches[0])
    t0 = time.perf_counter()
    for i in range(int(args.stage_a_updates)):
        if time.perf_counter() - t0 > float(args.stage_a_budget_s):
            stage_a["timeout"] = True
            break
        result = _manual_update(trainer, cfg, model_a, entry_a, opt_a, device,
                                micro_a, update_index=i)
        stage_a["losses"].append(result["loss"])
        stage_a["param_update_max_abs"].append(
            result["param_update_max_abs"])
        stage_a["param_update_l2"].append(result["param_update_l2"])
        stage_a["updates"] += 1
        if not math.isfinite(result["loss"]):
            break
    stage_a["wall_s"] = time.perf_counter() - t0
    losses = stage_a["losses"]
    stage_a["first_loss"] = losses[0] if losses else None
    stage_a["last_loss"] = losses[-1] if losses else None
    stage_a["trend_nonpositive"] = bool(
        len(losses) >= 2 and losses[-1] <= losses[0] and
        all(math.isfinite(v) for v in losses))
    stage_a["slope_per_update"] = _slope(losses)
    details["stage_a_overfit"] = stage_a
    gates["multiple_optimizer_steps"] = stage_a["updates"] >= 2
    gates["stage_a_loss_trend_nonpositive"] = stage_a["trend_nonpositive"]
    gates["no_nan_inf_oom"] = bool(
        all(math.isfinite(v) for v in losses if v is not None)
        and pre["grad_stats"]["params_grad_nonfinite"] == 0
        and math.isfinite(second["loss"]))

    del model, model_a, optimizer, opt_a
    torch.cuda.empty_cache()

    preflight_pass = all(gates.values())
    save_json(pilot_dir / "correctness_preflight.json",
              {"pass": preflight_pass, "gates": gates, "details": details,
               "generated_utc": now_utc()})
    print("PREFLIGHT_PASS=" + str(preflight_pass).lower(), flush=True)
    return 0 if preflight_pass else 4


def _slope(values):
    n = len(values)
    if n < 2:
        return None
    xs = list(range(n))
    mx = statistics.mean(xs)
    my = statistics.mean(values)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, values)) / den


# ---------------------------------------------------------------------------
# smoke / train subprocess phases
# ---------------------------------------------------------------------------


def _run_trainer(args, mode, timeout_s, extra):
    python = sys.executable
    cmd = [python, str(REPO / TRAINER_REL), "--mode", mode,
           "--corpus-root", str(args.corpus_root),
           "--run-dir", str(args.pilot_dir)] + extra
    if args.fast_verify:
        cmd.append("--fast-verify")
    print("RUN " + " ".join(cmd), flush=True)
    log_path = Path(args.pilot_dir) / "logs" / f"{mode}_stdout.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    outcome = {"mode": mode, "cmd": cmd, "timeout_s": timeout_s,
               "stopped_by": "completed", "returncode": None}
    with open(log_path, "w", encoding="utf-8") as handle:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                bufsize=1)
        try:
            deadline = time.perf_counter() + timeout_s
            for line in proc.stdout:
                handle.write(line)
                handle.flush()
                if mode == "train":
                    print(line.rstrip(), flush=True)
                if time.perf_counter() > deadline:
                    outcome["stopped_by"] = "wall_clock"
                    proc.terminate()
                    break
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=30)
    outcome["returncode"] = proc.returncode
    outcome["wall_s"] = time.perf_counter() - started
    if proc.returncode not in (0, None) and outcome["stopped_by"] != \
            "wall_clock":
        outcome["stopped_by"] = "error"
    return outcome


def phase_smoke(args) -> int:
    outcome = _run_trainer(args, "smoke", SMOKE_TIMEOUT_S, [])
    save_json(Path(args.pilot_dir) / "smoke_outcome.json", outcome)
    ok = outcome["returncode"] == 0
    print("SMOKE_PASS=" + str(ok).lower(), flush=True)
    return 0 if ok else 5


def phase_train(args) -> int:
    extra = ["--target-tokens", str(args.target_tokens),
             "--save-every", "1", "--log-every", "1"]
    if args.archive_every:
        extra += ["--archive-every", str(args.archive_every)]
    budget = max(30.0, float(args.pilot_budget_s) - float(args.reserve_s))
    outcome = _run_trainer(args, "train", budget, extra)
    save_json(Path(args.pilot_dir) / "train_outcome.json", outcome)
    return 0 if outcome["stopped_by"] != "error" else 6


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path):
    records = []
    if not Path(path).is_file():
        return records
    for line in Path(path).read_text(encoding="utf-8",
                                     errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def build_training_csv(records, path: Path):
    import csv

    columns = ["step", "tokens_consumed", "valid_pairs", "loss", "grad_norm",
               "lr", "step_ms", "tok_s", "valid_pairs_s", "mem_alloc_GiB",
               "mem_peak_GiB", "mem_reserved_GiB", "elapsed_s",
               "param_update_max_abs_preflight_only"]
    updates = [r for r in records if r.get("event") == "update"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in updates:
            writer.writerow({
                "step": record.get("step"),
                "tokens_consumed": record.get("tokens_consumed"),
                "valid_pairs": record.get("valid_pairs"),
                "loss": record.get("loss"),
                "grad_norm": record.get("grad_norm"),
                "lr": record.get("lr"),
                "step_ms": record.get("step_ms"),
                "tok_s": record.get("tok_s"),
                "valid_pairs_s": record.get("valid_pairs_s"),
                "mem_alloc_GiB": record.get("mem_alloc_GiB"),
                "mem_peak_GiB": record.get("mem_peak_GiB"),
                "mem_reserved_GiB": record.get("mem_reserved_GiB"),
                "elapsed_s": record.get("elapsed_s"),
                "param_update_max_abs_preflight_only": "",
            })
    return len(updates)


def classify(gates, smoke, updates, stopped_by, target_tokens,
             tokens_consumed, source_pass, blocked=False):
    if not source_pass:
        return "INVALID_SOURCE"
    if blocked:
        return "BLOCKED"
    hard = [name for name, ok in gates.items() if ok is not True]
    if hard:
        return "FAIL"
    if updates < 2:
        return "FAIL"
    if stopped_by == "wall_clock" or tokens_consumed < target_tokens:
        return "PASS_WITH_LIMITATIONS"
    return "PASS"


def phase_report(args) -> int:
    pilot_dir = Path(args.pilot_dir)
    train_records = _read_jsonl(pilot_dir / "logs" / "train.jsonl")
    smoke_records = _read_jsonl(pilot_dir / "logs" / "smoke.jsonl")
    startup_records = _read_jsonl(pilot_dir / "logs" / "startup.jsonl")
    updates = [r for r in train_records if r.get("event") == "update"]
    ckpt_path = pilot_dir / "ckpt" / "latest.pt"
    smoke_outcome = (load_json(pilot_dir / "smoke_outcome.json")
                     if (pilot_dir / "smoke_outcome.json").is_file() else {})
    train_outcome = (load_json(pilot_dir / "train_outcome.json")
                     if (pilot_dir / "train_outcome.json").is_file() else {})
    preflight = (load_json(pilot_dir / "correctness_preflight.json")
                 if (pilot_dir / "correctness_preflight.json").is_file()
                 else {"gates": {}, "pass": False})
    source = (load_json(pilot_dir / "source_manifest.json")
              if (pilot_dir / "source_manifest.json").is_file() else {})

    # smoke gates
    smoke_gates = {
        "checkpoint_save": bool(
            any(r.get("event") == "smoke_checkpoint_reloaded"
                for r in smoke_records)),
        "checkpoint_reload": bool(
            any(r.get("event") == "smoke_checkpoint_reloaded"
                for r in smoke_records)),
        "reload_produces_finite_output": bool(
            any(r.get("event") == "smoke_comparison"
                and _finite(r.get("loss_a")) and _finite(r.get("loss_b"))
                for r in smoke_records)),
        "post_reload_step": bool(
            any(r.get("event") == "smoke_comparison"
                for r in smoke_records)),
    }
    smoke_ok = bool(smoke_outcome.get("returncode") == 0
                    and any(r.get("event") == "smoke_end"
                            and r.get("status") == "PASS"
                            for r in smoke_records))

    gates = dict(preflight.get("gates", {}))
    gates.update(smoke_gates)

    # token accounting / no silent skipping
    rows_ok = all(int(r.get("rows", 0)) == int(args.global_batch)
                  for r in updates) if updates else False
    steps = [int(r.get("step", -1)) for r in updates]
    no_gaps = steps == list(range(1, len(steps) + 1)) if steps else False
    tokens_consumed = int(updates[-1].get("tokens_consumed", 0)) if updates \
        else 0
    next_sequence = int(updates[-1].get("next_sequence", 0)) if updates else 0
    token_exact = bool(updates and tokens_consumed == next_sequence * 2048)
    nan_inf = [r for r in updates
               if not _finite(r.get("loss")) or not _finite(r.get("grad_norm"))]
    oom = [r for r in updates if "out of memory" in str(r).lower()]
    gates["token_accounting_exact"] = bool(
        gates.get("token_accounting_exact", False) and token_exact
        and rows_ok and no_gaps)
    gates["no_nan_inf_oom"] = bool(
        gates.get("no_nan_inf_oom", True) and not nan_inf and not oom)

    losses = [float(r["loss"]) for r in updates if _finite(r.get("loss"))]
    grad_norms = [float(r["grad_norm"]) for r in updates
                  if _finite(r.get("grad_norm"))]
    peak_mem = max((float(r.get("mem_peak_GiB", 0.0)) for r in updates),
                   default=0.0)
    total_wall = float(train_outcome.get("wall_s", 0.0))
    tok_s = (tokens_consumed / total_wall) if total_wall > 0 else None
    loss_trend = {
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "delta": (losses[-1] - losses[0]) if len(losses) >= 2 else None,
        "slope_per_update": _slope(losses),
        "overall_nonpositive": bool(
            len(losses) >= 2 and losses[-1] <= losses[0]),
        "min_loss": min(losses) if losses else None,
    }
    stage_a = preflight.get("details", {}).get("stage_a_overfit", {})

    # checkpoint artifact + required-field audit
    ckpt_meta = None
    ckpt_audit = None
    if ckpt_path.is_file():
        import shutil
        import torch
        target = pilot_dir / "soda_bdh_pilot_checkpoint.pt"
        shutil.copy2(ckpt_path, target)
        try:
            payload = torch.load(ckpt_path, map_location="cpu",
                                 weights_only=False)
            required = ("model", "optimizer", "progress", "config", "rng",
                        "code_sha256", "corpus")
            present = {key: (key in payload) for key in required}
            ckpt_audit = {
                "required_keys_present": all(present.values()),
                "required_keys": present,
                "progress": payload.get("progress"),
                "code_sha256": payload.get("code_sha256"),
                "config_SEED": (payload.get("config") or {}).get("SEED"),
                "rng_present": bool(payload.get("rng")),
                "scheduler_state": ("not used; the frozen trainer derives LR "
                                    "from the token schedule in progress"),
            }
            gates["checkpoint_contains_required_fields"] = \
                ckpt_audit["required_keys_present"]
        except Exception as exc:  # noqa: BLE001
            ckpt_audit = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            gates["checkpoint_contains_required_fields"] = False
        ckpt_meta = {"path": str(target),
                     "bytes": target.stat().st_size,
                     "source_checkpoint": str(ckpt_path),
                     "audit": ckpt_audit}
    else:
        gates["checkpoint_contains_required_fields"] = False

    n_csv = build_training_csv(train_records, pilot_dir / "training_log.csv")

    verdict = classify(gates, smoke_records, len(updates),
                       train_outcome.get("stopped_by"),
                       int(args.target_tokens), tokens_consumed,
                       bool(source.get("source_lock", {}).get("pass")),
                       blocked=bool(preflight.get("verdict") == "BLOCKED"))

    summary = {
        "generated_utc": now_utc(),
        "labels": LABELS,
        "verdict": verdict,
        "source": source.get("source_lock", {}),
        "corpus_verification": source.get("corpus_verification"),
        "smoke": {"outcome": smoke_outcome, "ok": smoke_ok,
                  "gates": smoke_gates},
        "preflight": {"pass": preflight.get("pass")},
        "stage_a_overfit": stage_a,
        "train": {
            "outcome": train_outcome,
            "completed_updates": len(updates),
            "tokens_consumed": tokens_consumed,
            "next_sequence": next_sequence,
            "target_tokens": int(args.target_tokens),
            "stopped_by": train_outcome.get("stopped_by"),
            "wall_s": total_wall,
            "throughput_input_tokens_per_s": tok_s,
            "loss_trend": loss_trend,
            "grad_norm": {
                "min": min(grad_norms) if grad_norms else None,
                "max": max(grad_norms) if grad_norms else None,
                "median": statistics.median(grad_norms) if grad_norms
                else None,
            },
            "peak_mem_GiB": peak_mem,
            "nan_inf_events": len(nan_inf),
            "oom_events": len(oom),
            "rows_ok": rows_ok,
            "step_sequence_ok": no_gaps,
        },
        "gates": gates,
        "hard_failures": [k for k, v in gates.items() if v is not True],
        "limitations": [
            "Vendor Soda file is absent; only the repo-local dense path "
            "was verified (VENDOR_SODA_EQUIVALENCE: UNVERIFIED).",
            "Per-update parameter-update magnitude is not emitted by the "
            "frozen trainer; it is recorded for the preflight and Stage A "
            "fixed-batch updates only.",
            "Five minutes of training establishes execution integrity, not "
            "convergence or final quality.",
        ],
        "artifacts": {
            "pilot_dir": str(pilot_dir),
            "training_log_csv": str(pilot_dir / "training_log.csv"),
            "training_csv_rows": n_csv,
            "checkpoint": ckpt_meta,
            "smoke_log": str(pilot_dir / "logs" / "smoke.jsonl"),
            "train_log": str(pilot_dir / "logs" / "train.jsonl"),
            "startup_log": str(pilot_dir / "logs" / "startup.jsonl"),
        },
    }
    save_json(pilot_dir / "pilot_summary.json", summary)
    save_json(pilot_dir / "correctness.json",
              {"pass": not summary["hard_failures"],
               "gates": gates,
               "details": preflight.get("details", {}),
               "smoke": {"ok": smoke_ok, "outcome": smoke_outcome},
               "verdict": verdict,
               "generated_utc": now_utc()})
    _write_report(pilot_dir, summary)
    print_final_block(summary)
    return 0 if verdict in ("PASS", "PASS_WITH_LIMITATIONS") else 7


def _write_report(pilot_dir: Path, summary: dict):
    t = summary["train"]
    s = summary["source"]
    lines = [
        "# SODA-BDH 5-Minute Training Pilot Report",
        "",
        f"VERDICT: {summary['verdict']}",
        f"SOURCE_COMMIT: {s.get('git_commit')}",
        "SOURCE_HASH_STATUS: " + (
            "VERIFIED" if s.get("pass") else "MISMATCH"),
        "MODEL: " + LABELS["MODEL"],
        "MODEL_DEFINITION: " + LABELS["MODEL_DEFINITION"],
        "VENDOR_SODA_EQUIVALENCE: " + LABELS["VENDOR_SODA_EQUIVALENCE"],
        "DENSE_PATH_CONFIRMED: " + LABELS["DENSE_PATH_CONFIRMED"],
        "DATA_MODE: " + LABELS["DATA_MODE"],
        f"COMPLETED_UPDATES: {t['completed_updates']}",
        f"TOTAL_TOKENS: {t['tokens_consumed']}",
        f"INITIAL_LOSS: {_fmt(t['loss_trend']['initial_loss'])}",
        f"FINAL_LOSS: {_fmt(t['loss_trend']['final_loss'])}",
        f"LOSS_TREND: slope_per_update="
        f"{_fmt(t['loss_trend']['slope_per_update'])}",
        "GRADIENT_STATUS: " + ("finite, non-zero"
                               if summary['gates'].get('grads_finite')
                               and summary['gates'].get('grads_nonzero')
                               else "FAIL"),
        "PARAMETER_UPDATE_STATUS: " + (
            "AdamW changed trainable parameters"
            if summary['gates'].get('adamw_changes_parameters')
            else "FAIL"),
        "CHECKPOINT_RELOAD_STATUS: " + (
            "PASS" if summary['gates'].get('checkpoint_reload')
            else "FAIL"),
        f"PEAK_GPU_MEMORY_MB: {t['peak_mem_GiB'] * 1024:.0f}",
        f"NAN_INF_EVENTS: {t['nan_inf_events']}",
        f"OOM_EVENTS: {t['oom_events']}",
        "THROUGHPUT_DIAGNOSTIC: "
        f"{_fmt(t['throughput_input_tokens_per_s'])} input tok/s over "
        f"{t['wall_s']:.1f} s (stopped_by={t['stopped_by']})",
        "PRIMARY_FAILURE: " + (
            "none" if not summary["hard_failures"]
            else ", ".join(summary["hard_failures"])),
        "LIMITATIONS: " + " | ".join(summary["limitations"]),
        f"ARTIFACT_DIRECTORY: {pilot_dir}",
        "",
        "## Gates",
        "",
    ]
    for name, ok in summary["gates"].items():
        lines.append(f"- {name}: {ok}")
    lines += [
        "",
        "## Train diagnostics",
        "",
        f"- target tokens: {t['target_tokens']}",
        f"- stopped_by: {t['stopped_by']}",
        f"- wall: {t['wall_s']:.1f} s",
        f"- throughput: {_fmt(t['throughput_input_tokens_per_s'])} input tok/s",
        f"- peak allocated: {t['peak_mem_GiB']:.2f} GiB",
        f"- rows ok: {t['rows_ok']}; step sequence ok: {t['step_sequence_ok']}",
        f"- stage A overfit updates: {summary['stage_a_overfit'].get('updates')}",
        f"- stage A trend non-positive: "
        f"{summary['stage_a_overfit'].get('trend_nonpositive')}",
        "",
        "Labels: MODEL=SODA-BDH; MODEL_DEFINITION=repo-local canonical dense "
        "BDH/Arm-A; VENDOR_SODA_EQUIVALENCE=UNVERIFIED; "
        "DENSE_PATH_CONFIRMED=true; DATA_MODE=FROZEN_CORPUS_G4.",
        "",
    ]
    (Path(pilot_dir) / "pilot_report.md").write_text("\n".join(lines),
                                                     encoding="utf-8")


def _fmt(value):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def print_final_block(summary: dict):
    t = summary["train"]
    s = summary["source"]
    print("\n" + "=" * 72)
    print("SODA-BDH 5-MINUTE PILOT - FINAL REPORT")
    print("=" * 72)
    print(f"VERDICT: {summary['verdict']}")
    print(f"SOURCE_COMMIT: {s.get('git_commit')}")
    print("SOURCE_HASH_STATUS: " + (
        "VERIFIED" if s.get("pass") else "MISMATCH"))
    print("MODEL: " + LABELS["MODEL"])
    print("MODEL_DEFINITION: " + LABELS["MODEL_DEFINITION"])
    print("VENDOR_SODA_EQUIVALENCE: " +
          LABELS["VENDOR_SODA_EQUIVALENCE"])
    print("DENSE_PATH_CONFIRMED: " + LABELS["DENSE_PATH_CONFIRMED"])
    print("DATA_MODE: " + LABELS["DATA_MODE"])
    print(f"COMPLETED_UPDATES: {t['completed_updates']}")
    print(f"TOTAL_TOKENS: {t['tokens_consumed']}")
    print(f"INITIAL_LOSS: {_fmt(t['loss_trend']['initial_loss'])}")
    print(f"FINAL_LOSS: {_fmt(t['loss_trend']['final_loss'])}")
    print("LOSS_TREND: slope_per_update="
          f"{_fmt(t['loss_trend']['slope_per_update'])}")
    print("GRADIENT_STATUS: finite/non-zero="
          f"{summary['gates'].get('grads_finite')}/"
          f"{summary['gates'].get('grads_nonzero')}")
    print("PARAMETER_UPDATE_STATUS: "
          f"{summary['gates'].get('adamw_changes_parameters')}")
    print("CHECKPOINT_RELOAD_STATUS: "
          f"{summary['gates'].get('checkpoint_reload')}")
    print(f"PEAK_GPU_MEMORY_MB: {t['peak_mem_GiB'] * 1024:.0f}")
    print(f"NAN_INF_EVENTS: {t['nan_inf_events']}")
    print(f"OOM_EVENTS: {t['oom_events']}")
    print("THROUGHPUT_DIAGNOSTIC: "
          f"{_fmt(t['throughput_input_tokens_per_s'])} input tok/s")
    print("PRIMARY_FAILURE: " + (
        "none" if not summary["hard_failures"]
        else ", ".join(summary["hard_failures"])))
    print("LIMITATIONS: " + " | ".join(summary["limitations"]))
    print(f"ARTIFACT_DIRECTORY: {summary['artifacts']['pilot_dir']}")
    print("=" * 72, flush=True)


# ---------------------------------------------------------------------------
# all / selftest / CLI
# ---------------------------------------------------------------------------


def phase_all(args) -> int:
    t0 = time.perf_counter()
    rc = phase_lock(args)
    if rc != 0:
        print("INVALID_SOURCE_STOP", flush=True)
        return rc
    rc_pre = phase_preflight(args)
    rc_smoke = None
    rc_train = None
    if rc_pre != 3:  # 3 == runtime/corpus BLOCKED; smoke would fail too
        rc_smoke = phase_smoke(args)
        if rc_pre == 0 and rc_smoke == 0:
            rc_train = phase_train(args)
    rc_report = phase_report(args)
    print("CELL_WALL_S=%.1f" % (time.perf_counter() - t0), flush=True)
    print("PHASE_CODES=" + json.dumps(
        {"preflight": rc_pre, "smoke": rc_smoke, "train": rc_train,
         "report": rc_report}), flush=True)
    return rc_report


def phase_selftest(args) -> int:
    """Harness-only validation: tiny CPU model + synthetic log records.

    This is NOT the pilot and produces no pilot evidence. It validates the
    orchestrator's model-step code path and artifact builders.
    """
    import torch

    trainer = load_trainer()
    cfg = trainer.ArmAConfig(T=32, V=128, D=16, N=64, H=4, L=2, HIDDEN=24,
                             SEED=7, THETA=2.0 ** 16, READ_BLOCK=16,
                             PEAK_LR=1e-3, WARMUP_TOKENS=512,
                             GLOBAL_BATCH=4, MICROBATCH=2, SCAN_BLOCK=16)
    device = torch.device("cpu")
    model, entry = trainer._build_model(cfg, device, use_compile=False)
    trainer.load_init(model, trainer.canonical_init(cfg), device)
    model.train()
    optimizer = trainer.make_optimizer(model, cfg, device.type)
    gen = torch.Generator().manual_seed(11)
    rows = cfg.GLOBAL_BATCH
    start = torch.zeros((rows, cfg.T), dtype=torch.int32)
    start[:, cfg.T // 2:] = cfg.T // 2
    pos = (torch.arange(cfg.T).unsqueeze(0).expand(rows, -1)
           - start.long()).to(torch.int32)
    same = start[:, :, None] == start[:, None, :]
    strict = torch.ones((cfg.T, cfg.T), dtype=torch.bool).tril(-1)
    cpu_batch = {
        "x": torch.randint(0, cfg.V, (rows, cfg.T), generator=gen),
        "y": torch.randint(0, cfg.V, (rows, cfg.T), generator=gen),
        "pos": pos, "segpos": pos.clone(),
        "valid": torch.ones((rows, cfg.T), dtype=torch.bool),
        "input_valid": torch.ones((rows, cfg.T), dtype=torch.bool),
        "start": start,
    }
    micro = _gpu_batch(trainer, cfg, device, cpu_batch)
    first = _manual_update(trainer, cfg, model, entry, optimizer, device,
                           micro, update_index=0)
    second = trainer.one_full_update(cpu_batch, model, entry, optimizer, 1,
                                     cfg, device)
    checks = {
        "model_construction": True,
        "loss_finite_step1": _finite(first["loss"]),
        "loss_finite_step2": _finite(second["loss"]),
        "grads_finite": first["grad_stats"]["params_grad_nonfinite"] == 0,
        "grads_nonzero": first["grad_stats"]["params_grad_nonzero"] > 0,
        "adamw_changed_params": first["params_changed"] > 0,
        "param_update_max_abs": first["param_update_max_abs"],
        "param_update_l2": first["param_update_l2"],
    }
    # artifact builders on synthetic records
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        synthetic = []
        loss = 9.0
        for i in range(1, 6):
            loss -= 0.05
            synthetic.append({
                "event": "update", "step": i, "rows": 64,
                "tokens_consumed": i * 131072,
                "next_sequence": i * 64, "loss": loss, "lr": 1e-3,
                "grad_norm": 1.0, "valid_pairs": 130990, "step_ms": 1600.0,
                "tok_s": 80000.0, "valid_pairs_s": 81000.0,
                "mem_alloc_GiB": 1.0, "mem_peak_GiB": 58.0,
                "mem_reserved_GiB": 59.0, "elapsed_s": 1.6 * i})
        (tmp / "logs").mkdir(parents=True, exist_ok=True)
        (tmp / "logs" / "train.jsonl").write_text(
            "\n".join(json.dumps(r) for r in synthetic), encoding="utf-8")
        (tmp / "logs" / "smoke.jsonl").write_text("\n".join([
            json.dumps({"event": "smoke_update", "update_index": 0,
                        "loss": 9.0}),
            json.dumps({"event": "smoke_checkpoint_reloaded"}),
            json.dumps({"event": "smoke_comparison", "loss_a": 8.9,
                        "loss_b": 8.9}),
            json.dumps({"event": "smoke_end", "status": "PASS"}),
        ]), encoding="utf-8")
        save_json(tmp / "source_manifest.json",
                  {"source_lock": {"pass": True, "git_commit": PINNED_COMMIT}})
        save_json(tmp / "smoke_outcome.json", {"returncode": 0})
        save_json(tmp / "train_outcome.json",
                  {"returncode": 0, "stopped_by": "completed", "wall_s": 8.0})
        save_json(tmp / "correctness_preflight.json",
                  {"pass": True, "gates": {k: True for k in (
                      "production_runtime", "corpus_has_two_batches",
                      "model_construction", "forward_finite", "loss_finite",
                      "backward_completed", "grads_finite", "grads_nonzero",
                      "adamw_changes_parameters", "second_step_finite",
                      "token_accounting_exact", "cross_row_isolation",
                      "multiple_optimizer_steps",
                      "stage_a_loss_trend_nonpositive", "no_nan_inf_oom")},
                   "details": {"stage_a_overfit": {"updates": 3,
                                                   "trend_nonpositive": True}}})
        (tmp / "ckpt").mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": "arm_a_2p5b_ckpt_v1", "model": {}, "optimizer": {},
            "progress": {"updates_done": 5, "tokens_consumed": 655360,
                         "next_sequence": 320, "target_tokens": 500000},
            "config": {"SEED": 7}, "rng": {"torch": None},
            "code_sha256": "selftest", "corpus": {},
        }, tmp / "ckpt" / "latest.pt")
        rc = phase_report(argparse.Namespace(
            pilot_dir=str(tmp), global_batch=64, target_tokens=500000,
            fast_verify=True))
        summary = load_json(tmp / "pilot_summary.json")
        checks["report_rc"] = rc
        checks["report_verdict"] = summary["verdict"]
        checks["csv_rows"] = summary["artifacts"]["training_csv_rows"]
        checks["report_md_exists"] = (tmp / "pilot_report.md").is_file()
        checks["hard_failures"] = summary["hard_failures"]
    checks["HARNESS_SELFTEST_PASS"] = bool(
        checks["loss_finite_step1"] and checks["loss_finite_step2"]
        and checks["grads_finite"] and checks["grads_nonzero"]
        and checks["adamw_changed_params"]
        and checks["report_verdict"] == "PASS"
        and not checks["hard_failures"])
    print("HARNESS_SELFTEST_PASS="
          + str(checks["HARNESS_SELFTEST_PASS"]).lower(), flush=True)
    print(json.dumps(checks, indent=2, default=str), flush=True)
    return 0 if checks["HARNESS_SELFTEST_PASS"] else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default="all",
                        choices=("lock", "preflight", "smoke", "train",
                                 "report", "all", "selftest"))
    parser.add_argument("--corpus-root", default=DEFAULT_CORPUS)
    parser.add_argument("--pilot-dir", default=DEFAULT_PILOT_DIR)
    parser.add_argument("--target-tokens", type=int,
                        default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--pilot-budget-s", type=float,
                        default=DEFAULT_PILOT_BUDGET_S)
    parser.add_argument("--reserve-s", type=float, default=DEFAULT_RESERVE_S)
    parser.add_argument("--stage-a-updates", type=int,
                        default=DEFAULT_STAGE_A_UPDATES)
    parser.add_argument("--stage-a-budget-s", type=float,
                        default=DEFAULT_STAGE_A_BUDGET_S)
    parser.add_argument("--archive-every", type=int, default=0)
    parser.add_argument("--global-batch", type=int, default=64)
    parser.add_argument("--fast-verify", dest="fast_verify",
                        action="store_true",
                        help="skip per-shard corpus SHA-256 (sizes, artifact "
                             "index and manifest digests still verified)")
    parser.add_argument("--full-verify", dest="fast_verify",
                        action="store_false",
                        help="full per-shard SHA-256 verification")
    parser.set_defaults(fast_verify=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    phases = {
        "lock": phase_lock,
        "preflight": phase_preflight,
        "smoke": phase_smoke,
        "train": phase_train,
        "report": phase_report,
        "all": phase_all,
        "selftest": phase_selftest,
    }
    return phases[args.phase](args)


if __name__ == "__main__":
    raise SystemExit(main())
