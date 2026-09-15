"""Standalone trained-checkpoint sparsity census (campaign Phase 1).

Runs on a G4 (or any CUDA/CPU box) AFTER the Arm-A trainer is not using the
GPU. It only reads the checkpoint and corpus; nothing is written except the
output JSON.

Examples:
  python run_trained_sparsity_census.py \
      --checkpoint "/content/drive/.../ckpt/step_0000002000.pt" \
      --corpus-root "/content/drive/.../frozen_5b_v1" \
      --output "/content/drive/.../sparsity_census_262m.json" \
      --num-global-batches 64

  # identical sampled data at 2.5B
  python run_trained_sparsity_census.py \
      --checkpoint "/content/drive/.../ckpt/step_0000019074.pt" \
      --ranges-from "/content/drive/.../sparsity_census_262m.json" \
      --output "/content/drive/.../sparsity_census_2p5b.json"

Never run concurrently with the production trainer on the same GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from opt.census_metrics import TrainedCensusAccumulator
from opt.census_model import CensusArmA
from opt.model_ref import ArmAConfig

CKPT_FORMAT = "arm_a_2p5b_ckpt_v1"
BATCH_SIZE = 64
TOKENS_PER_UPDATE = 64 * 2048

PROD_CKPT_STEP2000 = (
    "/content/drive/Shareddrives/ICLR PHASE BDH/phase_bdh/"
    "runs/arm_a_2p5b_opt3c_all/ckpt/step_0000002000.pt"
)
PROD_CORPUS = (
    "/content/drive/Shareddrives/ICLR PHASE BDH/phase_bdh/"
    "corpus/stage2/frozen_5b_v1"
)
GRADES = {
    (2000, 262_144_000): "trained_early_262m",
    (19074, 2_500_067_328): "trained_final_2p5b",
}


class CensusError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise CensusError(message)


def sha256_file(path: Path, block: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sample_ranges(total_sequences: int, num_batches: int, regions: int,
                  seed: int, batch_size: int = BATCH_SIZE):
    """Deterministic stratified sequence ranges across the corpus."""
    require(num_batches >= 1 and regions >= 1, "bad sampling request")
    require(total_sequences >= batch_size, "corpus smaller than one batch")
    regions = min(regions, max(1, total_sequences // batch_size))
    rng = random.Random(seed)
    base = num_batches // regions
    extra = num_batches - base * regions
    span = total_sequences // regions
    ranges = []
    for region in range(regions):
        lo = region * span
        hi = min(lo + span, total_sequences)
        want = base + (1 if region < extra else 0)
        picks = []
        attempts = 0
        while len(picks) < want and attempts < 200000:
            attempts += 1
            start = rng.randrange(lo, hi - batch_size + 1)
            if all(abs(start - other) >= batch_size for other in picks):
                picks.append(start)
        require(len(picks) == want,
                f"could not sample {want} disjoint batches in region "
                f"{region} (span {lo}:{hi})")
        ranges.extend((start, start + batch_size) for start in sorted(picks))
    ranges.sort()
    return ranges


def contiguous_ranges(start_sequence: int, num_batches: int,
                      total_sequences: int,
                      batch_size: int = BATCH_SIZE):
    require(start_sequence >= 0, "start-sequence must be >= 0")
    require(start_sequence + num_batches * batch_size <= total_sequences,
            "contiguous range exceeds the corpus")
    return [(start_sequence + i * batch_size,
             start_sequence + (i + 1) * batch_size)
            for i in range(num_batches)]


def ranges_from_file(path: Path, total_sequences: int,
                     batch_size: int = BATCH_SIZE):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    ranges = [(int(a), int(b)) for a, b in payload["ranges"]]
    require(ranges, "ranges file is empty")
    for start, end in ranges:
        require(end - start == batch_size,
                f"range width mismatch in {path}: {start}:{end}")
        require(0 <= start < end <= total_sequences,
                f"range out of bounds in {path}: {start}:{end}")
    return ranges


def inspect_checkpoint(ckpt_path: Path, expected_config: dict,
                       corpus_artifact_sha256: str, total_sequences: int,
                       code_path: Path = None) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    require(isinstance(ckpt, dict), "checkpoint is not a dict payload")
    require(ckpt.get("format") == CKPT_FORMAT,
            f"checkpoint format {ckpt.get('format')!r} != {CKPT_FORMAT}")
    config = ckpt.get("config", {})
    mismatched = {
        key: (config.get(key), value) for key, value in expected_config.items()
        if config.get(key) != value
    }
    require(not mismatched,
            f"checkpoint config mismatch: {mismatched}")
    corpus = ckpt.get("corpus", {})
    require(corpus.get("artifact_hashes_sha256") == corpus_artifact_sha256,
            "checkpoint corpus artifact hash differs from the corpus")
    require(int(corpus.get("total_sequences", -1)) == total_sequences,
            "checkpoint corpus sequence count differs from the corpus")
    progress = ckpt.get("progress", {})
    step = int(progress.get("updates_done", -1))
    tokens = int(progress.get("tokens_consumed", -1))
    cursor = int(progress.get("next_sequence", -1))
    require(step >= 0 and tokens >= 0, "checkpoint progress missing")
    require(tokens == cursor * 2048,
            "checkpoint token/cursor accounting inconsistent")
    code_checked = False
    if code_path is not None and Path(code_path).is_file():
        local = sha256_file(Path(code_path))
        remote = ckpt.get("code_sha256")
        if remote:
            local_bytes = Path(code_path).read_bytes().replace(b"\r\n", b"\n")
            local = hashlib.sha256(local_bytes).hexdigest()
            require(local == remote,
                    "checkpoint code fingerprint differs from the local "
                    "trainer source; refusing to census the wrong code")
            code_checked = True
    require("model" in ckpt, "checkpoint has no model state")
    grade = GRADES.get((step, tokens), f"trained_custom_step_{step}")
    return {
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "checkpoint_step": step,
        "checkpoint_input_tokens": tokens,
        "checkpoint_cursor": cursor,
        "evidence_grade": grade,
        "code_fingerprint_checked": code_checked,
        "state_dict": ckpt["model"],
    }


def load_corpus_module():
    path = SCRIPT_ROOT / "training" / "arm_a_2p5b_trainer.py"
    require(path.is_file(),
            f"packed-row reconstruction source missing: {path}")
    spec = importlib.util.spec_from_file_location("arm_a_trainer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["arm_a_trainer"] = module
    spec.loader.exec_module(module)
    return module


def _prepare_batch(batch, lo, hi, device):
    x = batch["x"][lo:hi].to(device, dtype=torch.long)
    pos = batch["pos"][lo:hi].to(device, dtype=torch.int32)
    segpos = batch["segpos"][lo:hi].to(device, dtype=torch.int32)
    start = batch.get("start", batch.get("segment_start"))[lo:hi].to(
        device, dtype=torch.int32)
    if "input_valid" in batch:
        input_valid = batch["input_valid"][lo:hi].to(device, dtype=torch.bool)
        causal = torch.ones((x.shape[1], x.shape[1]), dtype=torch.bool,
                            device=device).tril(diagonal=-1)
        full_mask = (
            (start[:, :, None] == start[:, None, :])
            & input_valid[:, :, None]
            & input_valid[:, None, :]
            & causal.unsqueeze(0)
        )
    else:
        full_mask = batch["full_mask"][lo:hi].to(device)
    return x, pos, segpos, full_mask, start


@torch.no_grad()
def run_census_batches(model, batch_iter, device, microbatch, use_bf16,
                       fp32_probe_batches, accumulator, seed, log_every=10):
    """Stream batches (CPU dicts) through the model and accumulate metrics."""
    fp32_acc = TrainedCensusAccumulator(model.cfg)
    bf16_probe_acc = TrainedCensusAccumulator(model.cfg)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                               cache_enabled=False)
                if use_bf16 else nullcontext())
    rows = 0
    batches = 0
    for batch_index, batch in enumerate(batch_iter):
        total = int(batch["x"].shape[0])
        probe = batch_index < fp32_probe_batches
        for lo in range(0, total, microbatch):
            hi = min(lo + microbatch, total)
            rows_here = hi - lo
            x, pos, segpos, full_mask, start = _prepare_batch(
                batch, lo, hi, device)
            accumulator.new_batch(rows_here, generator)
            model.begin_forward(
                lambda level, xx, yy, uu, ss: accumulator.add(
                    level, xx, yy, uu, ss)
            )
            with autocast:
                model.forward_packed(x, pos, segpos, full_mask, start)
            if probe:
                bf16_probe_acc.new_batch(rows_here, generator)
                model.begin_forward(
                    lambda level, xx, yy, uu, ss: bf16_probe_acc.add(
                        level, xx, yy, uu, ss)
                )
                model.forward_packed(x, pos, segpos, full_mask, start)
                fp32_acc.new_batch(rows_here, generator)
                model.begin_forward(
                    lambda level, xx, yy, uu, ss: fp32_acc.add(
                        level, xx, yy, uu, ss)
                )
                model.forward_packed(x, pos, segpos, full_mask, start)
            rows += rows_here
        batches += 1
        if log_every and batches % log_every == 0:
            print(json.dumps({"progress": {"batches": batches, "rows": rows}}),
                  flush=True)
    return rows, batches, fp32_acc.finalize(), bf16_probe_acc.finalize()


def _compare_probe(fp32_summary, bf16_summary) -> dict:
    def get(summary, path):
        node = summary
        for key in path:
            node = node[key]
        return float(node)

    out = {}
    for name in ("x", "y", "u"):
        a = get(fp32_summary, ["global", f"{name}_positive_fraction"])
        b = get(bf16_summary, ["global", f"{name}_positive_fraction"])
        out[f"{name}_positive_fraction"] = {
            "fp32": a, "bf16": b, "abs_diff": abs(a - b),
        }
    return out


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default=PROD_CKPT_STEP2000,
        help="explicit checkpoint; defaults to the immutable step_0000002000 "
             "archive. latest.pt is intentionally never used as a default.",
    )
    parser.add_argument("--corpus-root", default=PROD_CORPUS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-global-batches", type=int, default=64)
    parser.add_argument("--regions", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--start-sequence", type=int, default=None,
                        help="contiguous range instead of stratified sampling")
    parser.add_argument("--ranges-from", default=None,
                        help="reuse the exact ranges recorded by a previous "
                             "census JSON (required for the 2.5B rerun)")
    parser.add_argument("--fp32-validation-batches", type=int, default=2)
    parser.add_argument("--fast-verify", action="store_true")
    parser.add_argument("--device", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    print("NOTE: run this census only when the production Arm-A trainer is "
          "not using this GPU; never concurrently with training.",
          flush=True)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = ArmAConfig()
    ckpt_path = Path(args.checkpoint)
    corpus_root = Path(args.corpus_root)
    out_path = Path(args.output)
    require(out_path.resolve() != ckpt_path.resolve(),
            "output path equals the checkpoint path")
    require(corpus_root.resolve() not in out_path.resolve().parents,
            "refusing to write inside the frozen corpus")

    trainer = load_corpus_module()
    corpus = trainer.FrozenPackedCorpus(
        corpus_root, cfg, None, fast_verify=args.fast_verify)
    total_sequences = corpus.total_sequences
    info = inspect_checkpoint(
        ckpt_path,
        expected_config={
            "T": cfg.T, "V": cfg.V, "D": cfg.D, "N": cfg.N,
            "H": cfg.H, "L": cfg.L, "HIDDEN": cfg.HIDDEN,
            "GLOBAL_BATCH": BATCH_SIZE, "MICROBATCH": 16,
        },
        corpus_artifact_sha256=corpus.contract.artifact_hashes_sha256,
        total_sequences=total_sequences,
        code_path=SCRIPT_ROOT / "training" / "arm_a_2p5b_trainer.py",
    )

    if args.ranges_from:
        ranges = ranges_from_file(Path(args.ranges_from), total_sequences)
    elif args.start_sequence is not None:
        ranges = contiguous_ranges(args.start_sequence,
                                   args.num_global_batches, total_sequences)
    else:
        ranges = sample_ranges(total_sequences, args.num_global_batches,
                               args.regions, args.seed)

    model = CensusArmA(cfg, device, scan_block=1024).to(device)
    model.load_state_dict(info.pop("state_dict"), strict=True)
    model.eval()

    def batch_iter():
        for start, _end in ranges:
            for _, batch in corpus.stream_batches(start, BATCH_SIZE):
                yield batch
                break

    accumulator = TrainedCensusAccumulator(cfg)
    started = time.perf_counter()
    rows, batches, fp32_summary, bf16_probe = run_census_batches(
        model, batch_iter(), device, args.microbatch,
        use_bf16=(device.type == "cuda"),
        fp32_probe_batches=args.fp32_validation_batches,
        accumulator=accumulator, seed=args.seed,
    )
    census = accumulator.finalize()
    payload = {
        "meta": {
            "campaign": "arm_a_sparse_resonant",
            "phase": "1_sparsity_census",
            "implementation": "CensusArmA/opt3c_all_b1024 (frozen)",
            "evidence_grade": info["evidence_grade"],
            "inference": "bf16_autocast" if device.type == "cuda" else "fp32",
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "checkpoint": info["checkpoint"],
            "checkpoint_sha256": info["checkpoint_sha256"],
            "checkpoint_step": info["checkpoint_step"],
            "checkpoint_input_tokens": info["checkpoint_input_tokens"],
            "checkpoint_cursor": info["checkpoint_cursor"],
            "code_fingerprint_checked": info["code_fingerprint_checked"],
            "corpus_root": str(corpus_root),
            "corpus_artifact_hashes_sha256":
                corpus.contract.artifact_hashes_sha256,
            "corpus_total_sequences": total_sequences,
            "sampling": {
                "mode": ("ranges_from" if args.ranges_from else
                         ("contiguous" if args.start_sequence is not None
                          else "stratified")),
                "seed": args.seed,
                "regions": args.regions if args.start_sequence is None else 1,
                "global_batches": batches,
                "rows": rows,
                "microbatch": args.microbatch,
            },
            "seconds": time.perf_counter() - started,
            "writes": "output JSON only; checkpoint and corpus untouched",
        },
        "ranges": [[start, end] for start, end in ranges],
        "census": census,
        "fp32_validation": {
            "batches": min(args.fp32_validation_batches, batches),
            "comparison_vs_bf16_probe": _compare_probe(
                fp32_summary, bf16_probe),
            "fp32_summary_headline": fp32_summary.get("global", {}),
            "bf16_probe_headline": bf16_probe.get("global", {}),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out_path),
        "evidence_grade": info["evidence_grade"],
        "checkpoint_step": info["checkpoint_step"],
        "checkpoint_input_tokens": info["checkpoint_input_tokens"],
        "batches": batches,
        "rows": rows,
        "global": census["global"],
        "decision": census["decision"],
        "mac_estimates": census["mac_estimates"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
