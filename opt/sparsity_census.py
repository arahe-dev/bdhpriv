"""Arm-A sparsity census (campaign Phase 1) — machine-readable JSON.

Examples:
  py -3.12 opt/sparsity_census.py --tiny --synthetic --canonical-init \
      --allow-random-init --out results/sparsity_census_smoke.json
  py -3.12 opt/sparsity_census.py --ckpt <latest.pt> \
      --corpus-root <frozen_5b_v1> --batches 8 --out results/sparsity_census.json

Rules (mission): a trained checkpoint is required for evidence. Canonical
init runs are marked `diagnostic_only` and must not be cited.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opt.census_model import CensusArmA, SparsityAccumulator
from opt.model_ref import ArmAConfig, canonical_init, load_init, \
    synthetic_packed_batch

TINY = dict(T=16, V=32, D=8, N=16, H=2, L=2, HIDDEN=8)


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _load_corpus_module():
    path = Path(__file__).resolve().parents[1] / "training" / \
        "arm_a_2p5b_trainer.py"
    spec = importlib.util.spec_from_file_location("arm_a_trainer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["arm_a_trainer"] = module
    spec.loader.exec_module(module)
    return module


def _load_weights(model, ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    return ckpt


def _checkpoint_config(ckpt) -> dict:
    config = ckpt.get("config") if isinstance(ckpt, dict) else None
    return config or {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=None,
                        help="trained Arm-A checkpoint (required for evidence)")
    parser.add_argument("--canonical-init", action="store_true")
    parser.add_argument("--allow-random-init", action="store_true",
                        help="explicitly mark a random-init census as diagnostic")
    parser.add_argument("--corpus-root", default=None,
                        help="frozen corpus root (production cfg only)")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--sample-rows", type=int, default=256)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--block-widths", type=int, nargs="*", default=None)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-mode", default="mixed")
    parser.add_argument("--fast-verify", action="store_true",
                        help="skip per-file corpus SHA-256 (index digest only)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="results/sparsity_census.json")
    args = parser.parse_args()

    if args.ckpt is None and not (args.canonical_init or args.allow_random_init):
        parser.error("provide --ckpt (trained) or --canonical-init "
                     "--allow-random-init (diagnostic)")
    if args.ckpt is not None and args.canonical_init:
        parser.error("--ckpt and --canonical-init are mutually exclusive")
    if args.corpus_root and args.tiny:
        parser.error("--corpus-root requires the production config")

    cfg = ArmAConfig(**TINY) if args.tiny else ArmAConfig()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    block_widths = tuple(args.block_widths) if args.block_widths else None
    acc = SparsityAccumulator(cfg, sample_rows=args.sample_rows,
                              block_widths=block_widths, bands=args.bands)
    model = CensusArmA(cfg, device,
                       scan_block=8 if args.tiny else 1024).to(device)
    model.eval()

    ckpt = None
    if args.ckpt:
        ckpt = _load_weights(model, Path(args.ckpt))
        weights_kind = "trained_checkpoint"
    else:
        load_init(model, canonical_init(cfg), device)
        weights_kind = "canonical_init_diagnostic"

    data_kind = "frozen_corpus" if args.corpus_root else "synthetic"
    if data_kind == "frozen_corpus" and not args.synthetic and cfg.T != 2048:
        raise SystemExit("corpus census requires the production config")

    generator = torch.Generator(device="cpu").manual_seed(cfg.SEED)
    started = time.perf_counter()
    rows_seen = 0
    batches_seen = 0

    corpus_iter = None
    trainer = None
    if data_kind == "frozen_corpus":
        trainer = _load_corpus_module()
        corpus = trainer.FrozenPackedCorpus(
            Path(args.corpus_root), cfg, None,
            fast_verify=args.fast_verify,
        )
        corpus_iter = corpus.stream_batches(
            0, trainer.PROD_CFG.GLOBAL_BATCH
        )

    with torch.no_grad():
        while batches_seen < args.batches:
            if data_kind == "frozen_corpus":
                try:
                    _, batch64 = next(corpus_iter)
                except StopIteration:
                    break
                total_rows = int(batch64["x"].shape[0])
                entry = {
                    key: batch64[key] for key in batch64
                }
                entry["_rows"] = total_rows
            else:
                total_rows = args.microbatch
                batch = synthetic_packed_batch(
                    cfg, total_rows, device, seed=cfg.SEED + batches_seen,
                    mode=args.synthetic_mode,
                )
                entry = {**batch, "_rows": total_rows}

            for lo in range(0, total_rows, args.microbatch):
                hi = min(lo + args.microbatch, total_rows)
                if data_kind == "frozen_corpus":
                    x = entry["x"][lo:hi].to(device, dtype=torch.long)
                    pos = entry["pos"][lo:hi].to(device, dtype=torch.int32)
                    segpos = entry["segpos"][lo:hi].to(device, dtype=torch.int32)
                    start = entry["start"][lo:hi].to(device, dtype=torch.int32)
                    input_valid = entry["input_valid"][lo:hi].to(
                        device, dtype=torch.bool)
                    causal = torch.ones((cfg.T, cfg.T), dtype=torch.bool,
                                        device=device).tril(diagonal=-1)
                    full_mask = (
                        (start[:, :, None] == start[:, None, :])
                        & input_valid[:, :, None]
                        & input_valid[:, None, :]
                        & causal.unsqueeze(0)
                    )
                else:
                    x = entry["x"][lo:hi].to(device, dtype=torch.long)
                    pos = entry["pos"][lo:hi].to(device, dtype=torch.int32)
                    segpos = entry["segpos"][lo:hi].to(device, dtype=torch.int32)
                    start = entry["segment_start"][lo:hi].to(device,
                                                             dtype=torch.int32)
                    full_mask = entry["full_mask"][lo:hi].to(device)
                rows = hi - lo
                acc.new_batch(rows, generator)
                model.begin_forward(
                    lambda level, xx, yy, uu, ss: acc.add(level, xx, yy, uu, ss)
                )
                model.forward_packed(x, pos, segpos, full_mask, start)
                rows_seen += rows
            batches_seen += 1

    census = acc.finalize()
    headline = _headline(census)
    out = {
        "meta": {
            "campaign": "arm_a_sparse_resonant",
            "phase": "1_sparsity_census",
            "git_rev": _git_rev(),
            "implementation": "census_model.CensusArmA (opt3c_all flags)",
            "weights": weights_kind,
            "ckpt": args.ckpt,
            "data": data_kind,
            "corpus_root": args.corpus_root,
            "synthetic_mode": args.synthetic_mode if data_kind == "synthetic"
            else None,
            "evidence_grade": (
                "production" if (weights_kind == "trained_checkpoint"
                                 and data_kind == "frozen_corpus")
                else "diagnostic_only"
            ),
            "config": {
                "T": cfg.T, "V": cfg.V, "D": cfg.D, "N": cfg.N,
                "H": cfg.H, "K": cfg.K, "L": cfg.L, "HIDDEN": cfg.HIDDEN,
            },
            "checkpoint_config": _checkpoint_config(ckpt) if ckpt else None,
            "batches": batches_seen,
            "rows": rows_seen,
            "sample_rows": census["sample_rows"],
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "seconds": time.perf_counter() - started,
        },
        "census": census,
        "headline": headline,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_path), "meta": out["meta"],
                      "headline": headline}, indent=2))


def _headline(census: dict) -> dict:
    keys = ("u", "x", "y")
    agg = {k: {"zero_fraction": [], "top_25pct_mass": []} for k in keys}
    block = {}
    pair_zero = []
    adj = []
    for layer in census["levels"]:
        for head in layer["heads"]:
            for key in keys:
                agg[key]["zero_fraction"].append(
                    head[key]["zero_fraction"])
                agg[key]["top_25pct_mass"].append(
                    head["top_mass_share"][key]["0.25"])
            pair_zero.append(head["x"]["pair_zero_fraction"])
            adj.append(head["adjacent_jaccard"]["x"])
            for width, values in head["block_occupancy"].items():
                block.setdefault(width, []).append(values["x_any"])
    return {
        "mean_zero_fraction": {
            k: sum(agg[k]["zero_fraction"]) / len(agg[k]["zero_fraction"])
            for k in keys
        },
        "mean_top_25pct_mass_share": {
            k: sum(agg[k]["top_25pct_mass"]) / len(agg[k]["top_25pct_mass"])
            for k in keys
        },
        "mean_x_pair_zero_fraction": sum(pair_zero) / len(pair_zero),
        "mean_x_adjacent_jaccard": sum(adj) / len(adj),
        "mean_block_any_active": {
            w: sum(v) / len(v) for w, v in sorted(block.items())
        },
    }


if __name__ == "__main__":
    main()
