"""PREPARED (not executed) E3 frozen-corpus census package.

This script is the explicit E3 upgrade path for the Arm-A population analysis.
It runs the SAME pass-1/pass-2 capture as collect.py, but streams the frozen
packed corpus instead of synthetic random-token batches, so the resulting
statistics can be labelled E3 (representative frozen-corpus evidence).

It requires the frozen corpus (see context/frozen_corpus_contract.json):

  /content/drive/Shareddrives/ICLR PHASE BDH/phase_bdh/corpus/stage2/frozen_5b_v1

and a CUDA GPU or a patient CPU. It is intentionally NOT run in the local
CPU-only mission session.

Usage (Colab):
  !cd /content/iclr-oc && python analysis/arm_a_science/e3_census.py \
      --ckpt latest --corpus-root "/content/drive/.../frozen_5b_v1" \
      --batches 8 --microbatch 8 --tag frozen_v1

Usage (local plan check, no data needed):
  python analysis/arm_a_science/e3_census.py --plan
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import (  # noqa: E402
    CHECKPOINTS, KEYS, LADDER, RESULTS_DIR, ArmAConfig, build_model,
    ckpt_sha256, load_state, make_sample_layout, rss_mb, save_json,
    band_slices,
)
from capture import Pass1Capture, Pass2Capture, build_lag_pairs  # noqa: E402
from collect import CORE_DEFS, LAG_LADDER, MAX_PAIRS, flatten  # noqa: E402

CORE_MASK_KEYS = ("x", "u")


def _load_corpus_module():
    path = (Path(__file__).resolve().parents[2] / "training"
            / "arm_a_2p5b_trainer.py")
    spec = importlib.util.spec_from_file_location("arm_a_trainer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["arm_a_trainer"] = module
    spec.loader.exec_module(module)
    return module


def build_rankings_and_core(cfg, s1, n_tokens_all):
    """Identical semantics to collect.py: cross-fit by batch parity."""
    mass = s1["mass_sum"]
    pos = s1["pos_count"]
    n_batches = mass["x"].shape[0]
    split_idx = {"A": [i for i in range(n_batches) if i % 2 == 0],
                 "B": [i for i in range(n_batches) if i % 2 == 1]}
    rankings = {k: {} for k in KEYS}
    rankings_pooled = {k: {} for k in KEYS}
    for k in KEYS:
        for level in range(cfg.L):
            rankings[k][level] = {}
            rankings_pooled[k][level] = {}
        for split, idxs in split_idx.items():
            m = mass[k][idxs].sum(0)
            for level in range(cfg.L):
                rankings[k][level][split] = np.argsort(
                    -m[level], axis=-1).astype(np.int32)
                rankings_pooled[k][level][split] = np.argsort(
                    -m[level].reshape(-1)).astype(np.int32)
    core_defs = {name: {k: {} for k in CORE_MASK_KEYS}
                 for name, _ in CORE_DEFS}
    for name, _ in CORE_DEFS:
        for k in CORE_MASK_KEYS:
            for split in ("A", "B"):
                for level in range(cfg.L):
                    core_defs[name][k].setdefault(level, {})[split] = None
    for name, frac in CORE_DEFS:
        for k in CORE_MASK_KEYS:
            for split, idxs in split_idx.items():
                pc_split = pos[k][idxs].sum(0)
                m_split = mass[k][idxs].sum(0)
                for level in range(cfg.L):
                    if frac is not None:
                        c = max(1, int(round(frac * cfg.K)))
                        order = np.argsort(-m_split[level], axis=-1)
                        mask = np.zeros((cfg.H, cfg.K), dtype=bool)
                        np.put_along_axis(mask, order[:, :c], True, axis=1)
                    elif name == "pact_ge_0p5":
                        mask = pc_split[level] >= 0.5 * n_tokens_all[split]
                    else:
                        mask = pc_split[level] >= 0.9 * n_tokens_all[split]
                    core_defs[name][k][level][split] = mask
    return rankings, rankings_pooled, core_defs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="latest",
                    choices=list(CHECKPOINTS) + ["random_init"])
    ap.add_argument("--ckpt-file", default=None,
                    help="explicit checkpoint path (e.g. a Drive copy); "
                         "overrides --ckpt for loading")
    ap.add_argument("--corpus-root", default=None)
    ap.add_argument("--fast-verify", action="store_true",
                    help="skip per-file corpus SHA-256 (index digest only)")
    ap.add_argument("--batches", type=int, default=8,
                    help="corpus batches of GLOBAL_BATCH rows to stream")
    ap.add_argument("--microbatch", type=int, default=8,
                    help="rows per forward (CPU memory control)")
    ap.add_argument("--start-sequence", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--context", type=int, default=2048,
                    help="truncate corpus windows to the first T tokens; "
                         "T<2048 is a within-training-context robustness run, "
                         "T>2048 is a separate length-extrapolation "
                         "experiment and must be reported separately")
    ap.add_argument("--plan", action="store_true",
                    help="print the plan without touching any data")
    args = ap.parse_args()

    plan = {
        "checkpoint": args.ckpt,
        "ckpt_file": args.ckpt_file,
        "corpus_root": args.corpus_root,
        "fast_verify": args.fast_verify,
        "batches": args.batches,
        "microbatch": args.microbatch,
        "start_sequence": args.start_sequence,
        "context": args.context,
        "device": args.device or ("cuda" if torch.cuda.is_available()
                                  else "cpu"),
        "outputs": [
            str(RESULTS_DIR / "raw_frozen" /
                f"pass1_frozen_{args.tag}.npz"),
            str(RESULTS_DIR / "raw_frozen" /
                f"pass2_frozen_{args.tag}.npz"),
            str(RESULTS_DIR / "raw_frozen" /
                f"ranks_frozen_{args.tag}.npz"),
            str(RESULTS_DIR / "raw_frozen" /
                f"meta_frozen_{args.tag}.json"),
        ],
        "note": "E3 evidence; run only on Colab/GPU or with explicit consent "
                "for a long CPU run",
    }
    if args.plan or not args.corpus_root:
        print(json.dumps({"plan": plan, "dry_run": True}, indent=2))
        return

    device = torch.device(plan["device"])
    cfg = ArmAConfig(T=args.context) if args.context != 2048 else ArmAConfig()
    if args.ckpt_file:
        ckpt_path = Path(args.ckpt_file)
        state, ckpt = load_state(ckpt_path)
        ckpt_sha = ckpt_sha256(ckpt_path)
    else:
        if args.ckpt == "random_init":
            raise SystemExit("random init is not E3 evidence; use a checkpoint")
        ckpt_path = CHECKPOINTS[args.ckpt]
        state, ckpt = load_state(ckpt_path)
        ckpt_sha = ckpt_sha256(ckpt_path)
    model = build_model(cfg, state, threads=args.threads).to(device)
    del state
    trainer = _load_corpus_module()
    corpus = trainer.FrozenPackedCorpus(Path(args.corpus_root), cfg, None,
                                        verify_files=True,
                                        fast_verify=args.fast_verify)
    stream = corpus.stream_batches(args.start_sequence,
                                   trainer.PROD_CFG.GLOBAL_BATCH)

    out_dir = RESULTS_DIR / "raw_frozen"
    out_dir.mkdir(parents=True, exist_ok=True)
    causal = torch.ones((cfg.T, cfg.T), dtype=torch.bool).tril(diagonal=-1)

    # microbatches are treated as independent analysis batches
    layouts = []
    batches = []
    t0 = time.perf_counter()
    for _ in range(args.batches):
        seq_index, batch64 = next(stream)
        total_rows = int(batch64["x"].shape[0])
        T = cfg.T
        for lo in range(0, total_rows, args.microbatch):
            hi = min(lo + args.microbatch, total_rows)
            rows = hi - lo
            start = batch64["start"][lo:hi, :T].long()
            entry = {
                "x": batch64["x"][lo:hi, :T].long(),
                "pos": batch64["pos"][lo:hi, :T].int(),
                "segpos": batch64["segpos"][lo:hi, :T].int(),
                "start": start,
                "input_valid": batch64["input_valid"][lo:hi, :T].bool(),
            }
            entry["full_mask"] = (
                (start[:, :, None] == start[:, None, :])
                & entry["input_valid"][:, :, None]
                & entry["input_valid"][:, None, :]
                & causal.unsqueeze(0))
            # synthetic-layout sampler needs a BatchSpec; the fields used are
            # only seed and rows, so reuse the mixed-mode spec deterministically
            from harness import BatchSpec
            spec = BatchSpec("mixed", 1000 + lo, rows, f"frozen_{lo}")
            layout = make_sample_layout(spec, cfg)
            layouts.append(layout)
            batches.append(entry)
    print(json.dumps({"streamed_microbatches": len(batches),
                      "seconds": time.perf_counter() - t0}, indent=2))

    # pass 1
    p1 = Pass1Capture(cfg, len(batches),
                      [layout.tierA_flat.size for layout in layouts])
    for i, (layout, entry) in enumerate(zip(layouts, batches)):
        p1.begin_batch(i, layout.tierA_flat)
        model.begin_forward(p1)
        with torch.no_grad():
            model.forward_packed(
                entry["x"].to(device), entry["pos"].to(device),
                entry["segpos"].to(device), entry["full_mask"].to(device),
                entry["start"].to(device))
    s1 = p1.finalize()
    flat1 = {}
    flatten("", s1, flat1)
    tok_meta = []
    for layout, entry in zip(layouts, batches):
        r = layout.tierA_flat // cfg.T
        p = layout.tierA_flat % cfg.T
        start_np = entry["start"].numpy()
        doc_id = np.zeros_like(start_np)
        for row in range(start_np.shape[0]):
            new = np.zeros(cfg.T, dtype=np.int64)
            new[1:] = start_np[row, 1:] != start_np[row, :-1]
            doc_id[row] = np.cumsum(new)
        doc_len = np.zeros_like(start_np)
        for row in range(start_np.shape[0]):
            for s0 in np.unique(start_np[row]):
                idx = np.where(start_np[row] == s0)[0]
                doc_len[row, idx] = idx[-1] - idx[0] + 1
        rel = np.arange(cfg.T)[None, :] - start_np
        tok_meta.append(np.stack([r, p, doc_id[r, p], doc_len[r, p],
                                  rel[r, p]], axis=-1))
    flat1["tok_meta"] = np.stack(tok_meta).astype(np.int32)
    flat1["tok_chunk"] = np.stack([l.tierA_chunk for l in layouts])
    flat1["tok_local"] = np.stack([l.tierA_local_pos for l in layouts])
    np.savez_compressed(out_dir / f"pass1_frozen_{args.tag}.npz", **flat1)

    n_tokens_all = {"A": 0.0, "B": 0.0}
    for i, entry in enumerate(batches):
        n_tokens_all["A" if i % 2 == 0 else "B"] += float(entry["x"].numel())
    rankings, rankings_pooled, core_defs = build_rankings_and_core(
        cfg, s1, n_tokens_all)

    # pass 2
    tierB_pairs = []
    for layout, entry in zip(layouts, batches):
        start_np = entry["start"].numpy()
        doc_id = np.zeros_like(start_np)
        for row in range(start_np.shape[0]):
            new = np.zeros(cfg.T, dtype=np.int64)
            new[1:] = start_np[row, 1:] != start_np[row, :-1]
            doc_id[row] = np.cumsum(new)
        tb_doc = doc_id.reshape(-1)[layout.tierB_flat]
        tierB_pairs.append(build_lag_pairs(
            layout, tb_doc, layout.tierB_row, layout.tierB_chunk,
            layout.tierB_local_pos, lags=LAG_LADDER, max_pairs=MAX_PAIRS))
    p2 = Pass2Capture(cfg, rankings, rankings_pooled, core_defs, tierB_pairs,
                      len(batches), detail=True, stability=True)
    for i, (layout, entry) in enumerate(zip(layouts, batches)):
        p2.begin_batch(i, layout.tierA_flat, layout.tierB_flat)
        model.begin_forward(p2)
        with torch.no_grad():
            model.forward_packed(
                entry["x"].to(device), entry["pos"].to(device),
                entry["segpos"].to(device), entry["full_mask"].to(device),
                entry["start"].to(device))
    s2 = p2.finalize()
    stab = s2.pop("stability", None)
    flat2 = {}
    flatten("", s2, flat2)
    if stab is not None:
        cat_order = ([f"lag_{l}" for l in LAG_LADDER]
                     + ["cross_row", "inter_128_256", "inter_256_512",
                        "inter_512_plus"])
        present = {e["category"] for e in stab}
        cat_names = [c for c in cat_order if c in present]
        cat_index = {c: i for i, c in enumerate(cat_names)}
        n_pairs = [len(e["spearman"]) for e in stab]
        max_p = max(n_pairs) if n_pairs else 0

        def pad(arrs, trailing):
            out = np.full((len(arrs), max_p, *trailing), np.nan,
                          dtype=np.float32)
            for i, a in enumerate(arrs):
                out[i, :len(a)] = np.asarray(a, dtype=np.float32)
            return out
        flat2["stab_jaccard"] = pad(
            [e["jaccard"] for e in stab], (len(LADDER),))
        flat2["stab_support"] = pad(
            [np.asarray(e["support_jaccard"])[:, None] for e in stab], (1,))
        flat2["stab_spearman"] = pad(
            [np.asarray(e["spearman"])[:, None] for e in stab], (1,))
        flat2["stab_rbo"] = pad(
            [np.asarray(e["rbo"])[:, None] for e in stab], (1,))
        flat2["stab_meta"] = np.asarray(
            [[e["level"], e["head"], {"x": 0, "u": 1}[e["key"]],
              cat_index[e["category"]]] for e in stab], dtype=np.int32)
        flat2["stab_categories"] = np.asarray(cat_names)
    np.savez_compressed(out_dir / f"pass2_frozen_{args.tag}.npz", **flat2)

    rank_arrays = {}
    for k in KEYS:
        for level in range(cfg.L):
            for split in ("A", "B"):
                rank_arrays[f"rank_{k}_L{level}_{split}"] = \
                    rankings[k][level][split]
                rank_arrays[f"rankpool_{k}_L{level}_{split}"] = \
                    rankings_pooled[k][level][split]
    core_arrays = {}
    for name, _ in CORE_DEFS:
        for k in CORE_MASK_KEYS:
            for level in range(cfg.L):
                for split in ("A", "B"):
                    core_arrays[f"core_{name}_{k}_L{level}_{split}"] = \
                        core_defs[name][k][level][split]
    np.savez_compressed(out_dir / f"ranks_frozen_{args.tag}.npz",
                        **rank_arrays, **core_arrays)
    save_json(out_dir / f"meta_frozen_{args.tag}.json", {
        "plan": plan,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "progress": ckpt.get("progress"),
        "corpus_root": args.corpus_root,
        "corpus_verify": "index digest + " + (
            "fast (per-file hashes skipped)" if args.fast_verify
            else "full per-file SHA-256"),
        "start_sequence": args.start_sequence,
        "context": args.context,
        "microbatches": len(batches),
        "rows_total": int(sum(e["x"].shape[0] for e in batches)),
        "tokens_total": int(sum(e["x"].numel() for e in batches)),
        "evidence_class": "E3 frozen-corpus streaming census",
        "device": str(device),
        "torch": torch.__version__,
        "rss_mb": rss_mb(),
        "ladder": list(LADDER),
        "pass1_arrays": {k: list(v.shape) for k, v in flat1.items()},
        "pass2_arrays": {k: list(v.shape) for k, v in flat2.items()},
        "note": "run analysis/arm_a_science/e3_finalize.py --tag <tag> to "
                "produce the mission-required e3_*.json files; the checkpoint "
                "key is 'frozen_<tag>'",
    })
    print(json.dumps({"done": True, "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
