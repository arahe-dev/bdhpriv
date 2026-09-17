"""Collect Arm-A activation-population statistics for one checkpoint.

Runs pass 1 (population statistics) and pass 2 (cross-fitted global top-N,
core/tail, Tier-B stability) over synthetic packed batches on CPU, then
writes compact NPZ + JSON. Read-only with respect to opt/.

Usage:
  py -3.12 analysis/arm_a_science/collect.py --ckpt latest
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import (  # noqa: E402
    CHECKPOINTS, CHECKPOINT_STEPS, KEYS, LADDER, MAIN_SPECS, RAW_DIR,
    ArmAConfig, BatchSpec, build_model, load_state, make_batch,
    make_sample_layout, rss_mb, save_json, token_meta,
)
from capture import Pass1Capture, Pass2Capture, build_lag_pairs  # noqa: E402

CORE_DEFS = (
    ("mass_top_1pct", 0.01),
    ("mass_top_6p25", 0.0625),
    ("mass_top_25", 0.25),
    ("pact_ge_0p5", None),
    ("pact_ge_0p9", None),
)
LAG_LADDER = (1, 2, 4, 8, 16, 32, 63, 127)
MAX_PAIRS = 32
RES_QS = (0.5, 0.8, 0.9)


def _frac_name(f: float) -> str:
    s = f"{f:.6f}".rstrip("0").rstrip(".")
    return s.replace("0.", "").replace(".", "p") or "0"


def build_core_defs(fracs=None):
    """Core definition ladder; defaults keep the E2-era mixed defs."""
    if fracs is None:
        return CORE_DEFS
    defs = [(f"mass_top_{_frac_name(f)}", float(f)) for f in fracs]
    defs += [("pact_ge_0p5", None), ("pact_ge_0p9", None)]
    return tuple(defs)


def context_specs(cfg):
    """Same spec design across context lengths, packed docs scaled to 0.7*T."""
    mean = 0.7 * cfg.T
    return (
        BatchSpec("scaled_mixed", 11, 4, "mixed_a", mean),
        BatchSpec("scaled_mixed", 23, 4, "mixed_b", mean),
        BatchSpec("single", 7, 4, "single_a"),
        BatchSpec("heavy", 5, 4, "heavy_a"),
    )


def flatten(prefix: str, obj, out: dict, skip_none: bool = True):
    if isinstance(obj, np.ndarray):
        out[prefix] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}_{k}" if prefix else str(k), v, out, skip_none)
    elif obj is None:
        if not skip_none:
            out[prefix] = np.asarray(0)
    else:
        out[prefix] = np.asarray(obj)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True,
                    choices=list(CHECKPOINTS) + ["random_init"])
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--out-dir", default=str(RAW_DIR))
    ap.add_argument("--no-stability", action="store_true")
    ap.add_argument("--no-core", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="2 batches x 1 row; fast plumbing test")
    ap.add_argument("--context", type=int, default=2048,
                    help="evaluation context length (T); weights are shared")
    ap.add_argument("--specs", choices=["main", "scaled"], default="main")
    ap.add_argument("--core-fracs", type=float, nargs="*", default=None,
                    help="mass-top core fractions ladder")
    ap.add_argument("--out-subdir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_subdir is None else \
        Path(args.out_dir) / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = ArmAConfig(T=args.context) if args.context != 2048 else ArmAConfig()
    core_ladder = build_core_defs(args.core_fracs)
    if args.ckpt == "random_init":
        from opt.model_ref import canonical_init
        init = canonical_init(cfg)
        state = {
            "embedding.weight": init["embedding"],
            "encoder": init["encoder"],
            "decoder_x": init["decoder_x"],
            "decoder_y": init["decoder_y"],
            "readout": init["readout"],
            "coordinator.Wc": init["coord_Wc"],
            "coordinator.bc": init["coord_bc"],
            "coordinator.alpha": init["coord_alpha"],
            "writer.W1": init["writer_W1"],
            "writer.W2": init["writer_W2"],
        }
        ckpt = {"config": {"init": "canonical_init", "SEED": cfg.SEED}}
        step = 0
    else:
        state, ckpt = load_state(CHECKPOINTS[args.ckpt])
        step = CHECKPOINT_STEPS[args.ckpt]
    model = build_model(cfg, state, threads=args.threads)
    del state

    if args.smoke:
        from harness import BatchSpec
        specs = [BatchSpec("mixed", 11, 1, "smoke_a"),
                 BatchSpec("mixed", 23, 1, "smoke_b")]
    elif args.specs == "scaled" or args.context != 2048:
        specs = list(context_specs(cfg))
    else:
        specs = list(MAIN_SPECS)
    layouts = [make_sample_layout(s, cfg) for s in specs]
    batches = [make_batch(s, cfg) for s in specs]
    metas = [token_meta(b["segment_start"].numpy(), cfg) for b in batches]
    S = layouts[0].tierA_flat.size

    tok_rows = []
    for i, layout in enumerate(layouts):
        fidx = layout.tierA_flat
        r, p = fidx // cfg.T, fidx % cfg.T
        tok_rows.append(np.stack(
            [r, p, metas[i]["doc_id"][r, p], metas[i]["doc_len"][r, p],
             metas[i]["doc_rel_pos"][r, p]], axis=-1))
    tok_meta = np.stack(tok_rows).astype(np.int32)  # (B, S, 5)
    tok_chunk = np.stack([layout.tierA_chunk for layout in layouts])
    tok_local = np.stack([layout.tierA_local_pos for layout in layouts])

    meta = {
        "checkpoint": args.ckpt,
        "checkpoint_step": step,
        "checkpoint_path": str(CHECKPOINTS.get(args.ckpt, "canonical_init")),
        "model_config": {k: getattr(cfg, k) for k in
                         ("T", "V", "D", "N", "H", "L", "HIDDEN", "SEED",
                          "THETA")},
        "specs": [s.__dict__ for s in specs],
        "layouts": [layout.to_meta() for layout in layouts],
        "ladder": list(LADDER),
        "bands": 16,
        "lag_ladder": list(LAG_LADDER),
        "max_pairs_per_category": MAX_PAIRS,
        "res_quantiles": list(RES_QS),
        "core_ladder": [n for n, _ in core_ladder],
        "data_evidence_class":
            "E1/E2 synthetic packed batches (random tokens); not frozen corpus",
        "torch": torch.__version__,
        "threads": args.threads,
        "rss_mb_start": rss_mb(),
        "n_tokens_per_batch": int(S),
        "n_tokens_all_per_batch": int(batches[0]["x"].numel()),
    }

    # ---------------- pass 1 ----------------
    t0 = time.perf_counter()
    p1 = Pass1Capture(cfg, len(specs),
                      [layout.tierA_flat.size for layout in layouts])
    for i, (layout, batch) in enumerate(zip(layouts, batches)):
        p1.begin_batch(i, layout.tierA_flat)
        model.begin_forward(p1)
        with torch.no_grad():
            model.forward_packed(
                batch["x"], batch["pos"], batch["segpos"],
                batch["full_mask"], batch["segment_start"])
    s1 = p1.finalize()
    pass1_s = time.perf_counter() - t0

    flat1 = {}
    flatten("", s1, flat1)
    flat1["tok_meta"] = tok_meta
    flat1["tok_chunk"] = tok_chunk
    flat1["tok_local"] = tok_local
    np.savez_compressed(out_dir / f"pass1_{args.ckpt}.npz", **flat1)

    # ---------------- rankings + core definitions ----------------
    n_batches = len(specs)
    split_idx = {"A": [i for i in range(n_batches) if i % 2 == 0],
                 "B": [i for i in range(n_batches) if i % 2 == 1]}
    n_all_split = {sp: float(sum(batches[i]["x"].numel() for i in idxs))
                   for sp, idxs in split_idx.items()}

    rankings = {k: {} for k in KEYS}
    rankings_pooled = {k: {} for k in KEYS}
    for k in KEYS:
        m_all = s1["mass_sum"][k]  # (B, L, H, K)
        pc_all = s1["pos_count"][k]
        for level in range(cfg.L):
            rankings[k][level] = {}
            rankings_pooled[k][level] = {}
        for split, idxs in split_idx.items():
            m = m_all[idxs].sum(0)  # (L, H, K)
            pc = pc_all[idxs].sum(0)
            for level in range(cfg.L):
                rankings[k][level][split] = np.argsort(
                    -m[level], axis=-1).astype(np.int32)
                pooled = m[level].reshape(-1)
                rankings_pooled[k][level][split] = np.argsort(
                    -pooled).astype(np.int32)

    core_defs = {name: {k: {} for k in ("x", "u")}
                 for name, _ in core_ladder}
    for name, _ in core_ladder:
        for k in ("x", "u"):
            for split in ("A", "B"):
                for level in range(cfg.L):
                    core_defs[name][k].setdefault(level, {})[split] = None

    for name, frac in core_ladder:
        for k in ("x", "u"):
            for split, idxs in split_idx.items():
                pc_split = s1["pos_count"][k][idxs].sum(0)
                m_split = s1["mass_sum"][k][idxs].sum(0)
                for level in range(cfg.L):
                    if frac is not None:
                        c = max(1, int(round(frac * cfg.K)))
                        order = np.argsort(-m_split[level], axis=-1)
                        mask = np.zeros((cfg.H, cfg.K), dtype=bool)
                        np.put_along_axis(mask, order[:, :c], True, axis=1)
                    elif name == "pact_ge_0p5":
                        mask = pc_split[level] >= 0.5 * n_all_split[split]
                    else:
                        mask = pc_split[level] >= 0.9 * n_all_split[split]
                    core_defs[name][k][level][split] = mask

    # ---------------- pass 2 ----------------
    tierB_pairs = []
    for i, layout in enumerate(layouts):
        tb_doc = metas[i]["doc_id"].reshape(-1)[layout.tierB_flat]
        tierB_pairs.append(build_lag_pairs(
            layout, tb_doc, layout.tierB_row, layout.tierB_chunk,
            layout.tierB_local_pos, lags=LAG_LADDER, max_pairs=MAX_PAIRS))

    t0 = time.perf_counter()
    p2 = Pass2Capture(cfg, rankings, rankings_pooled, core_defs, tierB_pairs,
                      len(specs), detail=not args.no_core,
                      stability=not args.no_stability, res_qs=RES_QS)
    for i, (layout, batch) in enumerate(zip(layouts, batches)):
        p2.begin_batch(i, layout.tierA_flat, layout.tierB_flat)
        model.begin_forward(p2)
        with torch.no_grad():
            model.forward_packed(
                batch["x"], batch["pos"], batch["segpos"],
                batch["full_mask"], batch["segment_start"])
    s2 = p2.finalize()
    pass2_s = time.perf_counter() - t0

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
        jac, sup, sp, rbo, meta_rows, n_pairs = [], [], [], [], [], []
        for e in stab:
            jac.append(e["jaccard"])
            sup.append(e["support_jaccard"])
            sp.append(e["spearman"])
            rbo.append(e["rbo"])
            n_pairs.append(len(e["spearman"]))
            meta_rows.append([e["level"], e["head"],
                              {"x": 0, "u": 1}[e["key"]],
                              cat_index[e["category"]]])
        max_p = max(n_pairs) if n_pairs else 0

        def pad_stack(arrs, trailing):
            out = np.full((len(arrs), max_p, *trailing), np.nan,
                          dtype=np.float32)
            for i, a in enumerate(arrs):
                if len(a):
                    out[i, :len(a)] = np.asarray(a, dtype=np.float32)
            return out

        flat2["stab_jaccard"] = pad_stack(jac, (len(LADDER),))
        flat2["stab_support"] = pad_stack([np.asarray(v)[:, None] for v in sup], (1,))
        flat2["stab_spearman"] = pad_stack([np.asarray(v)[:, None] for v in sp], (1,))
        flat2["stab_rbo"] = pad_stack([np.asarray(v)[:, None] for v in rbo], (1,))
        flat2["stab_n"] = np.asarray(n_pairs, dtype=np.int32)
        flat2["stab_meta"] = np.asarray(meta_rows, dtype=np.int32)
        flat2["stab_categories"] = np.asarray(cat_names)
    np.savez_compressed(out_dir / f"pass2_{args.ckpt}.npz", **flat2)

    rank_arrays = {}
    for k in KEYS:
        for level in range(cfg.L):
            for split in ("A", "B"):
                rank_arrays[f"rank_{k}_L{level}_{split}"] = \
                    rankings[k][level][split]
                rank_arrays[f"rankpool_{k}_L{level}_{split}"] = \
                    rankings_pooled[k][level][split]
    core_arrays = {}
    for name, _ in core_ladder:
        for k in ("x", "u"):
            for level in range(cfg.L):
                for split in ("A", "B"):
                    core_arrays[f"core_{name}_{k}_L{level}_{split}"] = \
                        core_defs[name][k][level][split]
    np.savez_compressed(out_dir / f"ranks_{args.ckpt}.npz",
                        **rank_arrays, **core_arrays)

    meta.update({
        "pass1_seconds": pass1_s,
        "pass2_seconds": pass2_s,
        "rss_mb_end": rss_mb(),
        "pass1_npz": str(out_dir / f"pass1_{args.ckpt}.npz"),
        "pass2_npz": str(out_dir / f"pass2_{args.ckpt}.npz"),
        "ranks_npz": str(out_dir / f"ranks_{args.ckpt}.npz"),
        "pass1_arrays": {k: list(v.shape) for k, v in flat1.items()},
        "pass2_arrays": {k: list(v.shape) for k, v in flat2.items()},
    })
    save_json(out_dir / f"meta_{args.ckpt}.json", meta)
    print(json.dumps({k: meta[k] for k in
                      ("checkpoint", "pass1_seconds", "pass2_seconds",
                       "rss_mb_end", "n_tokens_per_batch")}, indent=2))
    print("pass2 timers:", json.dumps(s2.get("timers", {}), indent=1))


if __name__ == "__main__":
    main()
