"""Overnight benchmark matrix. Runs INSIDE the container (CUDA torch).

Shape ladder preserves production K=4096/D=256/H=4, reduces T/L/B to fit
8GB. Variants: canonical ref (SAC ckpt) vs opt1 (scan+prefix, ckpt on/off),
eager first; compile pass second. OOM-safe: records OOM and continues.
Writes JSON lines to results/local_matrix.json.

Usage (in container):  python opt/bench_matrix.py [--compile] [--full]
"""

import argparse
import gc
import json
import sys
import traceback

import torch

sys.path.insert(0, ".")

from opt.bench_step import bench_variant
from opt.model_diet import DietArmA
from opt.model_opt import OptArmA
from opt.model_ref import ArmAConfig, NativeReadStage1ArmA, make_sac_policy
from functools import partial
from torch.utils.checkpoint import create_selective_checkpoint_contexts


def ladder(full):
    shapes = [
        ("T256_L2", dict(T=256, L=2), 2),
        ("T512_L2", dict(T=512, L=2), 2),
        ("T512_L8", dict(T=512, L=8), 1),
        ("T1024_L8", dict(T=1024, L=8), 1),
        ("T2048_L8", dict(T=2048, L=8), 1),
    ]
    return shapes if full else shapes[:3]


def variants(cfg, dev, blocks=()):
    sac = partial(create_selective_checkpoint_contexts, make_sac_policy(cfg))
    out = [
        ("ref_ckpt", lambda: NativeReadStage1ArmA(cfg, dev)),
        ("opt2_ckpt_b128", lambda: OptArmA(cfg, dev, scan_block=128, use_checkpoint=True, sac_context_fn=sac)),
        ("opt2_nockpt_b128", lambda: OptArmA(cfg, dev, scan_block=128, use_checkpoint=False)),
        ("opt2_nockpt_b256", lambda: OptArmA(cfg, dev, scan_block=256, use_checkpoint=False)),
        ("opt3_nockpt_b256", lambda: OptArmA(cfg, dev, scan_block=256, use_checkpoint=False, coord="dense")),
        ("opt3c_nockpt_b256", lambda: OptArmA(cfg, dev, scan_block=256, use_checkpoint=False, coord="dense", single_scan="chunkwise")),
        ("opt3_ckpt_b256", lambda: OptArmA(cfg, dev, scan_block=256, use_checkpoint=True, sac_context_fn=sac, coord="dense")),
        ("opt3c_where_b512", lambda: OptArmA(cfg, dev, scan_block=512, use_checkpoint=False, coord="dense", single_scan="chunkwise", packed_update="where")),
        ("opt3c_where_b1024", lambda: OptArmA(cfg, dev, scan_block=1024, use_checkpoint=False, coord="dense", single_scan="chunkwise", packed_update="where")),
        ("opt3c_zc_b512", lambda: OptArmA(cfg, dev, scan_block=512, use_checkpoint=False, coord="dense", single_scan="chunkwise", zero_carry=True)),
        ("opt3c_zc_b1024", lambda: OptArmA(cfg, dev, scan_block=1024, use_checkpoint=False, coord="dense", single_scan="chunkwise", zero_carry=True)),
        ("opt3c_dir_b1024", lambda: OptArmA(cfg, dev, scan_block=1024, use_checkpoint=False, coord="dense", single_scan="chunkwise", paper_layout="direct")),
        ("opt3c_rope_b1024", lambda: OptArmA(cfg, dev, scan_block=1024, use_checkpoint=False, coord="dense", single_scan="chunkwise", cache_rope=True)),
        ("opt3c_dirrope_b1024", lambda: OptArmA(cfg, dev, scan_block=1024, use_checkpoint=False, coord="dense", single_scan="chunkwise", paper_layout="direct", cache_rope=True)),
        ("opt3c_all_b1024", lambda: OptArmA(cfg, dev, scan_block=1024, use_checkpoint=False, coord="dense", single_scan="chunkwise", zero_carry=True, paper_layout="direct", cache_rope=True)),
        ("diet_nockpt_b1024", lambda: DietArmA(cfg, dev, scan_block=1024, use_checkpoint=False, coord="dense", single_scan="chunkwise")),
        ("diet_nockpt_b512", lambda: DietArmA(cfg, dev, scan_block=512, use_checkpoint=False, coord="dense", single_scan="chunkwise")),
        ("opt3p_nockpt_b512", lambda: OptArmA(cfg, dev, scan_block=512, use_checkpoint=False, coord="dense", single_scan="parallel")),
        ("opt3p_nockpt_b1024", lambda: OptArmA(cfg, dev, scan_block=1024, use_checkpoint=False, coord="dense", single_scan="parallel")),
        ("opt3h_nockpt_b512", lambda: OptArmA(cfg, dev, scan_block=512, use_checkpoint=False, coord="dense", single_scan="hybrid")),
        ("opt3h_nockpt_b1024", lambda: OptArmA(cfg, dev, scan_block=1024, use_checkpoint=False, coord="dense", single_scan="hybrid")),
        ("opt4_nockpt_b512", lambda: OptArmA(cfg, dev, scan_block=512, use_checkpoint=False, coord="dense", single_scan="static4")),
    ]
    for blk in blocks:
        out.append((f"opt3c_nockpt_b{blk}",
                    lambda blk=blk: OptArmA(cfg, dev, scan_block=blk, use_checkpoint=False,
                                            coord="dense", single_scan="chunkwise")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--warmups", type=int, default=3)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--shapes", default="", help="comma list like T512_L8,T2048_L8 (default all)")
    ap.add_argument("--variants", default="", help="comma list like ref_ckpt,opt2_nockpt_b256 (default all)")
    ap.add_argument("--compile-mode", default="default")
    ap.add_argument("--gb", type=int, default=0, help="override ladder global batch (0 = ladder default)")
    ap.add_argument("--seed", type=int, default=0, help="shuffle worklist order (0 = fixed order)")
    ap.add_argument("--blocks", default="", help="comma list overriding scan_block for opt* variants")
    ap.add_argument("--packed", default="single",
                    help="comma list of batch layouts (interleaved in one session): single,two,four,mixed,heavy")
    ap.add_argument("--assume-single-doc", action="store_true",
                    help="bypass [B,T,T] segment derivation + .item() graph break (single-doc batches only)")
    args = ap.parse_args()
    dev = torch.device("cuda")
    print({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "compiled": args.compile})
    out_path = "results/local_matrix.json"
    want_shapes = set(s.strip() for s in args.shapes.split(",") if s.strip())
    want_vars = set(s.strip() for s in args.variants.split(",") if s.strip())
    want_blocks = [int(s) for s in args.blocks.split(",") if s.strip()]
    want_packed = [s.strip() for s in args.packed.split(",") if s.strip()] or ["single"]
    # Probe available variant names once so typos/filters fail loudly.
    _probe_cfg = ArmAConfig(**dict(ladder(args.full)[0][1]))
    _known = {v for v, _ in variants(_probe_cfg, torch.device("cpu"), blocks=want_blocks)}
    _unknown = (want_vars - _known) or (want_shapes - {s for s, _, _ in ladder(args.full)})
    if _unknown:
        raise SystemExit(f"unknown --shapes/--variants names (use --blocks to generate opt3c_nockpt_b*): {_unknown}")
    import random as _random
    import time as _time
    session = _time.strftime("%Y%m%d-%H%M%S")
    work = []
    for sname, kw, gb in ladder(args.full):
        if want_shapes and sname not in want_shapes:
            continue
        if args.gb:
            gb = args.gb
        cfg = ArmAConfig(**kw)
        for vname, mk in variants(cfg, dev, blocks=want_blocks):
            if want_vars and vname not in want_vars:
                continue
            for mb in dict.fromkeys((gb, max(1, gb // 2))):
                for pk in want_packed:
                    work.append((sname, dict(kw), gb, mb, vname, mk, pk))
    if args.seed:
        _random.Random(args.seed).shuffle(work)
    for order, (sname, kw, gb, mb, vname, mk, pk) in enumerate(work):
        cfg = ArmAConfig(**kw)
        # Bypass is only valid for single-doc batches on non-canonical models
        # (the canonical forward takes exactly 4 args). Enforced, not assumed.
        byp = bool(args.assume_single_doc and pk == "single"
                   and not vname.startswith("ref"))
        row = {"shape": sname, "cfg": kw, "variant": vname,
               "global_batch": gb, "microbatch": mb,
               "compiled": args.compile, "compile_mode": args.compile_mode,
               "packed": pk, "bypass": byp,
               "session": session, "order": order, "seed": args.seed}
        try:
            gc.collect()
            torch.cuda.empty_cache()
            if hasattr(torch, "_dynamo"):
                torch._dynamo.reset()
            r = bench_variant(mk, cfg, dev, gb, mb, warmups=args.warmups,
                              samples=args.samples, compiled=args.compile,
                              compile_mode=args.compile_mode, packed=pk,
                              assume_single_doc=byp)
            row.update(r)
            row["status"] = "OK"
        except Exception as e:
            oom = "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError)
            row.update({"status": "OOM" if oom else "ERROR",
                        "error": f"{type(e).__name__}: {str(e)[:300]}"})
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            if hasattr(torch, "_dynamo"):
                torch._dynamo.reset()
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                    for k, v in row.items() if k != "times_ms"}),
              flush=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
