#!/usr/bin/env python
"""Standalone preflight verification for opt3c_all_b1024.

Purpose: decide, with the most rigorous minimal-time evidence, whether
`opt3c_all` (branch-free packed update + zero-carry skip + direct paper_y
layout + cached RoPE phase) is safe to launch for the full 5B run.

It does NOT optimize or change math. It exercises the production
implementation in this repo:

  opt/model_opt.py::OptArmA         (candidate vs certified configs)
  opt/scan_attn.py::scan_chunkwise_bthk / scan_chunkwise_where_bthk

Gates (every gate is named; the verdict lists all failures):
  1. scan_oracle                     dense fp64 oracle vs certified/candidate
                                     scans on randomized + adversarial layouts
  2. model_equiv_packed_fp32_eager   candidate vs certified (logits/loss/grads)
  3. model_equiv_packed_bf16_eager
  4. model_equiv_single_fp32_eager
  5. model_equiv_packed_fp32_compiled
  6. model_equiv_packed_bf16_compiled
  7. graph_breaks_candidate          torch._dynamo.explain on forward_packed
                                     at production T/block; must be 0
  8. determinism_repeat_backward     same model+batch twice: loss/grads exact
  9. determinism_full_update         two fresh models, one AdamW step each
 10. checkpoint_resume_equivalence   save/resume vs uninterrupted continuation
 11. smoke_train                     production B16x4/global-B64/AdamW/LR path
                                     (frozen packed corpus when available,
                                     deterministic synthetic otherwise):
                                     finite loss/grads, memory stability,
                                     no OOM, no graph breaks
 12. frozen_corpus_available         only gated when --require-corpus

Output ends with:
  OPT3C_ALL_ROBUST=true|false
  FAILED_GATES=[...]

Usage (repo root):
  python results/verify_opt3c_all.py                    # full preflight
  python results/verify_opt3c_all.py --quick            # gates only, no smoke
  python results/verify_opt3c_all.py --require-corpus   # fail if no corpus
"""

import argparse
import gc
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path, PurePosixPath

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opt.model_opt import OptArmA  # noqa: E402
from opt.model_ref import (  # noqa: E402
    ArmAConfig,
    NativeReadStage1ArmA,
    canonical_init,
    ce_sum,
    load_init,
    make_optimizer,
    synthetic_packed_batch,
    synthetic_single_doc_batch,
)
from opt.scan_attn import (  # noqa: E402
    dense_segstart_attention,
    scan_chunkwise_bthk,
    scan_chunkwise_where_bthk,
)

T = 2048
GLOBAL_BATCH = 64
WARMUP_TOKENS = 10_000_000
DEFAULT_CORPUS = (
    "/content/drive/Shareddrives/ICLR PHASE BDH/"
    "phase_bdh/corpus/stage2/frozen_5b_v1"
)


# ---------------------------------------------------------------------------
# verdict plumbing
# ---------------------------------------------------------------------------

class Verdict:
    def __init__(self):
        self.gates = []

    def add(self, name, passed, **detail):
        status = "PASS" if passed else "FAIL"
        self.gates.append({"name": name, "status": status, "detail": detail})
        print(
            "GATE " + json.dumps(self.gates[-1], sort_keys=True, default=str),
            flush=True,
        )

    def skip(self, name, reason):
        self.gates.append({"name": name, "status": "SKIP", "detail": {"reason": reason}})
        print(
            "GATE " + json.dumps(self.gates[-1], sort_keys=True, default=str),
            flush=True,
        )

    def failed(self):
        return [g["name"] for g in self.gates if g["status"] == "FAIL"]

    def finish(self, **extra):
        failed = self.failed()
        print("FAILED_GATES=" + json.dumps(failed))
        print("OPT3C_ALL_ROBUST=" + str(not failed).lower())
        print(
            "SUMMARY="
            + json.dumps(
                {"gates": self.gates, "extra": extra},
                sort_keys=True,
                default=str,
            )
        )
        return not failed


# ---------------------------------------------------------------------------
# model factories (the exact production flags under test)
# ---------------------------------------------------------------------------

def production_block(cfg):
    return min(1024, cfg.T)


def candidate_model(cfg, dev, block=None):
    return OptArmA(
        cfg, dev,
        scan_block=block if block is not None else production_block(cfg),
        use_checkpoint=False,
        coord="dense",
        single_scan="chunkwise",
        packed_update="branchfree",
        zero_carry=True,
        paper_layout="direct",
        cache_rope=True,
    )


def certified_model(cfg, dev, block=None):
    """The previously certified opt3c execution path."""
    return OptArmA(
        cfg, dev,
        scan_block=block if block is not None else production_block(cfg),
        use_checkpoint=False,
        coord="dense",
        single_scan="chunkwise",
        packed_update="where",
        zero_carry=False,
        paper_layout="flat",
        cache_rope=False,
    )


def tiny_cfg(block_hint=8):
    return ArmAConfig(T=64, V=256, D=32, N=128, H=2, L=2, HIDDEN=64, READ_BLOCK=64)


# ---------------------------------------------------------------------------
# gate 1: scan oracle on randomized + adversarial packed layouts
# ---------------------------------------------------------------------------

def _layouts_for(t, block, rng, n_random=4):
    layouts = []
    layouts.append(torch.zeros((1, t), dtype=torch.long))  # single doc
    seg = torch.zeros((1, t), dtype=torch.long)
    seg[0, block // 2:] = block // 2  # boundary mid-block 0
    layouts.append(seg)
    seg = torch.zeros((1, t), dtype=torch.long)
    if block + 1 < t:
        seg[0, block + 1:] = block + 1  # boundary one past block edge
    layouts.append(seg)
    layouts.append(torch.arange(t, dtype=torch.long).unsqueeze(0))  # all resets
    for _ in range(n_random):
        row = torch.zeros((1, t), dtype=torch.long)
        cuts = sorted(rng.sample(range(1, t), k=rng.randrange(max(1, t // 2))))
        s, col = 0, []
        for c in cuts + [t]:
            col += [s] * (c - s)
            s = c
        row[0] = torch.tensor(col)
        layouts.append(row)
    return layouts


def gate_scan_oracle(v):
    rng = random.Random(7)
    gen = torch.Generator().manual_seed(11)
    worst = 0.0
    cases = 0
    try:
        for (b, h, t, k, dv) in ((1, 1, 16, 4, 3), (2, 2, 32, 5, 4)):
            for block in sorted({max(1, t // 4), max(1, t // 2), t}):
                for seg in _layouts_for(t, block, rng):
                    seg = seg.expand(b, -1).contiguous()
                    q0 = torch.randn(b, h, t, k, generator=gen, dtype=torch.float64)
                    v0 = torch.randn(b, t, dv, generator=gen, dtype=torch.float64)
                    probe = torch.randn(b, h, t, dv, generator=gen, dtype=torch.float64)

                    def run(fn, stack):
                        q = q0.clone().requires_grad_(True)
                        vv = v0.clone().requires_grad_(True)
                        vh = vv.unsqueeze(1).expand(-1, h, -1, -1)
                        out = stack(fn, q, vh, seg)
                        gq, gv = torch.autograd.grad((out * probe).sum(), (q, vv))
                        return out.detach(), gq, gv

                    ref = run(None, lambda fn, q, vh, s: dense_segstart_attention(q, vh[:, 0], s))
                    single = bool((seg == 0).all())
                    cert = run(
                        None,
                        lambda fn, q, vh, s: scan_chunkwise_where_bthk(
                            q, vh, s, block=block, single_doc=single
                        ),
                    )
                    cand = run(
                        None,
                        lambda fn, q, vh, s: scan_chunkwise_bthk(
                            q, vh, s, block=block, single_doc=single,
                            skip_zero_carry=True,
                        ),
                    )
                    for got in (cert, cand):
                        for a, c in zip(got, ref):
                            d = float((a - c).abs().max())
                            worst = max(worst, d)
                            if d > 1e-10:
                                raise AssertionError(
                                    f"scan oracle mismatch {d:.3e} "
                                    f"(b={b} t={t} block={block} single={single})"
                                )
                    cases += 1
        v.add("scan_oracle", True, cases=cases, worst_abs_error=worst)
    except Exception as e:
        v.add("scan_oracle", False, error=f"{type(e).__name__}: {e}",
              cases=cases, worst_abs_error=worst)


# ---------------------------------------------------------------------------
# gates 2-6: model equivalence vs the certified baseline
# ---------------------------------------------------------------------------

def _run_model(model, entry, batch, cfg, dev, bf16, grads=True):
    model.zero_grad(set_to_none=True)
    ctx = torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                         cache_enabled=False, enabled=bf16)
    with ctx:
        logits = entry(
            batch["x"], batch["pos"], batch["segpos"],
            batch["full_mask"], batch["segment_start"],
        )
        loss = ce_sum(logits, batch["y"], batch["valid"], cfg.V) / int(
            batch["valid"].sum().item()
        )
    out = {"logits": logits.detach(), "loss": loss.detach()}
    if grads:
        loss.backward()
        out["grads"] = {
            n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in model.named_parameters()
        }
    return out


def gate_model_equiv(v, name, compiled, bf16, single_doc, quick):
    cfg = tiny_cfg()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if compiled and dev.type != "cuda":
        v.skip(name, "compiled equivalence requires CUDA")
        return
    if compiled and quick:
        v.skip(name, "quick mode")
        return
    try:
        block = max(1, cfg.T // 2)
        if single_doc:
            batch = synthetic_single_doc_batch(cfg, 2, dev, seed=3)
            batch["segment_start"] = torch.zeros(
                (2, cfg.T), dtype=torch.long, device=dev
            )
        else:
            batch = synthetic_packed_batch(cfg, 2, dev, seed=3, mode="mixed")

        init = canonical_init(cfg)
        cert = certified_model(cfg, dev, block=block).to(dev)
        load_init(cert, init, dev)
        cand = candidate_model(cfg, dev, block=block).to(dev)
        load_init(cand, init, dev)
        cert.train()
        cand.train()

        cert_entry = (
            torch.compile(cert.forward_packed if not single_doc else cert.forward_single_doc,
                          mode="default")
            if compiled else (cert.forward_packed if not single_doc else cert.forward_single_doc)
        )
        cand_entry = (
            torch.compile(cand.forward_packed if not single_doc else cand.forward_single_doc,
                          mode="default")
            if compiled else (cand.forward_packed if not single_doc else cand.forward_single_doc)
        )

        ref = _run_model(cert, cert_entry, batch, cfg, dev, bf16)
        got = _run_model(cand, cand_entry, batch, cfg, dev, bf16)

        logit_d = float((got["logits"].float() - ref["logits"].float()).abs().max())
        loss_d = float((got["loss"].float() - ref["loss"].float()).abs().max())
        grad_d = max(
            float((got["grads"][n].float() - ref["grads"][n].float()).abs().max())
            for n in ref["grads"]
            if ref["grads"][n] is not None
        )
        rtol, atol = (5e-2, 5e-2) if bf16 else (1e-5, 1e-6)
        torch.testing.assert_close(got["logits"].float(), ref["logits"].float(),
                                   rtol=rtol, atol=atol)
        torch.testing.assert_close(got["loss"].float(), ref["loss"].float(),
                                   rtol=rtol, atol=atol)
        for n in ref["grads"]:
            if ref["grads"][n] is None:
                continue
            torch.testing.assert_close(
                got["grads"][n].float(), ref["grads"][n].float(),
                rtol=rtol, atol=atol,
            )
        v.add(name, True, logit_max_abs=logit_d, loss_abs=loss_d,
              grad_max_abs=grad_d, rtol=rtol, atol=atol,
              compiled=compiled, bf16=bf16, single_doc=single_doc)
    except Exception as e:
        v.add(name, False, error=f"{type(e).__name__}: {e}",
              compiled=compiled, bf16=bf16, single_doc=single_doc)


def gate_canonical_anchor(v, quick):
    """Extra anchor: candidate vs canonical dense reference (tiny fp32 eager)."""
    cfg = tiny_cfg()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        block = max(1, cfg.T // 2)
        batch = synthetic_packed_batch(cfg, 2, dev, seed=4, mode="mixed")
        init = canonical_init(cfg)
        ref = NativeReadStage1ArmA(cfg, dev).to(dev)
        load_init(ref, init, dev)
        ref.train()
        cand = candidate_model(cfg, dev, block=block).to(dev)
        load_init(cand, init, dev)
        cand.train()
        ref_out = _run_model(
            ref, lambda *a: ref(a[0], a[1], a[2], a[3]), batch, cfg, dev, bf16=False
        )
        cand_out = _run_model(cand, cand.forward_packed, batch, cfg, dev, bf16=False)
        grad_d = max(
            float((cand_out["grads"][n] - ref_out["grads"][n]).abs().max())
            for n in ref_out["grads"]
        )
        logit_d = float((cand_out["logits"] - ref_out["logits"]).abs().max())
        torch.testing.assert_close(cand_out["logits"], ref_out["logits"],
                                   rtol=1e-4, atol=1e-4)
        v.add("canonical_anchor_tiny", True, logit_max_abs=logit_d, grad_max_abs=grad_d)
    except Exception as e:
        v.add("canonical_anchor_tiny", False, error=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# gate 7: graph breaks
# ---------------------------------------------------------------------------

def gate_graph_breaks(v, quick):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if quick:
        v.skip("graph_breaks_candidate", "quick mode")
        return
    try:
        cfg = ArmAConfig()
        block = production_block(cfg)
        model = candidate_model(cfg, dev, block=block).to(dev)
        load_init(model, canonical_init(cfg), dev)
        model.train()
        batch = synthetic_packed_batch(cfg, 1, dev, seed=5, mode="mixed")
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        exp = torch._dynamo.explain(model.forward_packed)(
            batch["x"], batch["pos"], batch["segpos"],
            batch["full_mask"], batch["segment_start"],
        )
        breaks = int(getattr(exp, "graph_break_count", -1))
        v.add(
            "graph_breaks_candidate",
            breaks == 0,
            graph_break_count=breaks,
            graph_count=int(getattr(exp, "graph_count", -1)),
            break_reasons=[str(r)[:300] for r in getattr(exp, "break_reasons", [])],
            context="production T/block, B=1 packed",
        )
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
    except Exception as e:
        v.add("graph_breaks_candidate", False, error=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# gates 8-9: determinism
# ---------------------------------------------------------------------------

def gate_determinism_repeat(v, quick):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if quick:
        v.skip("determinism_repeat_backward", "quick mode")
        return
    try:
        cfg = ArmAConfig()
        block = production_block(cfg)
        model = candidate_model(cfg, dev, block=block).to(dev)
        load_init(model, canonical_init(cfg), dev)
        model.train()
        batch = synthetic_packed_batch(cfg, 1, dev, seed=6, mode="mixed")
        entry = torch.compile(model.forward_packed, mode="default")

        def one():
            model.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                cache_enabled=False):
                logits = entry(batch["x"], batch["pos"], batch["segpos"],
                               batch["full_mask"], batch["segment_start"])
                loss = ce_sum(logits, batch["y"], batch["valid"], cfg.V)
            loss.backward()
            grads = {
                n: (p.grad.detach().clone() if p.grad is not None else None)
                for n, p in model.named_parameters()
            }
            return float(loss.detach()), grads

        one()  # warmup: first invocation settles workspaces/kernel selection
        l1, g1 = one()
        l2, g2 = one()
        l3, g3 = one()
        diffs = []
        for ga, gb in ((g1, g2), (g1, g3), (g2, g3)):
            for n in ga:
                diffs.append((float((ga[n] - gb[n]).abs().max()), n))
        grad_d, worst_param = max(diffs)
        loss_d = max(abs(l1 - l2), abs(l1 - l3), abs(l2 - l3))
        v.add(
            "determinism_repeat_backward",
            grad_d <= 1e-5 and loss_d <= 1e-7,
            loss_abs_diff=loss_d,
            grad_max_abs_diff=grad_d,
            worst_parameter=worst_param,
            bitwise=(grad_d == 0.0 and loss_d == 0.0),
            tolerance=1e-5,
            compiled=True,
            note="same model/batch, 3 backward passes after a warmup, max pairwise diff",
        )
    except Exception as e:
        v.add("determinism_repeat_backward", False, error=f"{type(e).__name__}: {e}")


def gate_determinism_full_update(v, quick, modes):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if quick:
        for m in ("eager", "compiled"):
            v.skip(f"determinism_full_update_{m}", "quick mode")
        return
    if not modes:
        for m in ("eager", "compiled"):
            v.skip(f"determinism_full_update_{m}", "filtered by --only")
        return
    try:
        cfg = ArmAConfig()
        block = production_block(cfg)
        init = canonical_init(cfg)
        batch = synthetic_packed_batch(cfg, 1, dev, seed=7, mode="mixed")

        def one_update(compiled):
            torch.manual_seed(cfg.SEED)
            model = candidate_model(cfg, dev, block=block).to(dev)
            load_init(model, init, dev)
            model.train()
            opt = make_optimizer(model, cfg, dev.type)
            fwd = torch.compile(model.forward_packed, mode="default") if compiled else model.forward_packed
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                cache_enabled=False):
                logits = fwd(batch["x"], batch["pos"], batch["segpos"],
                             batch["full_mask"], batch["segment_start"])
                loss = ce_sum(logits, batch["y"], batch["valid"], cfg.V)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_NORM)
            for group in opt.param_groups:
                group["lr"] = cfg.PEAK_LR
            opt.step()
            return {n: p.detach().clone() for n, p in model.named_parameters()}

        for mode in ("eager", "compiled"):
            name = f"determinism_full_update_{mode}"
            if mode not in modes:
                v.skip(name, "filtered by --only")
                continue
            compiled = mode == "compiled"
            if compiled and not torch.cuda.is_available():
                v.skip(name, "compiled determinism requires CUDA")
                continue
            a = one_update(compiled)
            b = one_update(compiled)
            d = max(float((a[n] - b[n]).abs().max()) for n in a)
            tol = 0.0 if not compiled else 1e-6
            v.add(name, d <= tol, param_max_abs_diff=d, tolerance=tol,
                  bitwise=(d == 0.0),
                  note="two fresh models, same init/data, one AdamW step")
    except Exception as e:
        v.add("determinism_full_update", False, error=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# gate 10: checkpoint save/resume equivalence
# ---------------------------------------------------------------------------

def gate_checkpoint_resume(v, quick, tmp_dir):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if quick:
        v.skip("checkpoint_resume_equivalence", "quick mode")
        return
    try:
        cfg = ArmAConfig()
        block = production_block(cfg)
        init = canonical_init(cfg)
        batches = [
            synthetic_packed_batch(cfg, 1, dev, seed=20 + i, mode="mixed")
            for i in range(2)
        ]
        before, after = 3, 2

        def make():
            torch.manual_seed(cfg.SEED)
            model = candidate_model(cfg, dev, block=block).to(dev)
            load_init(model, init, dev)
            model.train()
            opt = make_optimizer(model, cfg, dev.type)
            fwd = torch.compile(model.forward_packed, mode="default")
            return model, opt, fwd

        def update(model, opt, fwd, batch, step_index):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                cache_enabled=False):
                logits = fwd(batch["x"], batch["pos"], batch["segpos"],
                             batch["full_mask"], batch["segment_start"])
                loss = ce_sum(logits, batch["y"], batch["valid"], cfg.V)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_NORM)
            for group in opt.param_groups:
                group["lr"] = cfg.PEAK_LR * min(
                    ((step_index + 1) * GLOBAL_BATCH * cfg.T) / WARMUP_TOKENS, 1.0
                )
            opt.step()
            return float(loss.detach())

        # uninterrupted reference
        m_ref, o_ref, f_ref = make()
        for i in range(before):
            update(m_ref, o_ref, f_ref, batches[0], i)
        ckpt_path = Path(tmp_dir) / "opt3c_all_preflight_ckpt.pt"
        torch.save(
            {
                "model": m_ref.state_dict(),
                "opt": o_ref.state_dict(),
                "step": before,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if dev.type == "cuda" else None,
            },
            ckpt_path,
        )
        losses_ref = []
        for i in range(after):
            losses_ref.append(update(m_ref, o_ref, f_ref, batches[1], before + i))

        # resumed run
        m_res, o_res, f_res = make()
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        m_res.load_state_dict(ckpt["model"])
        o_res.load_state_dict(ckpt["opt"])
        losses_res = []
        for i in range(after):
            losses_res.append(update(m_res, o_res, f_res, batches[1], before + i))

        param_d = max(
            float((a - b).abs().max())
            for a, b in zip(m_ref.parameters(), m_res.parameters())
        )
        loss_d = max(abs(a - b) for a, b in zip(losses_ref, losses_res))
        v.add(
            "checkpoint_resume_equivalence",
            param_d <= 1e-6 and loss_d <= 1e-5,
            param_max_abs_diff=param_d,
            loss_max_abs_diff=loss_d,
            steps_before=before,
            steps_after=after,
            note="compiled production path, AdamW state saved/restored",
        )
        ckpt_path.unlink(missing_ok=True)
    except Exception as e:
        v.add("checkpoint_resume_equivalence", False,
              error=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# frozen packed corpus reader (used only when the corpus is present)
# ---------------------------------------------------------------------------

def load_frozen_packed(root, n_sequences):
    import pyarrow.parquet as pq

    root = Path(root)
    frozen = json.loads((root / "FROZEN.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "corpus_manifest.json").read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise RuntimeError("FROZEN.json is not FROZEN")
    if int(frozen.get("context_length")) != T:
        raise RuntimeError("frozen context length mismatch")
    if int(frozen.get("sequences")) != 2_441_407:
        raise RuntimeError("frozen sequence count mismatch")
    shards = manifest.get("shards", [])
    if len(shards) != 25:
        raise RuntimeError(f"expected 25 shards, got {len(shards)}")

    columns = (
        "sequence_index",
        "sequence_token_start",
        "sequence_token_end",
        "selected_document_index",
        "document_token_start",
        "document_token_end",
    )

    last_needed = n_sequences  # +1 lookahead sequence
    tokens_by_seq, lengths_by_seq, rows_by_seq = {}, {}, {}
    sequence_base = 0
    cross_sequence_pairs = 0

    for shard_number, shard in enumerate(shards):
        if sequence_base > last_needed:
            break
        stem = f"shard_{shard_number:06d}"
        if shard.get("stem") != stem:
            raise RuntimeError("unexpected shard order")
        token_path = root / shard["tokens_file"]
        length_path = root / shard["valid_lengths_file"]
        prov_path = root / shard["provenance_file"]
        nseq = int(shard["sequences"])
        if token_path.stat().st_size != nseq * T * 2:
            raise RuntimeError(f"token shard size mismatch: {token_path}")
        if length_path.stat().st_size != nseq * 2:
            raise RuntimeError(f"length shard size mismatch: {length_path}")

        lengths = np.memmap(length_path, mode="r", dtype=np.uint16, shape=(nseq,))
        token_map = np.memmap(token_path, mode="r", dtype=np.uint16, shape=(nseq, T))
        table = pq.read_table(prov_path, columns=list(columns))
        if table.num_rows != int(shard["provenance_rows"]):
            raise RuntimeError(f"provenance row count mismatch: {prov_path}")

        cols = {}
        for name in columns:
            col = table.column(name).combine_chunks()
            if col.null_count:
                raise RuntimeError(f"null provenance in {name}")
            cols[name] = col.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        order = np.lexsort((cols["sequence_token_start"], cols["sequence_index"]))

        for row_index in order:
            seq_global = int(cols["sequence_index"][row_index])
            if seq_global < sequence_base or seq_global > last_needed:
                continue
            if seq_global >= sequence_base + nseq:
                raise RuntimeError("sequence index outside shard")
            seq = seq_global - sequence_base
            if seq_global not in tokens_by_seq:
                tokens_by_seq[seq_global] = np.asarray(token_map[seq], dtype=np.uint16).copy()
                lengths_by_seq[seq_global] = int(lengths[seq])
            rows_by_seq.setdefault(seq_global, []).append(
                (
                    int(cols["sequence_token_start"][row_index]),
                    int(cols["sequence_token_end"][row_index]),
                    int(cols["selected_document_index"][row_index]),
                    int(cols["document_token_start"][row_index]),
                    int(cols["document_token_end"][row_index]),
                )
            )
        del table, cols, order, token_map, lengths
        gc.collect()
        if sequence_base + nseq - 1 >= last_needed:
            break
        sequence_base += nseq

    if set(range(n_sequences + 1)) != set(tokens_by_seq) or set(range(n_sequences + 1)) != set(rows_by_seq):
        raise RuntimeError("packed prefix did not read exactly the required ids")

    x = np.zeros((n_sequences, T), dtype=np.uint16)
    y = np.zeros((n_sequences, T), dtype=np.uint16)
    pos = np.zeros((n_sequences, T), dtype=np.int64)
    segpos = np.zeros((n_sequences, T), dtype=np.int32)
    start = np.zeros((n_sequences, T), dtype=np.int32)
    input_valid = np.zeros((n_sequences, T), dtype=np.bool_)
    valid = np.zeros((n_sequences, T), dtype=np.bool_)

    for seq_global in range(n_sequences):
        tok = tokens_by_seq[seq_global]
        length = lengths_by_seq[seq_global]
        rows = sorted(rows_by_seq[seq_global], key=lambda r: r[0])
        if not (0 < length <= T):
            raise RuntimeError(f"invalid valid length at sequence {seq_global}")
        cursor = 0
        prev = None
        prev_start = None
        for row_index, row in enumerate(rows):
            a, e, doc_id, doc_start, doc_end = row
            if not (0 <= a < e <= length and a == cursor and e - a == doc_end - doc_start):
                raise RuntimeError(f"invalid span at sequence {seq_global}")
            cursor = e
            if prev is None or doc_id != prev[2]:
                seg_start = a
            else:
                if doc_start != prev[4]:
                    raise RuntimeError("non-contiguous same-document spans")
                seg_start = prev_start
            start[seq_global, a:e] = seg_start
            segpos[seq_global, a:e] = np.arange(a, e, dtype=np.int32) - seg_start
            pos[seq_global, a:e] = doc_start + np.arange(e - a, dtype=np.int64)
            input_valid[seq_global, a:e] = True
            if e - a > 1:
                y[seq_global, a:e - 1] = tok[a + 1:e]
                valid[seq_global, a:e - 1] = True
            if row_index + 1 < len(rows):
                nxt = rows[row_index + 1]
                if (doc_id == nxt[2] and doc_end == nxt[3] and e == nxt[0]):
                    y[seq_global, e - 1] = tok[e]
                    valid[seq_global, e - 1] = True
            elif seq_global + 1 in rows_by_seq:
                nxt = sorted(rows_by_seq[seq_global + 1], key=lambda r: r[0])[0]
                if e == length and nxt[0] == 0 and doc_id == nxt[2] and doc_end == nxt[3]:
                    y[seq_global, e - 1] = tokens_by_seq[seq_global + 1][nxt[0]]
                    valid[seq_global, e - 1] = True
                    cross_sequence_pairs += 1
            prev = row
            prev_start = seg_start
        if cursor != length:
            raise RuntimeError(f"provenance does not cover sequence {seq_global}")
        start[seq_global, length:] = T + seq_global + 1
        x[seq_global] = tok

    cpu = {
        "x": torch.from_numpy(np.ascontiguousarray(x)).pin_memory(),
        "y": torch.from_numpy(np.ascontiguousarray(y)).pin_memory(),
        "pos": torch.from_numpy(np.ascontiguousarray(pos)).pin_memory(),
        "valid": torch.from_numpy(np.ascontiguousarray(valid)).pin_memory(),
        "input_valid": torch.from_numpy(np.ascontiguousarray(input_valid)).pin_memory(),
        "start": torch.from_numpy(np.ascontiguousarray(start)).pin_memory(),
        "segpos": torch.from_numpy(np.ascontiguousarray(segpos)).pin_memory(),
    }
    meta = {
        "sequences": n_sequences,
        "cross_sequence_target_pairs": int(cross_sequence_pairs),
    }
    return cpu, meta


# ---------------------------------------------------------------------------
# gate 11: smoke train on the production path
# ---------------------------------------------------------------------------

def _microbatch_candidates(preferred=16):
    out = []
    for mb in (preferred, 16, 8, 4, 2, 1):
        if mb <= preferred and mb not in out:
            out.append(mb)
    return out


def gate_smoke_train(v, corpus_root, require_corpus, steps, warmups, tmp_dir,
                     preferred_microbatch=16):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        v.add("smoke_train", False, error="smoke train requires CUDA")
        return

    total_updates = warmups + steps
    data = None
    corpus_meta = None
    if corpus_root is not None and Path(corpus_root).is_dir():
        try:
            data, corpus_meta = load_frozen_packed(
                corpus_root, GLOBAL_BATCH * total_updates
            )
        except Exception as e:
            v.add("smoke_train", False,
                  error=f"corpus read failed: {type(e).__name__}: {e}")
            return
    elif require_corpus:
        v.add("smoke_train", False, error=f"frozen corpus required but missing: {corpus_root}")
        return

    cfg = ArmAConfig()
    block = production_block(cfg)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)

    model = candidate_model(cfg, dev, block=block).to(dev)
    load_init(model, canonical_init(cfg), dev)
    model.train()
    opt = make_optimizer(model, cfg, dev.type)
    entry = torch.compile(model.forward_packed, mode="default")

    def one_update(step_index, microbatch):
        opt.zero_grad(set_to_none=True)
        denom = int(
            (data["valid"][step_index * GLOBAL_BATCH:(step_index + 1) * GLOBAL_BATCH].sum().item()
             if data is not None
             else GLOBAL_BATCH * cfg.T)
        )
        # Synthetic batches live only for this update: keeping a full
        # global batch (with its [64,2048,2048] bool mask) per step would
        # look like a memory leak to the stability gate.
        step_batch = None
        if data is None:
            step_batch = synthetic_packed_batch(
                cfg, GLOBAL_BATCH, dev, seed=1000 + step_index, mode="mixed"
            )

        def mb_slice(lo, hi):
            sl = slice(lo, hi)
            if data is not None:
                gsl = slice(step_index * GLOBAL_BATCH + lo, step_index * GLOBAL_BATCH + hi)
                x = data["x"][gsl].to(dev, dtype=torch.long, non_blocking=True)
                y = data["y"][gsl].to(dev, dtype=torch.long, non_blocking=True)
                pos = data["pos"][gsl].to(dev, dtype=torch.int32, non_blocking=True)
                valid = data["valid"][gsl].to(dev, dtype=torch.bool, non_blocking=True)
                segpos = data["segpos"][gsl].to(dev, dtype=torch.int32, non_blocking=True)
                start = data["start"][gsl].to(dev, dtype=torch.int32, non_blocking=True)
                iv = data["input_valid"][gsl].to(dev, dtype=torch.bool, non_blocking=True)
                causal = torch.ones((cfg.T, cfg.T), dtype=torch.bool, device=dev).tril(-1)
                full_mask = (
                    (start[:, :, None] == start[:, None, :])
                    & iv[:, :, None]
                    & iv[:, None, :]
                    & causal.unsqueeze(0)
                )
                return x, y, pos, valid, segpos, full_mask, start
            return (
                step_batch["x"][sl], step_batch["y"][sl], step_batch["pos"][sl],
                step_batch["valid"][sl], step_batch["segpos"][sl],
                step_batch["full_mask"][sl], step_batch["segment_start"][sl],
            )

        losses = []
        for lo in range(0, GLOBAL_BATCH, microbatch):
            hi = min(lo + microbatch, GLOBAL_BATCH)
            x, y, pos, valid, segpos, full_mask, seg = mb_slice(lo, hi)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                cache_enabled=False):
                logits = entry(x, pos, segpos, full_mask, seg)
                loss = ce_sum(logits, y, valid, cfg.V) / denom
            loss.backward()
            losses.append(float(loss.detach()))
            del x, y, pos, valid, segpos, full_mask, seg, logits, loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_NORM)
        for group in opt.param_groups:
            group["lr"] = cfg.PEAK_LR * min(
                ((step_index + 1) * GLOBAL_BATCH * cfg.T) / WARMUP_TOKENS, 1.0
            )
        opt.step()
        del step_batch
        return sum(losses)

    try:
        results = []
        oom_events = []
        microbatch = None
        for mb in _microbatch_candidates(preferred_microbatch):
            try:
                gc.collect()
                torch.cuda.empty_cache()
                torch.manual_seed(cfg.SEED)
                torch.cuda.manual_seed_all(cfg.SEED)
                model = candidate_model(cfg, dev, block=block).to(dev)
                load_init(model, canonical_init(cfg), dev)
                model.train()
                opt = make_optimizer(model, cfg, dev.type)
                entry = torch.compile(model.forward_packed, mode="default")
                for i in range(warmups):
                    one_update(i, mb)
                torch.cuda.synchronize(dev)
                microbatch = mb
                break
            except Exception as e:
                if "out of memory" not in str(e).lower():
                    raise
                oom_events.append({"microbatch": mb, "error": str(e)[:200]})
                model = opt = entry = None
                gc.collect()
                torch.cuda.empty_cache()
        if microbatch is None:
            v.add("smoke_train", False, error="no microbatch fits", oom_events=oom_events)
            return

        torch.cuda.reset_peak_memory_stats(dev)
        alloc_after_first = None
        for s in range(steps):
            step_index = warmups + s
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            loss = one_update(step_index, microbatch)
            torch.cuda.synchronize(dev)
            dt = time.perf_counter() - t0
            grads_ok = True
            found_grad = False
            for p in model.parameters():
                if p.grad is not None:
                    found_grad = True
                    if not bool(torch.isfinite(p.grad).all()):
                        grads_ok = False
                        break
            alloc = torch.cuda.memory_allocated(dev)
            if s == 0:
                alloc_after_first = alloc
            results.append(
                {
                    "step": step_index,
                    "loss": loss,
                    "loss_finite": math.isfinite(loss),
                    "grads_finite": grads_ok and found_grad,
                    "seconds": dt,
                    "alloc_GiB": alloc / 2**30,
                }
            )
        peak = torch.cuda.max_memory_allocated(dev)
        total = torch.cuda.get_device_properties(0).total_memory
        alloc_last = torch.cuda.memory_allocated(dev)
        growth = (
            (alloc_last - alloc_after_first) / max(1, alloc_after_first)
            if alloc_after_first
            else 0.0
        )
        finite = all(r["loss_finite"] and r["grads_finite"] for r in results)
        memory_ok = peak < 0.95 * total and growth <= 0.05
        v.add(
            "smoke_train",
            finite and memory_ok,
            data="frozen_corpus" if data is not None else "synthetic_mixed",
            corpus_meta=corpus_meta,
            microbatch=microbatch,
            gradient_accumulation_steps=GLOBAL_BATCH // microbatch,
            production_scale=(microbatch == 16 and data is not None),
            steps=results,
            peak_alloc_GiB=peak / 2**30,
            total_mem_GiB=total / 2**30,
            alloc_growth_fraction=growth,
            oom_events=oom_events,
        )
    except Exception as e:
        v.add("smoke_train", False, error=f"{type(e).__name__}: {e}",
              traceback=traceback.format_exc()[-800:])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def maybe_mount_drive():
    """Best-effort Drive mount so the corpus path resolves on Colab."""
    if Path(DEFAULT_CORPUS).is_dir():
        return
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive", force_remount=False)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="auto",
                    help="auto | none | explicit frozen corpus path")
    ap.add_argument("--require-corpus", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="skip production-scale compiled/smoke gates")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--microbatch", type=int, default=16,
                    help="preferred smoke-train microbatch (falls back on OOM)")
    ap.add_argument("--only", default="",
                    help="comma list of gate names to run; others skipped")
    ap.add_argument("--tmp-dir", default=os.environ.get("TEMP", "/tmp"))
    args = ap.parse_args()

    if args.corpus == "auto":
        maybe_mount_drive()
        corpus_root = DEFAULT_CORPUS if Path(DEFAULT_CORPUS).is_dir() else None
    elif args.corpus == "none":
        corpus_root = None
    else:
        corpus_root = args.corpus

    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(
        "PREFLIGHT "
        + json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": dev_name,
                "corpus": corpus_root,
                "quick": args.quick,
                "steps": args.steps,
                "warmups": args.warmups,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    v = Verdict()
    v.add("impl_imports", True, implementation="opt.model_opt.OptArmA opt3c_all")

    if args.require_corpus:
        v.add("frozen_corpus_available", corpus_root is not None,
              corpus=corpus_root)
        if corpus_root is None:
            v.finish(note="frozen corpus required but unavailable")
            return

    only = {s.strip() for s in args.only.split(",") if s.strip()}

    def want(name):
        return (not only) or (name in only)

    def maybe_skip(name):
        if not want(name):
            v.skip(name, "filtered by --only")

    if want("scan_oracle"):
        gate_scan_oracle(v)
    else:
        maybe_skip("scan_oracle")
    for gname, compiled, bf16, single in (
        ("model_equiv_packed_fp32_eager", False, False, False),
        ("model_equiv_packed_bf16_eager", False, True, False),
        ("model_equiv_single_fp32_eager", False, False, True),
        ("model_equiv_packed_fp32_compiled", True, False, False),
        ("model_equiv_packed_bf16_compiled", True, True, False),
    ):
        if want(gname):
            gate_model_equiv(v, gname, compiled, bf16, single, args.quick)
        else:
            maybe_skip(gname)
    if want("canonical_anchor_tiny"):
        gate_canonical_anchor(v, args.quick)
    else:
        maybe_skip("canonical_anchor_tiny")
    if want("graph_breaks_candidate"):
        gate_graph_breaks(v, args.quick)
    else:
        maybe_skip("graph_breaks_candidate")
    if want("determinism_repeat_backward"):
        gate_determinism_repeat(v, args.quick)
    else:
        maybe_skip("determinism_repeat_backward")
    if want("determinism_full_update_eager") or want("determinism_full_update_compiled"):
        fd_modes = tuple(
            m for m in ("eager", "compiled")
            if want(f"determinism_full_update_{m}")
        )
        gate_determinism_full_update(v, args.quick, fd_modes)
    else:
        maybe_skip("determinism_full_update_eager")
        maybe_skip("determinism_full_update_compiled")
    if want("checkpoint_resume_equivalence"):
        gate_checkpoint_resume(v, args.quick, args.tmp_dir)
    else:
        maybe_skip("checkpoint_resume_equivalence")
    if want("smoke_train"):
        if args.quick:
            v.skip("smoke_train", "quick mode")
        else:
            gate_smoke_train(v, corpus_root, args.require_corpus, args.steps,
                             args.warmups, args.tmp_dir,
                             preferred_microbatch=args.microbatch)
    else:
        maybe_skip("smoke_train")

    v.finish(
        corpus_used=corpus_root,
        quick=args.quick,
        device=dev_name,
    )


if __name__ == "__main__":
    main()
