"""Checkpoint evaluation: SFT val NLL, proxy LM NLL, parameter drift, native
BDH probes.

All LM losses are computed with the frozen ``forward_packed`` semantics and
exact target masks:
  * SFT validation NLL: assistant-response targets only (held-out split);
  * BASE_TEXT_PROXY NLL: ordinary next-token loss on non-instruction prose
    (explicit proxy because the frozen 5B corpus is not available locally).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from training.sft_probe import common, packing
else:
    from . import common, packing


def load_state(checkpoint: str | None):
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu",
                             weights_only=False)
        return payload["model"], payload
    return common.load_base_state(), None


def lr_loss_rows(rows, model, device, autocast=False):
    """Weighted mean CE over the packed rows' valid target positions."""
    total_loss = 0.0
    total_valid = 0
    for row in rows:
        cpu = packing.rows_to_cpu_batch([row])
        batch = packing.cpu_batch_to_device(cpu, device)
        with torch.no_grad():
            context = (
                torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if autocast and device.type == "cuda"
                else torch.no_grad()
            )
            with context:
                logits = model.forward_packed(
                    batch["x"],
                    batch["pos"],
                    batch["segpos"],
                    batch["full_mask"],
                    batch["start"],
                )
                per_token = F.cross_entropy(
                    logits.reshape(-1, model.cfg.V),
                    batch["y"].reshape(-1),
                    reduction="none",
                )
        valid = batch["valid"].reshape(-1)
        total_loss += float(per_token[valid].sum().item())
        total_valid += int(valid.sum().item())
        del batch, logits, per_token
    return total_loss / max(1, total_valid), total_valid


def eval_sft_val(model, device, data_dir: Path, autocast=False, limit=None):
    examples, plan = (
        packing.load_tokens_jsonl(data_dir / "val_tokens.jsonl"),
        packing.load_row_plan_jsonl(data_dir / "val_rows.jsonl"),
    )
    rows = []
    for entry in plan[:limit] if limit else plan:
        built = packing.pack_examples(
            [
                {
                    "index": i,
                    "prefix_ids": examples[i]["p"],
                    "response_ids": examples[i]["r"],
                }
                for i in entry["examples"]
            ],
            T=2048,
        )
        rows.extend(built)
    started = time.perf_counter()
    loss, count = lr_loss_rows(rows, model, device, autocast=autocast)
    return {
        "sft_val_nll": loss,
        "sft_val_target_tokens": count,
        "sft_val_rows": len(rows),
        "sft_val_seconds": time.perf_counter() - started,
        "autocast_bf16": bool(autocast),
    }


def load_proxy_rows(data_dir: Path, limit=None):
    rows = []
    with open(data_dir / "proxy_rows.jsonl", "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            record = __import__("json").loads(line)
            rows.append(
                packing.PackedRow(
                    x=np.asarray(record["x"], dtype=np.uint16),
                    y=np.asarray(record["y"], dtype=np.uint16),
                    pos=np.asarray(record["pos"], dtype=np.int32),
                    segpos=np.asarray(record["segpos"], dtype=np.int32),
                    start=np.asarray(record["start"], dtype=np.int32),
                    input_valid=np.asarray(
                        record["input_valid"], dtype=np.bool_
                    ),
                    valid=np.asarray(record["valid"], dtype=np.bool_),
                    example_indices=[],
                    target_tokens=int(record["target_tokens"]),
                    sequence_tokens=int(record["sequence_tokens"]),
                )
            )
    return rows


def eval_proxy(model, device, data_dir: Path, autocast=False, limit=None):
    rows = load_proxy_rows(data_dir, limit=limit)
    started = time.perf_counter()
    loss, count = lr_loss_rows(rows, model, device, autocast=autocast)
    return {
        "proxy_nll": loss,
        "proxy_target_tokens": count,
        "proxy_rows": len(rows),
        "proxy_seconds": time.perf_counter() - started,
        "proxy_label": "BASE_TEXT_PROXY",
        "autocast_bf16": bool(autocast),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-dir", default=str(common.DATA_DIR))
    parser.add_argument("--out-dir", default=str(common.OUT_DIR / "eval"))
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--params", action="store_true")
    parser.add_argument("--no-lm", action="store_true")
    parser.add_argument("--autocast", action="store_true")
    parser.add_argument(
        "--val-limit", type=int, default=0,
        help="debug only: limit validation rows",
    )
    parser.add_argument(
        "--base-hidden",
        default=str(common.OUT_DIR / "native" / "base_hidden.npz"),
    )
    parser.add_argument(
        "--base-json", default=str(common.OUT_DIR / "native" / "base.json")
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state, payload = load_state(args.checkpoint)
    model = common.build_model(state, device=device.type)
    model.to(device)
    model.eval()

    report = {
        "tag": args.tag,
        "checkpoint": args.checkpoint or str(common.BASE_CKPT),
        "is_base": args.checkpoint is None,
        "created_at": common.iso_now(),
    }
    if payload is not None:
        report["sft_meta"] = payload.get("sft")
        report["checkpoint_progress"] = payload.get("progress")

    if not args.no_lm:
        report["lm"] = {}
        report["lm"].update(
            eval_sft_val(
                model, device, data_dir,
                autocast=args.autocast,
                limit=args.val_limit or None,
            )
        )
        report["lm"].update(
            eval_proxy(model, device, data_dir, autocast=args.autocast)
        )

    if args.params:
        if args.checkpoint is None:
            report["params"] = None
        else:
            base_state = common.load_base_state()
            report["params"] = common.parameter_drift(base_state, state)
            del base_state

    if args.native:
        native_mod = native_probe_module()
        result = native_mod.run_probe(state, device=args.device)
        if args.checkpoint is not None:
            base = common.load_json(Path(args.base_json))
            base_hidden = dict(np.load(args.base_hidden))
            result = native_mod.compare_to_base(result, base, base_hidden)
        result.pop("_hidden", None)
        report["native"] = result

    common.save_json(out_dir / f"{args.tag}.json", report)
    summary = {
        "tag": args.tag,
        "sft_val_nll": report.get("lm", {}).get("sft_val_nll"),
        "proxy_nll": report.get("lm", {}).get("proxy_nll"),
        "total_relative_weight_drift": (
            report.get("params", {}) or {}
        ).get("groups", {}).get("TOTAL", {}).get("relative_update_norm"),
    }
    print(__import__("json").dumps(summary, indent=1))
    return 0


def native_probe_module():
    from training.sft_probe import native_probe

    return native_probe


if __name__ == "__main__":
    sys.exit(main())
