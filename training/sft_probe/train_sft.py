"""Full-parameter SFT calibration trainer for Arm-A.

Starts from the untouched frozen 2.5B checkpoint, reuses the frozen trainer's
model class, optimizer construction and update routine
(``one_full_update``: BF16 autocast + FP32 master AdamW fused, clip 1.0,
betas 0.9/0.95, wd 0.1), and applies the SFT loss only to assistant response
tokens. No architecture change, no LoRA, no special tokens, no cross-example
attention.

Dose accounting uses loss-bearing instruction target tokens; replay tokens are
counted separately.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from training.sft_probe import common, packing
else:
    from . import common, packing

T = 2048


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument(
        "--checkpoint-at",
        default="",
        help="comma-separated instruction target-token doses to save at",
    )
    parser.add_argument("--replay-pct", type=float, default=0.0)
    parser.add_argument("--tokens-per-update", type=int, default=8192)
    parser.add_argument("--microbatch-rows", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--init-from", default=None)
    parser.add_argument(
        "--reset-data-cursor",
        action="store_true",
        help=(
            "when resuming from --init-from on a different dataset, start "
            "the data plan at row 0 instead of the saved cursor"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--data-dir", default=str(common.DATA_DIR))
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--grad-probe-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=common.SPLIT_SEED)
    return parser.parse_args(argv)


def load_split(data_dir: Path, split: str):
    tokens = packing.load_tokens_jsonl(data_dir / f"{split}_tokens.jsonl")
    plan = packing.load_row_plan_jsonl(data_dir / f"{split}_rows.jsonl")
    return tokens, plan


def load_replay_rows(data_dir: Path):
    rows = []
    with open(data_dir / "replay_rows.jsonl", "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
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


def iter_stream(
    plan,
    examples,
    replay,
    replay_pct,
    start_row,
    replay_cursor,
    instruction_counter,
    replay_counter,
):
    """Deterministic row stream: instruction rows plus replay rows.

    Replay rows are inserted with the schedule
    ``floor((n_instr) * pct/100) > n_replay`` so the cumulative replay share
    converges to exactly ``pct`` by processed sequence tokens (all rows are
    2048 tokens).
    """
    row_cursor = start_row
    replay_index = replay_cursor
    while row_cursor < len(plan):
        entry = plan[row_cursor]
        examples_sel = [
            {
                "index": i,
                "prefix_ids": examples[i]["p"],
                "response_ids": examples[i]["r"],
            }
            for i in entry["examples"]
        ]
        built = packing.pack_examples(examples_sel, T=T)
        if len(built) != 1:
            raise RuntimeError(f"row plan row {row_cursor} did not pack to 1")
        instruction_counter += 1
        row_cursor += 1
        yield (
            "instruction",
            built[0],
            {
                "row_cursor": row_cursor,
                "replay_cursor": replay_index,
                "instruction_counter": instruction_counter,
                "replay_counter": replay_counter,
            },
        )
        if replay_pct > 0 and replay:
            wanted = (instruction_counter * replay_pct / (100.0 - replay_pct))
            while replay_counter < int(wanted):
                replay_row = replay[replay_index % len(replay)]
                replay_index += 1
                replay_counter += 1
                yield (
                    "replay",
                    replay_row,
                    {
                        "row_cursor": row_cursor,
                        "replay_cursor": replay_index,
                        "instruction_counter": instruction_counter,
                        "replay_counter": replay_counter,
                    },
                )


def grad_group_rms(model) -> dict:
    out = {}
    for group, names in common.PARAM_GROUPS.items():
        sq = 0.0
        count = 0
        for name in names:
            param = model.get_parameter(name)
            if param.grad is None:
                continue
            grad = param.grad.detach().to(torch.float64)
            sq += float((grad * grad).sum().item())
            count += grad.numel()
        out[group] = float(np.sqrt(sq / count)) if count else float("nan")
    return out


def param_group_norms(model) -> dict:
    out = {}
    for group, names in common.PARAM_GROUPS.items():
        sq = 0.0
        for name in names:
            param = model.get_parameter(name).detach().to(torch.float64)
            sq += float((param * param).sum().item())
        out[group] = float(np.sqrt(sq))
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir) if args.out_dir else common.CKPT_DIR / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    trainer = common.load_trainer()
    cfg = trainer.ArmAConfig()

    data_dir = Path(args.data_dir)
    examples, plan = load_split(data_dir, "train")
    replay = load_replay_rows(data_dir) if args.replay_pct > 0 else []
    if args.replay_pct > 0 and not replay:
        raise RuntimeError("replay requested but proxy_rows.jsonl is empty")

    start_row = 0
    replay_cursor = 0
    instruction_counter = 0
    replay_counter = 0
    updates_done = 0
    instruction_target_done = 0
    replay_target_done = 0
    previous_wall = 0.0

    state = common.load_base_state()
    optimizer_state = None
    if args.init_from:
        payload = torch.load(args.init_from, map_location="cpu",
                             weights_only=False)
        state = payload["model"]
        optimizer_state = payload.get("optimizer")
        meta = payload.get("sft", {})
        if args.reset_data_cursor:
            start_row = 0
            replay_cursor = 0
        else:
            start_row = int(meta.get("row_cursor", 0))
            replay_cursor = int(meta.get("replay_cursor", 0))
        instruction_counter = int(meta.get("instruction_counter", 0)) \
            if not args.reset_data_cursor else 0
        replay_counter = int(meta.get("replay_counter", 0)) \
            if not args.reset_data_cursor else 0
        updates_done = int(meta.get("updates_done", 0))
        instruction_target_done = int(
            meta.get("instruction_target_tokens", 0)
        )
        replay_target_done = int(meta.get("replay_target_tokens", 0))
        previous_wall = float(meta.get("wall_seconds", 0.0))

    model = common.build_model(state, device=device.type)
    del state
    model.to(device)
    model.train()

    sft_cfg = replace(
        cfg,
        PEAK_LR=float(args.lr),
        MICROBATCH=int(args.microbatch_rows),
        GLOBAL_BATCH=max(1, args.tokens_per_update // T),
        WARMUP_TOKENS=int(args.warmup_steps * args.tokens_per_update),
    )
    optimizer = common.make_optimizer(model, sft_cfg, device.type)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        optimizer_state = None

    compiled = (
        torch.compile(model.forward_packed, mode="default")
        if args.compile
        else model.forward_packed
    )

    checkpoint_doses = [
        int(x) for x in str(args.checkpoint_at).split(",") if x.strip()
    ]
    checkpoint_doses = sorted(set(checkpoint_doses))
    saved = set()

    log_handle = open(log_path, "a", encoding="utf-8")

    def log(event: str, **fields):
        record = {"event": event, "ts": common.iso_now(), **fields}
        line = json.dumps(record, sort_keys=True, default=str)
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    def write_checkpoint(tag: str, final: bool = False):
        meta = {
            "arm": args.arm,
            "lr": float(args.lr),
            "replay_pct": float(args.replay_pct),
            "updates_done": updates_done,
            "row_cursor": int(start_row),
            "replay_cursor": int(replay_cursor),
            "instruction_counter": int(instruction_counter),
            "replay_counter": int(replay_counter),
            "instruction_target_tokens": int(instruction_target_done),
            "replay_target_tokens": int(replay_target_done),
            "target_tokens": int(instruction_target_done),
            "wall_seconds": float(previous_wall),
            "tokens_per_update": int(args.tokens_per_update),
            "microbatch_rows": int(args.microbatch_rows),
            "warmup_steps": int(args.warmup_steps),
            "init_from": str(args.init_from) if args.init_from else None,
            "base_checkpoint_sha256": common.BASE_CKPT_SHA256,
            "data_manifest": str(data_dir / "data_manifest.json"),
        }
        payload = common.sft_checkpoint_payload(
            model,
            optimizer,
            cfg,
            meta,
            {
                "updates_done": updates_done,
                "tokens_consumed": instruction_target_done
                + replay_target_done,
                "target_tokens": int(instruction_target_done),
                "next_sequence": int(start_row),
            },
        )
        path = out_dir / f"{tag}.pt"
        common.save_checkpoint_atomic(path, payload)
        log("checkpoint_saved", tag=tag, path=str(path))

    log(
        "start",
        arm=args.arm,
        lr=args.lr,
        replay_pct=args.replay_pct,
        target_tokens=args.target_tokens,
        tokens_per_update=args.tokens_per_update,
        microbatch_rows=args.microbatch_rows,
        warmup_steps=args.warmup_steps,
        init_from=args.init_from,
        start_row=start_row,
        updates_done=updates_done,
        compile=bool(args.compile),
        torch=torch.__version__,
        device=str(device),
        gpu=(
            torch.cuda.get_device_name(0) if device.type == "cuda" else None
        ),
    )

    rows = []
    accum_started = time.perf_counter()
    update_index = updates_done
    stream = iter_stream(
        plan,
        examples,
        replay,
        args.replay_pct,
        start_row,
        replay_cursor,
        instruction_counter,
        replay_counter,
    )
    stopped = False
    for kind, row, cursors in stream:
        if not rows:
            accum_started = time.perf_counter()
        rows.append(row)
        start_row = cursors["row_cursor"]
        replay_cursor = cursors["replay_cursor"]
        if kind == "instruction":
            instruction_target_done += row.target_tokens
        else:
            replay_target_done += row.target_tokens
        if len(rows) < max(1, args.tokens_per_update // T):
            continue
        cpu_batch = packing.rows_to_cpu_batch(rows)
        sequence_tokens = int(sum(r.sequence_tokens for r in rows))
        rows = []
        stats = trainer.one_full_update(
            cpu_batch,
            model,
            compiled,
            optimizer,
            update_index,
            sft_cfg,
            device,
            check_grads=True,
        )
        updates_done += 1
        update_index += 1
        update_seconds = time.perf_counter() - accum_started
        previous_wall += update_seconds

        if not bool(torch.isfinite(model.readout).all()):
            log("hard_stop", reason="non-finite weights", update=updates_done)
            raise RuntimeError("non-finite weights after optimizer step")

        record = {
            "update": updates_done,
            "loss": stats["loss"],
            "lr": stats["lr"],
            "grad_norm": stats["grad_norm"],
            "sequence_tokens": sequence_tokens,
            "instruction_target_tokens": instruction_target_done,
            "replay_target_tokens": replay_target_done,
            "step_seconds": update_seconds,
            "tok_s": sequence_tokens / max(update_seconds, 1e-9),
            "peak_mem_gib": (
                torch.cuda.max_memory_allocated() / 2**30
                if device.type == "cuda"
                else None
            ),
        }
        if (
            args.grad_probe_every > 0
            and updates_done % args.grad_probe_every == 0
        ):
            record["grad_group_rms"] = grad_group_rms(model)
            record["param_group_norm"] = param_group_norms(model)
        log("update", **record)

        for dose in checkpoint_doses:
            if dose not in saved and instruction_target_done >= dose:
                saved.add(dose)
                write_checkpoint(f"dose_{dose}")
        if args.max_updates and updates_done >= args.max_updates:
            log("max_updates_reached", updates_done=updates_done)
            stopped = True
            break
        if instruction_target_done >= args.target_tokens:
            stopped = True
            break

    if not stopped and not rows and instruction_target_done < args.target_tokens:
        log("stream_exhausted",
            instruction_target_done=instruction_target_done)
    if rows:
        cpu_batch = packing.rows_to_cpu_batch(rows)
        stats = trainer.one_full_update(
            cpu_batch, model, compiled, optimizer, update_index, sft_cfg,
            device, check_grads=True,
        )
        updates_done += 1
        previous_wall += time.perf_counter() - accum_started
        log("short_update", loss=stats["loss"], rows=len(rows))

    write_checkpoint("final", final=True)
    log(
        "end",
        updates_done=updates_done,
        instruction_target_tokens=instruction_target_done,
        replay_target_tokens=replay_target_done,
        wall_seconds=previous_wall,
        target_tokens=args.target_tokens,
        reached_target=instruction_target_done >= args.target_tokens,
    )
    log_handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
