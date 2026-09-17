"""Native BDH sparse-population probes on a fixed deterministic probe set.

Runs the exact frozen Arm-A forward (same ``OptArmA`` operators, subclassed
only to record intermediates) on a frozen set of proxy-text windows, and
streams out the campaign-required per-level metrics:

  x = ReLU(v @ decoder_x)
  y = ReLU(LN(attention_result) @ decoder_y)
  u = x * y
  g = coordinator(v, segpos, full_mask)
  delta = writer(g * base)

Recorded per level: zero fraction, mean, RMS, top-64 / top-256 mass share,
global top-64 / top-256 neuron index sets, coordinator g statistics,
writer norms, and hidden-state drift versus the base checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from training.sft_probe import common
else:
    from . import common

SAMPLE_STRIDE = 4
PROBE_ROWS = 4
HEADS = 4
K = 4096


def build_probe_model(trainer, state, device):
    class _Probe(trainer.OptArmA):
        def __init__(self, cfg, dev):
            super().__init__(cfg, dev, scan_block=cfg.SCAN_BLOCK)
            self.capture = None
            self.level_index = 0

        def begin(self, capture):
            self.capture = capture
            self.level_index = 0

        def level(self, v, pos, segpos, full_mask, segment_start, cs, sn):
            cfg = self.cfg
            x_bt = self.project_x_native(v)
            a = self.ln(self.attention_scan(x_bt, v, pos, segment_start,
                                            cs, sn))
            ypre = F.relu(a @ self.decoder_y)
            prod = x_bt * ypre.permute(0, 2, 1, 3)
            paper_y_flat = prod.reshape(v.shape[0], cfg.T, cfg.N)
            base = self.ln(paper_y_flat @ self.encoder)
            g = self.coordinator(v, segpos, full_mask)
            delta = self.writer(g * base)
            v_new = self.ln(v + delta)
            if self.capture is not None:
                self.capture(
                    self.level_index,
                    {
                        "x": x_bt,
                        "y": ypre.permute(0, 2, 1, 3),
                        "u": prod,
                        "base": base,
                        "g": g,
                        "delta": delta,
                        "v_before": v,
                        "v_after": v_new,
                    },
                )
            self.level_index += 1
            return v_new

    model = _Probe(trainer.ArmAConfig(), device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def load_probe_rows(data_dir: Path, rows: int = PROBE_ROWS):
    records = []
    with open(data_dir / "proxy_rows.jsonl", "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= rows:
                break
            records.append(json.loads(line))
    return records


def rows_to_device(records, device):
    def stack(key, dtype):
        return torch.tensor(
            np.stack([r[key] for r in records]), dtype=dtype, device=device
        )

    x = stack("x", torch.long)
    pos = stack("pos", torch.int32)
    segpos = stack("segpos", torch.int32)
    start = stack("start", torch.int32)
    input_valid = stack("input_valid", torch.bool)
    causal = torch.ones(
        (x.shape[1], x.shape[1]), dtype=torch.bool, device=device
    ).tril(diagonal=-1)
    batch = {
        "x": x,
        "pos": pos,
        "segpos": segpos,
        "start": start,
        "input_valid": input_valid,
        "full_mask": (
            (start[:, :, None] == start[:, None, :])
            & input_valid[:, :, None]
            & input_valid[:, None, :]
            & causal.unsqueeze(0)
        ),
    }
    batch["valid"] = stack("valid", torch.bool)
    return batch


class Accumulator:
    def __init__(self, t: int, sample_stride: int = SAMPLE_STRIDE):
        self.sample_index = torch.arange(0, t, sample_stride)
        self.levels = {}
        self.hidden = {}

    def _level(self, level: int, device):
        state = self.levels.get(level)
        if state is None:
            state = {
                "count": 0,
                "pos_x": 0.0,
                "pos_y": 0.0,
                "pos_u": 0.0,
                "sum_x": 0.0,
                "sum_y": 0.0,
                "sum_u": 0.0,
                "sq_x": 0.0,
                "sq_y": 0.0,
                "sq_u": 0.0,
                "mass_x": torch.zeros(HEADS * K, dtype=torch.float64,
                                      device=device),
                "mass_y": torch.zeros(HEADS * K, dtype=torch.float64,
                                      device=device),
                "mass_u": torch.zeros(HEADS * K, dtype=torch.float64,
                                      device=device),
                "top64_mass_x": 0.0,
                "top256_mass_x": 0.0,
                "top64_mass_y": 0.0,
                "top256_mass_y": 0.0,
                "top64_mass_u": 0.0,
                "top256_mass_u": 0.0,
                "top_n": 0,
                "g_sum": 0.0,
                "g_sq": 0.0,
                "g_n": 0,
                "g_min": float("inf"),
                "g_max": float("-inf"),
                "base_sq": 0.0,
                "delta_sq": 0.0,
                "writer_n": 0,
                "ratio_sum": 0.0,
                "ratio_n": 0,
            }
            self.levels[level] = state
        return state

    @torch.no_grad()
    def __call__(self, level: int, tensors: dict):
        state = self._level(level, tensors["x"].device)
        b, t, h, k = tensors["x"].shape
        idx = self.sample_index.to(tensors["x"].device)
        x = tensors["x"]
        y = tensors["y"]
        u = tensors["u"]
        state["count"] += x.numel()
        state["pos_x"] += float((x > 0).sum().item())
        state["pos_y"] += float((y > 0).sum().item())
        state["pos_u"] += float((u > 0).sum().item())
        state["sum_x"] += float(x.detach().double().sum().item())
        state["sum_y"] += float(y.detach().double().sum().item())
        state["sum_u"] += float(u.detach().double().sum().item())
        state["sq_x"] += float((x.detach().double() ** 2).sum().item())
        state["sq_y"] += float((y.detach().double() ** 2).sum().item())
        state["sq_u"] += float((u.detach().double() ** 2).sum().item())
        for key, tensor in (("x", x), ("y", y), ("u", u)):
            sampled = tensor[:, idx, :, :].detach().double()
            flat = sampled.reshape(-1, k)
            total = flat.sum(dim=-1).clamp_min(1e-30)
            order = flat.sort(dim=-1, descending=True).values
            cumulative = order.cumsum(-1)
            state[f"top64_mass_{key}"] += float(
                (cumulative[:, 63] / total).sum().item()
            )
            state[f"top256_mass_{key}"] += float(
                (cumulative[:, 255] / total).sum().item()
            )
            mass = flat.reshape(-1, h, k).sum(dim=0)
            state[f"mass_{key}"] += mass.reshape(-1).double()
        state["top_n"] += int(x.shape[0] * idx.numel() * h)

        g = tensors["g"].detach().double()
        state["g_sum"] += float(g.sum().item())
        state["g_sq"] += float((g * g).sum().item())
        state["g_n"] += g.numel()
        state["g_min"] = min(state["g_min"], float(g.min().item()))
        state["g_max"] = max(state["g_max"], float(g.max().item()))

        base = tensors["base"].detach().double()
        delta = tensors["delta"].detach().double()
        v_before = tensors["v_before"].detach().double()
        state["base_sq"] += float((base * base).sum().item())
        state["delta_sq"] += float((delta * delta).sum().item())
        state["writer_n"] += base.numel()
        ratio = delta.norm(dim=-1) / v_before.norm(dim=-1).clamp_min(1e-12)
        state["ratio_sum"] += float(ratio.sum().item())
        state["ratio_n"] += ratio.numel()

        self.hidden[level] = (
            tensors["v_after"][:, idx, :].detach().to(torch.float32).cpu()
        )

    @torch.no_grad()
    def finalize(self, rho: float) -> dict:
        levels = []
        for level in sorted(self.levels):
            state = self.levels[level]
            count = max(1, state["count"])
            top_n = max(1, state["top_n"])
            entry = {
                "level": level,
                "rho": rho,
                "x": {
                    "zero_fraction": 1.0 - state["pos_x"] / count,
                    "mean": state["sum_x"] / count,
                    "rms": float(np.sqrt(state["sq_x"] / count)),
                    "top64_mass_share": state["top64_mass_x"] / top_n,
                    "top256_mass_share": state["top256_mass_x"] / top_n,
                },
                "y": {
                    "zero_fraction": 1.0 - state["pos_y"] / count,
                    "mean": state["sum_y"] / count,
                    "rms": float(np.sqrt(state["sq_y"] / count)),
                    "top64_mass_share": state["top64_mass_y"] / top_n,
                    "top256_mass_share": state["top256_mass_y"] / top_n,
                },
                "u": {
                    "zero_fraction": 1.0 - state["pos_u"] / count,
                    "mean": state["sum_u"] / count,
                    "rms": float(np.sqrt(state["sq_u"] / count)),
                    "top64_mass_share": state["top64_mass_u"] / top_n,
                    "top256_mass_share": state["top256_mass_u"] / top_n,
                },
                "coordinator": {
                    "g_mean": state["g_sum"] / max(1, state["g_n"]),
                    "g_std": float(np.sqrt(max(
                        0.0,
                        state["g_sq"] / max(1, state["g_n"])
                        - (state["g_sum"] / max(1, state["g_n"])) ** 2,
                    ))),
                    "g_min": state["g_min"],
                    "g_max": state["g_max"],
                },
                "writer": {
                    "base_rms": float(np.sqrt(
                        state["base_sq"] / max(1, state["writer_n"])
                    )),
                    "delta_rms": float(np.sqrt(
                        state["delta_sq"] / max(1, state["writer_n"])
                    )),
                    "delta_over_v_ratio": state["ratio_sum"]
                    / max(1, state["ratio_n"]),
                },
            }
            for key in ("x", "y", "u"):
                mass = state[f"mass_{key}"].reshape(HEADS, K)
                order = torch.argsort(-mass, dim=-1)
                entry[f"{key}_top64_indices"] = (
                    order[:, :64].to(torch.int32).cpu().tolist()
                )
                entry[f"{key}_top256_indices"] = (
                    order[:, :256].to(torch.int32).cpu().tolist()
                )
            levels.append(entry)
        hidden = {
            str(level): tensor.numpy() for level, tensor in self.hidden.items()
        }
        return {"levels": levels, "_hidden": hidden}


def hidden_drift(current: dict, base: dict) -> dict:
    out = {}
    for level in sorted(current):
        a = torch.from_numpy(current[level]).double()
        b = torch.from_numpy(base[level]).double()
        cos = F.cosine_similarity(a.reshape(-1, a.shape[-1]),
                                  b.reshape(-1, b.shape[-1]), dim=-1)
        rmsd = ((a - b) ** 2).mean().sqrt()
        out[level] = {
            "cosine_mean": float(cos.mean().item()),
            "cosine_min": float(cos.min().item()),
            "rms_diff": float(rmsd.item()),
        }
    return out


def topk_overlap(a, b) -> float:
    if a and isinstance(a[0], (list, tuple)):
        return float(np.mean([
            topk_overlap(x, y) for x, y in zip(a, b)
        ]))
    sa = set(int(v) for v in a)
    sb = set(int(v) for v in b)
    if not sa:
        return float("nan")
    return len(sa & sb) / len(sa)


def run_probe(state, device="cuda", data_dir=None):
    device = torch.device(device)
    trainer = common.load_trainer()
    data_dir = Path(data_dir) if data_dir else common.DATA_DIR
    records = load_probe_rows(data_dir)
    batch = rows_to_device(records, device)
    model = build_probe_model(trainer, state, device)
    rho = float(torch.sigmoid(model.coordinator.alpha).item())
    accumulator = Accumulator(t=batch["x"].shape[1])
    model.begin(accumulator)
    with torch.no_grad():
        model.forward_packed(
            batch["x"],
            batch["pos"],
            batch["segpos"],
            batch["full_mask"],
            batch["start"],
        )
    result = accumulator.finalize(rho)
    result["probe"] = {
        "rows": len(records),
        "tokens_per_row": int(batch["x"].shape[1]),
        "sampled_positions_per_row": int(len(accumulator.sample_index)),
        "sample_stride": SAMPLE_STRIDE,
        "target_tokens": int(batch["valid"].sum().item()),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def compare_to_base(sft: dict, base: dict, base_hidden: dict) -> dict:
    base_levels = {level["level"]: level for level in base["levels"]}
    for entry in sft["levels"]:
        reference = base_levels[entry["level"]]
        for key in ("x", "y", "u"):
            for topk in (64, 256):
                entry[f"{key}_top{topk}_overlap_vs_base"] = topk_overlap(
                    entry[f"{key}_top{topk}_indices"],
                    reference[f"{key}_top{topk}_indices"],
                )
    drift = hidden_drift(sft["_hidden"], base_hidden)
    for entry in sft["levels"]:
        entry["hidden_vs_base"] = drift[str(entry["level"])]
    return sft


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None,
                        help="trainer checkpoint; omit for the base 2.5B")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=str(common.OUT_DIR / "native"))
    parser.add_argument(
        "--base-json", default=str(common.OUT_DIR / "native" / "base.json")
    )
    parser.add_argument(
        "--base-hidden",
        default=str(common.OUT_DIR / "native" / "base_hidden.npz"),
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu",
                             weights_only=False)
        state = payload["model"]
    else:
        state = common.load_base_state()

    result = run_probe(state, device=args.device)
    result["tag"] = args.tag
    result["checkpoint"] = (
        str(args.checkpoint) if args.checkpoint else str(common.BASE_CKPT)
    )
    base_path = Path(args.base_json)
    if not args.checkpoint:
        hidden = result.pop("_hidden")
        common.save_json(base_path, result)
        np.savez_compressed(args.base_hidden, **hidden)
        print(f"wrote base probe to {base_path}")
        return 0
    base = common.load_json(base_path)
    base_hidden = dict(np.load(args.base_hidden))
    result = compare_to_base(result, base, base_hidden)
    hidden = result.pop("_hidden")
    np.savez_compressed(out_dir / f"{args.tag}_hidden.npz", **hidden)
    common.save_json(out_dir / f"{args.tag}.json", result)
    summary = {
        entry["level"]: {
            "x_top64_overlap": round(entry["x_top64_overlap_vs_base"], 4),
            "u_top64_overlap": round(entry["u_top64_overlap_vs_base"], 4),
            "x_zero_frac": round(entry["x"]["zero_fraction"], 4),
            "hidden_cosine": round(
                entry["hidden_vs_base"]["cosine_mean"], 6
            ),
        }
        for entry in result["levels"]
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
