"""Track A.1: full Arm-A 2.5B training-trajectory analysis.

Parses train.jsonl completely, applies robust smoothing (rolling median +
quantiles), quantifies stationarity, spikes and region behavior, and emits
machine-readable JSON plus the F1 figure.

Run: py -3.12 opt/arm_a_trajectory.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rolling_quantile(values: np.ndarray, window: int, q: float) -> np.ndarray:
    half = window // 2
    out = np.empty_like(values, dtype=np.float64)
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out[i] = np.quantile(values[lo:hi], q)
    return out


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    return rolling_quantile(values, window, 0.5)


def rolling_mad(values: np.ndarray, window: int) -> np.ndarray:
    med = rolling_median(values, window)
    half = window // 2
    out = np.empty_like(values, dtype=np.float64)
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out[i] = np.median(np.abs(values[lo:hi] - med[i]))
    return out


def rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    # average ties
    _, inverse, counts = np.unique(a, return_inverse=True,
                                   return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float((rx @ ry) / math.sqrt((rx @ rx) * (ry @ ry)))


def region_stats(tokens, values):
    return {
        "tokens": [int(tokens[0]), int(tokens[-1])],
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/arm_a_2p5b_opt3c_all")
    parser.add_argument("--out",
                        default="results/arm_a_training_trajectory.json")
    parser.add_argument("--fig", default="figures/fig_training.png")
    args = parser.parse_args()

    records = []
    log_path = Path(args.run_dir) / "logs" / "train.jsonl"
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    updates = [r for r in records if r.get("event") == "update"]
    if not updates:
        raise SystemExit("no update events found")

    steps = np.array([u["step"] for u in updates], dtype=np.float64)
    tokens = np.array([u["tokens_consumed"] for u in updates],
                      dtype=np.float64)
    loss = np.array([u["loss"] for u in updates], dtype=np.float64)
    lr = np.array([u["lr"] for u in updates], dtype=np.float64)
    grad = np.array([u["grad_norm"] for u in updates], dtype=np.float64)
    tok_s = np.array([u["tok_s"] for u in updates], dtype=np.float64)
    step_ms = np.array([u["step_ms"] for u in updates], dtype=np.float64)
    mem_peak = np.array([u.get("mem_peak_GiB", np.nan) for u in updates])
    mem_alloc = np.array([u.get("mem_alloc_GiB", np.nan) for u in updates])
    valid_pairs = np.array([u["valid_pairs"] for u in updates],
                           dtype=np.float64)
    elapsed = np.array([u["elapsed_s"] for u in updates], dtype=np.float64)

    window = 51
    loss_med = rolling_median(loss, window)
    loss_p10 = rolling_quantile(loss, window, 0.10)
    loss_p90 = rolling_quantile(loss, window, 0.90)
    loss_mad = rolling_mad(loss, window)
    tok_med = rolling_median(tok_s, window)

    warmup = tokens < 10_000_000
    regions = {
        "warmup": warmup,
        "early": (tokens >= 10_000_000) & (tokens < 500_000_000),
        "mid": (tokens >= 500_000_000) & (tokens < 2_000_000_000),
        "late": tokens >= 2_000_000_000,
    }
    region_report = {}
    for name, mask in regions.items():
        if mask.sum() == 0:
            continue
        region_report[name] = {
            "loss": region_stats(tokens[mask], loss[mask]),
            "tok_s": region_stats(tokens[mask], tok_s[mask]),
            "step_ms": region_stats(tokens[mask], step_ms[mask]),
            "grad_norm": region_stats(tokens[mask], grad[mask]),
            "lr_last": float(lr[mask][-1]),
        }

    rho = spearman(steps, tok_s)
    n = len(steps)
    t_stat = rho * math.sqrt((n - 2) / max(1e-12, 1 - rho ** 2))
    first200 = np.median(tok_s[:200])
    last200 = np.median(tok_s[-200:])
    drift = {
        "spearman_rho_step_vs_tok_s": rho,
        "t_statistic": t_stat,
        "first200_median_tok_s": float(first200),
        "last200_median_tok_s": float(last200),
        "ratio_last_first": float(last200 / first200),
        "step_ms_spearman": spearman(steps, step_ms),
    }

    spike_mask = loss > (loss_med + 3.0 * 1.4826 * loss_mad)
    spikes = [
        {
            "step": int(steps[i]),
            "tokens": int(tokens[i]),
            "loss": float(loss[i]),
            "rolling_median": float(loss_med[i]),
            "valid_pairs": int(valid_pairs[i]),
        }
        for i in np.where(spike_mask)[0]
    ]
    dips = [
        {
            "step": int(steps[i]),
            "tokens": int(tokens[i]),
            "loss": float(loss[i]),
            "rolling_median": float(loss_med[i]),
        }
        for i in np.where(loss < (loss_med - 3.0 * 1.4826 * loss_mad))[0]
    ]
    corr_valid_loss = float(np.corrcoef(valid_pairs, loss)[0, 1])

    thresholds = {}
    for thr in (4.0, 3.5, 3.0, 2.9, 2.85, 2.8):
        hits = np.where(loss_med <= thr)[0]
        if len(hits):
            thresholds[str(thr)] = int(tokens[hits[0]])

    late_window = loss_med[-200:]
    late_slope = float(np.polyfit(np.arange(len(late_window)), late_window, 1)[0])

    summary = {
        "run_dir": str(args.run_dir),
        "update_events": int(len(updates)),
        "log_every_updates": 10,
        "updates_covered": [int(steps[0]), int(steps[-1])],
        "tokens_covered": [int(tokens[0]), int(tokens[-1])],
        "wall_seconds": float(elapsed[-1]),
        "wall_hours": float(elapsed[-1] / 3600.0),
        "regions": region_report,
        "stationarity": drift,
        "spikes": {
            "count": len(spikes),
            "top_by_excess": sorted(
                spikes, key=lambda s: -(s["loss"] - s["rolling_median"])
            )[:10],
            "all_steps": [s["step"] for s in spikes],
        },
        "dips": {"count": len(dips), "examples": dips[:5]},
        "loss_valid_pairs_pearson": corr_valid_loss,
        "loss_threshold_tokens_rolling_median": thresholds,
        "late_rolling_median_slope_per_update": late_slope,
        "final": {
            "loss": float(loss[-1]),
            "loss_last50_mean": float(np.mean(loss[-50:])),
            "loss_last50_std": float(np.std(loss[-50:])),
            "loss_last500_mean": float(np.mean(loss[-500:])),
            "tok_s_last": float(tok_s[-1]),
            "step_ms_last": float(step_ms[-1]),
            "grad_norm_last": float(grad[-1]),
            "mem_peak_GiB_max": float(np.nanmax(mem_peak)),
            "lr_last": float(lr[-1]),
        },
        "curves": {
            "steps": steps.astype(int).tolist(),
            "tokens": tokens.astype(int).tolist(),
            "loss": loss.tolist(),
            "loss_rolling_median": loss_med.tolist(),
            "loss_rolling_p10": loss_p10.tolist(),
            "loss_rolling_p90": loss_p90.tolist(),
            "tok_s": tok_s.tolist(),
            "tok_s_rolling_median": tok_med.tolist(),
            "grad_norm": grad.tolist(),
            "elapsed_s": elapsed.tolist(),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=130)
    mt = tokens / 1e6
    axes[0, 0].plot(mt, loss, lw=0.7, alpha=0.45, label="loss")
    axes[0, 0].plot(mt, loss_med, lw=1.6, label="rolling median")
    axes[0, 0].plot(mt, loss_p10, lw=1.0, ls="--", label="rolling p10")
    axes[0, 0].plot(mt, loss_p90, lw=1.0, ls="--", label="rolling p90")
    axes[0, 0].set_xlabel("tokens (M)")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].set_title("Arm-A 2.5B: loss vs tokens")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(mt, tok_s, lw=0.6, alpha=0.4, label="tok/s")
    axes[0, 1].plot(mt, tok_med, lw=1.6, label="rolling median")
    axes[0, 1].set_xlabel("tokens (M)")
    axes[0, 1].set_ylabel("tok/s")
    axes[0, 1].set_title(f"throughput (rho={rho:.3f}, last/first={last200/first200:.3f})")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(mt, grad, lw=0.7, alpha=0.5)
    axes[1, 0].plot(mt, rolling_median(grad, window), lw=1.5)
    axes[1, 0].set_xlabel("tokens (M)")
    axes[1, 0].set_ylabel("grad norm")
    axes[1, 0].set_title("grad norm vs tokens")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(mt, step_ms, lw=0.6, alpha=0.4, label="step ms")
    axes[1, 1].plot(mt, rolling_median(step_ms, window), lw=1.5,
                    label="rolling median")
    ax2 = axes[1, 1].twinx()
    ax2.plot(mt, mem_peak, lw=1.0, color="tab:red", alpha=0.7,
             label="peak GiB")
    ax2.set_ylabel("peak GiB", color="tab:red")
    axes[1, 1].set_xlabel("tokens (M)")
    axes[1, 1].set_ylabel("step ms")
    axes[1, 1].set_title("full update time and peak memory")
    axes[1, 1].legend(fontsize=8, loc="upper left")
    axes[1, 1].grid(alpha=0.25)

    fig.tight_layout()
    fig_path = Path(args.fig)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)
    print(json.dumps({
        "out": str(out),
        "fig": str(fig_path),
        "wall_hours": summary["wall_hours"],
        "stationarity": drift,
        "spikes": len(spikes),
        "dips": len(dips),
        "loss_valid_pairs_pearson": corr_valid_loss,
        "thresholds": thresholds,
        "late_slope": late_slope,
        "final": summary["final"],
        "regions": {k: {"loss_median": v["loss"]["median"],
                        "tok_s_median": v["tok_s"]["median"]}
                    for k, v in region_report.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
