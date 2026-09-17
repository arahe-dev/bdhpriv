"""P7: dimensionless cross-system concentration comparison.

Compares per-unit concentration of:
  * Arm-A (one trained trajectory, synthetic packed batches, E1/E2)
      - x coordinate total activation mass (per level-head and pooled)
      - u = x*y coordinate total activation mass
      - coordinate activation frequency p_act
  * Gemma Scope 2 270M layer-12 SAEs (small/medium): encoder direction norms
    (weight-space energy; decoders are unit-normalized), thresholds.
  * Anthropic public features: activation density per feature.

Writes results/sae/cross_system_comparison.json. Every quantity carries its
denominator and semantic definition; no cross-unit raw comparisons.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
RAW = RESULTS / "arm_a_science" / "raw"
SAE = RESULTS / "sae"
ASSET = ROOT / "data" / "sae" / "gemma-scope-2-270m-pt" / "mlp_out"
ANTHROPIC = (ROOT / "data" / "sae" / "neuronpedia-sae-concepts" / "anthropic"
             / "train" / "monosemantic_2023.parquet")


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64).ravel())
    if x.size == 0 or x[-1] <= 0:
        return float("nan")
    n = x.size
    i = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (i * x).sum()) / (n * x.sum()) - (n + 1.0) / n)


def neff_over_n(x: np.ndarray) -> float:
    p = np.asarray(x, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return float("nan")
    p = p / p.sum()
    return float(np.exp(-(p * np.log(p)).sum()) / p.size)


def pr_over_n(x: np.ndarray) -> float:
    p = np.asarray(x, dtype=np.float64)
    p = p / max(p.sum(), 1e-300)
    return float(1.0 / (p * p).sum() / p.size)


def curve(x: np.ndarray, fracs=None) -> dict:
    if fracs is None:
        fracs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5]
    v = np.sort(np.asarray(x, dtype=np.float64).ravel())[::-1]
    tot = v.sum()
    n = v.size
    out = {}
    for f in fracs:
        c = max(1, int(round(f * n)))
        out[str(f)] = float(v[:c].sum() / tot) if tot > 0 else float("nan")
    return out


def units_for_share(x: np.ndarray, shares=(0.25, 0.5, 0.75, 0.9, 0.99)):
    v = np.sort(np.asarray(x, dtype=np.float64).ravel())[::-1]
    tot = v.sum()
    n = v.size
    csum = np.cumsum(v) / max(tot, 1e-300)
    out = {}
    for s in shares:
        idx = int(np.searchsorted(csum, s)) + 1
        out[str(s)] = min(idx, n) / n
    return out


def summarize(x: np.ndarray, extra: dict | None = None) -> dict:
    x = np.asarray(x, dtype=np.float64)
    out = {
        "n_units": int(x.size),
        "zero_fraction": float((x == 0).mean()),
        "gini": gini(x),
        "neff_over_n": neff_over_n(x),
        "participation_ratio_over_n": pr_over_n(x),
        "top_share_curve": curve(x),
        "unit_fraction_for_share": units_for_share(x),
        "quantiles": {f"p{q}": float(np.quantile(x, q))
                      for q in (0.01, 0.1, 0.5, 0.9, 0.99)},
    }
    if extra:
        out.update(extra)
    return out


def main():
    out = {
        "meta": {
            "purpose": "dimensionless concentration comparison; each entry "
                       "names its denominator and semantics",
            "warning": "these quantities are NOT the same measurement: "
                       "Arm-A mass = accumulated activation over sampled "
                       "tokens; Gemma encoder norm^2 = weight-space energy "
                       "(decoder rows are unit-normalized); Anthropic density "
                       "= activation frequency",
            "arm_a_evidence_class": "E1/E2 (synthetic packed batches)",
        },
        "systems": {},
    }
    # ---------- Arm-A ----------
    p1 = np.load(RAW / "pass1_latest.npz")
    n_batches, L, H, K = p1["mass_sum_x"].shape
    meta = json.loads((RAW / "meta_latest.json").read_text(encoding="utf-8"))
    rows = meta["specs"][0]["rows"]
    n_tokens = n_batches * rows * meta["model_config"]["T"]
    arm = {"denominator": "K=4096 coordinates per head; pooled N=16384"}
    for key in ("x", "u"):
        mass = p1[f"mass_sum_{key}"].sum(0)          # (L,H,K)
        pact = p1[f"pos_count_{key}"].sum(0) / n_tokens
        arm[f"{key}_mass_per_cell"] = summarize(mass.reshape(-1))
        arm[f"{key}_mass_pooled"] = summarize(
            mass.transpose(0, 2, 1).reshape(-1))
        arm[f"{key}_pact_per_cell"] = summarize(pact.reshape(-1))
        arm[f"{key}_pact_pooled"] = summarize(
            pact.transpose(0, 2, 1).reshape(-1))
    out["systems"]["arm_a_latest"] = arm

    # ---------- Gemma SAE ----------
    for which in ("small", "medium"):
        p = ASSET / f"layer_12_width_16k_l0_{which}" / "params.safetensors"
        with safe_open(str(p), framework="pt", device="cpu") as f:
            w_enc = f.get_tensor("w_enc").float().numpy()
            w_dec = f.get_tensor("w_dec").float().numpy()
            thr = f.get_tensor("threshold").float().numpy()
        enc_norm = np.linalg.norm(w_enc, axis=0)
        dec_norm = np.linalg.norm(w_dec, axis=1)
        out["systems"][f"gemma_{which}"] = {
            "denominator": "16384 features x 640 dims; decoder rows unit-norm",
            "encoder_norm_sq": summarize(
                enc_norm ** 2, {"semantics": "weight-space energy"}),
            "encoder_norm": summarize(
                enc_norm, {"semantics": "weight-space norm"}),
            "threshold_positive": summarize(
                thr, {"semantics": "jump_relu activation threshold"}),
            "decoder_norm": summarize(
                dec_norm, {"semantics": "unit-normalized by construction"}),
        }

    # ---------- Anthropic ----------
    tab = pq.read_table(ANTHROPIC, columns=["density", "max_activation"])
    density = tab.column("density").to_numpy(zero_copy_only=False).astype(
        np.float64)
    maxact = tab.column("max_activation").to_numpy(
        zero_copy_only=False).astype(np.float64)
    out["systems"]["anthropic_public"] = {
        "denominator": "2,149,712 public features (Claude 1 SAE features)",
        "density": summarize(
            density, {"semantics": "activation frequency per feature"}),
        "max_activation": summarize(
            maxact, {"semantics": "per-feature max activation"}),
    }
    # rank-frequency slopes (log-log) on positive values
    slopes = {}
    for name, x in (
            ("arm_a_x_mass_pooled",
             p1["mass_sum_x"].sum(0).transpose(0, 2, 1).reshape(-1)),
            ("arm_a_u_mass_pooled",
             p1["mass_sum_u"].sum(0).transpose(0, 2, 1).reshape(-1)),
            ("anthropic_density", density),
            ("gemma_small_encoder_norm_sq",
             np.linalg.norm(
                 safe_open(str(ASSET / "layer_12_width_16k_l0_small"
                               / "params.safetensors"),
                           framework="pt", device="cpu")
                 .get_tensor("w_enc").float().numpy(), axis=0) ** 2)):
        v = np.sort(np.asarray(x, dtype=np.float64))[::-1]
        v = v[v > 0]
        k = min(10000, v.size)
        ranks = np.arange(1, k + 1)
        slopes[name] = float(np.polyfit(np.log(ranks), np.log(v[:k]), 1)[0])
    out["rank_frequency_loglog_slope_top10k"] = slopes
    SAE.mkdir(parents=True, exist_ok=True)
    (SAE / "cross_system_comparison.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps({
        "arm_a_x_pooled": {k: round(v, 4) for k, v in
                           out["systems"]["arm_a_latest"][
                               "x_mass_pooled"].items()
                           if isinstance(v, float)},
        "anthropic_density": {k: round(v, 4) for k, v in
                              out["systems"]["anthropic_public"][
                                  "density"].items()
                              if isinstance(v, float)},
        "slopes": slopes,
    }, indent=2))


if __name__ == "__main__":
    main()
