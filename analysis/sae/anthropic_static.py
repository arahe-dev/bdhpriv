"""Anthropic public feature dataset: schema + supported-field analysis.

The dataset is public feature-level data for Claude 1 / Claude 3 Sonnet SAE
features (no SAE weights). Writes
  results/sae/anthropic_schema.json
  results/sae/anthropic_analysis.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "sae" / "neuronpedia-sae-concepts" / "anthropic" / "train"
OUT = ROOT / "results" / "sae"


def quantiles(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    qs = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
    out = {f"p{q*100:g}": float(np.quantile(x, q)) for q in qs}
    out.update({"mean": float(x.mean()), "std": float(x.std()),
                "min": float(x.min()), "max": float(x.max()),
                "n": int(x.size)})
    return out


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64).ravel())
    if x.size == 0 or x[-1] <= 0:
        return float("nan")
    n = x.size
    i = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (i * x).sum()) / (n * x.sum()) - (n + 1.0) / n)


def lorenz(x: np.ndarray, fracs=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5)) -> dict:
    v = np.sort(np.asarray(x, dtype=np.float64).ravel())[::-1]
    total = v.sum()
    n = v.size
    out = {}
    for f in fracs:
        c = max(1, int(round(f * n)))
        out[str(f)] = float(v[:c].sum() / total) if total > 0 else float("nan")
    return out


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() /
                 np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def schema_report():
    out = {"files": {}}
    for name in ("anthropic_concepts.parquet", "monosemantic_2023.parquet"):
        f = pq.ParquetFile(DATA / name)
        fields = []
        for field in f.schema_arrow:
            col = f.read_row_group(0, columns=[field.name]).column(0)
            vals = col.to_pylist()[:5]
            fields.append({
                "name": field.name,
                "type": str(field.type),
                "nullable": bool(field.nullable),
                "sample": vals,
            })
        out["files"][name] = {
            "rows": f.metadata.num_rows,
            "row_groups": f.metadata.num_row_groups,
            "columns": fields,
        }
    return out


def analyze():
    import pandas as pd

    m = pq.read_table(DATA / "monosemantic_2023.parquet").to_pandas()
    c = pq.read_table(DATA / "anthropic_concepts.parquet").to_pandas()
    out = {
        "note": "public feature-level data; not SAE weights",
        "monosemantic_2023": {
            "rows": int(len(m)),
            "columns": list(m.columns),
            "density": quantiles(m["density"].to_numpy(dtype=float)),
            "max_activation": quantiles(
                m["max_activation"].to_numpy(dtype=float)),
            "density_gini": gini(m["density"].to_numpy(dtype=float)),
            "density_lorenz_top_share": lorenz(
                m["density"].to_numpy(dtype=float)),
            "density_fraction_zero": float(
                (m["density"].to_numpy(dtype=float) == 0).mean()),
            "non_null": {col: int(m[col].notna().sum()) for col in m.columns},
            "by_model": {},
        },
        "anthropic_concepts": {
            "rows": int(len(c)),
            "columns": list(c.columns),
            "by_model": c["model"].value_counts().to_dict(),
            "by_group": c["group"].value_counts().to_dict(),
            "top_activation_value": quantiles(
                c["top_activation_value"].to_numpy(dtype=float)),
            "non_null": {col: int(c[col].notna().sum()) for col in c.columns},
        },
    }
    valid_d = np.isfinite(m["density"].to_numpy(dtype=float)) & \
        np.isfinite(m["max_activation"].to_numpy(dtype=float))
    out["monosemantic_2023"]["spearman_density_vs_max_activation"] = spearman(
        m["density"].to_numpy(dtype=float)[valid_d],
        m["max_activation"].to_numpy(dtype=float)[valid_d])
    for model, grp in m.groupby("model"):
        d = grp["density"].to_numpy(dtype=float)
        out["monosemantic_2023"]["by_model"][str(model)] = {
            "n": int(len(grp)),
            "density_median": float(np.nanmedian(d)),
            "density_mean": float(np.nanmean(d)),
            "density_p90": float(np.nanquantile(d, 0.9)),
            "gini": gini(d),
        }
    toks = c["top_activation_token"].astype(str)
    out["anthropic_concepts"]["top_tokens"] = (
        toks.value_counts().head(50).to_dict())
    out["anthropic_concepts"]["group_x_model"] = (
        c.groupby(["group", "model"]).size().reset_index(name="n")
        .to_dict(orient="records"))
    # rank-frequency slope of density (log-log)
    d = np.sort(m["density"].to_numpy(dtype=float))[::-1]
    d = d[d > 0]
    ranks = np.arange(1, d.size + 1)
    head = d[:min(10000, d.size)]
    out["monosemantic_2023"]["rank_frequency"] = {
        "loglog_slope_top10k": float(np.polyfit(np.log(ranks[:head.size]),
                                                np.log(head), 1)[0]),
        "effective_number_exp_entropy": float(np.exp(
            -((lambda p: p * np.log(p + 1e-300))(d[:head.size] /
              d[:head.size].sum())).sum())),
    }
    # feature index overlap across the two files
    mi = set(m["feature_index"].dropna().astype(int).tolist()[:200000])
    ci = set(int(x) for x in c["feature_index"].dropna().astype(int).tolist()
             if str(x).isdigit())
    out["cross_file"] = {
        "concept_feature_indices_digit_int": len(ci),
        "monosemantic_first_200k_distinct": len(mi),
        "intersection": len(mi & ci),
        "models_in_monosemantic": sorted(
            m["model"].dropna().unique().tolist()),
        "models_in_concepts": sorted(c["model"].dropna().unique().tolist()),
    }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = schema_report()
    (OUT / "anthropic_schema.json").write_text(
        json.dumps(s, indent=2, default=str), encoding="utf-8")
    a = analyze()
    (OUT / "anthropic_analysis.json").write_text(
        json.dumps(a, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "density_quantiles": {k: round(v, 6) for k, v in
                              a["monosemantic_2023"]["density"].items()
                              if isinstance(v, float)},
        "density_gini": a["monosemantic_2023"]["density_gini"],
        "rows": a["monosemantic_2023"]["rows"],
        "models": a["cross_file"]["models_in_monosemantic"],
    }, indent=2))


if __name__ == "__main__":
    main()
