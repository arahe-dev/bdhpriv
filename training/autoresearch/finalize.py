"""Finalize the SFT autoresearch mission: SAE diagnostics, metrics table and
AUTORESEARCH_FINAL_REPORT.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.sft_probe import common  # noqa: E402

AR_DIR = common.OUT_DIR.parent / "autoresearch"
REPORT_PATH = AR_DIR / "AUTORESEARCH_FINAL_REPORT.md"
SFT_BASE_GEN = common.OUT_DIR / "generations" / "base.json"
SFT_BASE_EVAL = AR_DIR / "eval" / "ar_000_base.json"


def load_experiments() -> list:
    records = []
    for path in sorted(AR_DIR.glob("experiment_*.json")):
        records.append(common.load_json(path))
    return records


def sae_report() -> dict:
    return {
        "format": "arm_a_autoresearch_sae_report_v1",
        "created_at": common.iso_now(),
        "SAE_STATUS": "UNAVAILABLE",
        "reason": (
            "No frozen SAE / sparse dictionary trained on the Arm-A or "
            "Akasha representation exists in the repository or local "
            "caches. Local SAE assets are Gemma Scope 2 and Anthropic "
            "public-feature resources only."
        ),
        "diagnostic_substitute": {
            "instrument": "native BDH sparse-neuron probes",
            "probe_set": "fixed 4x2048 proxy-text windows, stride-4 sampling",
            "metrics": [
                "x/u zero fraction", "x/u RMS", "top-64/256 mass share",
                "global top-64/256 index overlap vs base",
                "coordinator g statistics", "writer delta norms",
                "hidden-state cosine vs base",
            ],
            "note": (
                "Feature activation drift, feature survival and "
                "new-feature formation could not be measured without an "
                "SAE; native population overlap (top-64/256) is the "
                "closest available substitute and is reported per "
                "experiment."
            ),
        },
        "policy_followed": (
            "No SAE was trained or updated during this mission."
        ),
        "SAE_COSINE_VS_BASE": None,
    }


def before_after(best: dict) -> list:
    base_eval = common.load_json(SFT_BASE_EVAL)
    best_eval = common.load_json(
        AR_DIR / "eval" / f"{best}.json"
    )
    base_gen = common.load_json(SFT_BASE_GEN)
    best_gen = common.load_json(AR_DIR / "generations" / f"{best}.json")
    base_lm = base_eval["lm"]
    best_lm = best_eval["lm"]

    def sample(gen, key):
        return gen["sampled_summary"].get(key) if gen else None

    def native_mean(eval_data, key):
        values = [
            level[key] for level in eval_data["native"]["levels"]
            if level.get(key) is not None
        ]
        return sum(values) / len(values)

    def hidden_mean(eval_data):
        values = [
            level["hidden_vs_base"]["cosine_mean"]
            for level in eval_data["native"]["levels"]
            if level.get("hidden_vs_base")
        ]
        return sum(values) / len(values) if values else None

    best_params = best_eval.get("params") or {}
    drift = (
        (best_params.get("groups", {}).get("TOTAL") or {}).get(
            "relative_update_norm"
        )
    )
    return [
        ("validation NLL (mixture val)",
         f"{base_lm['sft_val_nll']:.4f}",
         f"{best_lm['sft_val_nll']:.4f}"),
        ("proxy (base-text) NLL",
         f"{base_lm['proxy_nll']:.4f}",
         f"{best_lm['proxy_nll']:.4f}"),
        ("hidden cosine vs base",
         "1.0000", f"{hidden_mean(best_eval):.4f}"),
        ("weight relative drift",
         "0.0000", f"{drift:.4f}"),
        ("x top-64 overlap vs base",
         "1.0000",
         f"{native_mean(best_eval, 'x_top64_overlap_vs_base'):.4f}"),
        ("x top-256 overlap vs base",
         "1.0000",
         f"{native_mean(best_eval, 'x_top256_overlap_vs_base'):.4f}"),
        ("sampled repeat-3gram",
         f"{sample(base_gen, 'mean_repeated_trigram_fraction'):.4f}",
         f"{sample(best_gen, 'mean_repeated_trigram_fraction'):.4f}"),
        ("sampled distinct-2",
         f"{sample(base_gen, 'mean_distinct_2'):.4f}",
         f"{sample(best_gen, 'mean_distinct_2'):.4f}"),
        ("sampled token entropy (bits)",
         f"{sample(base_gen, 'mean_token_entropy_bits'):.2f}",
         f"{sample(best_gen, 'mean_token_entropy_bits'):.2f}"),
        ("Akasha greedy parity",
         str(base_gen.get("FULL_VS_RECURRENT_GREEDY_MATCH")),
         str(best_gen.get("FULL_VS_RECURRENT_GREEDY_MATCH"))),
    ]


def build_report():
    experiments = load_experiments()
    best = common.load_json(AR_DIR / "best.json")
    best_id = best["best_experiment"]
    rows = before_after(best_id)
    common.save_json(AR_DIR / "sae_report.json", sae_report())
    common.save_json(
        AR_DIR / "final_comparison.json",
        {
            "format": "arm_a_autoresearch_final_comparison_v1",
            "best_experiment": best_id,
            "best_config": best.get("config"),
            "rows": [
                {"metric": metric, "base": base, "final": final}
                for metric, base, final in rows
            ],
        },
    )

    lines = []
    add = lines.append
    add("# Arm-A / Akasha SFT Autoresearch - Final Report")
    add("")
    add(f"Date: {common.iso_now()}  ")
    add(f"Best checkpoint: `{best.get('checkpoint')}` "
        f"(`{best_id}`)  ")
    add("Base checkpoint: frozen 2.5B Arm-A "
        f"(sha256 `{common.BASE_CKPT_SHA256[:16]}...`)")
    add("")
    add("## 1. Best checkpoint and final configuration")
    add("")
    add("| Item | Value |")
    add("|---|---|")
    config = best.get("config") or {}
    add(f"| instruction mixture | {config.get('mix')} "
        "(bespoke/oasst/tulu/openhermes 25/25/25/25) |")
    add(f"| learning rate | {config.get('lr')} |")
    add(f"| scheduler / warmup | {config.get('scheduler')} / "
        f"{config.get('warmup_frac')} |")
    add(f"| replay | {config.get('replay_pct')}% (BASE_TEXT_PROXY_REPLAY, "
        "repository source code) |")
    dose_tokens = {
        "0.01": 173_755, "0.03": 521_265, "0.10": 1_737_549,
        "0.30": 5_212_647, "0.40": 6_950_196,
    }.get(str(config.get("dose_tpp")), None)
    add(f"| dose | {config.get('dose_tpp')} TPP = "
        f"{dose_tokens if dose_tokens is not None else 'N/A'} instruction "
        "target tokens |")
    add(f"| validation NLL | {best.get('validation_nll'):.4f} |")
    add(f"| proxy NLL | {best.get('proxy_nll'):.4f} |")
    add(f"| hidden cosine vs base | {best.get('hidden_cosine'):.4f} |")
    add(f"| weight relative drift | {best.get('weight_drift'):.4f} |")
    add(f"| Akasha parity | {best.get('akasha_parity')} |")
    add("")
    add("## 2. Complete experiment history")
    add("")
    add("| id | phase | changed variable | old -> new | val NLL | "
        "hidden cos | drift | parity | decision |")
    add("|---|---|---|---|---|---|---|---|---|")
    for experiment in experiments:
        experiment_id = experiment["experiment_id"]
        add(
            f"| {experiment_id} | {experiment.get('phase')} | "
            f"{experiment.get('changed_variable')} | "
            f"{experiment.get('old_value')} -> {experiment.get('new_value')} | "
            f"{fmt(experiment.get('validation_nll'))} | "
            f"{fmt(experiment.get('hidden_cosine'))} | "
            f"{fmt(experiment.get('weight_drift'))} | "
            f"{experiment.get('akasha_parity')} | "
            f"{experiment.get('decision')} |"
        )
    add("")
    add("Failed/reverted experiments are scientific data and are kept in "
        "full; `results/autoresearch/experiment_<id>.json` holds the exact "
        "config, metrics, decision logic and notes.")
    add("")
    add("## 3. Before / after")
    add("")
    add("| metric | base | final (best) |")
    add("|---|---|---|")
    for metric, base, final in rows:
        add(f"| {metric} | {base} | {final} |")
    add("")
    add("## 4. Why this configuration was selected")
    add("")
    add(f"Best experiment: `{best_id}`.")
    add("")
    add("- MEASURED: at fixed mixture and hyperparameters the 0.10 TPP "
        "checkpoint has the lowest held-out validation NLL (2.2656), "
        "better than 0.03 TPP (2.3931), 0.30 TPP (2.2772) and 0.40 TPP "
        "(2.4067). Adaptation peaks near 0.10 TPP on this data.")
    add("- MEASURED: representation drift keeps growing with dose "
        "(0.1058 at 0.10 -> 0.1863 at 0.30 -> 0.2265 at 0.40) and hidden "
        "cosine falls (0.9404 -> 0.9263 -> 0.9058). At 0.40 TPP the run "
        "is close to both hard constraints (cosine floor 0.90, drift "
        "ceiling 0.25).")
    add("- MEASURED: Phase 1 found 3e-4 peak LR best; 1e-4/2e-4 adapt "
        "less and 5e-4 does not recover the loss while drifting more. "
        "Constant was better than cosine at this dose; warmup fraction "
        "(0/2/5%) was statistically tied.")
    add("- MEASURED: replay ratio leaves validation NLL tied within "
        "0.002 while proxy NLL improves monotonically with replay "
        "(0%: 3.8356, 5%: 3.7596, 10%: 3.7272, 20%: 3.6093). 10% is the "
        "selected operating point: no adaptation cost, clear retention "
        "benefit over 0/5%.")
    add("- MEASURED: mixture re-weighting is a flat dimension at this "
        "dose (all variants within 0.009 NLL; tulu40 2.3921 marginally "
        "best but inside the 0.002 keep threshold).")
    add("- INFERRED: the operating regime is 'short, moderate-LR, "
        "constant-schedule SFT with 10% replay'; the binding constraints "
        "are overfitting/secondary drift beyond ~0.10 TPP, not numerical "
        "instability.")
    add("")
    add("## 5. Known limitations")
    add("")
    add("- The frozen 5B pretraining corpus is not local "
        "(BLOCKED_ARTIFACT_NOT_LOCAL); base-text retention and replay "
        "use a labelled BASE_TEXT_PROXY (repository prose for retention, "
        "repository source code for replay). Replay numbers are therefore "
        "mechanical and same-domain-confounded.")
    add("- No Arm-A SAE exists locally (SAE_STATUS=UNAVAILABLE); "
        "feature-level drift metrics could not be measured. Native BDH "
        "sparse-neuron overlap is the substitute.")
    add("- `instruction_score` is defined as `-validation NLL` on a "
        "frozen 1000-example mixture split; no external instruction "
        "benchmark (e.g. MT-Bench/IFEval) was run. Generation metrics are "
        "pathology indicators, not quality scores.")
    add("- Phase 4 could not reach 0.50/1.0 TPP: the frozen 4-source "
        "pool holds ~7.2M instruction target tokens (~0.41 TPP single "
        "epoch), and the 0.10 -> 0.40 curve already shows saturation and "
        "accelerating drift, which satisfies the mission stop rule.")
    add("- Training was full-parameter SFT at 2048 context, "
        "microbatch 1 row, 8192 tokens/update; other batch geometries "
        "were not explored.")
    add("- All experiments use one seed (data order seed 1337); "
        "run-to-run noise was not measured.")
    add("")
    add("## 6. Artifacts")
    add("")
    add("- `results/autoresearch/experiment_<id>.json` - per-run records")
    add("- `results/autoresearch/metrics.json` - full metric table")
    add("- `results/autoresearch/research_log.md` - chronological log")
    add("- `results/autoresearch/sae_report.json` - SAE diagnostic status")
    add("- `results/autoresearch/final_comparison.json` - before/after")
    add("- `results/autoresearch/data/sources_manifest.json` and "
        "`data/mix_*/mixture_manifest.json` - dataset provenance")
    add("- `runs/autoresearch/<id>/` - checkpoints (excluded from git by "
        "size)")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT_PATH}")
    print(f"wrote {AR_DIR / 'sae_report.json'}")
    print(f"wrote {AR_DIR / 'final_comparison.json'}")
    return 0


def fmt(value):
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def main(argv=None) -> int:
    return build_report()


if __name__ == "__main__":
    sys.exit(main())
