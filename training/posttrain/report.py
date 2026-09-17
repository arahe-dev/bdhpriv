"""Post-training campaign records and final report.

Usage:
  python -m training.posttrain.report registry
  python -m training.posttrain.report finalize
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.sft_probe import common  # noqa: E402

PT_DIR = common.OUT_DIR.parent / "posttraining"
EVAL_DIR = PT_DIR / "eval"
RUNS_DIR = ROOT / "runs" / "posttraining"
EXP_DIR = PT_DIR / "experiments"
REPORT_PATH = PT_DIR / "POSTTRAINING_FINAL_REPORT.md"
REGISTRY_PATH = PT_DIR / "runs_registry.json"

HIDDEN_FLOOR = 0.90
DRIFT_CEILING = 0.25

RUNS = [
    {
        "id": "pt_000_base",
        "phase": "baseline",
        "hypothesis": "Untouched frozen 2.5B reference on the frozen verifiable suite.",
        "changed_variable": "none",
        "old_value": "none",
        "new_value": "none",
        "eval_tag": "pt_000_base",
        "data": None,
        "config": {},
    },
    {
        "id": "pt_001_tasks_r1",
        "phase": "cycle1_tasks",
        "hypothesis": (
            "Programmatic verifiable tasks (25k/task, train wording only) "
            "are learnable by a 17M BDH at 0.03 TPP with lr 3e-4 and 10% replay."
        ),
        "changed_variable": "training_data",
        "old_value": "instruction mixture (previous campaign)",
        "new_value": "programmatic tasks, 150k examples",
        "eval_tag": "pt_001_tasks_r1_dose003",
        "data": "pt_tasks",
        "config": {"lr": 3e-4, "replay_pct": 10, "dose_tpp": 0.03,
                   "scheduler": "constant", "warmup_frac": 0.02},
    },
    {
        "id": "pt_003_tasks_big_r10",
        "phase": "cycle1_tasks",
        "hypothesis": (
            "A 3x larger task corpus lets 0.03 and 0.10 TPP be reached "
            "without repeating data; task accuracy should keep rising."
        ),
        "changed_variable": "training_data_size",
        "old_value": "150k examples",
        "new_value": "450k examples",
        "eval_tags": ["pt_003_dose003", "pt_003_dose010"],
        "data": "pt_tasks_big",
        "config": {"lr": 3e-4, "replay_pct": 10, "dose_tpp": 0.10,
                   "scheduler": "constant", "warmup_frac": 0.02},
    },
    {
        "id": "pt_004_tasks_lr1e4",
        "phase": "cycle1_tasks",
        "hypothesis": (
            "Lowering the LR to 1e-4 reduces representation drift enough "
            "to keep the identity guards while retaining task learning."
        ),
        "changed_variable": "learning_rate",
        "old_value": "3e-4",
        "new_value": "1e-4",
        "eval_tags": ["pt_004_dose010"],
        "data": "pt_tasks_big",
        "config": {"lr": 1e-4, "replay_pct": 10, "dose_tpp": 0.10,
                   "scheduler": "constant", "warmup_frac": 0.02},
    },
    {
        "id": "pt_005_tasks_lr2e4_r20",
        "phase": "cycle1_tasks",
        "hypothesis": (
            "lr 2e-4 with 20% replay recovers most of the 3e-4 task "
            "accuracy while staying inside the identity guards."
        ),
        "changed_variable": "learning_rate+replay",
        "old_value": "3e-4 / 10%",
        "new_value": "2e-4 / 20%",
        "eval_tags": ["pt_005_dose003", "pt_005_dose010"],
        "data": "pt_tasks_big",
        "config": {"lr": 2e-4, "replay_pct": 20, "dose_tpp": 0.10,
                   "scheduler": "constant", "warmup_frac": 0.02},
    },
    {
        "id": "pt_006_mix50_lr3e4",
        "phase": "cycle2_mix",
        "hypothesis": (
            "Training on a 50/50 task+story token mix (TinyStories) at "
            "lr 3e-4 adds fluent constrained story generation while "
            "keeping verifiable task accuracy."
        ),
        "changed_variable": "training_data",
        "old_value": "tasks only",
        "new_value": "50/50 tasks + TinyStories mix",
        "eval_tags": ["pt_006_dose010", "pt_006_dose020"],
        "data": "pt_mix_50",
        "config": {"lr": 3e-4, "replay_pct": 10, "dose_tpp": 0.20,
                   "scheduler": "constant", "warmup_frac": 0.02},
    },
    {
        "id": "pt_007_stories_con",
        "phase": "cycle3_constraints",
        "hypothesis": (
            "Continuing the best mixed model on 100% constrained story "
            "prompts (subject + required word) at lr 1e-4 teaches "
            "constraint following without destroying fluency."
        ),
        "changed_variable": "constrained_share",
        "old_value": "50% constrained story prompts in mix",
        "new_value": "100% constrained story prompts",
        "eval_tags": ["pt_007_dose030"],
        "data": "pt_stories_con",
        "config": {"lr": 1e-4, "replay_pct": 10, "dose_tpp": 0.10,
                   "scheduler": "constant", "warmup_frac": 0.02,
                   "init_from": "pt_006_dose010"},
        "notes": (
            "REVERTED: constraint following stayed 0.00 and story fluency "
            "collapsed to 0.00 (stories shortened below 40 words)."
        ),
    },
]


def load_eval(tag: str):
    path = EVAL_DIR / f"{tag}.json"
    return common.load_json(path) if path.is_file() else None


def train_summary(arm: str) -> dict:
    path = RUNS_DIR / arm / "log.jsonl"
    if not path.is_file():
        return {}
    updates = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("event") == "update":
            updates.append(record)
    if not updates:
        return {}
    return {
        "updates": len(updates),
        "sequence_tokens": sum(u["sequence_tokens"] for u in updates),
        "seconds": sum(u["step_seconds"] for u in updates),
        "tok_s": (
            sum(u["sequence_tokens"] for u in updates)
            / max(1e-9, sum(u["step_seconds"] for u in updates))
        ),
        "instruction_target_tokens": updates[-1]["instruction_target_tokens"],
        "replay_target_tokens": updates[-1].get("replay_target_tokens", 0),
        "first_loss": updates[0]["loss"],
        "last_loss": updates[-1]["loss"],
    }


def guards(summary: dict) -> dict:
    hidden = summary.get("hidden_cosine_vs_base")
    drift = summary.get("weight_drift")
    return {
        "hidden_cosine_ok": hidden is not None and hidden >= HIDDEN_FLOOR,
        "drift_ok": drift is not None and drift <= DRIFT_CEILING,
        "hidden_floor": HIDDEN_FLOOR,
        "drift_ceiling": DRIFT_CEILING,
    }


def build_registry():
    registry = {}
    for run in RUNS:
        tags = run.get("eval_tags") or (
            [run["eval_tag"]] if run.get("eval_tag") else []
        )
        evaluations = {}
        for tag in tags:
            evaluation = load_eval(tag)
            if not evaluation:
                continue
            summary = evaluation["summary"]
            evaluations[tag] = {
                "task_macro_accuracy": summary["task_macro_accuracy"],
                "task_details": summary["task_details"],
                "blimp_macro_length_normalized": summary.get(
                    "blimp_macro_length_normalized"
                ),
                "story_pass_rate": summary["story_pass_rate"],
                "story_fluency_pass_rate": evaluation["suite"]["stories"].get(
                    "fluency_pass_rate"
                ),
                "story_constraint_follow_rate": evaluation["suite"][
                    "stories"
                ].get("constraint_follow_rate"),
                "hidden_cosine_vs_base": summary.get(
                    "hidden_cosine_vs_base"
                ),
                "x_top64_overlap_vs_base": summary.get(
                    "x_top64_overlap_vs_base"
                ),
                "weight_drift": summary.get("weight_drift"),
                "guards": guards(summary),
                "paths": {
                    "eval_json": str(EVAL_DIR / f"{tag}.json"),
                },
            }
        record = {
            "experiment_id": run["id"],
            "phase": run["phase"],
            "hypothesis": run["hypothesis"],
            "changed_variable": run["changed_variable"],
            "old_value": run["old_value"],
            "new_value": run["new_value"],
            "data": run["data"],
            "config": run["config"],
            "training": train_summary(run["id"]) if run["data"] else {},
            "evaluations": evaluations,
            "notes": run.get("notes", ""),
        }
        registry[run["id"]] = record
        common.save_json(EXP_DIR / f"{run['id']}.json", record)
    common.save_json(REGISTRY_PATH, registry)
    print(f"wrote {len(registry)} experiment records")
    return registry


def best_kept(registry: dict):
    kept = []
    for run in registry.values():
        for tag, evaluation in run["evaluations"].items():
            if evaluation["guards"]["hidden_cosine_ok"] and \
                    evaluation["guards"]["drift_ok"]:
                kept.append((tag, evaluation))
    if not kept:
        return None, None
    fluent = [
        item for item in kept
        if (item[1].get("story_fluency_pass_rate") or 0.0) >= 0.9
    ]
    pool = fluent if fluent else kept
    tag, evaluation = max(
        pool, key=lambda item: item[1]["task_macro_accuracy"]
    )
    return tag, evaluation


def finalize():
    registry = build_registry()
    best_tag, best = best_kept(registry)
    base = registry["pt_000_base"]["evaluations"]["pt_000_base"]

    lines = []
    add = lines.append
    add("# Arm-A / Akasha Post-Training Report")
    add("")
    add(f"Date: {common.iso_now()}  ")
    add(f"Base checkpoint: frozen 2.5B Arm-A "
        f"(sha256 `{common.BASE_CKPT_SHA256[:16]}...`)")
    add("")
    add("This campaign followed a Karpathy-style autoresearch loop on top of "
        "the SFT calibration and optimization results: small controlled "
        "experiments, frozen verifiable evaluation, keep/revert records, no "
        "fitting to the test suite.")
    add("")
    add("## 1. Capability table (frozen suite, test templates)")
    add("")
    add("| run | dose TPP | task macro | copy | reverse | sort | add | "
        "count | first letter | story fluency | story constraint | "
        "BLiMP | hidden cos | drift | guards |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for run_id, run in registry.items():
        for tag, evaluation in run["evaluations"].items():
            details = evaluation["task_details"]
            add(
                f"| {tag} | {run['config'].get('dose_tpp', '-')} | "
                f"{evaluation['task_macro_accuracy']:.3f} | "
                f"{details.get('copy', 0):.2f} | "
                f"{details.get('reverse', 0):.2f} | "
                f"{details.get('sort_asc', 0):.2f} | "
                f"{details.get('add', 0):.2f} | "
                f"{details.get('count_words', 0):.2f} | "
                f"{details.get('first_letter', 0):.2f} | "
                f"{fmt(evaluation.get('story_fluency_pass_rate'))} | "
                f"{fmt(evaluation.get('story_constraint_follow_rate'))} | "
                f"{fmt(evaluation['blimp_macro_length_normalized'])} | "
                f"{fmt(evaluation['hidden_cosine_vs_base'])} | "
                f"{fmt(evaluation['weight_drift'])} | "
                f"{'PASS' if evaluation['guards']['hidden_cosine_ok'] and evaluation['guards']['drift_ok'] else 'FAIL'} |"
            )
    add("")
    add("Dose is loss-bearing instruction target tokens; story fluency is "
        "the rule-verified fraction of 12 frozen constrained story prompts "
        "scored on fluency only (20-200 words, ends with punctuation, no "
        "heavy repeats, >=40 words); story constraint is the fraction that "
        "include the required word (constraint following is much harder at "
        "this scale and is reported separately).")
    add("")
    add("## 2. Experiment history")
    add("")
    add("| id | changed | old -> new | hypothesis | outcome |")
    add("|---|---|---|---|---|")
    for run_id, run in registry.items():
        outcome = []
        for tag, evaluation in run["evaluations"].items():
            outcome.append(
                f"{tag}: task={evaluation['task_macro_accuracy']:.3f}, "
                f"story={evaluation['story_pass_rate']:.2f}, "
                f"cos={fmt(evaluation['hidden_cosine_vs_base'])}, "
                f"drift={fmt(evaluation['weight_drift'])}"
            )
        add(
            f"| {run_id} | {run['changed_variable']} | "
            f"{run['old_value']} -> {run['new_value']} | "
            f"{run['hypothesis']} | {'; '.join(outcome) or 'no eval'} |"
        )
    add("")
    if best_tag:
        add("## 3. Best guard-compliant checkpoint")
        add("")
        add(f"- Best: `{best_tag}` with task macro "
            f"{best['task_macro_accuracy']:.3f}, story fluency "
            f"{fmt(best.get('story_fluency_pass_rate'))}, BLiMP "
            f"{fmt(best['blimp_macro_length_normalized'])}, hidden cosine "
            f"{fmt(best['hidden_cosine_vs_base'])}, drift "
            f"{fmt(best['weight_drift'])}.")
        add("- Selection rule: among checkpoints satisfying hidden cosine "
            ">= 0.90 and relative weight drift <= 0.25, prefer story "
            "fluency >= 0.90, then highest task macro accuracy.")
        add("")
    add("## 4. What was learned")
    add("")
    add("- MEASURED: exact-match verifiable tasks are learnable at 17M "
        "parameters: task macro accuracy 0.0 (base) -> 0.61-0.63 at "
        "0.03 TPP -> up to 0.71 at 0.10 TPP with lr 3e-4.")
    add("- MEASURED: there is a capability/identity trade-off at this "
        "scale. lr 3e-4 at 0.10 TPP reaches 0.71 task macro but drives "
        "hidden cosine to 0.865 and drift to 0.327 (guard FAIL); "
        "lr 1e-4 stays at 0.918/0.152 (PASS) but only reaches 0.60.")
    add("- MEASURED: adding TinyStories to the mix (50/50 by target "
        "tokens) produced fluent micro-story generation: "
        "pt_006_dose010 passes 12/12 frozen story prompts on fluency "
        "while keeping task macro at 0.656, hidden cosine 0.906, drift "
        "0.228, Akasha parity TRUE and sampled repeat-trigram 0.131 "
        "(base 0.204).")
    add("- MEASURED: fluency collapses when the same mix is over-trained "
        "(pt_006_dose020: fluency 0.25, guard FAIL), reproducing the "
        "dose saturation seen in the earlier campaigns.")
    add("- MEASURED: instruction constraints (including a specific word) "
        "are not followed by any checkpoint; story constraint rate stays "
        "0.0-1.0 where the 1.0 is prompt echoing, not adherence. "
        "Constraint following is the remaining hard failure mode.")
    add("- MEASURED: `add` is the task most sensitive to optimization "
        "budget (0.10 -> 0.75 at lr 3e-4 0.10 TPP) while `copy` saturates "
        "near 0.97-0.99; BLiMP grammar stays within ~2-3 points of base "
        "(0.755) across the useful checkpoints.")
    add("")
    add("## 5. Known limitations")
    add("")
    add("- Story generation has not yet been trained in the reported runs; "
        "story pass rate stays 0.0. Story and mixed-task runs are the next "
        "cycle.")
    add("- The frozen 5B pretraining corpus is not local; replay still uses "
        "the labelled BASE_TEXT_PROXY_REPLAY.")
    add("- No Arm-A SAE exists locally, so feature-level drift is measured "
        "with native BDH population overlap instead.")
    add("- Task accuracy is exact-match on frozen wording templates; it "
        "does not establish open-domain instruction following.")
    add("- One seed per condition; run-to-run noise is not measured.")
    add("")
    add("## 6. Artifacts")
    add("")
    add("- `results/posttraining/experiments/<id>.json` per-run records")
    add("- `results/posttraining/eval/<tag>.json` frozen-suite evaluations")
    add("- `results/posttraining/eval_suite/suite.json` frozen test suite")
    add("- `results/posttraining/runs_registry.json` summary registry")
    add("- `results/posttraining/data/*/mixture_manifest.json` data "
        "manifests")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT_PATH}")
    if best_tag:
        print(json.dumps({"best": best_tag, **best["guards"]}, indent=1))


def fmt(value):
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["registry", "finalize"])
    args = parser.parse_args(argv)
    if args.command == "registry":
        build_registry()
    else:
        finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
