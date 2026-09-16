"""Generate the frozen 350k mission-state manifest (no GPU work)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = [
    "opt/routed_expert.py",
    "opt/model_opt.py",
    "opt/model_ref.py",
    "opt/scan_attn.py",
    "opt/bench_expert_moe.py",
    "opt/doe_config.py",
    "opt/onehour_microbatch.py",
    "opt/onehour_reprofile.py",
    "results/g4_sparse_350k_cell.py",
]


def blob_hash(path: Path) -> str:
    return subprocess.check_output(
        ["git", "hash-object", str(path)], cwd=ROOT, text=True).strip()


def head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main():
    mb2 = load("results/onehour_mb_cfg/sparse_mb2.json")
    b3 = load("results/350k_b3_cache.json")
    b4 = load("results/350k_b4_writer.json")
    repro = load("results/350k_local_reprofile.json")
    mb1 = load("results/onehour_mb_cfg/sparse_mb1_clean.json")
    dense_mb1 = load("results/onehour_mb_cfg/dense_mb1.json")

    manifest = {
        "mission": "350k_g4_sparse",
        "status": "FROZEN_LOCAL_PENDING_G4",
        "frozen_at_commit": head(),
        "candidate_source_blobs": {
            path: blob_hash(ROOT / path) for path in SOURCE_FILES
        },
        "stack0": {
            "architecture": "M8/Ke512/top1 fixed cyclic contiguous window, "
                            "head-shared, compact exact-capacity executor",
            "source": "opt/routed_expert.py RoutedExpertArmA with "
                      "window_route_sets(8,1,cyclic=True), G=128 groups, "
                      "fixed route table (once per forward)",
            "microbatch_contract": {
                "global_sequences": 64,
                "sequence_length": 2048,
                "global_tokens_per_update": 131072,
                "microbatch_sequences": 2,
                "accumulation_steps": 32,
                "optimizer_step": "one clip + one fused AdamW step per "
                                  "global update",
            },
            "model_contract": {
                "T": 2048, "L": 8, "D": 256, "N": 16384, "H": 4,
                "precision": "bf16 autocast + fp32 master params",
                "loss": "sum CE over valid targets / total valid "
                        "(unchanged)",
                "clip_norm": 1.0,
                "optimizer": "fused AdamW lr 1e-3 betas 0.9/0.95 eps 1e-8 "
                             "wd 0.1",
                "packed_row_reconstruction": "frozen corpus semantics, "
                                             "synthetic packed mixed for "
                                             "benchmarks",
            },
            "capacity": {"capacity_tokens_per_expert_mb2": 512,
                         "padding_fraction": 0.0},
        },
        "local_benchmark_distribution_ms": {
            "sparse_mb2_medians_ms": [mb2["median_ms"],
                                      b3["records"]["cache_off"]["median_ms"],
                                      b4["sparse_mb2_normal_ms"]],
            "sparse_mb2_ms_arrays": {
                "onehour_microbatch": mb2["ms"],
                "b3_cache_off": b3["records"]["cache_off"]["ms"],
                "b3_cache_on": b3["records"]["cache_on"]["ms"],
            },
            "sparse_mb1_clean_ms": mb1["ms"],
            "reprofile_mb2": {"median_ms": repro["median_ms"],
                              "p10_ms": repro["p10_ms"],
                              "p90_ms": repro["p90_ms"],
                              "ms": repro["ms"]},
            "dense_mb1_control_ms": dense_mb1["ms"],
            "same_session_ratios": [4.22, 4.02, 4.05],
        },
        "memory": {
            "sparse_mb2_peak_GiB": [mb2["peak_mem_GiB"],
                                    repro["peak_mem_GiB"], 2.36],
            "sparse_mb1_peak_GiB": mb1["peak_mem_GiB"],
            "dense_mb1_peak_GiB": dense_mb1["peak_mem_GiB"],
            "dense_mb2_peak_GiB": 7.79,
            "dense_mb2_status": "memory-thrash collapse on 8 GiB",
        },
        "correctness_status": {
            "routed_executor": "opt/test_routed_expert.py 4/4 PASS "
                               "(all-active bitwise 0.0, sparse FP64 oracle "
                               "0.0, document-boundary leak test, builders)",
            "expertized_equivalence": "opt/test_expert_equivalence.py 5/5 "
                                      "PASS (FP32 bitwise, FP64 4.4e-12)",
            "learned_router": "opt/test_learned_router.py R0-R6 PASS "
                              "(ST forward exactly hard, surrogate gradient "
                              "1.1e-16, inactive experts zero grad, "
                              "graph breaks 0)",
        },
        "b3_status": {
            "decision": "KILLED",
            "effect_pct": -0.066,
            "evidence": "results/350k_b3_cache.json",
            "note": "gather-cache code reverted; executor identical to "
                    "frozen baseline",
        },
        "b4_status": {
            "decision": "RETRACTED_PERMANENTLY",
            "note": "writer ablations (B1 17.1 ms and B4 91.5% share) are "
                    "invalid for cost attribution because zeroing the "
                    "writer severs the recurrent autograd graph. Do not use "
                    "either number as writer-runtime evidence.",
            "evidence": "results/350k_b4_writer.json",
        },
        "labels": {
            "MEASURED": [
                "local sparse mb2 medians 2707-2804 ms per 131072-token "
                "global update",
                "local packed throughput 46.7-48.4k tok/s",
                "same-session ratios 4.02-4.22x",
                "memory numbers above",
                "G4 dense production 80.3k tok/s (prior mission, frozen)",
            ],
            "INFERRED": [
                "G4 sparse projection ~349k tok/s / ~1.99 h via local-to-G4 "
                "dense ratio 7.24x (low-medium confidence)",
            ],
            "SPECULATIVE": [
                "any untested fusion savings; writer cost figures "
                "(retracted)",
            ],
        },
        "g4_package": {
            "cell": "results/g4_sparse_350k_cell.py",
            "candidates": ["A_SPEED_CHAMPION", "B_SCIENTIFIC_CHAMPION"],
            "geometry_plan": "B8 -> B16 -> B32 -> (B64 if improving / B4 if "
                             "improving smaller)",
        },
        "next_actions_on_g4_json": [
            "ingest G4 measurements",
            "compare transfer ratio against local",
            "update production champion",
            "decide on one targeted G4 follow-up",
            "update paper-safe systems claims",
        ],
        "local_optimization": "STOPPED per owner instruction",
    }
    out = ROOT / "results/350k_frozen_state.json"
    out.write_text(json.dumps(manifest, indent=2, default=str),
                   encoding="utf-8")
    print(json.dumps({"out": str(out),
                      "frozen_at_commit": manifest["frozen_at_commit"],
                      "blobs": manifest["candidate_source_blobs"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
