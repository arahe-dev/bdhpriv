"""B=2 packed-document isolation diagnostic for the routed executor (CPU).

Reproduces artifacts/arm_a_speed_verification/parity_failure_B2.json.
No implementation changes: this is a read-only diagnostic.

py -3.12 opt/verify_speed_parity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opt.model_opt import OptArmA
from opt.model_ref import ArmAConfig, canonical_init, load_init, \
    synthetic_packed_batch
from opt.routed_expert import (
    RoutedExpertArmA,
    build_route_tensors,
    group_route_table,
)

OUT = Path(__file__).resolve().parents[1] / \
    "artifacts/arm_a_speed_verification/parity_failure_B2.json"


def main():
    cfg = ArmAConfig(T=64, V=32, D=32, N=128, H=4, L=2, HIDDEN=64)
    device = torch.device("cpu")
    dense = OptArmA(cfg, device, scan_block=64, use_checkpoint=False,
                    coord="dense", single_scan="chunkwise",
                    packed_update="branchfree", zero_carry=True,
                    paper_layout="direct", cache_rope=True)
    load_init(dense, canonical_init(cfg), device)
    dense.eval()
    model = RoutedExpertArmA(cfg, device, experts=8, expert_width=4,
                             scan_block=64)
    model.load_canonical(dense.state_dict())
    model.eval()
    batch = synthetic_packed_batch(cfg, 2, device, seed=7000, mode="mixed")
    table = group_route_table([tuple(range(8))], cfg.T // 32)
    route2 = build_route_tensors(table, 32, 2, cfg.T, 8, device)
    route1 = build_route_tensors(table, 32, 1, cfg.T, 8, device)

    def slice_row(b, index):
        return {k: (v[index:index + 1]
                    if torch.is_tensor(v) and v.dim() > 0 else v)
                for k, v in b.items()}

    with torch.no_grad():
        both = model.forward_route(batch["x"], batch["pos"], batch["segpos"],
                                   batch["full_mask"],
                                   batch["segment_start"], route2)
        rows = []
        for index in range(2):
            sb = slice_row(batch, index)
            rows.append(model.forward_route(
                sb["x"], sb["pos"], sb["segpos"], sb["full_mask"],
                sb["segment_start"], route1))
        reference = torch.cat(rows, 0)
        dense_out = dense.forward_packed(
            batch["x"], batch["pos"], batch["segpos"], batch["full_mask"],
            batch["segment_start"])
        offset = batch["segment_start"].clone()
        offset[1] += 1000
        both_offset = model.forward_route(
            batch["x"], batch["pos"], batch["segpos"], batch["full_mask"],
            offset, route2)

    evidence = {
        "diagnostic": "B=2 cross-row packed-document isolation test (CPU, "
                      "fp32, exact code path)",
        "config": {"T": 64, "D": 32, "N": 128, "H": 4, "K": 32, "M": 8,
                   "Ke": 4, "B": 2},
        "max_abs_diff": {
            "routed_B2_vs_per_row_reference":
                float((both - reference).abs().max()),
            "dense_B2_vs_per_row_reference":
                float((dense_out - reference).abs().max()),
            "routed_B2_seg_offset_vs_per_row_reference":
                float((both_offset - reference).abs().max()),
        },
        "conclusion": "routed executor concatenates rows into expert streams "
                      "without offsetting segment_start, so identical "
                      "segment values from different rows are treated as the "
                      "same document (cross-row attention leakage). Dense "
                      "path is per-row exact. B=1 routed is exact; B>1 routed "
                      "violates declared packed-document semantics.",
        "impact": "all sparse measured configurations with microbatch>1 "
                  "(mb2, mb4, planned G4 B8/B16/B32) are numerically invalid "
                  "as declared semantics; mb1 (B=1) measurements remain valid",
    }
    OUT.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence["max_abs_diff"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
