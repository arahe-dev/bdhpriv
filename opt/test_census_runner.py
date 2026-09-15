"""Local tests for opt/run_trained_sparsity_census.py (no GPU/corpus needed).

Run: py -3.12 opt/test_census_runner.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opt.census_metrics import TrainedCensusAccumulator
from opt.census_model import CensusArmA
from opt.model_ref import ArmAConfig, canonical_init, load_init, \
    synthetic_packed_batch
from opt.run_trained_sparsity_census import (
    CensusError,
    contiguous_ranges,
    inspect_checkpoint,
    ranges_from_file,
    run_census_batches,
    sample_ranges,
)

TINY = dict(T=16, V=32, D=8, N=16, H=2, L=2, HIDDEN=8)
TOTAL = 20000


def test_sampling():
    a = sample_ranges(TOTAL, 64, 16, seed=1337)
    b = sample_ranges(TOTAL, 64, 16, seed=1337)
    c = sample_ranges(TOTAL, 64, 16, seed=9)
    assert a == b, "sampling must be deterministic for a fixed seed"
    assert a != c, "different seeds must give different samples"
    assert len(a) == 64, len(a)
    spans = {}
    for start, end in a:
        assert end - start == 64
        assert 0 <= start < end <= TOTAL
    starts = sorted(s for s, _ in a)
    for i in range(len(starts) - 1):
        assert starts[i + 1] - starts[i] >= 64, "ranges overlap"
        spans[starts[i] // (TOTAL // 16) + 1] = True
    assert len(spans) >= 16, f"expected >=16 regions, got {len(spans)}"
    contig = contiguous_ranges(1000, 8, TOTAL)
    assert contig == [(1000 + 64 * i, 1000 + 64 * (i + 1)) for i in range(8)]
    try:
        contiguous_ranges(TOTAL - 100, 8, TOTAL)
        raise AssertionError("out-of-bounds contiguous range accepted")
    except CensusError:
        pass
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ranges.json"
        path.write_text(json.dumps({"ranges": [[int(s), int(e)] for s, e in a]}),
                        encoding="utf-8")
        reused = ranges_from_file(path, TOTAL)
        assert reused == a
        bad = Path(td) / "bad.json"
        bad.write_text(json.dumps({"ranges": [[0, 32]]}), encoding="utf-8")
        try:
            ranges_from_file(bad, TOTAL)
            raise AssertionError("wrong-width range accepted")
        except CensusError:
            pass


def _fake_checkpoint(path: Path, cfg: ArmAConfig, tokens_per_update=131072):
    step = 2000
    tokens = step * tokens_per_update
    payload = {
        "format": "arm_a_2p5b_ckpt_v1",
        "progress": {
            "updates_done": step,
            "tokens_consumed": tokens,
            "next_sequence": tokens // cfg.T,
        },
        "config": {
            "T": cfg.T, "V": cfg.V, "D": cfg.D, "N": cfg.N, "H": cfg.H,
            "L": cfg.L, "HIDDEN": cfg.HIDDEN, "GLOBAL_BATCH": 64,
            "MICROBATCH": 16,
        },
        "corpus": {
            "artifact_hashes_sha256": "a" * 64,
            "total_sequences": TOTAL,
        },
        "code_sha256": None,
        "model": {"embedding.weight": torch.zeros(cfg.V, cfg.D)},
    }
    torch.save(payload, path)
    return payload


def test_inspect_checkpoint():
    cfg = ArmAConfig()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "step_0000002000.pt"
        _fake_checkpoint(path, cfg)
        info = inspect_checkpoint(
            path,
            expected_config={
                "T": cfg.T, "V": cfg.V, "D": cfg.D, "N": cfg.N,
                "H": cfg.H, "L": cfg.L, "HIDDEN": cfg.HIDDEN,
                "GLOBAL_BATCH": 64, "MICROBATCH": 16,
            },
            corpus_artifact_sha256="a" * 64,
            total_sequences=TOTAL,
            code_path=None,
        )
        assert info["checkpoint_step"] == 2000
        assert info["checkpoint_input_tokens"] == 2000 * 131072
        assert info["evidence_grade"] == "trained_early_262m"
        assert len(info["checkpoint_sha256"]) == 64

        bad_corpus = Path(td) / "bad_corpus.pt"
        payload = _fake_checkpoint(bad_corpus, cfg)
        payload["corpus"]["artifact_hashes_sha256"] = "b" * 64
        torch.save(payload, bad_corpus)
        try:
            inspect_checkpoint(bad_corpus, {"T": cfg.T, "V": cfg.V, "D": cfg.D,
                                            "N": cfg.N, "H": cfg.H, "L": cfg.L,
                                            "HIDDEN": cfg.HIDDEN,
                                            "GLOBAL_BATCH": 64,
                                            "MICROBATCH": 16},
                               "a" * 64, TOTAL, None)
            raise AssertionError("corpus hash mismatch accepted")
        except CensusError:
            pass

        bad_config = Path(td) / "bad_config.pt"
        payload = _fake_checkpoint(bad_config, cfg)
        payload["config"]["T"] = cfg.T + 1
        torch.save(payload, bad_config)
        try:
            inspect_checkpoint(bad_config, {"T": cfg.T, "V": cfg.V, "D": cfg.D,
                                            "N": cfg.N, "H": cfg.H, "L": cfg.L,
                                            "HIDDEN": cfg.HIDDEN,
                                            "GLOBAL_BATCH": 64,
                                            "MICROBATCH": 16},
                               "a" * 64, TOTAL, None)
            raise AssertionError("config mismatch accepted")
        except CensusError:
            pass


def test_end_to_end_synthetic():
    cfg = ArmAConfig(**TINY)
    device = torch.device("cpu")
    model = CensusArmA(cfg, device, scan_block=cfg.K)
    load_init(model, canonical_init(cfg), device)
    model.eval()
    batches = [
        synthetic_packed_batch(cfg, 8, device, seed=5 + i, mode="mixed")
        for i in range(3)
    ]

    def batch_iter():
        for batch in batches:
            yield {k: v.cpu() if torch.is_tensor(v) else v
                   for k, v in batch.items()}

    acc = TrainedCensusAccumulator(cfg, block_widths=(2, 4), bands=4,
                                   sample_rows=8)
    rows, count, fp32_summary, bf16_probe = run_census_batches(
        model, batch_iter(), device, microbatch=4, use_bf16=False,
        fp32_probe_batches=1, accumulator=acc, seed=11,
    )
    assert rows == 24 and count == 3, (rows, count)
    census = acc.finalize()
    for key in ("levels", "global", "cross_level_support_jaccard",
                "mac_estimates", "decision"):
        assert key in census, key
    assert len(census["levels"]) == cfg.L
    assert len(census["levels"][0]["heads"]) == cfg.H
    decision_keys = {
        "exact_scalar_sparse_candidate", "pair_sparse_candidate",
        "block_sparse_candidate", "best_block", "moe_structured_candidate",
        "frequency_specialization_candidate", "reason", "evidence_note",
    }
    assert set(census["decision"]) == decision_keys, set(census["decision"])
    head = census["levels"][0]["heads"][0]
    for key in ("x", "y", "u"):
        stats = head[key]
        assert 0.0 <= stats["positive_fraction"] <= 1.0
        assert stats["active_count"]["p10"] <= stats["active_count"]["p99"]
        assert set(stats["top_mass_share"]) == {
            "0.0625", "0.125", "0.25", "0.5"}
    for width in ("2", "4"):
        block = head["block_occupancy"][width]
        assert set(block) == {"x", "u", "q"}
    assert "x_positive_fraction" in census["global"]
    assert "band_cv_x_mean" in census["global"]
    assert fp32_summary["global"]["u_positive_fraction"] >= 0.0
    assert bf16_probe["global"]["u_positive_fraction"] >= 0.0
    macs = census["mac_estimates"]
    assert 0.0 <= macs["ideal_scalar_removable_fraction"] <= 1.0
    for width, value in macs["block_removable_fraction"].items():
        assert 0.0 <= value <= 1.0, (width, value)
    print(json.dumps({"decision": census["decision"],
                      "global": census["global"]}, indent=2))


def test_decision_thresholds():
    cfg = ArmAConfig(**TINY)
    device = torch.device("cpu")
    acc = TrainedCensusAccumulator(cfg, block_widths=(2, 4), bands=4,
                                   sample_rows=4)
    model = CensusArmA(cfg, device, scan_block=cfg.K)
    load_init(model, canonical_init(cfg), device)
    model.eval()
    batch = synthetic_packed_batch(cfg, 4, device, seed=1, mode="mixed")
    acc.new_batch(4, torch.Generator().manual_seed(0))
    model.begin_forward(lambda l, x, y, u, s: acc.add(l, x, y, u, s))
    model.forward_packed(batch["x"], batch["pos"], batch["segpos"],
                         batch["full_mask"], batch["segment_start"])
    decision = acc.finalize()["decision"]
    assert isinstance(decision["exact_scalar_sparse_candidate"], bool)
    assert decision["reason"]


def main():
    tests = [
        ("sampling", test_sampling),
        ("inspect_checkpoint", test_inspect_checkpoint),
        ("end_to_end_synthetic", test_end_to_end_synthetic),
        ("decision_thresholds", test_decision_thresholds),
    ]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
    print("TEST_SUMMARY " + json.dumps(
        {"total": len(tests), "failed": failures}))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
