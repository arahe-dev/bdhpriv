from __future__ import annotations

import pytest

from akasha.bench.correctness import fp32_parity
from akasha.config import FP32_ATOL, FP32_RTOL, REQUIRED_V0_LENGTHS


@pytest.fixture(scope="session")
def parity_result(slow_enabled):
    lengths = list(REQUIRED_V0_LENGTHS)
    if not slow_enabled:
        lengths = [length for length in lengths if length <= 512]
    return fp32_parity(lengths=lengths, patterns=("random", "repeated", "segments"))


def test_fp32_full_vs_recurrent_within_contract(parity_result):
    assert parity_result["cases"], "no parity cases executed"
    for case in parity_result["cases"]:
        assert case["allclose_atol_rtol"], (
            f"{case['label']}: max_abs={case['max_abs_error']:.3e} "
            f"violations={case['tolerance_violations']} "
            f"(atol={FP32_ATOL}, rtol={FP32_RTOL})"
        )
        assert case["argmax_agreement"] == 1.0, case


def test_2048_token_parity(parity_result, slow_enabled):
    if not slow_enabled:
        pytest.skip("--skip-slow: 1024/2048 production parity not executed")
    assert parity_result["length_2048_pass"], parity_result["by_length"]["2048"]


def test_observed_error_distribution_is_reported(parity_result):
    assert parity_result["max_abs_error"] >= 0.0
    assert 0.0 <= parity_result["min_argmax_agreement"] <= 1.0
    assert parity_result["total_tolerance_violations"] == 0
