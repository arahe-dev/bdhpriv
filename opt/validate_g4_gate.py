"""Host-side validator for the G4 confirmation cell's correctness gates.

Executes results/g4_confirm_autoresearch_iii.py with Colab/GPU-only parts
stubbed (CPU torch, fake CUDA capabilities, stubbed google.colab) and
ARM_A_GATE_ONLY=1 so main() is skipped, then runs the two pure gates:
  - tiny_packed_correctness: dense fp64 oracle vs certified/candidate scans
  - tiny_model_gate: certified vs candidate stacked model (logits/loss/grads)

This runs the REAL gate code from the cell (no copies, no drift).
Usage: python opt/validate_g4_gate.py
"""

import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CELL = os.path.join(ROOT, "results", "g4_confirm_autoresearch_iii.py")


def install_stubs():
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda *a, **k: None)
    google.colab = colab
    sys.modules["google"] = google
    sys.modules["google.colab"] = colab

    torch.cuda.is_available = lambda: True
    torch.cuda.is_bf16_supported = lambda: True
    torch.cuda.get_device_name = (
        lambda *a, **k: "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    )
    torch.cuda.get_device_capability = lambda *a, **k: (12, 0)
    try:
        torch.__version__ = "2.11.0+cu128"
    except Exception:
        pass
    try:
        torch.version.cuda = "12.8"
    except Exception:
        pass


def patched_source():
    with open(CELL, "r", encoding="utf-8") as f:
        src = f.read()

    guard = (
        'if "torch" in sys.modules:\n'
        '    raise RuntimeError("Restart the Colab runtime before running this cell.")'
    )
    assert guard in src, "runtime-restart guard not found"
    src = src.replace(guard, "pass")

    assert 'DEVICE = torch.device("cuda")' in src, "DEVICE line not found"
    src = src.replace('DEVICE = torch.device("cuda")', 'DEVICE = torch.device("cpu")')

    for line in (
        "torch.backends.cuda.enable_flash_sdp(True)",
        "torch.backends.cuda.enable_mem_efficient_sdp(False)",
        "torch.backends.cuda.enable_math_sdp(False)",
    ):
        assert line in src, f"backends line not found: {line}"
        src = src.replace(line, "pass")

    return src


def main():
    install_stubs()
    os.environ["ARM_A_GATE_ONLY"] = "1"
    ns = {"__name__": "__gate_validator__"}
    exec(compile(patched_source(), CELL, "exec"), ns)

    print("cell executed with gates only; running gates...")
    gate = ns["tiny_packed_correctness"]()
    print("TINY_GATE=", gate)
    model_gate = ns["tiny_model_gate"]()
    print("TINY_MODEL_GATE=", model_gate)

    assert gate["status"] == "PASS"
    assert model_gate["status"] == "PASS"
    print("VALIDATION_PASS=true")


if __name__ == "__main__":
    main()
