"""Local GPU validator for results/g4_opt3c_all_final.py.

Executes the cell with Colab-only parts stubbed (google.colab, Blackwell
GPU-name check, restart guard) and calls its gate functions directly so the
merged preflight is exercised on this machine's GPU before it ships.

Usage (container): python opt/validate_final_cell.py [--gates a,b,c]
Default gates: model_equivalence, graph_breaks, determinism, checkpoint.
(Smoke train is excluded locally: it requires the frozen corpus.)
"""

import argparse
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CELL = os.path.join(ROOT, "results", "g4_opt3c_all_final.py")


def install_stubs():
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda *a, **k: None)
    google.colab = colab
    sys.modules["google"] = google
    sys.modules["google.colab"] = colab


def patched_source():
    with open(CELL, "r", encoding="utf-8") as f:
        src = f.read()

    guard = (
        'if "torch" in sys.modules:\n'
        '    raise RuntimeError("Restart the Colab runtime before running this cell.")'
    )
    assert guard in src, "runtime-restart guard not found"
    src = src.replace(guard, "pass")

    # Local GPU is not a Blackwell; bypass only the GPU-name assertion so the
    # gates themselves run unmodified. The version checks stay active.
    name_check = (
        'if "RTX PRO 6000 Blackwell" not in GPU_NAME or GPU_CAPABILITY != (12, 0):'
    )
    assert name_check in src, "GPU name check not found"
    src = src.replace(name_check, "if False:")

    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gates",
        default="model_equivalence,graph_breaks,determinism,checkpoint",
    )
    args = ap.parse_args()
    wanted = {s.strip() for s in args.gates.split(",") if s.strip()}

    install_stubs()
    os.environ["ARM_A_GATE_ONLY"] = "1"
    ns = {"__name__": "__final_cell_validator__"}
    exec(compile(patched_source(), CELL, "exec"), ns)

    dispatch = {
        "model_equivalence": ns["gate_model_equivalence"],
        "graph_breaks": ns["gate_graph_breaks_g4"],
        "determinism": ns["gate_determinism"],
        "checkpoint": ns["gate_checkpoint_resume"],
    }
    for name in ("model_equivalence", "graph_breaks", "determinism", "checkpoint"):
        if name not in wanted:
            continue
        print(f"--- running gate group: {name} ---", flush=True)
        dispatch[name]()

    gates = ns["GATES"]
    failed = [g["name"] for g in gates if g["status"] == "FAIL"]
    print("VALIDATOR_GATES=" + str(len(gates)))
    print("VALIDATOR_FAILED=" + str(failed))
    print("VALIDATOR_PASS=" + str(not failed).lower())


if __name__ == "__main__":
    main()
