"""Run Akasha diagnostics with an explicit CPU/RAM execution contract.

This launcher is for Linux workers that do not provide Apple MLX. It uses the
upstream PyTorch reference implementation on CPU and refuses a CUDA-enabled
PyTorch build. The refusal is deliberate: setting ``CUDA_VISIBLE_DEVICES``
after importing torch is not a reliable RAM-only guarantee.

Examples::

    python -m scripts.akasha_ram_only --report
    python -m scripts.akasha_ram_only --pytest tests/akasha --skip-slow -q
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Sequence


# Set this before importing torch or any module that may import torch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]


def environment_report() -> dict:
    """Return the selected backend and reject CUDA builds."""
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mlx_available": importlib.util.find_spec("mlx") is not None,
        "torch_available": importlib.util.find_spec("torch") is not None,
        "requested_backend": "cpu_ram",
        "cuda_allowed": False,
    }
    if not report["torch_available"]:
        report["selected_device"] = "unavailable until CPU torch is installed"
        return report

    import torch

    report["torch"] = torch.__version__
    report["torch_cuda_build"] = torch.version.cuda
    if torch.version.cuda is not None:
        raise RuntimeError(
            "RAM-only Akasha guard rejected a CUDA-enabled torch build; "
            "install a CPU-only PyTorch wheel"
        )
    report["selected_device"] = "cpu"
    return report


def run_pytest(args: Sequence[str]) -> int:
    """Run the requested test command with the CPU lock applied."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        cwd=ROOT,
        env=env,
        check=False,
    ).returncode


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv or argv[0] == "--report":
        print(json.dumps(environment_report(), indent=2, sort_keys=True))
        return 0
    if argv[0] == "--pytest":
        return run_pytest(argv[1:] or ("tests/akasha", "--skip-slow", "-q"))
    raise SystemExit(
        "usage: python -m scripts.akasha_ram_only "
        "[--report | --pytest [pytest args...]]"
    )


if __name__ == "__main__":
    raise SystemExit(main())
