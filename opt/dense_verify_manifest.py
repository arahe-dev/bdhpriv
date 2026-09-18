"""Dense BDH source/config lock + environment capture.

Writes artifacts/dense_bdh_verification/{source_manifest.json,environment.json}.
Run inside the container: python opt/dense_verify_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "dense_bdh_verification"
SOURCES = {
    "control_opt_arm_a": "opt/model_opt.py",
    "control_scan": "opt/scan_attn.py",
    "comparator_reference": "opt/model_ref.py",
    "oracle": "reference/scan_coordinator_oracle.py",
    "canonical_timing_cell": "reference/arm_a_timing_cell_colab.py",
    "dense_verify_harness": "opt/dense_verify.py",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sh(cmd):
    try:
        return subprocess.check_output(cmd, text=True,
                                       stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # noqa: BLE001
        return f"error: {type(exc).__name__}: {exc}"


def main():
    ART.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((ROOT / "results" / "350k_frozen_state.json")
                        .read_text(encoding="utf-8"))
    contract = frozen["candidate_source_blobs"]
    entries = {}
    for label, rel in SOURCES.items():
        path = ROOT / rel
        git_blob = subprocess.check_output(
            ["git", "hash-object", rel], cwd=str(ROOT), text=True).strip()
        frozen_blob = contract.get(rel)
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--", rel], cwd=str(ROOT),
            text=True).strip()
        entries[label] = {
            "path": rel,
            "sha256_raw": sha256(path),
            "git_blob_hash": git_blob,
            "frozen_contract_blob": frozen_blob,
            "matches_frozen_contract": (
                None if frozen_blob is None else frozen_blob == git_blob
            ),
            "working_tree_clean_for_path": dirty == "",
            "bytes": path.stat().st_size,
        }
    manifest = {
        "role_assignments": {
            "CONTROL": "dense BDH = opt/model_opt.py OptArmA opt3c flags",
            "COMPARATOR": "pinned executable reference = "
                          "opt/model_ref.py NativeReadStage1ArmA",
            "DUT": "same as CONTROL (current dense Arm-A); no separate DUT",
            "OUT_OF_SCOPE": "sparse BDH, routed experts, top-k execution",
        },
        "git_head": sh(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain": sh(["git", "status", "--porcelain"]),
        "frozen_commit": frozen["frozen_at_commit"],
        "sources": entries,
        "original_soda_vendor_file": {
            "present": False,
            "note": "vendor/canonical/icrl_reintegrated_bdh_baseline_v1.py "
                    "is absent from this workspace (README_FIRST.md pins its "
                    "SHA-256 only); original Soda comparator is unavailable",
        },
    }
    (ART / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    env = {
        "container": sh(["hostname"]),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": None, "cuda": None, "triton": None, "cudnn": None,
        "gpu": None, "nvidia_smi": None, "cpu": None, "memory": None,
        "env_vars": {k: os.environ.get(k) for k in (
            "PYTORCH_CUDA_ALLOC_CONF", "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR", "TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS",
        )},
    }
    import torch
    env["torch"] = torch.__version__
    env["cuda"] = torch.version.cuda
    try:
        import triton
        env["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        env["triton"] = None
    env["cudnn"] = torch.backends.cudnn.version()
    env["gpu"] = torch.cuda.get_device_name(0)
    env["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    env["gpu_memory_MiB"] = (torch.cuda.get_device_properties(0).total_memory
                             // 2**20)
    env["nvidia_smi"] = sh([
        "nvidia-smi",
        "--query-gpu=name,driver_version,vbios_version,memory.total,"
        "clocks.max.sm,power.limit,temperature.gpu",
        "--format=csv,noheader"])
    env["cpu"] = sh(["sh", "-c",
                     "grep -m1 'model name' /proc/cpuinfo; nproc"])
    env["memory"] = sh(["sh", "-c",
                        "grep -E 'MemTotal|MemAvailable' /proc/meminfo"])
    (ART / "environment.json").write_text(
        json.dumps(env, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"source_manifest": str(ART / "source_manifest.json"),
                      "environment": str(ART / "environment.json"),
                      "contract_matches": {
                          k: v["matches_frozen_contract"]
                          for k, v in entries.items()}},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
