"""Host-side dense BDH source lock (git + SHA-256 provenance).

Writes artifacts/dense_bdh_verification/source_manifest.json.
py -3.12 opt/dense_verify_source_lock.py
"""

from __future__ import annotations

import hashlib
import json
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
    "dense_verify_orchestrator": "opt/dense_verify_orchestrate.py",
    "dense_verify_reporter": "opt/dense_verify_report.py",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(args):
    return subprocess.check_output(["git"] + args, cwd=str(ROOT),
                                   text=True).strip()


def main():
    frozen = json.loads((ROOT / "results" / "350k_frozen_state.json")
                        .read_text(encoding="utf-8"))
    contract = frozen["candidate_source_blobs"]
    entries = {}
    for label, rel in SOURCES.items():
        path = ROOT / rel
        blob = git(["hash-object", rel])
        frozen_blob = contract.get(rel)
        dirty = git(["status", "--porcelain", "--", rel])
        entries[label] = {
            "path": rel,
            "sha256_raw": sha256(path),
            "git_blob_hash": blob,
            "frozen_contract_blob": frozen_blob,
            "matches_frozen_contract": (
                None if frozen_blob is None else frozen_blob == blob
            ),
            "working_tree_clean_for_path": dirty == "",
            "bytes": path.stat().st_size,
        }
    manifest = {
        "role_assignments": {
            "CONTROL": "dense BDH = opt/model_opt.py OptArmA (opt3c flags)",
            "COMPARATOR": "pinned executable dense reference = "
                          "opt/model_ref.py NativeReadStage1ArmA",
            "DUT": "same as CONTROL (current dense Arm-A); no separate DUT",
            "OUT_OF_SCOPE": "sparse BDH, routed experts, top-k execution, "
                            "microbatch-dependent sparse paths",
        },
        "git_head": git(["rev-parse", "HEAD"]),
        "git_branch": git(["branch", "--show-current"]),
        "git_status_porcelain": git(["status", "--porcelain"]),
        "frozen_commit": frozen["frozen_at_commit"],
        "sources": entries,
        "original_soda_vendor_file": {
            "present": False,
            "pinned_sha256": "947f8b33e740adede3e13382f8cb9e8e374d845de75"
                             "e282bda5feb05c582f624",
            "note": "vendor/canonical/icrl_reintegrated_bdh_baseline_v1.py is "
                    "absent from this workspace (README_FIRST.md pins only "
                    "its SHA-256); original Soda comparator is BLOCKED",
        },
    }
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "git_head": manifest["git_head"],
        "frozen_commit": manifest["frozen_commit"],
        "sources": {k: {"matches": v["matches_frozen_contract"],
                        "clean": v["working_tree_clean_for_path"]}
                    for k, v in entries.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
