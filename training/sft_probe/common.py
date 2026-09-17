"""Shared utilities for the Arm-A SFT calibration campaign.

Campaign-owned probe code. It imports the frozen trainer module read-only and
never modifies it: the model class, scan, coordinator, writer, LayerNorm,
RoPE, optimizer construction and update routine are all taken from
``training/arm_a_2p5b_trainer.py`` @ 0dcbb87.

Evidence discipline: every number produced by this package is either MEASURED
on the real checkpoint or explicitly labelled otherwise in the campaign
report.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TRAINER_PATH = ROOT / "training" / "arm_a_2p5b_trainer.py"
TRAINER_SHA256 = (
    "1985fa42042033c842c7ed0faea2c34ead6516bb1fe56753426b6774ee2d0b49"
)

BASE_CKPT = ROOT / "runs" / "arm_a_2p5b_opt3c_all" / "ckpt" / "latest.pt"
BASE_CKPT_SHA256 = (
    "fe7e6c2c6ac0d12630018812f735486c45003678adbfcb4a88a476d886c47efb"
)
BASE_CKPT_TOKENS = 2_500_067_328
BASE_CKPT_UPDATES = 19_074
BASE_WEIGHTS_FINGERPRINT = (
    "dbc2acc115e795f27b1ded7624cbffb9e90f6c49dcb54fad0cb2705f9627a934"
)

TOKENIZER_PATH = Path(
    r"C:\Users\arahe\OneDrive\Documents\ChatGPT\icrl"
    r"\phase_bdh_corpus_stage1\tokenizer\tokenizer.json"
)
TOKENIZER_SHA256 = (
    "9a05bca4c065d9952a01ee1e3f4b6e23e39200e1f33da65e5dad1819419f71c3"
)
TOKENIZER_IDENTITY = "bytelevel-bpe-8192-9a05bca4c065d995"

OUT_DIR = ROOT / "results" / "sft_probe"
DATA_DIR = OUT_DIR / "data"
CKPT_DIR = ROOT / "runs" / "sft_probe"
GEN_DIR = OUT_DIR / "generations"

N_PARAM = 17_375_489
SPLIT_SEED = 1337
SAMPLING_SEED = 20260916

TARGET_TPP = {
    "dose_001": 0.01 * N_PARAM,
    "dose_003": 0.03 * N_PARAM,
    "dose_010": 0.10 * N_PARAM,
    "dose_030": 0.30 * N_PARAM,
}

PAD_ID = 0

_TRAINER_MOD = None


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path, block: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_trainer():
    """Import the frozen trainer module without executing its CLI."""
    global _TRAINER_MOD
    if _TRAINER_MOD is not None:
        return _TRAINER_MOD
    spec = importlib.util.spec_from_file_location(
        "arm_a_2p5b_trainer_frozen", TRAINER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _TRAINER_MOD = module
    return module


def load_tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(TOKENIZER_PATH))


def verify_frozen_assets(verify_checkpoint: bool = True) -> dict:
    trainer_hash = sha256_file(TRAINER_PATH)
    tokenizer_hash = sha256_file(TOKENIZER_PATH)
    report = {
        "trainer_path": str(TRAINER_PATH),
        "trainer_sha256": trainer_hash,
        "trainer_pin_match": trainer_hash == TRAINER_SHA256,
        "tokenizer_path": str(TOKENIZER_PATH),
        "tokenizer_sha256": tokenizer_hash,
        "tokenizer_pin_match": tokenizer_hash == TOKENIZER_SHA256,
        "tokenizer_identity": TOKENIZER_IDENTITY,
        "base_checkpoint_path": str(BASE_CKPT),
    }
    if not report["trainer_pin_match"]:
        raise RuntimeError("frozen trainer hash mismatch; refusing to proceed")
    if not report["tokenizer_pin_match"]:
        raise RuntimeError("tokenizer hash mismatch; refusing to proceed")
    if verify_checkpoint:
        base_hash = sha256_file(BASE_CKPT)
        report["base_checkpoint_sha256"] = base_hash
        report["base_checkpoint_pin_match"] = base_hash == BASE_CKPT_SHA256
        if not report["base_checkpoint_pin_match"]:
            raise RuntimeError(
                "base checkpoint hash mismatch; refusing to proceed"
            )
    return report


def load_base_payload() -> dict:
    return torch.load(BASE_CKPT, map_location="cpu", weights_only=False)


def load_base_state() -> Dict[str, torch.Tensor]:
    payload = load_base_payload()
    state = payload["model"]
    if int(payload["progress"]["updates_done"]) != BASE_CKPT_UPDATES:
        raise RuntimeError("base checkpoint update count mismatch")
    if int(payload["progress"]["tokens_consumed"]) != BASE_CKPT_TOKENS:
        raise RuntimeError("base checkpoint token count mismatch")
    return state


def build_model(
    state: Optional[Dict[str, torch.Tensor]] = None,
    device: str = "cpu",
    scan_block: int = 1024,
):
    trainer = load_trainer()
    cfg = trainer.ArmAConfig()
    model = trainer.OptArmA(cfg, torch.device(device), scan_block=scan_block)
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model


def frozen_config_dict(cfg) -> dict:
    return load_trainer().frozen_config_dict(cfg)


def make_optimizer(model, cfg, device_type: str):
    return load_trainer().make_optimizer(model, cfg, device_type)


def save_checkpoint_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        torch.save(_to_cpu_tree(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _to_cpu_tree(obj):
    if torch.is_tensor(obj):
        return obj.detach().to("cpu")
    if isinstance(obj, dict):
        return {k: _to_cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu_tree(v) for v in obj)
    return obj


def sft_checkpoint_payload(
    model,
    optimizer,
    cfg,
    sft_meta: dict,
    progress: dict,
) -> dict:
    """Payload accepted by the Akasha converter (arm_a_2p5b_ckpt_v1)."""
    trainer = load_trainer()
    return {
        "format": trainer.CKPT_FORMAT,
        "implementation": trainer.IMPLEMENTATION_VERSION,
        "code_sha256": TRAINER_SHA256,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "config": trainer.frozen_config_dict(cfg),
        "corpus": {
            "corpus_id": trainer.PROD_CONTRACT.corpus_id,
            "artifact_hashes_sha256": trainer.PROD_CONTRACT.artifact_hashes_sha256,
            "logical_replay_sha256": trainer.PROD_CONTRACT.logical_replay_sha256,
            "total_sequences": int(trainer.PROD_CONTRACT.sequences),
        },
        "progress": {
            "updates_done": int(progress["updates_done"]),
            "tokens_consumed": int(progress["tokens_consumed"]),
            "target_tokens": int(progress["target_tokens"]),
            "next_sequence": int(progress.get("next_sequence", 0)),
        },
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        },
        "saved_at": iso_now(),
        "sft": sft_meta,
    }


PARAM_GROUPS = {
    "embedding": ("embedding.weight",),
    "encoder": ("encoder",),
    "decoder_x": ("decoder_x",),
    "decoder_y": ("decoder_y",),
    "coordinator.Wc": ("coordinator.Wc",),
    "coordinator.bc": ("coordinator.bc",),
    "coordinator.alpha": ("coordinator.alpha",),
    "writer.W1": ("writer.W1",),
    "writer.W2": ("writer.W2",),
    "readout": ("readout",),
}


def parameter_group_norms(state: Dict[str, torch.Tensor]) -> Dict[str, float]:
    out = {}
    for group, names in PARAM_GROUPS.items():
        total = 0.0
        for name in names:
            tensor = state[name].detach().to(torch.float64)
            total += float((tensor * tensor).sum().item())
        out[group] = float(np.sqrt(total))
    return out


def parameter_drift(
    base_state: Dict[str, torch.Tensor],
    current_state: Dict[str, torch.Tensor],
) -> dict:
    base_norms = parameter_group_norms(base_state)
    cur_norms = parameter_group_norms(current_state)
    total_base_sq = 0.0
    total_delta_sq = 0.0
    groups = {}
    for group, names in PARAM_GROUPS.items():
        delta_sq = 0.0
        for name in names:
            delta = (
                current_state[name].detach().to(torch.float64)
                - base_state[name].detach().to(torch.float64)
            )
            delta_sq += float((delta * delta).sum().item())
        delta_norm = float(np.sqrt(delta_sq))
        rel = (
            delta_norm / base_norms[group] if base_norms[group] > 0 else float("nan")
        )
        groups[group] = {
            "param_count": int(
                sum(base_state[n].numel() for n in names)
            ),
            "base_norm": base_norms[group],
            "sft_norm": cur_norms[group],
            "delta_norm": delta_norm,
            "relative_update_norm": rel,
            "cosine_base_vs_sft": _cosine_state(
                base_state, current_state, names
            ),
        }
        total_base_sq += base_norms[group] ** 2
        total_delta_sq += delta_sq
    total_base = float(np.sqrt(total_base_sq))
    total_delta = float(np.sqrt(total_delta_sq))
    groups["TOTAL"] = {
        "param_count": int(sum(v.numel() for v in base_state.values())),
        "base_norm": total_base,
        "delta_norm": total_delta,
        "relative_update_norm": total_delta / total_base,
        "cosine_base_vs_sft": _cosine_state(
            base_state, current_state, tuple(base_state)
        ),
    }
    return {"groups": groups}


def _cosine_state(base, current, names) -> float:
    dot = 0.0
    b_sq = 0.0
    c_sq = 0.0
    for name in names:
        b = base[name].detach().to(torch.float64).reshape(-1)
        c = current[name].detach().to(torch.float64).reshape(-1)
        dot += float((b * c).sum().item())
        b_sq += float((b * b).sum().item())
        c_sq += float((c * c).sum().item())
    if b_sq == 0 or c_sq == 0:
        return float("nan")
    return dot / float(np.sqrt(b_sq) * np.sqrt(c_sq))


def save_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f"not JSON-serializable: {type(o)}")
