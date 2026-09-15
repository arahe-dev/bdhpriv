#!/usr/bin/env python
"""CPU tests for training/arm_a_2p5b_trainer.py against a synthetic corpus.

The fixture is a two-shard frozen corpus with 16 sequences in the exact
on-disk format (FROZEN.json / corpus_manifest.json / artifact_hashes.json /
tokens.bin / valid_lengths.bin / provenance.parquet). It deliberately covers:
  - multi-document packed sequences,
  - documents split across adjacent rows (continues_inside),
  - a document continued across sequences within one shard,
  - a document continued across the shard boundary (cross-shard lookahead),
  - padding tails (valid_length < T) and a length-1 sequence,
  - repeated same-document rows.

Every semantic check derives expected tensors from the generator's own
placements, then compares bitwise with the trainer's streamed rows.

Run: py -3.12 training/test_stream_and_ckpt.py
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "arm_a_trainer", HERE / "arm_a_2p5b_trainer.py"
)
trainer = importlib.util.module_from_spec(_spec)
sys.modules["arm_a_trainer"] = trainer
_spec.loader.exec_module(trainer)

T = 32
SMALL_CFG = trainer.ArmAConfig(
    T=T, V=128, D=16, N=64, H=4, L=2, HIDDEN=24,
    SEED=7, THETA=2.0**16, READ_BLOCK=16, PEAK_LR=1e-3,
    WARMUP_TOKENS=512, GLOBAL_BATCH=4, MICROBATCH=2, SCAN_BLOCK=16,
)

# (length, [(doc label, document offset, token count)]) per global sequence.
PLAN = [
    (32, [("A", 0, 9), ("B", 0, 10), ("C", 0, 6), ("C", 6, 7)]),
    (27, [("D", 0, 27)]),
    (32, [("E", 0, 20), ("F", 0, 12)]),
    (32, [("F", 12, 8), ("G", 0, 5), ("G", 5, 7), ("H", 0, 12)]),
    (32, [("I", 0, 32)]),
    (1, [("J", 0, 1)]),
    (32, [("Y", 0, 32)]),
    (32, [("K", 0, 17), ("L", 0, 15)]),
    (32, [("L", 15, 7), ("L", 22, 8), ("M", 0, 17)]),
    (20, [("O", 0, 20)]),
    (32, [("P", 0, 10), ("Q", 0, 11), ("Q", 11, 11)]),
    (5, [("R", 0, 5)]),
    (32, [("S", 0, 32)]),
    (32, [("T", 0, 16), ("U", 0, 16)]),
    (32, [("V", 0, 16), ("W", 0, 16)]),
    (8, [("X", 0, 8)]),
]
DOC_LEN = {
    "A": 9, "B": 10, "C": 13, "D": 27, "E": 20, "F": 20, "G": 12, "H": 12,
    "I": 32, "J": 1, "Y": 32, "K": 17, "L": 30, "M": 17, "O": 20, "P": 10,
    "Q": 22, "R": 5, "S": 32, "T": 16, "U": 16, "V": 16, "W": 16, "X": 8,
}
SHARD_SPLIT = 8


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def write_json_bytes(path: Path, obj) -> bytes:
    data = json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")
    Path(path).write_bytes(data)
    return data


def build_meta(cfg):
    doc_ids = {}
    doc_tokens = {}
    next_doc_id = 100

    def doc_token_value(did, offset):
        return (did * 97 + offset * 13) % (cfg.V - 1) + 1

    sequences = []
    for length, placements in PLAN:
        rows = []
        cursor = 0
        for label, offset, count in placements:
            if label not in doc_ids:
                doc_ids[label] = next_doc_id
                next_doc_id += 1
                did = doc_ids[label]
                doc_tokens[did] = np.array(
                    [doc_token_value(did, j) for j in range(DOC_LEN[label])],
                    dtype=np.uint16,
                )
            did = doc_ids[label]
            rows.append({
                "a": cursor, "e": cursor + count, "doc": did, "label": label,
                "dstart": offset, "dend": offset + count,
            })
            cursor += count
        assert cursor == length, (length, cursor)
        sequences.append(rows)

    meta = {
        "seqlens": [], "rows": sequences, "tokens": [], "doc_of": [],
        "pos": [], "carry": [], "next_first": [], "shards": [],
    }
    for g, (length, _placements) in enumerate(PLAN):
        x = np.zeros(cfg.T, dtype=np.uint16)
        doc_of = np.full(cfg.T, -1, dtype=np.int64)
        pos = np.zeros(cfg.T, dtype=np.int64)
        for row in sequences[g]:
            a, e = row["a"], row["e"]
            did = row["doc"]
            x[a:e] = doc_tokens[did][row["dstart"]:row["dend"]]
            doc_of[a:e] = did
            pos[a:e] = np.arange(row["dstart"], row["dend"], dtype=np.int64)
        meta["seqlens"].append(length)
        meta["tokens"].append(x)
        meta["doc_of"].append(doc_of)
        meta["pos"].append(pos)

    for g in range(len(PLAN) - 1):
        last = sequences[g][-1]
        first = sequences[g + 1][0]
        carry = (
            last["label"] == first["label"]
            and last["dend"] == first["dstart"]
            and last["e"] == meta["seqlens"][g]
            and first["a"] == 0
        )
        meta["carry"].append(bool(carry))
    meta["carry"].append(False)

    for g in range(len(PLAN)):
        meta["next_first"].append(
            int(meta["tokens"][g + 1][0]) if g + 1 < len(PLAN) else None
        )

    base = 0
    shards = []
    for split in (SHARD_SPLIT, len(PLAN) - SHARD_SPLIT):
        shards.append(list(range(base, base + split)))
        base += split
    meta["shards"] = shards
    meta["doc_count"] = len(doc_ids)
    return meta


def build_fixture(root: Path, cfg):
    root = Path(root)
    (root / "train").mkdir(parents=True)
    meta = build_meta(cfg)

    shard_decls = []
    artifact_files = []

    for shard_number, seqs in enumerate(meta["shards"]):
        stem = f"shard_{shard_number:06d}"
        tokens = np.stack([meta["tokens"][g] for g in seqs])
        lengths = np.array([meta["seqlens"][g] for g in seqs], dtype=np.uint16)
        token_path = root / "train" / f"{stem}.tokens.bin"
        length_path = root / "train" / f"{stem}.valid_lengths.bin"
        prov_path = root / "train" / f"{stem}.provenance.parquet"
        token_path.write_bytes(tokens.astype("<u2").tobytes())
        length_path.write_bytes(lengths.astype("<u2").tobytes())

        records = []
        for g in seqs:
            for row in meta["rows"][g]:
                records.append({
                    "sequence_index": g,
                    "sequence_token_start": row["a"],
                    "sequence_token_end": row["e"],
                    "selected_document_index": row["doc"],
                    "document_token_start": row["dstart"],
                    "document_token_end": row["dend"],
                })
        table = pa.table({
            name: pa.array([r[name] for r in records], type=pa.int64())
            for name in trainer.PROVENANCE_COLUMNS
        })
        pq.write_table(table, prov_path)

        for path in (token_path, length_path, prov_path):
            rel = path.relative_to(root).as_posix()
            artifact_files.append({
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        shard_decls.append({
            "stem": stem,
            "tokens_file": f"train/{stem}.tokens.bin",
            "valid_lengths_file": f"train/{stem}.valid_lengths.bin",
            "provenance_file": f"train/{stem}.provenance.parquet",
            "sequences": len(seqs),
            "provenance_rows": len(records),
        })

    real_tokens = sum(meta["seqlens"])
    physical_tokens = len(PLAN) * cfg.T
    padding_tokens = physical_tokens - real_tokens

    manifest = {
        "status": "FROZEN",
        "real_tokens": real_tokens,
        "physical_tokens": physical_tokens,
        "padding_tokens": padding_tokens,
        "selected_documents": meta["doc_count"],
        "sequences": len(PLAN),
        "logical_replay_sha256": sha256_bytes(b"fixture-logical-replay"),
        "stream_tokens": {"train": real_tokens},
        "shards": shard_decls,
    }
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    (root / "corpus_manifest.json").write_bytes(manifest_bytes)
    artifact_files.append({
        "path": "corpus_manifest.json",
        "bytes": len(manifest_bytes),
        "sha256": sha256_bytes(manifest_bytes),
    })

    artifact_index = {"algorithm": "sha256", "files": artifact_files}
    artifact_bytes = json.dumps(
        artifact_index, indent=2, sort_keys=True
    ).encode("utf-8")
    (root / "artifact_hashes.json").write_bytes(artifact_bytes)
    artifact_digest = sha256_bytes(artifact_bytes)

    frozen = {
        "status": "FROZEN",
        "corpus_id": "arm_a_fixture_v1",
        "context_length": cfg.T,
        "tokenizer_sha256": sha256_bytes(b"fixture-tokenizer"),
        "logical_replay_sha256": manifest["logical_replay_sha256"],
        "artifact_hashes_sha256": artifact_digest,
        "real_training_tokens": real_tokens,
        "sequences": len(PLAN),
        "padding_tokens_masked": padding_tokens,
    }
    write_json_bytes(root / "FROZEN.json", frozen)

    contract = trainer.CorpusContract(
        corpus_id="arm_a_fixture_v1",
        context_length=cfg.T,
        tokenizer_sha256=frozen["tokenizer_sha256"],
        logical_replay_sha256=frozen["logical_replay_sha256"],
        artifact_hashes_sha256=artifact_digest,
        real_training_tokens=real_tokens,
        sequences=len(PLAN),
        padding_tokens_masked=padding_tokens,
        selected_documents=meta["doc_count"],
        physical_tokens=physical_tokens,
        shard_count=len(meta["shards"]),
    )
    return contract, meta


def load_corpus(root, cfg, contract, verify=True, fast=False):
    return trainer.FrozenPackedCorpus(
        root, cfg, contract=contract, verify_files=verify, fast_verify=fast
    )


def expected_row(meta, g, cfg):
    length = meta["seqlens"][g]
    x = meta["tokens"][g]
    doc_of = meta["doc_of"][g]
    pos = meta["pos"][g]
    input_valid = np.zeros(cfg.T, dtype=np.bool_)
    input_valid[:length] = True
    valid = np.zeros(cfg.T, dtype=np.bool_)
    y = np.zeros(cfg.T, dtype=np.uint16)
    for t in range(length - 1):
        if doc_of[t] == doc_of[t + 1]:
            valid[t] = True
            y[t] = x[t + 1]
    if meta["carry"][g]:
        valid[length - 1] = True
        y[length - 1] = meta["next_first"][g]
    start = np.zeros(cfg.T, dtype=np.int32)
    segpos = np.zeros(cfg.T, dtype=np.int32)
    t = 0
    while t < length:
        run_end = t
        while run_end + 1 < length and doc_of[run_end + 1] == doc_of[t]:
            run_end += 1
        start[t:run_end + 1] = t
        segpos[t:run_end + 1] = np.arange(run_end + 1 - t, dtype=np.int32)
        t = run_end + 1
    start[length:] = np.int32(cfg.T + g + 1)
    return {
        "x": x.astype(np.uint16), "y": y, "pos": pos, "segpos": segpos,
        "start": start, "input_valid": input_valid, "valid": valid,
    }


def assert_row_equal(got, want, label):
    for key, expected in want.items():
        actual = got[key]
        if not isinstance(actual, torch.Tensor):
            actual = torch.from_numpy(np.ascontiguousarray(actual))
        assert torch.equal(actual, torch.from_numpy(expected)), (
            f"{label}: {key} differs\n"
            f"  got={actual.tolist()}\n want={expected.tolist()}"
        )


def test_fixture_semantics(tmp: Path, root: Path, contract, meta):
    corpus = load_corpus(root, SMALL_CFG, contract)
    assert corpus.total_sequences == len(PLAN)
    seen = []
    for g, row in corpus.iter_rows(0):
        seen.append(g)
        assert_row_equal(row, expected_row(meta, g, SMALL_CFG), f"seq{g}")
    assert seen == list(range(len(PLAN))), seen
    assert sum(meta["carry"]) == 2, meta["carry"]


def test_contract_failclosed(tmp: Path, root: Path, contract, meta):
    load_corpus(root, SMALL_CFG, contract)

    tampered = tmp / "tamper_byte"
    shutil.copytree(root, tampered)
    token = tampered / "train" / "shard_000000.tokens.bin"
    data = bytearray(token.read_bytes())
    data[3] ^= 0x01
    token.write_bytes(bytes(data))
    try:
        load_corpus(tampered, SMALL_CFG, contract)
        raise AssertionError("tampered token shard was accepted")
    except trainer.CorpusContractError:
        pass

    bad_status = tmp / "tamper_status"
    shutil.copytree(root, bad_status)
    frozen = json.loads((bad_status / "FROZEN.json").read_text(encoding="utf-8"))
    frozen["status"] = "DRAFT"
    (bad_status / "FROZEN.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True), encoding="utf-8"
    )
    try:
        load_corpus(bad_status, SMALL_CFG, contract)
        raise AssertionError("non-FROZEN status was accepted")
    except trainer.CorpusContractError:
        pass

    bad_index = tmp / "tamper_index"
    shutil.copytree(root, bad_index)
    index = json.loads(
        (bad_index / "artifact_hashes.json").read_text(encoding="utf-8")
    )
    index["files"][0]["sha256"] = "0" * 64
    (bad_index / "artifact_hashes.json").write_text(
        json.dumps(index, indent=2, sort_keys=True), encoding="utf-8"
    )
    try:
        load_corpus(bad_index, SMALL_CFG, contract)
        raise AssertionError("modified artifact index was accepted")
    except trainer.CorpusContractError:
        pass

    bad_lengths = tmp / "tamper_lengths"
    shutil.copytree(root, bad_lengths)
    lengths = bad_lengths / "train" / "shard_000001.valid_lengths.bin"
    payload = bytearray(lengths.read_bytes())
    payload[0:2] = (0).to_bytes(2, "little")
    lengths.write_bytes(bytes(payload))
    corpus = load_corpus(bad_lengths, SMALL_CFG, contract, verify=False)
    try:
        next(corpus.iter_rows(0))
        raise AssertionError("zero valid length was accepted")
    except trainer.CorpusContractError:
        pass

    out = io.StringIO()
    original_contract = trainer.PROD_CONTRACT
    original_cfg = trainer.PROD_CFG
    trainer.PROD_CONTRACT = contract
    trainer.PROD_CFG = SMALL_CFG
    try:
        with contextlib.redirect_stdout(out):
            code = trainer.main([
                "--mode", "smoke",
                "--corpus-root", str(tampered),
                "--run-dir", str(tmp / "cli_run"),
                "--no-compile", "--no-graph-check",
            ])
    finally:
        trainer.PROD_CONTRACT = original_contract
        trainer.PROD_CFG = original_cfg
    assert code == 2, f"CLI contract failure returned {code}"
    assert "ARM_A_2P5B_TRAINER_READY=false" in out.getvalue()


def test_batch_streaming(tmp: Path, root: Path, contract, meta):
    corpus = load_corpus(root, SMALL_CFG, contract)
    batch_size = SMALL_CFG.GLOBAL_BATCH

    batches = list(corpus.stream_batches(0, batch_size))
    assert [c for c, _ in batches] == [0, 4, 8, 12], [c for c, _ in batches]
    rows = [row for _, row in corpus.iter_rows(0)]
    for index, (_, batch) in enumerate(batches):
        for offset in range(batch_size):
            for key in batch:
                expected = torch.from_numpy(
                    expected_row(meta, index * batch_size + offset, SMALL_CFG)[key]
                )
                assert torch.equal(batch[key][offset], expected), (
                    f"batch {index} row {offset} key {key}"
                )

    resumed = list(corpus.stream_batches(4, batch_size))
    assert [c for c, _ in resumed] == [4, 8, 12]
    for (_, batch_a), (_, batch_b) in zip(batches[1:], resumed):
        for key in batch_a:
            assert torch.equal(batch_a[key], batch_b[key]), key

    odd = list(corpus.stream_batches(0, 5))
    assert [c for c, _ in odd] == [0, 5, 10], [c for c, _ in odd]

    tail = list(corpus.stream_batches(0, len(PLAN)))
    assert len(tail) == 1
    assert len(list(corpus.stream_batches(12, batch_size))) == 1
    assert list(corpus.stream_batches(len(PLAN), batch_size)) == []
    assert len(list(corpus.iter_rows(len(PLAN)))) == 0
    assert len(list(corpus.iter_rows(len(PLAN) - 1))) == 1

    dtypes = {k: v.dtype for k, v in batches[0][1].items()}
    assert dtypes == {
        "x": torch.uint16, "y": torch.uint16, "pos": torch.int64,
        "segpos": torch.int32, "start": torch.int32,
        "input_valid": torch.bool, "valid": torch.bool,
    }, dtypes


def test_cross_shard_lookahead(tmp: Path, root: Path, contract, meta):
    assert meta["carry"][SHARD_SPLIT - 1], "fixture lost the cross-shard carry"
    corpus = load_corpus(root, SMALL_CFG, contract)
    g = SHARD_SPLIT - 1
    _, row = next(corpus.iter_rows(g))
    length = meta["seqlens"][g]
    assert bool(row["valid"][length - 1]), "cross-shard target missing"
    assert int(row["y"][length - 1]) == meta["next_first"][g]

    padded = 1
    assert meta["seqlens"][padded] < SMALL_CFG.T
    _, row_padded = next(corpus.iter_rows(padded))
    assert not bool(row_padded["input_valid"][meta["seqlens"][padded]])
    assert not bool(row_padded["valid"][meta["seqlens"][padded]])
    assert int(row_padded["start"][meta["seqlens"][padded]]) == (
        SMALL_CFG.T + padded + 1
    )
    assert int(row_padded["input_valid"].sum()) == meta["seqlens"][padded]


def _make_trained_state(tmp: Path, root: Path, contract, meta, steps=2):
    device = torch.device("cpu")
    corpus = load_corpus(root, SMALL_CFG, contract)
    model, entry = trainer._build_model(SMALL_CFG, device, use_compile=False)
    trainer.load_init(model, trainer.canonical_init(SMALL_CFG), device)
    optimizer = trainer.make_optimizer(model, SMALL_CFG, device.type)
    stream = corpus.stream_batches(0, SMALL_CFG.GLOBAL_BATCH)
    losses = []
    for step in range(steps):
        _, batch = next(stream)
        result = trainer.one_full_update(
            batch, model, entry, optimizer, step, SMALL_CFG, device
        )
        losses.append(result["loss"])
    return corpus, model, optimizer, losses


def test_checkpoint_roundtrip(tmp: Path, root: Path, contract, meta):
    device = torch.device("cpu")
    run_dir = tmp / "ckpt_run"
    corpus, model, optimizer, losses = _make_trained_state(
        tmp, root, contract, meta
    )
    code_fp = trainer.code_fingerprint()
    payload = trainer.build_checkpoint(
        SMALL_CFG, corpus, model, optimizer, updates_done=2,
        tokens_consumed=2 * SMALL_CFG.GLOBAL_BATCH * SMALL_CFG.T,
        target_tokens=4 * SMALL_CFG.GLOBAL_BATCH * SMALL_CFG.T,
        code_fp=code_fp, session_stats={"last_loss": losses[-1]},
    )
    ckpt_path = run_dir / "ckpt" / "latest.pt"
    trainer.save_checkpoint_atomic(ckpt_path, payload)
    assert ckpt_path.is_file()
    assert not list(ckpt_path.parent.glob("*.tmp")), "atomic save left a tmp file"

    model_b, entry_b = trainer._build_model(SMALL_CFG, device, use_compile=False)
    optimizer_b = trainer.make_optimizer(model_b, SMALL_CFG, device.type)
    info = trainer.validate_and_load_checkpoint(
        ckpt_path, SMALL_CFG, corpus, model_b, optimizer_b, device, code_fp
    )
    assert info["updates_done"] == 2
    assert info["tokens_consumed"] == 2 * SMALL_CFG.GLOBAL_BATCH * SMALL_CFG.T
    assert info["next_sequence"] == 2 * SMALL_CFG.GLOBAL_BATCH

    for (name_a, param_a), (name_b, param_b) in zip(
        model.named_parameters(), model_b.named_parameters()
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b), f"parameter {name_a} differs"
    state_a = optimizer.state_dict()
    state_b = optimizer_b.state_dict()
    assert state_a["param_groups"] == state_b["param_groups"]
    assert set(state_a["state"]) == set(state_b["state"])
    for key in state_a["state"]:
        for field in state_a["state"][key]:
            value_a = state_a["state"][key][field]
            value_b = state_b["state"][key][field]
            assert torch.equal(value_a, value_b), f"optimizer {key}.{field}"

    corrupt_dir = tmp / "ckpt_corrupt_latest"
    corrupt_dir.mkdir(parents=True)
    step_path = corrupt_dir / "step_0000000001.pt"
    trainer.save_checkpoint_atomic(step_path, payload)
    (corrupt_dir / "latest.pt").write_bytes(b"not-a-checkpoint")
    logger = trainer.RunLogger(None)
    model_c, _ = trainer._build_model(SMALL_CFG, device, use_compile=False)
    optimizer_c = trainer.make_optimizer(model_c, SMALL_CFG, device.type)
    with contextlib.redirect_stdout(io.StringIO()):
        selected = trainer.find_latest_valid_checkpoint(
            corrupt_dir, SMALL_CFG, corpus, model_c, optimizer_c, device,
            code_fp, logger,
        )
    assert selected is not None and selected["path"].endswith("step_0000000001.pt")
    logger.close()

    (corrupt_dir / "step_0000000001.pt").write_bytes(b"broken")
    model_d, _ = trainer._build_model(SMALL_CFG, device, use_compile=False)
    optimizer_d = trainer.make_optimizer(model_d, SMALL_CFG, device.type)
    logger = trainer.RunLogger(None)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.find_latest_valid_checkpoint(
                corrupt_dir, SMALL_CFG, corpus, model_d, optimizer_d, device,
                code_fp, logger,
            )
        raise AssertionError("all-corrupt checkpoint dir was accepted")
    except trainer.TrainerError:
        pass
    logger.close()


def _tamper_case(tmp, name, root, contract, mutate):
    device = torch.device("cpu")
    corpus = load_corpus(root, SMALL_CFG, contract)
    _, model, optimizer, _ = _make_trained_state(tmp, root, contract, None)
    code_fp = trainer.code_fingerprint()
    payload = trainer.build_checkpoint(
        SMALL_CFG, corpus, model, optimizer, updates_done=1,
        tokens_consumed=SMALL_CFG.GLOBAL_BATCH * SMALL_CFG.T,
        target_tokens=None, code_fp=code_fp,
    )
    mutate(payload)
    case_dir = tmp / name
    case_dir.mkdir(parents=True)
    path = case_dir / "latest.pt"
    trainer.save_checkpoint_atomic(path, payload)
    model_b, _ = trainer._build_model(SMALL_CFG, device, use_compile=False)
    optimizer_b = trainer.make_optimizer(model_b, SMALL_CFG, device.type)
    raised = None
    try:
        trainer.validate_and_load_checkpoint(
            path, SMALL_CFG, corpus, model_b, optimizer_b, device, code_fp
        )
    except Exception as exc:  # noqa: BLE001
        raised = exc
    assert isinstance(raised, trainer.TrainerError), (
        f"{name}: expected TrainerError, got {raised!r}"
    )


def test_checkpoint_failure_modes(tmp: Path, root: Path, contract, meta):
    device = torch.device("cpu")
    corpus = load_corpus(root, SMALL_CFG, contract)
    cases = {
        "bad_format": lambda p: p.update({"format": "other"}),
        "bad_implementation": lambda p: p.update({"implementation": "x"}),
        "bad_code": lambda p: p.update({"code_sha256": "0" * 64}),
        "bad_config": lambda p: p["config"].update({"T": SMALL_CFG.T + 1}),
        "bad_tokens": lambda p: p["progress"].update(
            {"tokens_consumed": p["progress"]["tokens_consumed"] + 1}
        ),
        "bad_cursor": lambda p: p["progress"].update(
            {"next_sequence": p["progress"]["next_sequence"] + 1}
        ),
        "bad_corpus": lambda p: p["corpus"].update(
            {"artifact_hashes_sha256": "0" * 64}
        ),
    }
    for name, mutate in cases.items():
        _tamper_case(tmp, name, root, contract, mutate)

    _, model, optimizer, _ = _make_trained_state(
        tmp, root, contract, None, steps=1
    )
    code_fp = trainer.code_fingerprint()
    payload = trainer.build_checkpoint(
        SMALL_CFG, corpus, model, optimizer, updates_done=1,
        tokens_consumed=SMALL_CFG.GLOBAL_BATCH * SMALL_CFG.T,
        target_tokens=None, code_fp=code_fp,
    )
    payload["code_sha256"] = "0" * 64
    path = tmp / "allow_code" / "latest.pt"
    trainer.save_checkpoint_atomic(path, payload)
    model_b, _ = trainer._build_model(SMALL_CFG, device, use_compile=False)
    optimizer_b = trainer.make_optimizer(model_b, SMALL_CFG, device.type)
    trainer.validate_and_load_checkpoint(
        path, SMALL_CFG, corpus, model_b, optimizer_b, device, code_fp,
        allow_code_change=True,
    )
    assert torch.equal(
        model.embedding.weight, model_b.embedding.weight
    ), "allow_code_change resume produced wrong weights"


def test_smoke_mode(tmp: Path, root: Path, contract, meta):
    device = torch.device("cpu")
    corpus = load_corpus(root, SMALL_CFG, contract)
    run_dir = tmp / "smoke_pass"
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ok = trainer.run_smoke(
            SMALL_CFG, corpus, run_dir, device, use_compile=False,
            check_graph_breaks=True,
        )
    assert ok, out.getvalue()
    assert "SMOKE_PASS=true" in out.getvalue()
    assert "ARM_A_2P5B_TRAINER_READY=true" in out.getvalue()
    assert not (run_dir / "smoke_ckpt").exists()

    run_dir_fail = tmp / "smoke_fail"
    original = trainer.one_full_update
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected update failure")
        return original(*args, **kwargs)

    trainer.one_full_update = flaky
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            ok = trainer.run_smoke(
                SMALL_CFG, corpus, run_dir_fail, device, use_compile=False,
                check_graph_breaks=False,
            )
    finally:
        trainer.one_full_update = original
    assert not ok
    assert "SMOKE_PASS=false" in out.getvalue()
    assert "ARM_A_2P5B_TRAINER_READY=false" in out.getvalue()

    cli_run = tmp / "cli_smoke"
    out = io.StringIO()
    original_contract = trainer.PROD_CONTRACT
    original_cfg = trainer.PROD_CFG
    trainer.PROD_CONTRACT = contract
    trainer.PROD_CFG = SMALL_CFG
    try:
        with contextlib.redirect_stdout(out):
            code = trainer.main([
                "--mode", "smoke",
                "--corpus-root", str(root),
                "--run-dir", str(cli_run),
                "--no-compile", "--no-graph-check",
            ])
    finally:
        trainer.PROD_CONTRACT = original_contract
        trainer.PROD_CFG = original_cfg
    assert code == 0, out.getvalue()
    assert "ARM_A_2P5B_TRAINER_READY=true" in out.getvalue()


def _latest_state(run_dir: Path):
    ckpt = torch.load(
        run_dir / "ckpt" / "latest.pt", map_location="cpu", weights_only=False
    )
    return ckpt["model"], ckpt["progress"]


def test_train_and_resume(tmp: Path, root: Path, contract, meta):
    device = torch.device("cpu")
    corpus = load_corpus(root, SMALL_CFG, contract)
    target = 4 * SMALL_CFG.GLOBAL_BATCH * SMALL_CFG.T

    run_a = tmp / "train_uninterrupted"
    with contextlib.redirect_stdout(io.StringIO()):
        assert trainer.run_train(
            SMALL_CFG, corpus, run_a, device, target_tokens=target,
            save_every=1, archive_every=1, log_every=0,
            use_compile=False, check_graph_breaks=False,
        )
    state_a, progress_a = _latest_state(run_a)
    assert progress_a["updates_done"] == 4, progress_a
    assert progress_a["tokens_consumed"] == target
    assert progress_a["next_sequence"] == 4 * SMALL_CFG.GLOBAL_BATCH

    run_b = tmp / "train_resumed"
    with contextlib.redirect_stdout(io.StringIO()):
        assert trainer.run_train(
            SMALL_CFG, corpus, run_b, device, target_tokens=target // 2,
            save_every=1, archive_every=1, log_every=0,
            use_compile=False, check_graph_breaks=False,
        )
    with contextlib.redirect_stdout(io.StringIO()):
        assert trainer.run_train(
            SMALL_CFG, corpus, run_b, device, target_tokens=target,
            save_every=1, archive_every=1, log_every=0,
            use_compile=False, check_graph_breaks=False,
        )
    state_b, progress_b = _latest_state(run_b)
    assert progress_b["updates_done"] == 4, progress_b
    max_diff = 0.0
    for name in state_a:
        diff = float((state_a[name] - state_b[name]).abs().max())
        max_diff = max(max_diff, diff)
    assert max_diff == 0.0, f"resume changed the weights: {max_diff:.3e}"

    with contextlib.redirect_stdout(io.StringIO()):
        assert trainer.run_train(
            SMALL_CFG, corpus, run_b, device, target_tokens=target,
            save_every=1, archive_every=1, log_every=0,
            use_compile=False, check_graph_breaks=False,
        )
    state_c, progress_c = _latest_state(run_b)
    assert progress_c["updates_done"] == 4
    for name in state_b:
        assert torch.equal(state_b[name], state_c[name])

    events = [
        json.loads(line)["event"]
        for line in (run_b / "logs" / "train.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert "resumed" in events, events
    assert "already_complete" in events, events
    assert events.count("session_end") >= 2


def test_lr_schedule_invariance(tmp: Path, root: Path, contract, meta):
    cfg = SMALL_CFG
    per_update = cfg.GLOBAL_BATCH * cfg.T
    peak_update = int(np.ceil(cfg.WARMUP_TOKENS / per_update))
    assert trainer.lr_for_update(1, cfg) < trainer.lr_for_update(peak_update, cfg)
    assert trainer.lr_for_update(peak_update, cfg) == cfg.PEAK_LR
    assert trainer.lr_for_update(peak_update + 1000, cfg) == cfg.PEAK_LR
    assert trainer.parse_target("full") is None
    assert trainer.parse_target("2500000000") == 2_500_000_000
    assert trainer.frozen_config_dict(cfg)["flags"] == trainer.FROZEN_FLAGS


def main():
    results = []
    with tempfile.TemporaryDirectory(prefix="arm_a_tests_") as td:
        base = Path(td)
        root = base / "fixture"
        contract, meta = build_fixture(root, SMALL_CFG)
        tests = [
            ("fixture_semantics", test_fixture_semantics),
            ("contract_failclosed", test_contract_failclosed),
            ("batch_streaming", test_batch_streaming),
            ("cross_shard_lookahead", test_cross_shard_lookahead),
            ("checkpoint_roundtrip", test_checkpoint_roundtrip),
            ("checkpoint_failure_modes", test_checkpoint_failure_modes),
            ("smoke_mode", test_smoke_mode),
            ("train_and_resume", test_train_and_resume),
            ("lr_schedule_invariance", test_lr_schedule_invariance),
        ]
        for name, fn in tests:
            case_dir = base / f"case_{name}"
            case_dir.mkdir()
            try:
                fn(case_dir, root, contract, meta)
                results.append((name, True, ""))
                print(f"PASS {name}", flush=True)
            except Exception as exc:  # noqa: BLE001
                results.append((name, False, f"{type(exc).__name__}: {exc}"))
                print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()

    failed = [name for name, ok, _ in results if not ok]
    print("\nTEST_SUMMARY " + json.dumps(
        {"total": len(results), "failed": failed}, sort_keys=True
    ))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
