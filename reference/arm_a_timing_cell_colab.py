# Complete replacement cell; preparation was checked statically without running the benchmark.
# No dependency installation or downloads. Drive access is read-only.

import contextlib
import csv
import gc
import hashlib
import io
import json
import math
import subprocess
import time
from pathlib import Path, PurePosixPath

RUNTIME_PROJECTION_SCOPE = "PURE_CAUSAL_FULL_SAME_DOCUMENT_T2048_WINDOWS_ONLY"
RUNTIME_PROJECTION_NOTE = (
    "Throughput and extrapolation cover only retained full same-document T=2048 windows. "
    "Final packed-corpus training runtime is not certified until the retained data quantity is accepted."
)

result = {
    "status": "STARTING",
    "runtime_projection_scope": RUNTIME_PROJECTION_SCOPE,
    "final_training_runtime_certified": False,
    "runtime_projection_note": RUNTIME_PROJECTION_NOTE,
    "drive_modified": False,
    "corpus_root": "/content/drive/Shareddrives/ICLR PHASE BDH/phase_bdh/corpus/stage2/frozen_5b_v1",
    "validated_artifacts": [],
    "corpus_census": None,
    "model": None,
    "runtime": {"status": "NOT_RUN"},
    "A_TIMING_READY": False,
    "ASSUMPTION_CHECKS": {},
}
phase = "contract"

ARCHITECTURE_DELTA_REQUIREMENTS = {
    "MoE": {
        "adapter": "apply_moe(base_arm_a: nn.Module) -> nn.Module",
        "forward_contract": "preserve forward(idx, pos, segpos, full_mask) -> logits, or provide an explicit trainer adapter",
        "required_spec": [
            "canonical insertion/replacement point",
            "expert count, routing top-k, capacity/overflow and dispatch/combine semantics",
            "router auxiliary-loss definition, reduction, coefficient, and whether nonzero",
            "trainable parameter census and routing metrics",
        ],
    },
    "Phase/Oscillation": {
        "adapter": "apply_phase(base_arm_a: nn.Module) -> nn.Module",
        "forward_contract": "preserve forward(idx, pos, segpos, full_mask) -> logits, or provide an explicit trainer adapter",
        "required_spec": [
            "canonical insertion point and exact phase/oscillation equations",
            "trainable parameters and configuration",
            "any added state shape, update/reset semantics, and gradient handling",
            "any auxiliary loss or required runtime metrics",
        ],
    },
    "combined_arm_D": "apply_phase(apply_moe(base_arm_a)); both adapters must preserve the trainer contract or declare the same explicit adapter.",
}


try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from functools import partial
    from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is unavailable in this Colab runtime") from exc

    from google.colab import drive

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        drive.mount("/content/drive", force_remount=False)

    CORPUS = Path(result["corpus_root"])
    T, VOCAB = 2048, 8192
    D = 256
    N = 16_384
    H = 4
    K = N // H
    L = 8
    HIDDEN = 1040
    SEED = 1337
    INIT_STD = 0.02
    THETA = 2**16
    READ_BLOCK = 256
    PEAK_LR = 1e-3
    WARMUP_TOKENS = 10_000_000
    BETAS = (0.9, 0.95)
    EPS = 1e-8
    WEIGHT_DECAY = 0.1
    CLIP_NORM = 1.0
    TARGET_USABLE_TOKENS = 2_500_000_000
    TARGET_REAL_INPUT_TOKENS = (2_500_000_000, 5_000_000_000)
    GLOBAL_BATCH = 64
    WARMUPS, SAMPLES = 3, 10
    COLAB_CU_PER_GPU_HOUR = 8.9
    REMAINING_CU = 385.13
    GLOBAL_TOKENS = GLOBAL_BATCH * T
    REQUIRED_SAMPLE_WINDOWS = GLOBAL_BATCH * (WARMUPS + SAMPLES)

    EXPECTED_FROZEN = {
        "status": "FROZEN",
        "corpus_id": "phase_bdh_stage2_5b_v1",
        "context_length": T,
        "tokenizer_sha256": "9a05bca4c065d9952a01ee1e3f4b6e23e39200e1f33da65e5dad1819419f71c3",
        "logical_replay_sha256": "c60bd7df4a8ef7329d20ec5b0f022f4579fca20c95657a90f79a2e9e67edb043",
        "artifact_hashes_sha256": "2fe45d14ff8ae36ded7f1b57af9e7e200233fcd3726354e1a6186af7ee0ea6a3",
        "real_training_tokens": 5_000_000_000,
        "sequences": 2_441_407,
        "padding_tokens_masked": 1_536,
    }
    EXPECTED_MANIFEST = {
        "status": "FROZEN",
        "real_tokens": 5_000_000_000,
        "physical_tokens": 5_000_001_536,
        "padding_tokens": 1_536,
        "selected_documents": 3_498_441,
        "sequences": 2_441_407,
    }

    def require(condition, message):
        if not condition:
            raise RuntimeError("CONTRACT_MISMATCH: " + message)

    def sha256_bytes(data):
        return hashlib.sha256(data).hexdigest()

    def sha256_file(path):
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            while True:
                block = f.read(16 * 1024 * 1024)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()

    def safe_corpus_path(relative_path):
        rel = PurePosixPath(str(relative_path))
        require(not rel.is_absolute() and ".." not in rel.parts, f"Unsafe artifact path: {relative_path}")
        require("\\" not in str(relative_path), f"Non-canonical artifact path: {relative_path}")
        return CORPUS.joinpath(*rel.parts)

    frozen_path = CORPUS / "FROZEN.json"
    artifact_path = CORPUS / "artifact_hashes.json"
    manifest_path = CORPUS / "corpus_manifest.json"

    frozen_bytes = frozen_path.read_bytes()
    artifact_bytes = artifact_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    artifact_index = json.loads(artifact_bytes)
    manifest = json.loads(manifest_bytes)

    for key, expected in EXPECTED_FROZEN.items():
        require(frozen.get(key) == expected, f"FROZEN.json field {key!r} differs from the pinned contract.")
    for key, expected in EXPECTED_MANIFEST.items():
        require(manifest.get(key) == expected, f"corpus_manifest.json field {key!r} differs from the pinned contract.")
    require(manifest.get("logical_replay_sha256") == EXPECTED_FROZEN["logical_replay_sha256"],
            "Manifest logical replay SHA differs from FROZEN.json.")
    require(artifact_index.get("algorithm") == "sha256", "artifact_hashes.json does not declare sha256.")
    require(sha256_bytes(artifact_bytes) == EXPECTED_FROZEN["artifact_hashes_sha256"],
            "artifact_hashes.json SHA-256 differs from FROZEN.json.")

    records = artifact_index.get("files", [])
    by_path = {}
    for record in records:
        rel = record.get("path")
        require(isinstance(rel, str) and rel not in by_path, f"Missing or duplicate artifact-index path: {rel!r}")
        by_path[rel] = record

    manifest_record = by_path.get("corpus_manifest.json")
    require(manifest_record is not None, "corpus_manifest.json is absent from artifact_hashes.json.")
    require(len(manifest_bytes) == int(manifest_record["bytes"]), "corpus_manifest.json byte count mismatch.")
    require(sha256_bytes(manifest_bytes) == manifest_record["sha256"], "corpus_manifest.json SHA-256 mismatch.")

    shards = manifest.get("shards", [])
    require(len(shards) == 25, f"Expected 25 declared shards; found {len(shards)}.")
    require(sum(int(s["sequences"]) for s in shards) == EXPECTED_MANIFEST["sequences"],
            "Shard sequence counts do not sum to the frozen total.")
    require(sum(int(v) for v in manifest.get("stream_tokens", {}).values()) == EXPECTED_MANIFEST["real_tokens"],
            "Manifest stream token counts do not sum to the frozen real-token total.")

    declared_token_paths, declared_length_paths, declared_provenance_paths = set(), set(), set()
    for shard_number, shard in enumerate(shards):
        stem = f"shard_{shard_number:06d}"
        require(shard.get("stem") == stem, f"Unexpected shard order or stem at index {shard_number}.")
        require(shard.get("tokens_file") == f"train/{stem}.tokens.bin", f"Unexpected token path for {stem}.")
        require(shard.get("valid_lengths_file") == f"train/{stem}.valid_lengths.bin",
                f"Unexpected valid-length path for {stem}.")
        require(shard.get("provenance_file") == f"train/{stem}.provenance.parquet",
                f"Unexpected provenance path for {stem}.")
        require(int(shard["sequences"]) > 0 and int(shard["provenance_rows"]) > 0,
                f"Empty shard declaration for {stem}.")
        declared_token_paths.add(shard["tokens_file"])
        declared_length_paths.add(shard["valid_lengths_file"])
        declared_provenance_paths.add(shard["provenance_file"])

    require({p for p in by_path if p.endswith(".tokens.bin")} == declared_token_paths,
            "Artifact index token shards differ from the corpus manifest.")
    require({p for p in by_path if p.endswith(".valid_lengths.bin")} == declared_length_paths,
            "Artifact index valid-length shards differ from the corpus manifest.")
    require({p for p in by_path if p.endswith(".provenance.parquet")} == declared_provenance_paths,
            "Artifact index provenance shards differ from the corpus manifest.")

    result["validated_artifacts"].append({
        "path": str(frozen_path),
        "bytes": len(frozen_bytes),
        "sha256": sha256_bytes(frozen_bytes),
        "verified_by": "pinned FROZEN.json fields and matching artifact-index digest",
    })
    result["validated_artifacts"].append({
        "path": str(artifact_path),
        "bytes": len(artifact_bytes),
        "expected_sha256": EXPECTED_FROZEN["artifact_hashes_sha256"],
        "sha256": sha256_bytes(artifact_bytes),
    })
    result["validated_artifacts"].append({
        "path": str(manifest_path),
        "bytes": len(manifest_bytes),
        "expected_sha256": manifest_record["sha256"],
        "sha256": sha256_bytes(manifest_bytes),
    })

    # Verify every declared shard file before reading corpus contents or benchmarking.
    for rel in sorted(declared_token_paths | declared_length_paths | declared_provenance_paths):
        record = by_path[rel]
        path = safe_corpus_path(rel)
        require(path.is_file(), f"Missing declared artifact: {path}")
        actual_bytes = path.stat().st_size
        require(actual_bytes == int(record["bytes"]), f"Byte count mismatch: {path}")
        actual_sha = sha256_file(path)
        require(actual_sha == record["sha256"], f"SHA-256 mismatch: {path}")
        result["validated_artifacts"].append({
            "path": str(path),
            "bytes": actual_bytes,
            "expected_sha256": record["sha256"],
            "sha256": actual_sha,
        })

    result["contract"] = {
        "status": "PASS",
        "corpus_id": frozen["corpus_id"],
        "context_length": int(frozen["context_length"]),
        "real_training_tokens": int(frozen["real_training_tokens"]),
        "sequences": int(frozen["sequences"]),
        "selected_documents": int(manifest["selected_documents"]),
        "shard_count": len(shards),
        "validated_shard_file_count": len(declared_token_paths | declared_length_paths | declared_provenance_paths),
        "validated_artifact_file_count": len(result["validated_artifacts"]),
    }

    phase = "census"
    num_docs = int(manifest["selected_documents"])
    doc_lengths = np.zeros(num_docs, dtype=np.int64)
    seen_docs = np.zeros(num_docs, dtype=np.bool_)
    sequence_total = 0
    sequence_base = 0
    physical_total = 0
    real_total = 0
    padding_total = 0
    max_token_id = -1
    min_token_id = VOCAB
    full_windows = 0
    tail_tokens = 0
    exact_pairs = 0
    pairs_in_full_windows = 0
    window_x, window_y, window_pos, window_valid = [], [], [], []
    current_doc = {"id": None, "end": 0, "sample_chunks": []}

    def finish_document():
        doc_id = current_doc["id"]
        if doc_id is None:
            return
        length = int(current_doc["end"])
        require(length > 0, f"Empty document provenance for selected document {doc_id}.")
        doc_lengths[doc_id] = length
        seen_docs[doc_id] = True

        q, rem = divmod(length, T)
        full_windows_count = q
        full_pairs = q * T - (1 if q > 0 and rem == 0 else 0)
        nonlocal_counters["full_windows"] += full_windows_count
        nonlocal_counters["tail_tokens"] += rem
        nonlocal_counters["exact_pairs"] += length - 1
        nonlocal_counters["pairs_in_full_windows"] += full_pairs

        remaining = REQUIRED_SAMPLE_WINDOWS - len(window_x)
        if remaining > 0 and q > 0:
            chunks = current_doc["sample_chunks"]
            prefix = (
                np.concatenate(chunks)
                if len(chunks) > 1
                else (chunks[0] if chunks else np.empty(0, dtype=np.uint16))
            )
            take = min(q, remaining)
            for wi in range(take):
                start = wi * T
                xw = np.asarray(prefix[start:start + T], dtype=np.int64)
                require(xw.size == T, f"Could not materialize full T={T} window for document {doc_id}.")
                yw = np.zeros(T, dtype=np.int64)
                yw[:-1] = np.asarray(prefix[start + 1:start + T], dtype=np.int64)
                has_final_target = start + T < length
                validw = np.ones(T, dtype=np.bool_)
                if has_final_target:
                    require(start + T < prefix.size, f"Missing lookahead token for document {doc_id}.")
                    yw[-1] = int(prefix[start + T])
                else:
                    validw[-1] = False
                window_x.append(xw)
                window_y.append(yw)
                window_pos.append(np.arange(start, start + T, dtype=np.int64))
                window_valid.append(validw)

        current_doc["id"] = None
        current_doc["end"] = 0
        current_doc["sample_chunks"] = []

    nonlocal_counters = {
        "full_windows": 0,
        "tail_tokens": 0,
        "exact_pairs": 0,
        "pairs_in_full_windows": 0,
    }
    provenance_columns = [
        "sequence_index",
        "sequence_token_start",
        "sequence_token_end",
        "selected_document_index",
        "document_token_start",
        "document_token_end",
    ]

    for shard in shards:
        nseq = int(shard["sequences"])
        token_path = safe_corpus_path(shard["tokens_file"])
        length_path = safe_corpus_path(shard["valid_lengths_file"])
        provenance_path = safe_corpus_path(shard["provenance_file"])

        require(token_path.stat().st_size == nseq * T * 2, f"Unexpected token-shard size: {token_path}")
        require(length_path.stat().st_size == nseq * 2, f"Unexpected valid-length-shard size: {length_path}")

        lengths = np.memmap(length_path, mode="r", dtype=np.uint16, shape=(nseq,))
        require(bool(np.all((lengths > 0) & (lengths <= T))), f"Invalid valid_lengths values in {length_path}.")
        shard_real = int(lengths.sum(dtype=np.uint64))
        sequence_total += nseq
        physical_total += nseq * T
        real_total += shard_real
        padding_total += nseq * T - shard_real

        token_map = np.memmap(token_path, mode="r", dtype=np.uint16, shape=(nseq, T))
        table = pq.read_table(provenance_path, columns=provenance_columns)
        require(table.num_rows == int(shard["provenance_rows"]),
                f"Provenance row count mismatch: {provenance_path}")

        columns = {}
        for name in provenance_columns:
            arrow_column = table.column(name).combine_chunks()
            require(arrow_column.null_count == 0, f"Null provenance values in {provenance_path}:{name}")
            columns[name] = arrow_column.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)

        order = np.lexsort((columns["sequence_token_start"], columns["sequence_index"]))
        cursor = np.zeros(nseq, dtype=np.int32)

        for row_index in order:
            seq_global = int(columns["sequence_index"][row_index])
            require(
                sequence_base <= seq_global < sequence_base + nseq,
                f"Out-of-range corpus-global sequence_index in {provenance_path}.",
            )
            seq = seq_global - sequence_base
            a = int(columns["sequence_token_start"][row_index])
            e = int(columns["sequence_token_end"][row_index])
            doc_id = int(columns["selected_document_index"][row_index])
            doc_start = int(columns["document_token_start"][row_index])
            doc_end = int(columns["document_token_end"][row_index])

            require(0 <= seq < nseq, f"Out-of-range sequence_index in {provenance_path}.")
            require(0 <= a < e <= T and e <= int(lengths[seq]),
                    f"Invalid sequence span in {provenance_path}, sequence {seq}.")
            require(0 <= doc_id < num_docs, f"Out-of-range selected_document_index in {provenance_path}.")
            require(doc_start >= 0 and doc_end > doc_start and e - a == doc_end - doc_start,
                    f"Document/sequence span mismatch in {provenance_path}.")
            require(a == int(cursor[seq]), f"Gap or overlap in provenance coverage: {provenance_path}, sequence {seq}.")
            cursor[seq] = e

            if current_doc["id"] is None or doc_id != current_doc["id"]:
                if current_doc["id"] is not None:
                    finish_document()
                require(not bool(seen_docs[doc_id]), f"Document {doc_id} reappears after its stream run closed.")
                require(doc_start == 0, f"Document {doc_id} does not start at token offset 0.")
                current_doc["id"] = doc_id
                current_doc["end"] = 0
                current_doc["sample_chunks"] = []
            else:
                require(doc_start == int(current_doc["end"]),
                        f"Non-contiguous document offsets for document {doc_id}.")

            require(doc_start == int(current_doc["end"]),
                    f"Document offset gap/overlap for document {doc_id}.")
            current_doc["end"] = doc_end

            segment = token_map[seq, a:e]
            seg_min, seg_max = int(segment.min()), int(segment.max())
            min_token_id = min(min_token_id, seg_min)
            max_token_id = max(max_token_id, seg_max)
            require(seg_max < VOCAB, f"Token id {seg_max} exceeds vocab={VOCAB} in {token_path}.")

            remaining = REQUIRED_SAMPLE_WINDOWS - len(window_x)
            if remaining > 0:
                sample_prefix_cap = remaining * T + 1
                take_tokens = max(0, min(doc_end, sample_prefix_cap) - doc_start)
                if take_tokens:
                    current_doc["sample_chunks"].append(
                        np.asarray(token_map[seq, a:a + take_tokens], dtype=np.uint16).copy()
                    )

        require(bool(np.array_equal(cursor.astype(np.int64), lengths.astype(np.int64))),
                f"Provenance does not cover every valid token in {provenance_path}.")
        del table, columns, order, cursor, token_map, lengths
        gc.collect()
        sequence_base += nseq

    finish_document()
    require(sequence_base == 2_441_407, "Cumulative sequence base differs from the frozen sequence total.")

    require(sequence_total == EXPECTED_MANIFEST["sequences"], "Scanned sequence total differs from contract.")
    require(physical_total == EXPECTED_MANIFEST["physical_tokens"], "Scanned physical-token total differs from contract.")
    require(real_total == EXPECTED_MANIFEST["real_tokens"], "Scanned real-token total differs from contract.")
    require(padding_total == EXPECTED_MANIFEST["padding_tokens"], "Scanned padding-token total differs from contract.")
    require(int(seen_docs.sum()) == num_docs, "Scanned selected-document count differs from manifest.")
    require(bool(np.all(doc_lengths > 0)), "One or more selected documents have no token spans.")
    require(int(doc_lengths.sum(dtype=np.int64)) == real_total, "Document lengths do not sum to real-token total.")
    require(min_token_id >= 0 and max_token_id < VOCAB, "Token-id range validation failed.")
    require(len(window_x) >= REQUIRED_SAMPLE_WINDOWS,
            f"Only {len(window_x)} full same-document windows found; need {REQUIRED_SAMPLE_WINDOWS} for benchmark.")

    percentile_values = np.percentile(doc_lengths, [1, 10, 50, 90, 99]).tolist()
    full_window_input_tokens = int(nonlocal_counters["full_windows"] * T)
    corpus_pair_support = bool(nonlocal_counters["exact_pairs"] >= TARGET_USABLE_TOKENS)
    full_window_support = bool(full_window_input_tokens >= TARGET_USABLE_TOKENS)

    result["corpus_census"] = {
        "context_length": T,
        "vocab_size": VOCAB,
        "token_dtype": "uint16",
        "real_tokens": int(real_total),
        "physical_tokens": int(physical_total),
        "padding_tokens_masked": int(padding_total),
        "max_real_token_id": int(max_token_id),
        "min_real_token_id": int(min_token_id),
        "sequence_count": int(sequence_total),
        "selected_document_count": int(seen_docs.sum()),
        "document_length_tokens": {
            "p1": float(percentile_values[0]),
            "p10": float(percentile_values[1]),
            "p50": float(percentile_values[2]),
            "p90": float(percentile_values[3]),
            "p99": float(percentile_values[4]),
            "max": int(doc_lengths.max()),
        },
        "exact_usable_next_token_pairs": int(nonlocal_counters["exact_pairs"]),
        "usable_next_token_pair_fraction_of_real_tokens": float(nonlocal_counters["exact_pairs"] / real_total),
        "full_2048_token_same_document_windows": int(nonlocal_counters["full_windows"]),
        "real_input_tokens_in_nonoverlapping_full_windows": full_window_input_tokens,
        "full_window_retention_fraction": float(full_window_input_tokens / real_total),
        "usable_pairs_covered_by_full_windows": int(nonlocal_counters["pairs_in_full_windows"]),
        "tokens_discarded_as_per_document_window_tails": int(nonlocal_counters["tail_tokens"]),
        "usable_pairs_not_covered_by_full_windows": int(
            nonlocal_counters["exact_pairs"] - nonlocal_counters["pairs_in_full_windows"]
        ),
        "cross_document_stream_transitions_masked": int(max(0, num_docs - 1)),
        "training_batch_source": "non-overlapping, full T windows reconstructed from the provenance-ordered token shards",
        "sampled_windows_for_warmup_and_measurement": int(REQUIRED_SAMPLE_WINDOWS),
    }

    # Canonical Arm-A census: shared paper-core E/Dx/Dy plus the frozen Stage-1 extension.
    component_parameters = {
        "embedding": VOCAB * D,
        "encoder_E": N * D,
        "decoder_Dx": N * D,
        "decoder_Dy": N * D,
        "readout": D * VOCAB,
        "coordinator_Wc": D * D,
        "coordinator_bc": D,
        "coordinator_alpha": 1,
        "writer_W1": D * HIDDEN,
        "writer_W2": HIDDEN * D,
    }
    analytic_core = component_parameters["encoder_E"] + component_parameters["decoder_Dx"] + component_parameters["decoder_Dy"]
    analytic_total = sum(component_parameters.values())
    analytic_nonembedding = analytic_total - component_parameters["embedding"]
    require(analytic_core == 3 * N * D, "BDH shared E/Dx/Dy core parameter census is inconsistent.")
    result["model"] = {
        "architecture": {
            "name": "canonical NativeReadStage1ArmA",
            "N_neurons": N,
            "d_model": D,
            "heads": H,
            "neurons_per_head": K,
            "layers": L,
            "weights_shared_across_layers": True,
            "tied_input_output_embedding": False,
            "context_length": T,
            "vocabulary": VOCAB,
            "attention": "Q=K, strict-past same-document blocked causal prefix read; no softmax/scale",
            "activation": "positive ReLU neuron activations",
            "position_encoding": "canonical document-local pairwise RoPE, theta=2**16",
            "coordinator": "canonical scalar sigmoid(alpha) gate",
            "writer": "canonical dense two-matrix writer, hidden=1040",
            "persistent_inter_chunk_state": False,
            "depth_parameter_sharing": "single E/Dx/Dy/coordinator/writer set reused for each of L layers",
        },
        "parameter_counts": {
            "components": {k: int(v) for k, v in component_parameters.items()},
            "paper_BDH_core_E_Dx_Dy": int(analytic_core),
            "total_stored_trainable": int(analytic_total),
            "active_per_token_dense": int(analytic_total),
            "nonembedding": int(analytic_nonembedding),
        },
        "canonical_source": {
            "path": "vendor/canonical/icrl_reintegrated_bdh_baseline_v1.py",
            "sha256": "947f8b33e740adede3e13382f8cb9e8e374d845de75e282bda5feb05c582f624",
            "scale_override": {"N": N, "D": D, "H": H, "K": K, "L": L, "T": T, "V": VOCAB},
        },
    }

    result["ASSUMPTION_CHECKS"] = {
        "corpus_sufficiency_without_repetition": {
            "target_usable_next_token_pairs": TARGET_USABLE_TOKENS,
            "target_real_input_tokens": list(TARGET_REAL_INPUT_TOKENS),
            "corpus_real_tokens": int(real_total),
            "exact_usable_next_token_pairs": int(nonlocal_counters["exact_pairs"]),
            "full_T_same_document_window_input_tokens": int(full_window_input_tokens),
            "pass_2p5B_by_full_windows": bool(full_window_input_tokens >= TARGET_REAL_INPUT_TOKENS[0]),
            "pass_5B_by_full_windows": bool(full_window_input_tokens >= TARGET_REAL_INPUT_TOKENS[1]),
        },
        "full_step_runtime_projections": {"status": "NOT_MEASURED"},
    }

    phase = "runtime"
    result["runtime"] = {"status": "NOT_RUN"}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable; run this cell on a CUDA Colab runtime.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The detected CUDA GPU does not support BF16, required for this benchmark.")

    device = torch.device("cuda")
    DEVICE = device
    props = torch.cuda.get_device_properties(device)
    smi_response = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    smi_rows = [
        [field.strip() for field in row]
        for row in csv.reader(io.StringIO(smi_response.stdout))
        if row
    ]
    if not smi_rows or any(len(row) != 3 or not row[0] for row in smi_rows):
        raise RuntimeError("nvidia-smi did not return valid GPU UUID/name/VRAM data.")
    torch_uuid = str(getattr(props, "uuid", "")).strip().lower().removeprefix("gpu-")
    matching_rows = [
        row for row in smi_rows
        if row[0].lower().removeprefix("gpu-") == torch_uuid
    ]
    if len(matching_rows) == 1:
        smi_row = matching_rows[0]
    elif len(smi_rows) == 1:
        smi_row = smi_rows[0]
    else:
        raise RuntimeError("Cannot unambiguously match the CUDA device to an nvidia-smi GPU UUID.")
    gpu_facts = {
        "gpu": torch.cuda.get_device_name(device),
        "gpu_uuid": smi_row[0],
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_identity": f"sm_{props.major}{props.minor}",
        "gpu_total_memory_bytes": int(props.total_memory),
        "nvidia_smi_gpu_name": smi_row[1],
        "nvidia_smi_vram_mib": int(smi_row[2]),
    }
    result["runtime"].update(gpu_facts)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)

    # The following model, initialization, RoPE, and selective-checkpoint definitions are
    # verbatim from the pinned canonical source; only the visible scale constants above differ.
    CANONICAL_SOURCE_SHA256 = "947f8b33e740adede3e13382f8cb9e8e374d845de75e282bda5feb05c582f624"

    def canonical_init():
        g = torch.Generator(device='cpu')
        g.manual_seed(SEED)
        def rnd(shape):
            return torch.randn(shape, generator=g, dtype=torch.float32) * INIT_STD
        return {
            'embedding': rnd((VOCAB, D)),
            'encoder': rnd((N, D)),
            'decoder_x': rnd((H, D, K)),
            'decoder_y': rnd((H, D, K)),
            'readout': rnd((D, VOCAB)),
            'coord_Wc': rnd((D, D)),
            'coord_bc': torch.zeros(D, dtype=torch.float32),
            'coord_alpha': torch.zeros((), dtype=torch.float32),
            'writer_W1': rnd((D, HIDDEN)),
            'writer_W2': rnd((HIDDEN, D)),
        }

    CAUSAL = torch.ones((T, T), device=DEVICE, dtype=torch.bool).tril(diagonal=-1)
    ROPE_PAIR_FREQ = (
        1.0 / (THETA ** ((2.0 * torch.arange(K // 2, dtype=torch.float32, device=DEVICE)) / K))
        / (2.0 * math.pi)
    )

    def rope_bthk(q, pos, freq):
        # q [B,T,H,K] — same pairwise rotation, different physical layout.
        b, t, h, k = q.shape
        qp = q.reshape(b, t, h, k // 2, 2)
        phase = pos.float().unsqueeze(-1).unsqueeze(-1) * freq.view(1, 1, 1, -1)
        phase = torch.remainder(phase, 1.0) * (2.0 * math.pi)
        cs = torch.cos(phase).to(q.dtype)
        sn = torch.sin(phase).to(q.dtype)
        qe, qo = qp[..., 0], qp[..., 1]
        return torch.stack((qe * cs - qo * sn, qo * cs + qe * sn), dim=-1).reshape_as(q)

    class DenseWriter(nn.Module):
        def __init__(self):
            super().__init__()
            self.W1 = nn.Parameter(torch.empty(D, HIDDEN))
            self.W2 = nn.Parameter(torch.empty(HIDDEN, D))
        def forward(self, x):
            return F.relu(x @ self.W1) @ self.W2


    class Coordinator(nn.Module):
        def __init__(self):
            super().__init__()
            self.Wc = nn.Parameter(torch.empty(D, D))
            self.bc = nn.Parameter(torch.zeros(D))
            self.alpha = nn.Parameter(torch.zeros(()))
        def forward(self, v, segpos, full_mask):
            z = v @ self.Wc + self.bc
            prev_sum = torch.bmm(full_mask.to(dtype=z.dtype), z)
            den = segpos.clamp_min(1).to(z.dtype).unsqueeze(-1)
            c = prev_sum / den - z
            rho = torch.sigmoid(self.alpha)
            return 1.0 + rho.to(c.dtype) * torch.tanh(c)


    _BMM = torch.ops.aten.bmm.default
    _MATMUL = torch.ops.aten.matmul.default


    def native_read_sac_policy(ctx, op, *args, **kwargs):
        # Save ONLY QK score blocks, exactly analogous to the accepted full-Gram SAC policy.
        if op in (_BMM, _MATMUL) and len(args) >= 2:
            a, b = args[0], args[1]
            if hasattr(a, 'shape') and hasattr(b, 'shape') and len(a.shape) >= 3 and len(b.shape) >= 3:
                try:
                    if (
                        int(a.shape[-1]) == K
                        and int(b.shape[-2]) == K
                        and int(a.shape[-2]) <= READ_BLOCK
                        and int(b.shape[-1]) <= T
                    ):
                        return CheckpointPolicy.MUST_SAVE
                except Exception:
                    pass
        return CheckpointPolicy.PREFER_RECOMPUTE


    NATIVE_SAC_CONTEXT_FN = partial(create_selective_checkpoint_contexts, native_read_sac_policy)


    class NativeReadStage1ArmA(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(VOCAB, D)
            self.encoder = nn.Parameter(torch.empty(N, D))
            self.decoder_x = nn.Parameter(torch.empty(H, D, K))
            self.decoder_y = nn.Parameter(torch.empty(H, D, K))
            self.readout = nn.Parameter(torch.empty(D, VOCAB))
            self.coordinator = Coordinator()
            self.writer = DenseWriter()
            self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)

        def project_x_native(self, v):
            w_wide = self.decoder_x.permute(1, 0, 2).reshape(D, N)
            return F.relu((v.reshape(v.shape[0] * T, D) @ w_wide).reshape(v.shape[0], T, H, K))

        def attention_native(self, x_bt, v, pos, full_mask):
            q_bt = rope_bthk(x_bt, pos, ROPE_PAIR_FREQ)
            qh = q_bt.permute(0, 2, 1, 3)
            vh = v.unsqueeze(1).expand(-1, H, -1, -1).contiguous()
            outs = []
            for s0 in range(0, T, READ_BLOCK):
                s1 = s0 + READ_BLOCK
                qb = qh[:, :, s0:s1, :]
                kp = qh[:, :, :s1, :]
                scores = qb @ kp.transpose(-1, -2)
                scores = scores.masked_fill(~full_mask[:, None, s0:s1, :s1], 0.0)
                outs.append(scores @ vh[:, :, :s1, :])
            return torch.cat(outs, dim=2)

        def level(self, v, pos, segpos, full_mask):
            # ---------------- EXACT BDH PAPER CORE ----------------
            x_bt = self.project_x_native(v)                       # physical [B,T,H,K]
            a = self.ln(self.attention_native(x_bt, v, pos, full_mask))
            ypre = F.relu(a @ self.decoder_y)                     # unchanged [B,H,T,K]

            # Only a view back to the baseline logical orientation; write path remains unchanged.
            paper_y = x_bt.permute(0, 2, 1, 3) * ypre
            paper_y_flat = paper_y.transpose(1, 2).reshape(v.shape[0], T, N)
            base = self.ln(paper_y_flat @ self.encoder)

            # ---------------- FROZEN STAGE-1 EXTENSION ------------
            g = self.coordinator(v, segpos, full_mask)
            delta = self.writer(g * base)
            return self.ln(v + delta)

        def forward(self, idx, pos, segpos, full_mask):
            v = self.ln(self.embedding(idx))
            for _ in range(L):
                def level_fn(vv):
                    return self.level(vv, pos, segpos, full_mask)
                v = checkpoint(
                    level_fn, v,
                    use_reentrant=False,
                    preserve_rng_state=False,
                    context_fn=NATIVE_SAC_CONTEXT_FN,
                )
            return v @ self.readout

        @torch.no_grad()
        def forward_eval(self, idx, pos, segpos, full_mask):
            v = self.ln(self.embedding(idx))
            for _ in range(L):
                v = self.level(v, pos, segpos, full_mask)
            return v @ self.readout

    def load_init(model, init):
        with torch.no_grad():
            model.embedding.weight.copy_(init['embedding'].to(DEVICE))
            model.encoder.copy_(init['encoder'].to(DEVICE))
            model.decoder_x.copy_(init['decoder_x'].to(DEVICE))
            model.decoder_y.copy_(init['decoder_y'].to(DEVICE))
            model.readout.copy_(init['readout'].to(DEVICE))
            model.coordinator.Wc.copy_(init['coord_Wc'].to(DEVICE))
            model.coordinator.bc.copy_(init['coord_bc'].to(DEVICE))
            model.coordinator.alpha.copy_(init['coord_alpha'].to(DEVICE))
            model.writer.W1.copy_(init['writer_W1'].to(DEVICE))
            model.writer.W2.copy_(init['writer_W2'].to(DEVICE))

    def ce_sum(logits, target, valid):
        per = F.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1), reduction='none')
        return per[valid.reshape(-1)].sum(dtype=torch.float32)

    # Provenance supplies full non-overlapping windows wholly within one selected document.
    # Each row is therefore one independent same-document T=2048 training example.
    sample_x = np.stack(window_x[:REQUIRED_SAMPLE_WINDOWS]).astype(np.uint16, copy=False)
    sample_y = np.stack(window_y[:REQUIRED_SAMPLE_WINDOWS]).astype(np.uint16, copy=False)
    sample_pos = np.stack(window_pos[:REQUIRED_SAMPLE_WINDOWS]).astype(np.int64, copy=False)
    sample_valid = np.stack(window_valid[:REQUIRED_SAMPLE_WINDOWS]).astype(np.bool_, copy=False)
    sample_rows = int(sample_x.shape[0])
    require(sample_rows == REQUIRED_SAMPLE_WINDOWS, "The frozen corpus did not materialize all 3+10 benchmark batches.")

    cpu = {
        "x": torch.from_numpy(np.ascontiguousarray(sample_x)).pin_memory(),
        "y": torch.from_numpy(np.ascontiguousarray(sample_y)).pin_memory(),
        "pos": torch.from_numpy(np.ascontiguousarray(sample_pos)).pin_memory(),
        "valid": torch.from_numpy(np.ascontiguousarray(sample_valid)).pin_memory(),
        "start": torch.zeros((sample_rows, T), dtype=torch.int32).pin_memory(),
        "segpos": torch.arange(T, dtype=torch.int32).unsqueeze(0).expand(sample_rows, -1).contiguous().pin_memory(),
    }
    valid_pairs_per_update = [
        int(sample_valid[i * GLOBAL_BATCH:(i + 1) * GLOBAL_BATCH].sum())
        for i in range(WARMUPS + SAMPLES)
    ]
    require(all(x > 0 for x in valid_pairs_per_update), "A benchmark batch has no usable next-token pairs.")

    def reset_rng():
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    def make_optimizer(model):
        decay, no_decay = [], []
        for parameter in model.parameters():
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
        groups = [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(
            groups, lr=PEAK_LR, betas=BETAS, eps=EPS, fused=True
        )

    def lr_for_update(update_1based):
        return PEAK_LR * min((update_1based * GLOBAL_TOKENS) / WARMUP_TOKENS, 1.0)

    def make_model_and_optimizer():
        reset_rng()
        model = NativeReadStage1ArmA().to(DEVICE)
        init = canonical_init()
        load_init(model, init)
        del init
        model.train()
        optimizer = make_optimizer(model)
        compiled_model = torch.compile(model, mode="default")
        return model, optimizer, compiled_model

    def is_cuda_oom(error):
        seen = set()
        current = error
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, torch.cuda.OutOfMemoryError):
                return True
            message = str(current).lower()
            if "cuda out of memory" in message or "cuda error: out of memory" in message:
                return True
            current = current.__cause__ or current.__context__
        return False

    def one_full_update(compiled_model, model, optimizer, update_index, microbatch):
        row0 = update_index * GLOBAL_BATCH
        denom = int(cpu["valid"][row0:row0 + GLOBAL_BATCH].sum().item())
        require(denom > 0, f"Update {update_index + 1} has no valid targets.")
        lr = lr_for_update(update_index + 1)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)

        for offset in range(0, GLOBAL_BATCH, microbatch):
            lo = row0 + offset
            hi = lo + microbatch
            x = cpu["x"][lo:hi].to(DEVICE, dtype=torch.long, non_blocking=True)
            y = cpu["y"][lo:hi].to(DEVICE, dtype=torch.long, non_blocking=True)
            pos = cpu["pos"][lo:hi].to(DEVICE, dtype=torch.int32, non_blocking=True)
            valid = cpu["valid"][lo:hi].to(DEVICE, dtype=torch.bool, non_blocking=True)
            start = cpu["start"][lo:hi].to(DEVICE, dtype=torch.int32, non_blocking=True)
            segpos = cpu["segpos"][lo:hi].to(DEVICE, dtype=torch.int32, non_blocking=True)
            full_mask = (start[:, :, None] == start[:, None, :]) & CAUSAL.unsqueeze(0)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False):
                logits = compiled_model(x, pos, segpos, full_mask)
                loss_sum = ce_sum(logits, y, valid)
                loss = loss_sum / denom
            loss.backward()
            del x, y, pos, valid, start, segpos, full_mask, logits, loss_sum, loss

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()

    oom_attempts = []
    selected_microbatch = None
    model = optimizer = compiled_model = None
    for candidate_microbatch in (64, 32, 16):
        try:
            gc.collect()
            torch.cuda.empty_cache()
            if hasattr(torch, "_dynamo"):
                torch._dynamo.reset()
            model, optimizer, compiled_model = make_model_and_optimizer()
            actual_parameter_count = sum(parameter.numel() for parameter in model.parameters())
            require(actual_parameter_count == analytic_total,
                    f"Instantiated parameters {actual_parameter_count:,} != analytical {analytic_total:,}.")
            # Three distinct corpus batches; compilation occurs on the first warmup.
            # A failed candidate is discarded before trying the next genuine-OOM fallback.
            for warmup_index in range(WARMUPS):
                one_full_update(compiled_model, model, optimizer, warmup_index, candidate_microbatch)
            selected_microbatch = candidate_microbatch
            break
        except Exception as exc:
            if not is_cuda_oom(exc):
                raise
            oom_attempts.append({
                "microbatch_sequences": candidate_microbatch,
                "gradient_accumulation_steps": GLOBAL_BATCH // candidate_microbatch,
                "exception": f"{type(exc).__name__}: {str(exc)[:1200]}",
            })
            del exc
            model = optimizer = compiled_model = None
            gc.collect()
            torch.cuda.empty_cache()
            if hasattr(torch, "_dynamo"):
                torch._dynamo.reset()
            if candidate_microbatch == 16:
                raise RuntimeError("Genuine CUDA OOM at B64, B32x2, and B16x4.") from None

    require(selected_microbatch is not None, "No supported microbatch configuration completed warmup.")
    torch.cuda.synchronize(DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    step_times_ms = []
    for sample_index in range(SAMPLES):
        batch_index = WARMUPS + sample_index
        torch.cuda.synchronize(DEVICE)
        start_time = time.perf_counter()
        one_full_update(compiled_model, model, optimizer, batch_index, selected_microbatch)
        torch.cuda.synchronize(DEVICE)
        step_times_ms.append((time.perf_counter() - start_time) * 1000.0)

    median_ms = float(np.percentile(step_times_ms, 50))
    p10_ms = float(np.percentile(step_times_ms, 10))
    p90_ms = float(np.percentile(step_times_ms, 90))
    median_valid_pairs = float(np.median(valid_pairs_per_update[WARMUPS:]))
    input_tokens_per_update = int(GLOBAL_BATCH * T)
    input_tok_s = float(input_tokens_per_update / (median_ms / 1000.0))
    usable_pair_s = float(median_valid_pairs / (median_ms / 1000.0))
    projections = {}
    for target_tokens in TARGET_REAL_INPUT_TOKENS:
        projected_hours = float(target_tokens / input_tok_s / 3600.0)
        target_label = f"{target_tokens / 1_000_000_000:g}B_real_input_tokens"
        projections[target_label] = {
            "target_real_input_tokens": int(target_tokens),
            "hours_at_median_input_tok_s": projected_hours,
            "compute_units_at_8p9_per_gpu_hour": float(projected_hours * COLAB_CU_PER_GPU_HOUR),
            "usable_pairs_hours_diagnostic": float(target_tokens / usable_pair_s / 3600.0),
            "usable_pairs_CU_diagnostic": float(target_tokens / usable_pair_s / 3600.0 * COLAB_CU_PER_GPU_HOUR),
        }

    result["model"]["parameter_counts"].update({
        "instantiated_total_trainable": int(actual_parameter_count),
        "instantiated_counts_match_analytic": True,
    })
    peak_allocated_bytes = int(torch.cuda.max_memory_allocated(DEVICE))
    peak_reserved_bytes = int(torch.cuda.max_memory_reserved(DEVICE))
    result["runtime"] = {
        "status": "MEASURED",
        **gpu_facts,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "precision": "BF16 autocast; canonical FP32 master parameters",
        "compile": "torch.compile(model, mode='default')",
        "optimizer": {
            "name": "fused AdamW",
            "lr_peak": PEAK_LR,
            "betas": list(BETAS),
            "eps": EPS,
            "weight_decay": WEIGHT_DECAY,
            "parameter_decay_groups": "ndim>=2 decay; vectors/scalars no decay",
            "gradient_clip_norm": CLIP_NORM,
        },
        "batch": {
            "global_sequences": GLOBAL_BATCH,
            "tokens_per_update": input_tokens_per_update,
            "attempt_order": ["B64x1", "B32x2", "B16x4"],
            "fallback_policy": "genuine CUDA OOM only",
            "b64_genuine_oom_fallback_used": any(x["microbatch_sequences"] == 64 for x in oom_attempts),
            "selected_configuration": f"B{selected_microbatch}x{GLOBAL_BATCH // selected_microbatch}",
            "microbatch_sequences": int(selected_microbatch),
            "gradient_accumulation_steps": int(GLOBAL_BATCH // selected_microbatch),
            "oom_fallback_attempts": oom_attempts,
            "unique_corpus_batches": WARMUPS + SAMPLES,
            "replayed_batches": False,
        },
        "warmup_updates": WARMUPS,
        "timed_updates": SAMPLES,
        "step_times_ms": [float(x) for x in step_times_ms],
        "median_step_ms": median_ms,
        "p10_step_ms": p10_ms,
        "p90_step_ms": p90_ms,
        "input_tokens_per_update": input_tokens_per_update,
        "valid_next_token_pairs_per_update_samples": [int(x) for x in valid_pairs_per_update[WARMUPS:]],
        "median_valid_next_token_pairs_per_update": median_valid_pairs,
        "input_tokens_per_second": input_tok_s,
        "usable_next_token_pairs_per_second": usable_pair_s,
        "peak_allocated_vram_bytes": peak_allocated_bytes,
        "peak_reserved_vram_bytes": peak_reserved_bytes,
        "peak_allocated_vram_GiB": float(peak_allocated_bytes / 2**30),
        "peak_reserved_vram_GiB": float(peak_reserved_bytes / 2**30),
        "projection_scope": RUNTIME_PROJECTION_SCOPE,
        "projection_note": RUNTIME_PROJECTION_NOTE,
        "final_training_runtime_certified": False,
        "projections": projections,
    }
    result["ASSUMPTION_CHECKS"]["full_step_runtime_projections"] = {
        "status": "MEASURED",
        "basis": "median synchronized full optimizer update; projections use real input-token throughput",
        "projections": projections,
        "final_training_runtime_certified": False,
    }
    result["A_TIMING_READY"] = True
    result["status"] = "PASS"
except Exception as exc:
    result["status"] = "CONTRACT_FAIL_CLOSED" if phase in ("contract", "census") else "BENCHMARK_FAILED"
    result["error"] = f"{type(exc).__name__}: {exc}"
    if result.get("runtime", {}).get("status") == "NOT_RUN":
        result["runtime"]["error"] = result["error"]

print("ARCHITECTURE_DELTA_REQUIREMENTS")
print(json.dumps(ARCHITECTURE_DELTA_REQUIREMENTS, indent=2, sort_keys=True))
print("A_TIMING_READY=" + str(bool(result.get("A_TIMING_READY", False))).lower())
print(json.dumps(result, indent=2, sort_keys=True))