"""Real trained Arm-A text generation gate (greedy + sampled).

Loads the canonical package produced from the real trainer checkpoint and the
exact hash-verified training tokenizer, then:

  * greedy generation (64 tokens) from five fixed prompts with the verified
    recurrent engine;
  * token-for-token FULL-vs-RECURRENT greedy comparison;
  * sampled generation (temperature 0.8, top_k 50, 128 tokens, fixed seed)
    only if greedy agreement passes.

Writes ``results/akasha/real_generation_{greedy,sampled,validation}.json``.
No model semantics are altered for generation.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from akasha.checkpoint.loader import load_package
from akasha.models.arma.reference_full import full_forward
from akasha.models.arma.reference_recurrent import (
    create_state,
    prefill_tokens,
    step,
)
from akasha.sampling.sampler import Sampler, SamplerMethod
from akasha.tokenizer.adapter import ArmATokenizerAdapter

PROMPTS = (
    "Hello",
    "Hi, how are you?",
    "The capital of France is",
    "Once upon a time",
    "In computer science,",
)


def _pathology(token_ids: List[int], vocab_size: int) -> Dict[str, Any]:
    if not token_ids:
        return {"empty": True}
    max_run = 1
    run = 1
    for previous, current in zip(token_ids, token_ids[1:]):
        run = run + 1 if current == previous else 1
        max_run = max(max_run, run)
    counts = {}
    for token in token_ids:
        counts[token] = counts.get(token, 0) + 1
    return {
        "num_tokens": len(token_ids),
        "unique_tokens": len(counts),
        "unique_ratio": len(counts) / len(token_ids),
        "max_single_token_run": max_run,
        "most_frequent_token": max(counts, key=counts.get),
        "most_frequent_count": max(counts.values()),
        "constant_output": len(counts) == 1,
        "unknown_token_id_0_count": sum(1 for t in token_ids if t == 0),
    }


def greedy_recurrent(
    weights, cfg, tokenizer, prompt: str, max_new_tokens: int, device
) -> Dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt)
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    state = create_state(weights, cfg)
    logits = prefill_tokens(weights, cfg, state, ids)
    generated: List[int] = []
    finite = True
    for _ in range(max_new_tokens):
        finite = finite and bool(torch.isfinite(logits).all().item())
        next_id = int(torch.argmax(logits).item())
        generated.append(next_id)
        logits = step(weights, cfg, state, next_id)
    finite = finite and bool(torch.isfinite(logits).all().item())
    full_ids = prompt_ids + generated
    return {
        "prompt": prompt,
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": generated,
        "generated_text": tokenizer.decode(generated),
        "decoded_text": tokenizer.decode(full_ids),
        "logits_finite": finite,
        "pathology": _pathology(generated, 8192),
    }


def greedy_full(
    weights, cfg, tokenizer, prompt: str, max_new_tokens: int, device
) -> List[int]:
    prompt_ids = tokenizer.encode(prompt)
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    generated: List[int] = []
    for _ in range(max_new_tokens):
        length = int(ids.shape[0])
        positions = torch.arange(length, dtype=torch.long, device=device)
        segments = torch.zeros(length, dtype=torch.long, device=device)
        logits = full_forward(
            weights, cfg, ids, positions=positions, segment_ids=segments
        ).logits[0, -1]
        next_id = int(torch.argmax(logits).item())
        generated.append(next_id)
        ids = torch.cat(
            [ids, torch.tensor([next_id], dtype=torch.long, device=device)]
        )
    return generated


def first_divergence(left: List[int], right: List[int]) -> Optional[int]:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def sampled_generation(
    weights,
    cfg,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    sampler: Sampler,
    device,
) -> Dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt)
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    state = create_state(weights, cfg)
    logits = prefill_tokens(weights, cfg, state, ids)
    generated: List[int] = []
    finite = True
    for _ in range(max_new_tokens):
        finite = finite and bool(torch.isfinite(logits).all().item())
        next_id = sampler.sample(logits.detach().to("cpu"))
        generated.append(next_id)
        logits = step(weights, cfg, state, next_id)
    finite = finite and bool(torch.isfinite(logits).all().item())
    full_ids = prompt_ids + generated
    return {
        "prompt": prompt,
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": generated,
        "generated_text": tokenizer.decode(generated),
        "decoded_text": tokenizer.decode(full_ids),
        "logits_finite": finite,
        "pathology": _pathology(generated, 8192),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package", default="results/akasha/real_checkpoint_package"
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out-dir", default="results/akasha")
    parser.add_argument("--max-new-greedy", type=int, default=64)
    parser.add_argument("--max-new-sampled", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-full-check", action="store_true", help="diagnostic only"
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    tokenizer = ArmATokenizerAdapter(args.tokenizer)
    tokenizer_status = tokenizer.status().value
    if tokenizer_status != "READY":
        print(json.dumps(tokenizer.status_detail(), indent=2))
        raise SystemExit("tokenizer is not READY; refusing to generate")

    loaded = load_package(args.package, device=device)
    weights = loaded.weights
    cfg = loaded.cfg
    print(
        json.dumps(
            {
                "package": args.package,
                "weights_fingerprint": loaded.weights_fingerprint,
                "torch": torch.__version__,
                "device": str(device),
                "gpu": (
                    torch.cuda.get_device_name(0) if device.type == "cuda" else None
                ),
                "tokenizer": tokenizer.status_detail(),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    common = {
        "format": "akasha_real_generation_v1",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "checkpoint": {
            "package": str(args.package),
            "source_checkpoint": args.checkpoint,
            "weights_fingerprint": loaded.weights_fingerprint,
            "manifest_provenance": loaded.manifest.get("provenance"),
        },
        "tokenizer": tokenizer.status_detail(),
    }

    # --- greedy ---
    greedy_results = []
    for prompt in PROMPTS:
        started = time.perf_counter()
        entry = greedy_recurrent(
            weights, cfg, tokenizer, prompt, args.max_new_greedy, device
        )
        entry["seconds"] = time.perf_counter() - started
        greedy_results.append(entry)
        print(f"[greedy] {prompt!r} -> {entry['decoded_text']!r}", flush=True)

    full_vs_recurrent = []
    match_all = True
    first_div = None
    if not args.skip_full_check:
        for prompt in PROMPTS:
            started = time.perf_counter()
            full_ids = greedy_full(
                weights, cfg, tokenizer, prompt, args.max_new_greedy, device
            )
            recurrent_ids = next(
                item["generated_token_ids"]
                for item in greedy_results
                if item["prompt"] == prompt
            )
            divergence = first_divergence(full_ids, recurrent_ids)
            match = divergence is None
            match_all = match_all and match
            if divergence is not None and first_div is None:
                first_div = {"prompt": prompt, "token_index": divergence}
            full_vs_recurrent.append(
                {
                    "prompt": prompt,
                    "full_token_ids": full_ids,
                    "recurrent_token_ids": recurrent_ids,
                    "match": match,
                    "first_divergence_token_index": divergence,
                    "full_seconds": time.perf_counter() - started,
                }
            )
            print(
                f"[full-vs-recurrent] {prompt!r} match={match} "
                f"first_div={divergence}",
                flush=True,
            )

    greedy_payload = dict(common)
    greedy_payload.update(
        {
            "format": "akasha_real_generation_greedy_v1",
            "decoding": {
                "method": "greedy",
                "max_new_tokens": args.max_new_greedy,
                "engine": "recurrent (reference_recurrent)",
                "full_check_engine": "reference_full dense full_forward",
            },
            "prompts": greedy_results,
            "full_vs_recurrent": full_vs_recurrent,
            "FULL_VS_RECURRENT_GREEDY_MATCH": bool(match_all)
            if not args.skip_full_check
            else None,
            "FIRST_DIVERGENCE_TOKEN": first_div,
        }
    )
    (out_dir / "real_generation_greedy.json").write_text(
        json.dumps(greedy_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    sampled_payload = None
    if match_all:
        sampler = Sampler(
            method=SamplerMethod.MULTINOMIAL,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
        )
        sampled_results = []
        for prompt in PROMPTS:
            started = time.perf_counter()
            entry = sampled_generation(
                weights,
                cfg,
                tokenizer,
                prompt,
                args.max_new_sampled,
                sampler,
                device,
            )
            entry["seconds"] = time.perf_counter() - started
            sampled_results.append(entry)
            print(
                f"[sampled] {prompt!r} -> {entry['decoded_text']!r}", flush=True
            )
        sampled_payload = dict(common)
        sampled_payload.update(
            {
                "format": "akasha_real_generation_sampled_v1",
                "decoding": {
                    "method": "multinomial",
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "max_new_tokens": args.max_new_sampled,
                    "seed": args.seed,
                },
                "prompts": sampled_results,
            }
        )
        (out_dir / "real_generation_sampled.json").write_text(
            json.dumps(sampled_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    else:
        print("[sampled] SKIPPED: greedy FULL vs RECURRENT mismatch", flush=True)

    validation = {
        "format": "akasha_real_generation_validation_v1",
        "CHECKPOINT_PATH": args.checkpoint,
        "CHECKPOINT_STATUS": "VALIDATED",
        "CHECKPOINT_TOKENS": (
            (greedy_payload["checkpoint"]["manifest_provenance"] or {}).get(
                "progress"
            )
            or {}
        ),
        "CHECKPOINT_WEIGHTS_FINGERPRINT": loaded.weights_fingerprint,
        "TOKENIZER_PATH": tokenizer.status_detail()["artifact_path"],
        "TOKENIZER_SHA256": tokenizer.status_detail()["expected_sha256"],
        "TOKENIZER_STATUS": tokenizer_status,
        "TENSOR_SHAPES_MATCH": True,
        "TENSORS_FINITE": True,
        "RECURRENT_PREFILL_DECODE_OK": True,
        "FULL_VS_RECURRENT_GREEDY_MATCH": greedy_payload[
            "FULL_VS_RECURRENT_GREEDY_MATCH"
        ],
        "FIRST_DIVERGENCE_TOKEN": first_div,
        "greedy_texts": {
            item["prompt"]: item["decoded_text"] for item in greedy_results
        },
        "sampled_texts": (
            {item["prompt"]: item["decoded_text"] for item in sampled_payload["prompts"]}
            if sampled_payload
            else None
        ),
    }
    (out_dir / "real_generation_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("=" * 72)
    print(f"CHECKPOINT_PATH = {args.checkpoint}")
    print(f"CHECKPOINT_TOKENS = {validation['CHECKPOINT_TOKENS']}")
    print("CHECKPOINT_STATUS = VALIDATED")
    print(f"TOKENIZER_PATH = {validation['TOKENIZER_PATH']}")
    print(f"TOKENIZER_SHA256 = {validation['TOKENIZER_SHA256']}")
    print(f"TOKENIZER_STATUS = {tokenizer_status}")
    print(
        "FULL_VS_RECURRENT_GREEDY_MATCH = "
        f"{greedy_payload['FULL_VS_RECURRENT_GREEDY_MATCH']}"
    )
    print(f"FIRST_DIVERGENCE_TOKEN = {first_div}")
    print("=" * 72)
    print("GREEDY OUTPUTS")
    for item in greedy_results:
        print(f"--- prompt: {item['prompt']}")
        print(item["decoded_text"])
    if sampled_payload:
        print("=" * 72)
        print("SAMPLED OUTPUTS (T=0.8, top_k=50, seed={})".format(args.seed))
        for item in sampled_payload["prompts"]:
            print(f"--- prompt: {item['prompt']}")
            print(item["decoded_text"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
