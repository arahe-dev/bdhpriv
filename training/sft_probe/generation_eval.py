"""Generation suite for SFT calibration checkpoints.

Runs the frozen 40-prompt suite (results/sft_probe/eval_prompts.json) through
the verified Akasha reference engines:

  * greedy, 64 new tokens, recurrent engine (all prompts);
  * sampled, temperature 0.8, top_k 50, seed 20260916, 128 new tokens;
  * full dense vs recurrent greedy token-for-token parity on the canonical
    five prompts.

Computes non-subjective pathology diagnostics and saves raw token ids and
decoded text so a human can inspect the outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from training.sft_probe import common
else:
    from . import common


def load_weights(checkpoint: str | None, device=None):
    from akasha.checkpoint.loader import map_trainer_state_dict
    from akasha.models.arma.config import production_config

    cfg = production_config()
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu",
                             weights_only=False)
        state = payload["model"]
    else:
        state = common.load_base_state()
    weights = map_trainer_state_dict(state, cfg, strict=True)
    weights = weights.to(device=device, dtype=torch.float32)
    return weights, cfg


def load_prompt_suite(path: Path):
    suite = json.loads(path.read_text(encoding="utf-8"))
    return suite


def ngram_stats(tokens, n):
    if len(tokens) < n:
        return {"total": 0, "unique": 0, "repeated_fraction": None,
                "distinct": None}
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    unique = len(set(grams))
    total = len(grams)
    return {
        "total": total,
        "unique": unique,
        "repeated_fraction": 1.0 - unique / total,
        "distinct": unique / total,
    }


def longest_repeated_substring(tokens):
    n = len(tokens)
    if n < 2:
        return 0
    best = 0
    previous = [0] * (n + 1)
    for i in range(1, n + 1):
        current = [0] * (n + 1)
        for j in range(1, n + 1):
            if tokens[i - 1] == tokens[j - 1] and i != j:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def token_entropy(tokens):
    if not tokens:
        return None
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    total = len(tokens)
    return float(-sum(
        (c / total) * math.log2(c / total) for c in counts.values()
    ))


def max_run(tokens):
    if not tokens:
        return 0
    best = run = 1
    for previous, current in zip(tokens, tokens[1:]):
        run = run + 1 if current == previous else 1
        best = max(best, run)
    return best


def pathology(tokens):
    unigram = ngram_stats(tokens, 1)
    bigram = ngram_stats(tokens, 2)
    trigram = ngram_stats(tokens, 3)
    run = max_run(tokens)
    return {
        "num_tokens": len(tokens),
        "unique_tokens": unigram["unique"],
        "repeated_unigram_fraction": unigram["repeated_fraction"],
        "repeated_bigram_fraction": bigram["repeated_fraction"],
        "repeated_trigram_fraction": trigram["repeated_fraction"],
        "distinct_2": bigram["distinct"],
        "distinct_3": trigram["distinct"],
        "longest_repeated_substring": longest_repeated_substring(tokens),
        "token_entropy_bits": token_entropy(tokens),
        "max_single_token_run": run,
        "constant_token_output": len(set(tokens)) == 1 if tokens else None,
        "loop_run_ge_8": run >= 8,
        "immediate_eos_rate": None,
        "immediate_eos_note": (
            "N/A: the real tokenizer/corpus contract has no EOS token "
            "(vocab 8192, single special token <unk> id 0)"
        ),
    }


def summarize(pathologies):
    def mean(key):
        values = [
            p[key] for p in pathologies if p.get(key) is not None
        ]
        return float(sum(values) / len(values)) if values else None

    return {
        "prompt_count": len(pathologies),
        "mean_repeated_unigram_fraction": mean("repeated_unigram_fraction"),
        "mean_repeated_bigram_fraction": mean("repeated_bigram_fraction"),
        "mean_repeated_trigram_fraction": mean("repeated_trigram_fraction"),
        "mean_distinct_2": mean("distinct_2"),
        "mean_distinct_3": mean("distinct_3"),
        "mean_token_entropy_bits": mean("token_entropy_bits"),
        "mean_longest_repeated_substring": mean(
            "longest_repeated_substring"
        ),
        "constant_token_rate": (
            sum(1 for p in pathologies if p.get("constant_token_output"))
            / len(pathologies)
        ),
        "loop_run_ge_8_rate": (
            sum(1 for p in pathologies if p.get("loop_run_ge_8"))
            / len(pathologies)
        ),
    }


def main(argv=None) -> int:
    from akasha.bench.real_generation import (
        first_divergence,
        greedy_full,
        greedy_recurrent,
        sampled_generation,
    )
    from akasha.sampling.sampler import Sampler, SamplerMethod
    from akasha.tokenizer.adapter import ArmATokenizerAdapter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--suite", default=str(common.OUT_DIR / "eval_prompts.json")
    )
    parser.add_argument(
        "--out-dir", default=str(common.OUT_DIR / "generations")
    )
    parser.add_argument(
        "--metrics",
        default=str(common.OUT_DIR / "generation_metrics.json"),
    )
    parser.add_argument("--skip-sampled", action="store_true")
    args = parser.parse_args(argv)

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(common.SAMPLING_SEED)

    tokenizer = ArmATokenizerAdapter(str(common.TOKENIZER_PATH))
    if tokenizer.status().value != "READY":
        raise RuntimeError("tokenizer is not READY")
    suite = load_prompt_suite(Path(args.suite))
    weights, cfg = load_weights(args.checkpoint, device=device)

    results = {
        "tag": args.tag,
        "checkpoint": args.checkpoint or str(common.BASE_CKPT),
        "weights_fingerprint": __import__(
            "akasha.models.arma.ops", fromlist=["weights_fingerprint"]
        ).weights_fingerprint(weights),
        "decoding": {
            "greedy_new_tokens": suite["greedy_new_tokens"],
            "sampled_new_tokens": suite["sampled_new_tokens"],
            "temperature": suite["sampled_temperature"],
            "top_k": suite["sampled_top_k"],
            "seed": 20260916,
            "sampler_stream": (
                "single Sampler RNG stream continued across prompts in "
                "suite order"
            ),
        },
        "greedy": [],
        "sampled": [],
        "full_vs_recurrent": [],
    }

    for entry in suite["prompts"]:
        greedy = greedy_recurrent(
            weights, cfg, tokenizer, entry["prompt"],
            suite["greedy_new_tokens"], device,
        )
        greedy["id"] = entry["id"]
        greedy["category"] = entry["category"]
        greedy["pathology"] = pathology(greedy["generated_token_ids"])
        results["greedy"].append(greedy)

    sampler = Sampler(
        method=SamplerMethod.MULTINOMIAL,
        temperature=suite["sampled_temperature"],
        top_k=suite["sampled_top_k"],
        seed=20260916,
    )
    if not args.skip_sampled:
        for entry in suite["prompts"]:
            sampled = sampled_generation(
                weights, cfg, tokenizer, entry["prompt"],
                suite["sampled_new_tokens"], sampler, device,
            )
            sampled["id"] = entry["id"]
            sampled["category"] = entry["category"]
            sampled["pathology"] = pathology(sampled["generated_token_ids"])
            results["sampled"].append(sampled)

    match_all = True
    first_div = None
    canonical = suite["canonical_five"]
    for prompt in canonical:
        full_ids = greedy_full(
            weights, cfg, tokenizer, prompt,
            suite["greedy_new_tokens"], device,
        )
        recurrent_ids = next(
            item["generated_token_ids"]
            for item in results["greedy"]
            if item["prompt"] == prompt
        )
        divergence = first_divergence(full_ids, recurrent_ids)
        match = divergence is None
        match_all = match_all and match
        if divergence is not None and first_div is None:
            first_div = {"prompt": prompt, "token_index": divergence}
        results["full_vs_recurrent"].append(
            {
                "prompt": prompt,
                "full_token_ids": full_ids,
                "recurrent_token_ids": recurrent_ids,
                "match": match,
                "first_divergence_token_index": divergence,
            }
        )
    results["FULL_VS_RECURRENT_GREEDY_MATCH"] = bool(match_all)
    results["FIRST_DIVERGENCE_TOKEN"] = first_div

    results["greedy_summary"] = summarize(
        [g["pathology"] for g in results["greedy"]]
    )
    if results["sampled"]:
        results["sampled_summary"] = summarize(
            [g["pathology"] for g in results["sampled"]]
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    common.save_json(out_dir / f"{args.tag}.json", results)

    metrics_path = Path(args.metrics)
    metrics = (
        common.load_json(metrics_path) if metrics_path.is_file() else {}
    )
    metrics[args.tag] = {
        "checkpoint": results["checkpoint"],
        "greedy_summary": results["greedy_summary"],
        "sampled_summary": results.get("sampled_summary"),
        "FULL_VS_RECURRENT_GREEDY_MATCH": results[
            "FULL_VS_RECURRENT_GREEDY_MATCH"
        ],
        "FIRST_DIVERGENCE_TOKEN": results["FIRST_DIVERGENCE_TOKEN"],
    }
    common.save_json(metrics_path, metrics)

    print(json.dumps(
        {
            "tag": args.tag,
            "greedy": results["greedy_summary"],
            "sampled": results.get("sampled_summary"),
            "parity": results["FULL_VS_RECURRENT_GREEDY_MATCH"],
        },
        indent=1,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
