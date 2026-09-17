# AKASHA REAL CHECKPOINT TEXT GENERATION GATE

**Date:** 2026-09-16  
**Status:** real trained checkpoint + exact tokenizer loaded; text generation executed  
**V0 reference commit:** `2080aa6` (untouched; no profiling, no optimization, no serving)

---

## 1. Artifacts

| Item | Value |
|---|---|
| `CHECKPOINT_PATH` | `C:\iclr-oc\runs\arm_a_2p5b_opt3c_all\ckpt\latest.pt` |
| Checkpoint file SHA-256 | `fe7e6c2c6ac0d12630018812f735486c45003678adbfcb4a88a476d886c47efb` |
| Trainer `code_sha256` in checkpoint | `1985fa42042033c842c7ed0faea2c34ead6516bb1fe56753426b6774ee2d0b49` (equals pinned trainer SHA-256) |
| Training tokens consumed | `2500067328` (target 2500000000) |
| Updates done | `19074` |
| `CHECKPOINT_STATUS` | `VALIDATED` |
| Weights fingerprint | `dbc2acc115e795f27b1ded7624cbffb9e90f6c49dcb54fad0cb2705f9627a934` |
| Canonical package | `results/akasha/real_checkpoint_package` (`model.safetensors` + `manifest.json`) |
| `TOKENIZER_PATH` | `C:\Users\arahe\OneDrive\Documents\ChatGPT\icrl\phase_bdh_corpus_stage1\tokenizer\tokenizer.json` |
| `TOKENIZER_SHA256` | `9a05bca4c065d9952a01ee1e3f4b6e23e39200e1f33da65e5dad1819419f71c3` |
| `TOKENIZER_STATUS` | `READY` |
| Tokenizer identity | `bytelevel-bpe-8192-9a05bca4c065d995` (vocab 8192, single special token `<unk>` id 0) |
| Device | cuda — NVIDIA GeForce RTX 4060 Laptop GPU; torch 2.10.0+cu128 |

The checkpoint is the final training artifact (`latest.pt`, updates_done=19074, tokens_consumed=2,500,067,328), not an intermediate archive. Its embedded trainer fingerprint matches the frozen source `training/arm_a_2p5b_trainer.py` at commit `0dcbb87` exactly.

## 2. Validation (existing validator + generation-time checks)

| Check | Result |
|---|---|
| `TRAINED_CHECKPOINT_LOCAL_STATUS` | `VALIDATED` |
| Tensor names / shapes | all 10 canonical tensors match the Arm-A manifest |
| Tensors finite | true |
| Manifest / package fingerprint | verified |
| Tokenizer SHA-256 match | `READY` (exact expected hash) |
| Recurrent prefill/decode executes | true (smoke prompt + decode step, finite logits) |
| `last_hidden` snapshot round trip | true |
| Validator smoke next token | `261` |

---

## 3. Greedy generation (64 new tokens, recurrent engine)

All five mission prompts, first deterministic run, no cherry-picking:

### `Hello`

- prompt token IDs: `[40, 610, 79]`
- generated token IDs: `[12, 354, 5552, 261, 280, 281, 288, 266, 422, 79, 263, 12, 936, 354, 5552, 559, 5834, 1855, 307, 788, 371, 464, 14, 199, 199, 41, 5552, 559, 5834, 642, 354, 5552, 261, 280, 281, 288, 266, 422, 79, 263, 12, 936, 354, 5552, 559, 5834, 642, 354, 5552, 261, 280, 281, 288, 266, 422, 79, 263, 14, 199, 199, 41, 5552, 559, 5834]`
- stats: unique 23/64 (ratio 0.36), max single-token run 2, most frequent token 5552 x7, `<unk>` count 0, logits finite: True

```text
Hello, I'm a fan of the Moon, but I'm not sure what to do with it.

I'm not sure if I'm a fan of the Moon, but I'm not sure if I'm a fan of the Moon.

I'm not sure
```

### `Hi, how are you?`

- prompt token IDs: `[40, 73, 12, 1671, 401, 826, 31]`
- generated token IDs: `[199, 199, 41, 5552, 559, 5834, 1855, 826, 1185, 12, 936, 354, 5552, 559, 5834, 1855, 826, 1185, 14, 199, 199, 41, 5552, 559, 5834, 642, 826, 401, 559, 14, 199, 199, 41, 5552, 559, 5834, 642, 826, 401, 559, 14, 199, 199, 41, 5552, 559, 5834, 642, 826, 401, 559, 14, 199, 199, 41, 5552, 559, 5834, 642, 826, 401, 559, 14, 199]`
- stats: unique 14/64 (ratio 0.22), max single-token run 2, most frequent token 199 x11, `<unk>` count 0, logits finite: True

```text
Hi, how are you?

I'm not sure what you mean, but I'm not sure what you mean.

I'm not sure if you are not.

I'm not sure if you are not.

I'm not sure if you are not.

I'm not sure if you are not.

```

### `The capital of France is`

- prompt token IDs: `[550, 1945, 2369, 288, 383, 414, 341, 314]`
- generated token IDs: `[261, 280, 543, 3098, 12, 280, 543, 3098, 12, 301, 280, 543, 3098, 12, 280, 543, 3098, 12, 301, 280, 543, 3098, 12, 280, 543, 3098, 12, 301, 280, 543, 3098, 14, 1139, 1535, 83, 261, 280, 543, 3098, 12, 280, 543, 3098, 12, 280, 543, 3098, 12, 280, 543, 3098, 12, 280, 543, 3098, 12, 301, 280, 543, 3098, 14, 199, 199, 550]`
- stats: unique 12/64 (ratio 0.19), max single-token run 2, most frequent token 280 x13, `<unk>` count 0, logits finite: True

```text
The capital of France is a fantastic, fantastic, and fantastic, fantastic, and fantastic, fantastic, and fantastic. It’s a fantastic, fantastic, fantastic, fantastic, fantastic, and fantastic.

The
```

### `Once upon a time`

- prompt token IDs: `[47, 3716, 5421, 261, 884]`
- generated token IDs: `[12, 266, 430, 395, 272, 1197, 1148, 295, 4669, 14, 399, 430, 395, 272, 1197, 1148, 295, 4669, 291, 266, 430, 395, 272, 1197, 14, 399, 430, 395, 272, 1197, 1148, 295, 4669, 291, 266, 430, 395, 272, 1197, 14, 399, 430, 395, 272, 1197, 1148, 295, 4669, 291, 266, 430, 395, 272, 1197, 14, 399, 430, 395, 272, 1197, 1148, 295, 4669, 291]`
- stats: unique 12/64 (ratio 0.19), max single-token run 1, most frequent token 430 x8, `<unk>` count 0, logits finite: True

```text
Once upon a time, the British were born. The British were born in the British. The British were born in the British. The British were born in the British. The British were born in
```

### `In computer science,`

- prompt token IDs: `[528, 5514, 7270, 12]`
- generated token IDs: `[266, 1467, 2022, 6422, 307, 2870, 314, 266, 7342, 288, 266, 1171, 14, 399, 1171, 314, 266, 1171, 288, 266, 1171, 288, 266, 1171, 288, 266, 1171, 288, 266, 1171, 288, 266, 1171, 288, 266, 1171, 288, 266, 1171, 288, 266, 1171, 288, 266, 1171, 288, 266, 1171, 288, 266, 1171, 288, 266, 1171, 288, 266, 1171, 288, 266, 1171, 288, 266, 1171, 288]`
- stats: unique 12/64 (ratio 0.19), max single-token run 1, most frequent token 266 x19, `<unk>` count 0, logits finite: True

```text
In computer science, the most important thing to understand is the importance of the process. The process is the process of the process of the process of the process of the process of the process of the process of the process of the process of the process of the process of the process of the process of the process of the process of the process of
```

## 4. FULL-vs-RECURRENT greedy check

Source-faithful dense full-prefix inference (`reference_full.full_forward`) vs the verified recurrent engine (`reference_recurrent`), token-for-token:

| Prompt | Match | First divergence |
|---|---|---|
| `Hello` | True | None |
| `Hi, how are you?` | True | None |
| `The capital of France is` | True | None |
| `Once upon a time` | True | None |
| `In computer science,` | True | None |

**`FULL_VS_RECURRENT_GREEDY_MATCH` = True**  
**`FIRST_DIVERGENCE_TOKEN` = None**

No correctness defect detected; sampled generation proceeded.

## 5. Sampled generation (temperature 0.8, top_k 50, seed 20260916, 128 new tokens)

Single deterministic seeded run over the same five prompts:

### `Hello`

- generated token IDs (first 24 shown): `[12, 358, 1196, 340, 408, 23, 12, 2496, 24, 199, 199, 33, 1196, 340, 408, 23, 12, 2496, 24, 429, 199, 41, 497, 1657]` …
- stats: unique 81/128 (ratio 0.63), max single-token run 2, most frequent token 12 x6, `<unk>` count 0, logits finite: True

```text
Hello, April 07, 2008

April 07, 2008


I can't hear that but I am a guy and I am looking for new ones. As I walked through the stairs, the guy was waiting to find out.
It was in the night club that we were hopping. One day, I had to start looking at the guy and saw the guy playing. He stood in dice and smiled and said, "I should have been in the house?"
I couldn't tell why
```

### `Hi, how are you?`

- generated token IDs (first 24 shown): `[2889, 826, 258, 270, 2955, 307, 1593, 1845, 3437, 301, 4558, 31, 199, 199, 46, 628, 83, 462, 266, 1637, 329, 391, 199, 199]` …
- stats: unique 78/128 (ratio 0.61), max single-token run 4, most frequent token 31 x11, `<unk>` count 0, logits finite: True

```text
Hi, how are you? Is you talking to your audience and why?

News from the Church

Wednesday, May 08, 2008

Did you cough your hand and pull your hand out and swing your hand over to the pinnacle of bed?? Every day, you'll take your head away from a bed and turn to the bed?

The dress should be about a thousand foot wide. Think it's a bad bastard???

A little of this????
```

### `The capital of France is`

- generated token IDs (first 24 shown): `[266, 1128, 387, 266, 1945, 2369, 288, 266, 383, 414, 341, 12, 301, 266, 1945, 2369, 288, 383, 414, 341, 744, 559, 1143, 921]` …
- stats: unique 64/128 (ratio 0.50), max single-token run 2, most frequent token 266 x10, `<unk>` count 0, logits finite: True

```text
The capital of France is the same as the capital of the France, and the capital of France has not been so good.

Why is the capital of France so important?

Sometimes, it is very difficult to see how this can be done with France, but it is one of the most important things to believe. It is one of the problems that can only be solved by getting one in the field. So, when France’s federal government has passed the capital of France, France is the only one who knows what
```

### `Once upon a time`

- generated token IDs (first 24 shown): `[31, 4028, 2974, 464, 3439, 652, 31, 199, 199, 37, 2008, 2294, 12, 1637, 5182, 301, 361, 327, 281, 65, 2141, 3243, 261, 330]` …
- stats: unique 98/128 (ratio 0.77), max single-token run 2, most frequent token 199 x7, `<unk>` count 0, logits finite: True

```text
Once upon a time? How did it turn out?

Every year, Chris and Solana had made a living, but there were some bad old, unrealistic things going for. Earlier in Searching Life, John Warren has been able to show her son’s true sense of honour — a son of the Rome, who in 2008 was a horrible tale to him:


What are you talking about? 

Hold this.

For years, when we have been told of the st
```

### `In computer science,`

- generated token IDs (first 24 shown): `[266, 292, 2975, 5293, 288, 266, 4077, 314, 1361, 365, 261, 342, 7069, 1521, 3644, 1389, 14, 399, 891, 1455, 767, 365, 266, 4064]` …
- stats: unique 44/128 (ratio 0.34), max single-token run 1, most frequent token 266 x13, `<unk>` count 0, logits finite: True

```text
In computer science, the digital technology of the US is based on a hybrid device. The system operates on the machine and the device is monitored by the machine, and there is a digital technology, called a computer. The device is designed to move in the direction and direction from which it goes. The device monitors the digital devices and is able to control the digital devices. The device is connected to a computer, and is also connected to the device. The device is connected to the device, and the device monitors the digital devices and is connected to the device
```

## 6. Pathology classification

This is **not** an LM benchmark. Statements are limited to the ten examples above.

| Mode | Classification | Evidence |
|---|---|---|
| Greedy | **C — pathological repetition/collapse** (after a B-level opening) | each prompt produces 1–2 fluent sentences, then phrase loops: “the process of the process”, “fantastic, fantastic”, “The British were born in the British”, “I'm not sure if you are not” |
| Sampled (T=0.8, top_k=50) | **B — weak but structured language** | varied, locally grammatical, globally incoherent text; repetition appears only in tails (“the device is connected to the device”) |

Observed pathologies / absences:

- Short-phrase loops: **yes** (greedy, all five prompts; sampled tails).
- Single-token constant loops: no (max single-token run = 2 greedy / 4 sampled).
- Immediate EOS: not applicable (tokenizer has no EOS token; `<unk>` id 0 was never emitted).
- Invalid decoding: none (all token sequences decode cleanly).
- NaN / Inf logits: none (all `logits_finite = true`).
- Constant output: no.

**`GENERATION_CLASSIFICATION` = B (weak but structured language) overall; greedy decoding collapses to C (short-phrase repetition) after the opening sentences.**

## 7. Required summary

```text
CHECKPOINT_PATH = C:\iclr-oc\runs\arm_a_2p5b_opt3c_all\ckpt\latest.pt
CHECKPOINT_TOKENS = 2500067328
CHECKPOINT_STATUS = VALIDATED
TOKENIZER_PATH = C:\Users\arahe\OneDrive\Documents\ChatGPT\icrl\phase_bdh_corpus_stage1\tokenizer\tokenizer.json
TOKENIZER_SHA256 = 9a05bca4c065d9952a01ee1e3f4b6e23e39200e1f33da65e5dad1819419f71c3
TOKENIZER_STATUS = READY
FULL_VS_RECURRENT_GREEDY_MATCH = True
FIRST_DIVERGENCE_TOKEN = None
GENERATION_CLASSIFICATION = B (weak but structured); greedy = C (short-phrase repetition)
```

## 8. Boundaries respected

- No V0.5 profiling was run (Stage B–G never executed).
- `reference_full.py` / `reference_recurrent.py` were not modified.
- No optimization, no Triton, no DeltaLog, no serving work.
- The tokenizer artifact was hash-verified, not recreated.

## 9. Evidence files

- `results/akasha/real_generation_greedy.json`
- `results/akasha/real_generation_sampled.json`
- `results/akasha/real_generation_validation.json`
- `results/akasha/real_checkpoint_package/` (canonical package)
- `campaigns/AKASHA_REAL_GENERATION_REPORT.md` (this file)
