# AKASHA V0 REPORT — EXACT DENSE ARM-A RECURRENT INFERENCE

**Date:** 2026-09-16
**Status:** `AKASHA_V0_REFERENCE_PASS = true` — dense PyTorch recurrent oracle frozen.
**Scope of V0:** correctness only. No Triton, CUDA, HTTP, sparse routing, quantization,
SGLang/vLLM/llama.cpp, DeltaLog, or multi-GPU. Optimization begins only after this gate.

---

## 1. Required summary fields

| Field | Value |
|---|---|
| `SOURCE_COMMIT` | `0dcbb87` (`0dcbb878d24b99b5808359c889e97143c3cec00b`) |
| `TRAINER_SHA256` | `1985fa42042033c842c7ed0faea2c34ead6516bb1fe56753426b6774ee2d0b49` |
| `AKASHA_V0_REFERENCE_PASS` | `true` |
| `TOKENIZER_STATUS` | `BLOCKED_ARTIFACT_NOT_LOCAL` |
| `TRAINED_CHECKPOINT_STATUS` | `BLOCKED` (no local trainer checkpoint; validator prepared) |
| `MAX_FP64_ERROR` | `5.329070518200751e-15` (attention algebra; coordinator/token-major/chunked scan are `0.0`) |
| `MAX_FP32_LOGIT_ERROR` | `1.3113021850585938e-06` over all 33 production cases |
| `ARGMAX_AGREEMENT` | `1.0` (min over all 33 production cases) |
| `2048_TOKEN_PARITY` | `PASS` (random, repeated, and multi-segment inputs) |
| `SEGMENT_RESET_PASS` | `true` |
| `CLONE_PASS` | `true` |
| `SERIALIZATION_PASS` | `true` |

Contract tolerances used: `atol = 1e-5`, `rtol = 1e-4` (FP32), `atol = 1e-11` (FP64).
Observed FP32 error is two orders of magnitude inside `atol`; no tolerance was relaxed.

## 2. What was built (owned subtree)

```
akasha/
  config.py                        provenance + artifact-status constants
  models/arma/
    config.py                      frozen ArmAConfig (K = N//H)
    manifest.py                    source commit, trainer SHA-256, tensor shapes,
                                   LayerNorm eps=1e-5, RoPE convention
    ops.py                         shared ops: affine-free LN, interleaved RoPE,
                                   wide projection, weights fingerprint
    state.py                       AkashaState {S, C, position, segment_count,
                                   last_hidden, context_policy, segment_id},
                                   reset/clone/serialization schema
    reference_full.py              source-faithful level-major oracle, certified
                                   chunkwise scan + dense coordinator, debug tensors
    reference_recurrent.py         token-major recurrence (read-before-write,
                                   per-head decoder_y, coordinator mean, window reset)
  runtime/  model.py, session.py   prefill/decode API + generation sessions
  checkpoint/ loader.py, convert_arma.py, validate_real_checkpoint.py
  sampling/ sampler.py             greedy/multinomial sampler, cloneable RNG
  tokenizer/ adapter.py            hash-verified artifact adapter (never fabricates)
  bench/correctness.py             FP64 gates, FP32 parity suite, artifact writer
tests/akasha/                      53 tests (all passing)
campaigns/AKASHA_V0_REPORT.md
results/akasha/v0_{correctness,manifest,memory_contract,blockers,real_checkpoint}.json
```

Frozen semantics implemented exactly: `x = ReLU(v @ Dx)`, `q = RoPE(x, position)`,
strict-past read `a[h] = q[h]ᵀ S[l,h]` **before** `S[l,h] += outer(q[h], v)`,
`a = LN(a)`, `y = ReLU(a @ Dy[h])`, `u = x * y`, `base = LN(flat(u) @ E)`,
`z = v @ Wc + bc`, `c = C[l]/max(n,1) - z`, `g = 1 + sigmoid(alpha) tanh(c)`,
`C[l] += z`, `delta = ReLU((g·base) @ W1) @ W2`, `v = LN(v + delta)`, 8 shared levels,
`logits = v @ readout`.

## 3. Proof chain

1. **FP64 algebra gates (tiny: D=8, H=2, K=16, L=2, V=32, hidden=12)**
   - recurrent read-then-write vs strict-past dense attention: `5.33e-15`
   - recurrent coordinator prefix vs dense masked mean: `0.0`
   - token-major vs level-major logits: `0.0`; hidden states `4.34e-19`
   - chunked scan (`block=2,4,5,8,16`) vs dense: `0.0`
2. **Direct transcription against the trainer source** (`tests/akasha/test_trainer_transcription.py`)
   - `tensor.equal` on `rope_pair_freq`/`rope_phase` and `scan_chunkwise_candidate`
   - `OptArmA.forward_packed` at tiny dims vs Akasha full and recurrent: `<= 1e-10` (FP64)
3. **FP32 production parity (33 cases)** — full reference vs recurrent reference at
   lengths `1, 2, 7, 31, 32, 127, 128, 511, 512, 1024, 2048`, patterns
   random / repeated / multi-segment. Result: all within `atol/rtol`, argmax
   agreement `1.0`, `length_2048_pass = true`.
4. **Session semantics** — reset, clone, serialization (including `last_hidden`),
   sampler-RNG sidecar, prefill-next-token regression (`prefill_tokens` returns the
   logits of the last prompt token; the next operation must consume a new token).

## 4. Defects found by the gates (and fixed, not tolerated around)

- The dense coordinator used a wrong einsum contraction (`bts,btd->btd`) that
  multiplied the mask by the current token's `z` instead of summing previous `z`.
  Fixed to `bts,bsd->btd`; confirmed by the FP64 coordinator gate.
- The recurrent `y` projection applied `decoder_y[level]` to every head; the trainer's
  `decoder_y` is per-head `[H,D,K]`. Fixed to `einsum("hd,hdk->hk", a, decoder_y)`.
  This was caught only because the tiny gate uses `H=2`; it is a genuine semantics bug.
- The certified chunkwise scan identifies segments by *start index* (`seg < t0` carry
  logic). `reference_full` now translates arbitrary contiguous segment labels to start
  indices before chunked execution. A naive single-chunk carry formula remains
  insufficient across a reset; segmented prefill follows the trainer mask logic.

## 5. State / memory contract (`results/akasha/v0_memory_contract.json`)

| Item | Value |
|---|---|
| `S` elements | `33,554,432` (`8 × 4 × 4096 × 256`) |
| `C` elements | `2,048` (`8 × 256`) |
| FP32 bytes (S + C) | `134,225,920` |
| FP32 state | **`128.0078125 MiB`** |
| `last_hidden` | `256` elements (serialized, not part of the mathematical counter) |

Hardware correction: the production G4 is the **RTX PRO 6000 Blackwell Server Edition**
(96 GB GDDR7, **1597 GB/s**). Analytical state-only traffic floor (one read + one write
of 128 MiB per generated token) is `~5,949 token-steps/s`. This is an arithmetic bound,
**not a benchmark result and not a target**.

## 6. Artifact status

- **Tokenizer:** `BLOCKED_ARTIFACT_NOT_LOCAL`. Identity
  `bytelevel-bpe-8192-9a05bca4c065d995`, expected SHA-256
  `9a05bca4c065d9952a01ee1e3f4b6e23e39200e1f33da65e5dad1819419f71c3`. The adapter
  hashes and loads an artifact only when supplied; no equivalent tokenizer was
  fabricated and no text-prompt CLI exists yet. Token-ID inference is unaffected.
- **Trained checkpoint:** `BLOCKED` (not local). Converter, package loader, manifest
  validation and state-dict mapping are validated against source-compatible generated
  fixtures. One command is prepared for when the Drive checkpoint exists:

  ```powershell
  py -3.12 -m akasha.checkpoint.validate_real_checkpoint `
      --checkpoint "<...>/ckpt/latest.pt" `
      --out-dir results/akasha/real_checkpoint_package `
      --json-out results/akasha/v0_real_checkpoint.json
  ```

  Without an artifact it prints `TRAINED_CHECKPOINT_LOCAL_STATUS = BLOCKED` and exits 0.

## 7. How to reproduce

```powershell
# full gate suite (includes 1024/2048 fp32 parity; ~15 min CPU)
py -3.12 -m pytest tests/akasha -q

# artifacts (fp64 gates + 33-case fp32 parity + gate flags)
py -3.12 -m akasha.bench.correctness --out-dir results/akasha

# fast iteration without 1024/2048
py -3.12 -m pytest tests/akasha --skip-slow -q
```

Environment used: CPython 3.11/3.12, torch 2.10.0 CPU, pytest 8.4/9.0,
safetensors 0.6.2. Production runtime is torch 2.11.0+cu128 / CUDA 12.8 on sm_120;
V0 ran on CPU, and FP32 matmul precision is `highest` (no TF32).

## 8. Explicit non-goals (deferred until after V0 freeze)

DeltaLog, Triton kernels, chunkwise prefill kernels, indexed state pools, CUDA graphs,
CPU SIMD, BF16/FP16/FP8 state, serving adapters, speculative decoding, prefix radix
cache, multi-GPU. The first post-V0 task is an honest PyTorch recurrent profile
(tokens/s, state read/write cost, projection cost, launch count, VRAM, HBM counters).
`CONTINUOUS_EXPERIMENTAL` context remains marked UNVALIDATED MODEL EXTRAPOLATION;
`TRAINING_WINDOW` is the default policy and resets `S`/`C`/`segment_count` at the
2048-token boundary while `position` continues.

## 9. Unresolved blockers

1. Exact tokenizer artifact not local (does not block token-ID inference).
2. Final trained checkpoint not local (converter/validator prepared, not yet executed).
3. Production GPU benchmarks not run (explicitly out of V0 scope).
