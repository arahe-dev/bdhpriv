# Arm-A SFT Calibration Campaign (NOT the final SFT)

Date: 2026-09-17T06:10:44Z  
Git HEAD: `2080aa6bbf06e08400e7abafa8bb11fba74b0365`  
Frozen architecture source: `training/arm_a_2p5b_trainer.py` @ `0dcbb87` (SHA-256 `1985fa42042033c8...`)

This campaign is a small, controlled dose-response SFT experiment on the real 17,375,489-parameter Arm-A BDH. It is **not** the final SFT run; no 100M-token run was launched and no architecture, Akasha, or frozen-checkpoint change was made.

---
## 1. Environment and pins

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (8.00 GiB) |
| PyTorch / CUDA | 2.10.0+cu128 / 12.8 |
| Base checkpoint | `C:\iclr-oc\runs\arm_a_2p5b_opt3c_all\ckpt\latest.pt` |
| Base checkpoint SHA-256 | `fe7e6c2c6ac0d12630018812f735486c45003678adbfcb4a88a476d886c47efb` |
| Base training tokens / updates | 2500067328 / 19074 |
| Tokenizer | `bytelevel-bpe-8192-9a05bca4c065d995` SHA-256 `9a05bca4c065d995...` (verified) |
| Parameter count | 17375489 |

## 2. Data

- DATASET_NAME: Alpaca-GPT4 (vicgalle/alpaca-gpt4)
- DATASET_SOURCE: https://huggingface.co/datasets/vicgalle/alpaca-gpt4
- DATASET_REVISION: `f7e3ded725cb81e8e564e32feb12860f376f2b51`
- DATASET_FILE_SHA256: `bdd9b3f1aa3688ee2015550974c3a14b27fec20e4cfb459fd6800dc14030b9e6`
- RAW_EXAMPLE_COUNT: 52002
- TRAIN_EXAMPLE_COUNT: 51002 kept / 51002 raw
- VALIDATION_EXAMPLE_COUNT: 1000 kept
- TARGET_TOKEN_COUNT (train, packed): 9716324
- TOTAL_TOKEN_COUNT (train, packed): 11416416
- Deterministic split seed: 1337; shuffle once, never reshuffled between arms.
- Truncated responses: 0 train / 0 validation; skipped over-long prompts: 0 train.

BASE_CORPUS_RETENTION_STATUS = `BLOCKED_ARTIFACT_NOT_LOCAL`. The frozen `phase_bdh_stage2_5b_v1` corpus is not present on this machine, so retention and replay use a clearly labelled **BASE_TEXT_PROXY** (43 windows, 63507 target tokens of repository prose).

## 3. Method

- Full-parameter SFT, no LoRA, no new special tokens.
- Format: `User: <instruction>\n[Input: <input>\n]Assistant: <response>\n`; loss on response + terminating newline only; prompt and response share one attention segment, examples never attend to one another.
- Frozen model semantics: same `OptArmA`, scan, coordinator, writer, LayerNorm, RoPE, BF16 autocast + FP32 master AdamW (betas 0.9/0.95, eps 1e-8, wd 0.1, clip 1.0), `one_full_update`.
- `torch.compile(mode="default")` used; full inductor cache hits after the first arm.
- MICROBATCH=1 row (2048 tokens) per forward, 4 rows accumulated per optimizer update = 8192 sequence tokens/update; constant LR after a 20-update linear warmup.

## 4. Final checkpoint table

| ARM | LR | REPLAY_PCT | SFT_TPP | TARGET_TOKENS | TOTAL_TOKENS | OPT_STEPS | WALL_TIME_S | SFT_VAL_NLL | BASE_LM_NLL | BASE_LM_DELTA_PCT | REPEAT_3GRAM | DISTINCT_2 | DISTINCT_3 | HIDDEN_COSINE_VS_BASE | TOP64_OVERLAP_VS_BASE | TOP256_OVERLAP_VS_BASE | SAE_COSINE_VS_BASE | TOTAL_RELATIVE_WEIGHT_DRIFT | AKASHA_PARITY |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | N/A | N/A | 0.00 | 0 | 2500067328 | 19074 | N/A | 2.6642 | 3.7699 | 0.00 | 0.2044 | 0.7156 | 0.7956 | 1.00000 | 1.0000 | 1.0000 | N/A (SAE_STATUS=UNAVAILABLE) | N/A | True |
| a1_3e5 | 3e-05 | 0 | 0.01 | 175522 | 207677 | 28 | 22.3 | 2.5590 | 3.7857 | 0.42 | 0.2159 | 0.6988 | 0.7841 | 0.99228 | 0.9287 | 0.9563 | N/A (SAE_STATUS=UNAVAILABLE) | N/A | True |
| a1_3e5 | 3e-05 | 0 | 0.03 | 522619 | 619231 | 83 | 60.5 | 2.4931 | 3.7645 | -0.14 | 0.2014 | 0.7098 | 0.7986 | 0.98951 | 0.9121 | 0.9460 | N/A (SAE_STATUS=UNAVAILABLE) | 0.0055 | True |
| a2_1e4 | 0.0001 | 0 | 0.01 | 175522 | 207677 | 28 | 22.3 | 2.5091 | 3.7838 | 0.37 | 0.1460 | 0.7555 | 0.8540 | 0.98728 | 0.9116 | 0.9473 | N/A (SAE_STATUS=UNAVAILABLE) | N/A | True |
| a2_1e4 | 0.0001 | 0 | 0.03 | 522619 | 619231 | 83 | 60.5 | 2.4328 | 3.7801 | 0.27 | 0.1724 | 0.7374 | 0.8276 | 0.98126 | 0.8789 | 0.9294 | N/A (SAE_STATUS=UNAVAILABLE) | 0.0154 | True |
| a3_3e4 | 0.0003 | 0 | 0.01 | 175522 | 207677 | 28 | 22.3 | 2.4758 | 3.8211 | 1.36 | 0.1740 | 0.7390 | 0.8260 | 0.97551 | 0.8745 | 0.9238 | N/A (SAE_STATUS=UNAVAILABLE) | N/A | True |
| a3_3e4 | 0.0003 | 0 | 0.03 | 522619 | 619231 | 83 | 60.5 | 2.3924 | 3.8294 | 1.58 | 0.1962 | 0.7126 | 0.8038 | 0.96837 | 0.8394 | 0.9061 | N/A (SAE_STATUS=UNAVAILABLE) | 0.0396 | True |
| b1_rep0 | 0.0003 | 0 | 0.03 | 522619 | 619231 | 83 | 60.5 | 2.3932 | 3.8263 | 1.50 | 0.1639 | 0.7478 | 0.8361 | 0.96584 | 0.8320 | 0.9017 | N/A (SAE_STATUS=UNAVAILABLE) | N/A | True |
| b1_rep0 | 0.0003 | 0 | 0.10 | 1739241 | 2047510 | 275 | 193.9 | 2.2912 | 3.8949 | 3.31 | 0.1808 | 0.7337 | 0.8192 | 0.95095 | 0.7925 | 0.8761 | N/A (SAE_STATUS=UNAVAILABLE) | 0.0879 | True |
| b2_rep10 | 0.0003 | 10 | 0.03 | 522619 | 673468 | 92 | 66.8 | 2.3965 | 3.7005 | -1.84 | 0.2133 | 0.7004 | 0.7867 | 0.96739 | 0.8257 | 0.8943 | N/A (SAE_STATUS=UNAVAILABLE) | N/A | True |
| b2_rep10 | 0.0003 | 10 | 0.10 | 1742478 | 2243954 | 306 | 215.4 | 2.2964 | 3.6174 | -4.04 | 0.1724 | 0.7390 | 0.8276 | 0.94912 | 0.7896 | 0.8749 | N/A (SAE_STATUS=UNAVAILABLE) | 0.0917 | True |
| c_dose030_from_b2_rep10 | 0.0003 | 10 | 0.30 | 5213863 | 4463831 | 912 | 423.5 | 2.1952 | 3.5166 | -6.72 | 0.1710 | 0.7423 | 0.8290 | 0.91807 | 0.7104 | 0.8297 | N/A (SAE_STATUS=UNAVAILABLE) | 0.1812 | True |

`REPEAT_3GRAM`, `DISTINCT_2`, `DISTINCT_3` are sampled-decoding suite means (temperature 0.8, top_k 50, seed 20260916). `TOTAL_TOKENS` counts processed sequence tokens (prompt + response + replay); `TARGET_TOKENS` counts loss-bearing instruction tokens.

WALL_TIME_S is the training wall time of that process; the base row was trained on the remote RTX PRO 6000 G4 runtime, not measured locally. The 0.30 row continued from the 0.10 checkpoint: its TARGET_TOKENS/OPT_STEPS (3,475,098 additional target tokens, 912 total trajectory steps) and SFT_TPP are cumulative for the trajectory, while TOTAL_TOKENS and WALL_TIME_S are incremental for that process. Measured local throughput: 10464 sequence tok/s; campaign GPU training time 1014 s (pre-campaign estimate 3518 s including evaluation; actual total under 3 h).

## 5. Phase A - learning-rate micro-sweep (to 0.03 TPP)

| tag | lr | sft_val_nll | proxy_nll | base_lm_delta_pct | repeat_3gram | hidden_cosine_vs_base | total_relative_weight_drift |
|---|---|---|---|---|---|---|---|
| a1_3e5_dose001 | 0.0000 | 2.5590 | 3.7857 | 0.4201 | 0.2159 | 0.9923 | N/A |
| a1_3e5_dose003 | 0.0000 | 2.4931 | 3.7645 | -0.1431 | 0.2014 | 0.9895 | 0.0055 |
| a2_1e4_dose001 | 0.0001 | 2.5091 | 3.7838 | 0.3681 | 0.1460 | 0.9873 | N/A |
| a2_1e4_dose003 | 0.0001 | 2.4328 | 3.7801 | 0.2712 | 0.1724 | 0.9813 | 0.0154 |
| a3_3e4_dose001 | 0.0003 | 2.4758 | 3.8211 | 1.3583 | 0.1740 | 0.9755 | N/A |
| a3_3e4_dose003 | 0.0003 | 2.3924 | 3.8294 | 1.5784 | 0.1962 | 0.9684 | 0.0396 |

**SELECTED_LR = 0.0003** (a3_3e4).

SELECTION_REASON: MEASURED: lowest held-out SFT NLL at 0.03 TPP among numerically stable arms with non-pathological generation.

## 6. Phase B - replay test at the selected LR (to 0.10 TPP)

| run | SFT_VAL_NLL | proxy_NLL | retention_penalty | repeat_3gram | hidden_cos | top64_overlap | total_drift | parity |
|---|---|---|---|---|---|---|---|---|
| b1_rep0 | 2.2912 | 3.8949 | 0.1250 | 0.1808 | 0.9509 | 0.7925 | 0.0879 | True |
| b2_rep10 | 2.2964 | 3.6174 | -0.1525 | 0.1724 | 0.9491 | 0.7896 | 0.0917 | True |

REPLAY_SOURCE: BASE_TEXT_PROXY (repository prose). The frozen 5B corpus is BLOCKED_ARTIFACT_NOT_LOCAL, so original-corpus replay could not be used; this is an explicit approximation.

**SELECTED_REPLAY = 10%**

SELECTION_REASON: MEASURED: 10% replay reduces the base-proxy penalty by 222.0% while held-out SFT NLL is within 0.23% of the 0% run; 10% replay selected.

## 7. Phase C - dose extension to 0.30 TPP

- LR: 0.0003; replay: 10.0%
- Continued from: runs/sft_probe/b2_rep10/dose_1737549.pt
- SFT_VAL_NLL: 2.195177578398706; proxy_NLL: 3.516636036354137 (-6.718169641239348% vs base)
- repeat_3gram: 0.17103174603174603; distinct_2: 0.7423228346456693; distinct_3: 0.8289682539682539
- hidden_cosine_vs_base: 0.9180700301161442; top64_overlap: 0.71044921875; total_relative_weight_drift: 0.18122466381515648

DERIVED dose-response shape (held-out SFT NLL; note the 0.01/0.03 rows are 0% replay while 0.10/0.30 are the selected 10% replay recipe, so this is a trend, not a clean ablation):

| stage | SFT_TPP | SFT_VAL_NLL | marginal NLL/TPP |
|---|---|---|---|
| base | 0.0 | 2.6642 | - |
| selected 0% replay 0.01 | 0.01 | 2.4758 | 18.84 |
| selected 10% replay 0.03 | 0.03 | 2.3924 | 4.17 |
| selected 10% replay 0.10 | 0.1 | 2.2964 | 1.37 |
| selected 10% replay 0.30 | 0.3 | 2.1952 | 0.51 |

RISK NOTE (INFERRED): replay rows are repository source code while the retention proxy is repository prose, so the negative BASE_LM_DELTA_PCT of the replay arms overstates pure anti-forgetting. The 0% replay arms show the unconfounded direction: proxy loss worsens with dose.

## 8. Akasha correctness

- a1_3e5_dose003: FULL_VS_RECURRENT_GREEDY_MATCH=True, FIRST_DIVERGENCE_TOKEN=None
- a2_1e4_dose003: FULL_VS_RECURRENT_GREEDY_MATCH=True, FIRST_DIVERGENCE_TOKEN=None
- a3_3e4_dose003: FULL_VS_RECURRENT_GREEDY_MATCH=True, FIRST_DIVERGENCE_TOKEN=None
- b1_rep0_dose010: FULL_VS_RECURRENT_GREEDY_MATCH=True, FIRST_DIVERGENCE_TOKEN=None
- b2_rep10_dose010: FULL_VS_RECURRENT_GREEDY_MATCH=True, FIRST_DIVERGENCE_TOKEN=None
- base: FULL_VS_RECURRENT_GREEDY_MATCH=True, FIRST_DIVERGENCE_TOKEN=None
- c_dose030: FULL_VS_RECURRENT_GREEDY_MATCH=True, FIRST_DIVERGENCE_TOKEN=None

Raw generation artifacts (exact token ids, decoded text and pathology diagnostics for all 40 prompts, greedy + sampled) are saved per checkpoint under `results/sft_probe/generations/<tag>.json`; aggregate diagnostics are in `results/sft_probe/generation_metrics.json`.

## 9. Blocker

- BASE_CORPUS_RETENTION_STATUS = BLOCKED_ARTIFACT_NOT_LOCAL (proxy used; see blockers.json)
- SAE_STATUS = UNAVAILABLE (no frozen Arm-A SAE; native BDH sparse-neuron probes substituted)
- An earlier Phase B 10%-replay run was invalidated because its replay text overlapped the retention proxy (proxy NLL collapsed by 35%). It was quarantined under results/sft_probe/_invalidated and rerun with disjoint replay text; only the clean run is reported.

## 10. Headline fields

```
BASE_CHECKPOINT = C:\iclr-oc\runs\arm_a_2p5b_opt3c_all\ckpt\latest.pt sha256=fe7e6c2c6ac0d126... tokens=2500067328 updates=19074
TOKENIZER_VERIFIED = true (bytelevel-bpe-8192-9a05bca4c065d995 sha256=9a05bca4c065d995...)
DATASET = Alpaca-GPT4 vicgalle/alpaca-gpt4 rev f7e3ded725cb81e8e564e32feb12860f376f2b51 sha256=bdd9b3f1aa3688ee...
GPU = NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GiB)
PARAMETERS = 17375489
BASE_TRAINING_TOKENS = 2500067328
PHASE_A_COMPLETE = true
LR_3E5_RESULT = SFT_val_NLL=2.4931, proxy_NLL=3.7645 (delta -0.14%), repeat_3gram=0.201, hidden_cos=0.9895
LR_1E4_RESULT = SFT_val_NLL=2.4328, proxy_NLL=3.7801 (delta +0.27%), repeat_3gram=0.172, hidden_cos=0.9813
LR_3E4_RESULT = SFT_val_NLL=2.3924, proxy_NLL=3.8294 (delta +1.58%), repeat_3gram=0.196, hidden_cos=0.9684
SELECTED_LR = 0.0003
PHASE_B_COMPLETE = true
REPLAY_0_RESULT = SFT_val_NLL=2.2912, proxy_NLL=3.8949 (delta +3.31%), repeat_3gram=0.181, hidden_cos=0.9509
REPLAY_10_RESULT = SFT_val_NLL=2.2964, proxy_NLL=3.6174 (delta -4.04%), repeat_3gram=0.172, hidden_cos=0.9491
SELECTED_REPLAY = 10%
PHASE_C_COMPLETE = true
DOSE_001_RESULT = SFT_val_NLL=2.4758, proxy_NLL=3.8211 (delta +1.36%), repeat_3gram=0.174, hidden_cos=0.9755
DOSE_003_RESULT = SFT_val_NLL=2.3924, proxy_NLL=3.8294 (delta +1.58%), repeat_3gram=0.196, hidden_cos=0.9684
DOSE_010_RESULT = SFT_val_NLL=2.2964, proxy_NLL=3.6174 (delta -4.04%), repeat_3gram=0.172, hidden_cos=0.9491
DOSE_030_RESULT = SFT_val_NLL=2.1952, proxy_NLL=3.5166 (delta -6.72%), repeat_3gram=0.171, hidden_cos=0.9181
BASE_RETENTION_TREND = MEASURED: with 0% replay the proxy penalty grows with dose (+1.58% at 0.03, +3.31% at 0.10). With 10% replay the proxy loss is below base at 0.10 and 0.30 (-4.04% / -6.72%); INFERRED with a confound: the replay text is repository source code and the retention proxy is repository prose, so part of this retention benefit is likely same-domain transfer rather than pure anti-forgetting.
INSTRUCTION_ADAPTATION_TREND = MEASURED: held-out SFT NLL keeps falling through 0.30 TPP: base 2.6642 -> 2.4758 at 0.01 -> 2.3924 at 0.03 -> 2.2964 at 0.10 -> 2.1952 at 0.30. No saturation was observed (marginal gain per TPP decelerates ~8x from the 0.01-0.03 segment to the 0.10-0.30 segment but stays positive).
REPETITION_TREND = MEASURED: sampled repeated-trigram fraction is flat-to-better than base (base 0.204 -> 0.196 at 0.03 -> 0.172 at 0.10 -> 0.171 at 0.30); greedy repetition drops (0.719 -> 0.468) and no constant-token or long-loop collapse appears (generation_metrics.json).
NATIVE_BDH_DRIFT_TREND = MEASURED: hidden cosine vs base falls with dose (0.968 at 0.03 -> 0.949 at 0.10 -> 0.918 at 0.30); x top-64 overlap falls 0.839 -> 0.790 -> 0.710; native sparsity is preserved (native_bdh_drift.json).
SAE_STATUS = UNAVAILABLE
SAE_DRIFT_TREND = N/A: no frozen Arm-A SAE exists locally; native BDH probes used instead.
AKASHA_PARITY = true
MEASURED_LOCAL_TRAIN_TOK_S = 10464.3 sequence tok/s
TOTAL_CAMPAIGN_GPU_TIME = 1014.2 s
EVIDENCE_SUPPORTED_MINIMUM_SFT_TPP = 0.01
EVIDENCE_SUPPORTED_SATURATION_POINT = >0.30 (not observed)
EVIDENCE_SUPPORTED_MAX_SAFE_TPP = 0.3
RECOMMENDED_FINAL_SFT_DOSE = 0.1
RECOMMENDED_FINAL_SFT_LR = 0.0003
RECOMMENDED_REPLAY_PCT = 10%
```

## 11. Provisional recommendations (do not launch this)

- RECOMMENDED_FINAL_SFT_LR: 0.0003
- RECOMMENDED_FINAL_SFT_DOSE: 0.1 TPP (= 1,737,549 loss-bearing instruction target tokens) if the cost/benefit point is chosen, or up to 0.30 TPP if instruction fidelity dominates; the curve had not flattened at 0.30 TPP
- RECOMMENDED_REPLAY_PCT: 10%
- Basis:
  - MEASURED: held-out SFT NLL improves strongly already at 0.01 TPP (2.4758 vs base 2.6642); 0.01 TPP is the evidence-supported minimum.
  - MEASURED: the held-out SFT NLL curve is still descending at 0.30 TPP (2.1952); marginal gain per TPP is 4.2 (0.01-0.03), 1.4 (0.03-0.10) and 0.5 (0.10-0.30) NLL per TPP. Returns decelerate ~8x but do NOT flatten by 0.30 TPP; saturation is INFERRED to lie above 0.30.
  - MEASURED: 0.30 TPP completed with the selected 10% replay recipe without NaN/Inf weights or losses, with non-pathological sampled generation and the lowest SFT val NLL. Safety beyond 0.30 TPP and at 0% replay beyond 0.10 TPP was not measured.
  - INFERRED: 0.10 TPP captures most of the fast adaptation phase (SFT NLL 2.2964) with materially less representation drift than 0.30 TPP; 0.30 TPP adds instruction fit at ~1/8 the per-token rate and shows the first degenerate generation cases. The final recipe should use 0.10 TPP as the cost/benefit point and may extend to 0.30 TPP only if instruction fidelity dominates retention.
  - MEASURED: 0.30 TPP proxy delta -6.72% and repeat_3gram 0.171 (10% replay recipe).

These are provisional empirical recommendations from THIS small campaign only. They are not a validated final recipe and this campaign does not launch one.

## 12. Evidence labels

- MEASURED: all NLLs, overlaps, drift norms and generation diagnostics in the tables above come from saved artifacts under `results/sft_probe/`.
- DERIVED: SFT_TPP = target tokens / 17,375,489; percentage deltas are arithmetic on measured values.
- INFERRED: phase selections and the provisional dose recommendations; these combine several measurements.
- SPECULATIVE: anything about larger token budgets, other data mixtures, or final-recipe behaviour is not claimed.

## 13. Stop notice

This campaign intentionally stops at 0.30 TPP. The final SFT recipe must be designed from these measurements; do not launch 100M tokens or any larger run from this report.

## 14. Reproduce

```
python -m training.sft_probe.prepare_data
python -m training.sft_probe.run_campaign --phase all
```

Stages are resumable and skip existing artifacts; `--force` recomputes. Every training and evaluation process re-verifies the frozen trainer, tokenizer and base-checkpoint hashes.