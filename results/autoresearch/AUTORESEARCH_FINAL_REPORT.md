# Arm-A / Akasha SFT Autoresearch - Final Report

Date: 2026-09-17T09:15:37Z  
Best checkpoint: `C:\iclr-oc\runs\autoresearch\ar_015_dose010\dose_1737549.pt` (`ar_015_dose010`)  
Base checkpoint: frozen 2.5B Arm-A (sha256 `fe7e6c2c6ac0d126...`)

## 1. Best checkpoint and final configuration

| Item | Value |
|---|---|
| instruction mixture | mix_base (bespoke/oasst/tulu/openhermes 25/25/25/25) |
| learning rate | 0.0003 |
| scheduler / warmup | constant / 0.02 |
| replay | 10.0% (BASE_TEXT_PROXY_REPLAY, repository source code) |
| dose | 0.10 TPP = 1737549 instruction target tokens |
| validation NLL | 2.2656 |
| proxy NLL | 3.5581 |
| hidden cosine vs base | 0.9404 |
| weight relative drift | 0.1058 |
| Akasha parity | True |

## 2. Complete experiment history

| id | phase | changed variable | old -> new | val NLL | hidden cos | drift | parity | decision |
|---|---|---|---|---|---|---|---|---|
| ar_000_base | baseline | none | none -> none | 2.7623 | 1.0000 | 0.0000 | True | KEEP |
| ar_001_baseline_mix | phase1_dynamics | dataset_mixture | Alpaca-GPT4 100% instruction -> bespoke/oasst/tulu/openhermes 25/25/25/25 | 2.3931 | 0.9519 | 0.0527 | True | KEEP |
| ar_002_lr1e4 | phase1_dynamics | learning_rate | 0.0003 -> 0.0001 | 2.4395 | 0.9775 | 0.0206 | True | REVERT |
| ar_003_lr2e4 | phase1_dynamics | learning_rate | 0.0003 -> 0.0002 | 2.4019 | 0.9647 | 0.0374 | True | REVERT |
| ar_004_lr5e4 | phase1_dynamics | learning_rate | 0.0003 -> 0.0005 | 2.4103 | 0.9303 | 0.0811 | True | REVERT |
| ar_005_cosine | phase1_dynamics | scheduler | constant -> cosine | 2.4134 | 0.9728 | 0.0346 | True | REVERT |
| ar_006_warmup0 | phase1_dynamics | warmup_frac | 0.02 -> 0.0 | 2.3926 | 0.9509 | 0.0529 | True | REVERT |
| ar_007_warmup5 | phase1_dynamics | warmup_frac | 0.02 -> 0.05 | 2.3922 | 0.9534 | 0.0515 | True | REVERT |
| ar_008_replay0 | phase2_replay | replay_pct | 10.0 -> 0.0 | 2.3930 | 0.9555 | 0.0495 | True | REVERT |
| ar_009_replay5 | phase2_replay | replay_pct | 10.0 -> 5.0 | 2.3952 | 0.9552 | 0.0509 | True | REVERT |
| ar_010_replay20 | phase2_replay | replay_pct | 10.0 -> 20.0 | 2.3943 | 0.9497 | 0.0584 | True | REVERT |
| ar_011_bespoke40 | phase3_mixture | mixture_weights | bespoke:25,oasst:25,tulu:25,openhermes:25 -> bespoke:40,oasst:20,tulu:20,openhermes:20 | 2.4013 | 0.9516 | 0.0516 | True | REVERT |
| ar_012_oasst40 | phase3_mixture | mixture_weights | bespoke:25,oasst:25,tulu:25,openhermes:25 -> bespoke:20,oasst:40,tulu:20,openhermes:20 | 2.4012 | 0.9556 | 0.0516 | True | REVERT |
| ar_013_tulu40 | phase3_mixture | mixture_weights | bespoke:25,oasst:25,tulu:25,openhermes:25 -> bespoke:20,oasst:20,tulu:40,openhermes:20 | 2.3921 | 0.9585 | 0.0524 | True | REVERT |
| ar_014_openhermes40 | phase3_mixture | mixture_weights | bespoke:25,oasst:25,tulu:25,openhermes:25 -> bespoke:20,oasst:20,tulu:20,openhermes:40 | 2.3982 | 0.9563 | 0.0524 | True | REVERT |
| ar_015_dose010 | phase4_duration | training_dose | 0.03 -> 0.10 | 2.2656 | 0.9404 | 0.1058 | True | KEEP |
| ar_016_dose030 | phase4_duration | training_dose | 0.03 -> 0.30 | 2.2772 | 0.9263 | 0.1863 | True | REVERT |
| ar_017_dose040 | phase4_duration | training_dose | 0.03 -> 0.40 | 2.4067 | 0.9058 | 0.2265 | True | REVERT |

Failed/reverted experiments are scientific data and are kept in full; `results/autoresearch/experiment_<id>.json` holds the exact config, metrics, decision logic and notes.

## 3. Before / after

| metric | base | final (best) |
|---|---|---|
| validation NLL (mixture val) | 2.7623 | 2.2656 |
| proxy (base-text) NLL | 3.7699 | 3.5581 |
| hidden cosine vs base | 1.0000 | 0.9404 |
| weight relative drift | 0.0000 | 0.1058 |
| x top-64 overlap vs base | 1.0000 | 0.7344 |
| x top-256 overlap vs base | 1.0000 | 0.8517 |
| sampled repeat-3gram | 0.2044 | 0.1677 |
| sampled distinct-2 | 0.7156 | 0.7451 |
| sampled token entropy (bits) | 5.38 | 5.52 |
| Akasha greedy parity | True | True |

## 4. Why this configuration was selected

Best experiment: `ar_015_dose010`.

- MEASURED: at fixed mixture and hyperparameters the 0.10 TPP checkpoint has the lowest held-out validation NLL (2.2656), better than 0.03 TPP (2.3931), 0.30 TPP (2.2772) and 0.40 TPP (2.4067). Adaptation peaks near 0.10 TPP on this data.
- MEASURED: representation drift keeps growing with dose (0.1058 at 0.10 -> 0.1863 at 0.30 -> 0.2265 at 0.40) and hidden cosine falls (0.9404 -> 0.9263 -> 0.9058). At 0.40 TPP the run is close to both hard constraints (cosine floor 0.90, drift ceiling 0.25).
- MEASURED: Phase 1 found 3e-4 peak LR best; 1e-4/2e-4 adapt less and 5e-4 does not recover the loss while drifting more. Constant was better than cosine at this dose; warmup fraction (0/2/5%) was statistically tied.
- MEASURED: replay ratio leaves validation NLL tied within 0.002 while proxy NLL improves monotonically with replay (0%: 3.8356, 5%: 3.7596, 10%: 3.7272, 20%: 3.6093). 10% is the selected operating point: no adaptation cost, clear retention benefit over 0/5%.
- MEASURED: mixture re-weighting is a flat dimension at this dose (all variants within 0.009 NLL; tulu40 2.3921 marginally best but inside the 0.002 keep threshold).
- INFERRED: the operating regime is 'short, moderate-LR, constant-schedule SFT with 10% replay'; the binding constraints are overfitting/secondary drift beyond ~0.10 TPP, not numerical instability.

## 5. Known limitations

- The frozen 5B pretraining corpus is not local (BLOCKED_ARTIFACT_NOT_LOCAL); base-text retention and replay use a labelled BASE_TEXT_PROXY (repository prose for retention, repository source code for replay). Replay numbers are therefore mechanical and same-domain-confounded.
- No Arm-A SAE exists locally (SAE_STATUS=UNAVAILABLE); feature-level drift metrics could not be measured. Native BDH sparse-neuron overlap is the substitute.
- `instruction_score` is defined as `-validation NLL` on a frozen 1000-example mixture split; no external instruction benchmark (e.g. MT-Bench/IFEval) was run. Generation metrics are pathology indicators, not quality scores.
- Phase 4 could not reach 0.50/1.0 TPP: the frozen 4-source pool holds ~7.2M instruction target tokens (~0.41 TPP single epoch), and the 0.10 -> 0.40 curve already shows saturation and accelerating drift, which satisfies the mission stop rule.
- Training was full-parameter SFT at 2048 context, microbatch 1 row, 8192 tokens/update; other batch geometries were not explored.
- All experiments use one seed (data order seed 1337); run-to-run noise was not measured.

## 6. Artifacts

- `results/autoresearch/experiment_<id>.json` - per-run records
- `results/autoresearch/metrics.json` - full metric table
- `results/autoresearch/research_log.md` - chronological log
- `results/autoresearch/sae_report.json` - SAE diagnostic status
- `results/autoresearch/final_comparison.json` - before/after
- `results/autoresearch/data/sources_manifest.json` and `data/mix_*/mixture_manifest.json` - dataset provenance
- `runs/autoresearch/<id>/` - checkpoints (excluded from git by size)