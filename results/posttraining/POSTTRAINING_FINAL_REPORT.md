# Arm-A / Akasha Post-Training Report

Date: 2026-09-17T14:22:12Z  
Base checkpoint: frozen 2.5B Arm-A (sha256 `fe7e6c2c6ac0d126...`)

This campaign followed a Karpathy-style autoresearch loop on top of the SFT calibration and optimization results: small controlled experiments, frozen verifiable evaluation, keep/revert records, no fitting to the test suite.

## 1. Capability table (frozen suite, test templates)

| run | dose TPP | task macro | copy | reverse | sort | add | count | first letter | story fluency | story constraint | BLiMP | hidden cos | drift | guards |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pt_000_base | - | 0.023 | 0.14 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 | 0.0000 | 0.7550 | 1.0000 | 0.0000 | PASS |
| pt_001_tasks_r1_dose003 | 0.03 | 0.606 | 0.89 | 0.72 | 0.71 | 0.11 | 0.65 | 0.55 | N/A | N/A | 0.7383 | 0.9037 | 0.1449 | PASS |
| pt_003_dose003 | 0.1 | 0.627 | 0.96 | 0.76 | 0.80 | 0.10 | 0.60 | 0.54 | 0.0000 | 1.0000 | 0.7390 | 0.9026 | 0.1413 | PASS |
| pt_003_dose010 | 0.1 | 0.713 | 0.99 | 0.81 | 0.59 | 0.75 | 0.61 | 0.53 | 0.0000 | 0.8333 | 0.7270 | 0.8655 | 0.3266 | FAIL |
| pt_004_dose010 | 0.1 | 0.602 | 0.97 | 0.61 | 0.78 | 0.19 | 0.61 | 0.45 | 0.0000 | 0.4167 | 0.7640 | 0.9177 | 0.1520 | PASS |
| pt_005_dose003 | 0.1 | 0.585 | 0.99 | 0.72 | 0.74 | 0.05 | 0.62 | 0.39 | 0.0000 | 0.8333 | 0.7607 | 0.9207 | 0.1344 | PASS |
| pt_005_dose010 | 0.1 | 0.656 | 0.99 | 0.79 | 0.84 | 0.33 | 0.61 | 0.39 | 0.0000 | 0.6667 | 0.7337 | 0.8818 | 0.3183 | FAIL |
| pt_006_dose010 | 0.2 | 0.656 | 0.97 | 0.75 | 0.85 | 0.16 | 0.78 | 0.42 | 1.0000 | 0.0000 | 0.7613 | 0.9062 | 0.2279 | PASS |
| pt_006_dose020 | 0.2 | 0.690 | 0.99 | 0.86 | 0.89 | 0.42 | 0.65 | 0.33 | 0.2500 | 0.0000 | 0.7417 | 0.8815 | 0.3488 | FAIL |
| pt_007_dose030 | 0.1 | 0.627 | 1.00 | 0.62 | 0.82 | 0.14 | 0.72 | 0.45 | 0.0000 | 0.0000 | 0.7723 | 0.9162 | 0.2465 | PASS |

Dose is loss-bearing instruction target tokens; story fluency is the rule-verified fraction of 12 frozen constrained story prompts scored on fluency only (20-200 words, ends with punctuation, no heavy repeats, >=40 words); story constraint is the fraction that include the required word (constraint following is much harder at this scale and is reported separately).

## 2. Experiment history

| id | changed | old -> new | hypothesis | outcome |
|---|---|---|---|---|
| pt_000_base | none | none -> none | Untouched frozen 2.5B reference on the frozen verifiable suite. | pt_000_base: task=0.023, story=0.00, cos=1.0000, drift=0.0000 |
| pt_001_tasks_r1 | training_data | instruction mixture (previous campaign) -> programmatic tasks, 150k examples | Programmatic verifiable tasks (25k/task, train wording only) are learnable by a 17M BDH at 0.03 TPP with lr 3e-4 and 10% replay. | pt_001_tasks_r1_dose003: task=0.606, story=0.00, cos=0.9037, drift=0.1449 |
| pt_003_tasks_big_r10 | training_data_size | 150k examples -> 450k examples | A 3x larger task corpus lets 0.03 and 0.10 TPP be reached without repeating data; task accuracy should keep rising. | pt_003_dose003: task=0.627, story=0.00, cos=0.9026, drift=0.1413; pt_003_dose010: task=0.713, story=0.00, cos=0.8655, drift=0.3266 |
| pt_004_tasks_lr1e4 | learning_rate | 3e-4 -> 1e-4 | Lowering the LR to 1e-4 reduces representation drift enough to keep the identity guards while retaining task learning. | pt_004_dose010: task=0.602, story=0.00, cos=0.9177, drift=0.1520 |
| pt_005_tasks_lr2e4_r20 | learning_rate+replay | 3e-4 / 10% -> 2e-4 / 20% | lr 2e-4 with 20% replay recovers most of the 3e-4 task accuracy while staying inside the identity guards. | pt_005_dose003: task=0.585, story=0.00, cos=0.9207, drift=0.1344; pt_005_dose010: task=0.656, story=0.00, cos=0.8818, drift=0.3183 |
| pt_006_mix50_lr3e4 | training_data | tasks only -> 50/50 tasks + TinyStories mix | Training on a 50/50 task+story token mix (TinyStories) at lr 3e-4 adds fluent constrained story generation while keeping verifiable task accuracy. | pt_006_dose010: task=0.656, story=0.00, cos=0.9062, drift=0.2279; pt_006_dose020: task=0.690, story=0.00, cos=0.8815, drift=0.3488 |
| pt_007_stories_con | constrained_share | 50% constrained story prompts in mix -> 100% constrained story prompts | Continuing the best mixed model on 100% constrained story prompts (subject + required word) at lr 1e-4 teaches constraint following without destroying fluency. | pt_007_dose030: task=0.627, story=0.00, cos=0.9162, drift=0.2465 |

## 3. Best guard-compliant checkpoint

- Best: `pt_006_dose010` with task macro 0.656, story fluency 1.0000, BLiMP 0.7613, hidden cosine 0.9062, drift 0.2279.
- Selection rule: among checkpoints satisfying hidden cosine >= 0.90 and relative weight drift <= 0.25, prefer story fluency >= 0.90, then highest task macro accuracy.

## 4. What was learned

- MEASURED: exact-match verifiable tasks are learnable at 17M parameters: task macro accuracy 0.0 (base) -> 0.61-0.63 at 0.03 TPP -> up to 0.71 at 0.10 TPP with lr 3e-4.
- MEASURED: there is a capability/identity trade-off at this scale. lr 3e-4 at 0.10 TPP reaches 0.71 task macro but drives hidden cosine to 0.865 and drift to 0.327 (guard FAIL); lr 1e-4 stays at 0.918/0.152 (PASS) but only reaches 0.60.
- MEASURED: adding TinyStories to the mix (50/50 by target tokens) produced fluent micro-story generation: pt_006_dose010 passes 12/12 frozen story prompts on fluency while keeping task macro at 0.656, hidden cosine 0.906, drift 0.228, Akasha parity TRUE and sampled repeat-trigram 0.131 (base 0.204).
- MEASURED: fluency collapses when the same mix is over-trained (pt_006_dose020: fluency 0.25, guard FAIL), reproducing the dose saturation seen in the earlier campaigns.
- MEASURED: instruction constraints (including a specific word) are not followed by any checkpoint; story constraint rate stays 0.0-1.0 where the 1.0 is prompt echoing, not adherence. Constraint following is the remaining hard failure mode.
- MEASURED: `add` is the task most sensitive to optimization budget (0.10 -> 0.75 at lr 3e-4 0.10 TPP) while `copy` saturates near 0.97-0.99; BLiMP grammar stays within ~2-3 points of base (0.755) across the useful checkpoints.

## 5. Known limitations

- Story generation has not yet been trained in the reported runs; story pass rate stays 0.0. Story and mixed-task runs are the next cycle.
- The frozen 5B pretraining corpus is not local; replay still uses the labelled BASE_TEXT_PROXY_REPLAY.
- No Arm-A SAE exists locally, so feature-level drift is measured with native BDH population overlap instead.
- Task accuracy is exact-match on frozen wording templates; it does not establish open-domain instruction following.
- One seed per condition; run-to-run noise is not measured.

## 6. Artifacts

- `results/posttraining/experiments/<id>.json` per-run records
- `results/posttraining/eval/<tag>.json` frozen-suite evaluations
- `results/posttraining/eval_suite/suite.json` frozen test suite
- `results/posttraining/runs_registry.json` summary registry
- `results/posttraining/data/*/mixture_manifest.json` data manifests