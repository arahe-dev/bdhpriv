# E3 G4/Colab science cell (run on the GPU machine — not locally)

Status: **prepared, frozen, not executed.** The CPU analysis agent has no GPU
and no frozen corpus. This is the single minimal cell to run on the G4/Colab
machine after (or alongside) the G4 systems validation. It only performs
forward passes on a frozen checkpoint; it does not train and does not modify
`opt/`, `training/`, or the corpus.

The machine-readable contract is `results/arm_a_science/e3_contract.json`
(predeclared primary quantities and outcome bands; do not redesign after
seeing results).

---

## Prerequisites (one-time, on the G4 machine)

```bash
# repo with this analysis code
test -d /content/iclr-oc || git clone <this-repo-url> /content/iclr-oc
cd /content/iclr-oc
# checkpoint (adjust the path to where the G4 machine keeps the run)
CKPT="/content/drive/Shareddrives/ICLR PHASE BDH/phase_bdh/runs/arm_a_2p5b_opt3c_all/ckpt/latest.pt"
# frozen corpus root from context/frozen_corpus_contract.json
CORPUS="/content/drive/Shareddrives/ICLR PHASE BDH/phase_bdh/corpus/stage2/frozen_5b_v1"
```

## THE CELL (copy-paste, one block)

```bash
%%bash
set -euo pipefail
cd /content/iclr-oc
CKPT="${CKPT:-/content/drive/Shareddrives/ICLR PHASE BDH/phase_bdh/runs/arm_a_2p5b_opt3c_all/ckpt/latest.pt}"
CORPUS="${CORPUS:-/content/drive/Shareddrives/ICLR PHASE BDH/phase_bdh/corpus/stage2/frozen_5b_v1}"

# native context, tag v1 (E3-A/B/C primary)
python analysis/arm_a_science/e3_census.py \
  --ckpt-file "$CKPT" --corpus-root "$CORPUS" \
  --batches 8 --microbatch 8 --tag v1 --fast-verify

# context strata (cheap truncations of the same corpus rows)
python analysis/arm_a_science/e3_census.py \
  --ckpt-file "$CKPT" --corpus-root "$CORPUS" \
  --batches 8 --microbatch 8 --context 1024 --tag ctx1024 --fast-verify
python analysis/arm_a_science/e3_census.py \
  --ckpt-file "$CKPT" --corpus-root "$CORPUS" \
  --batches 8 --microbatch 8 --context 512 --tag ctx512 --fast-verify

# analysis -> mission-required outputs (v1 names unsuffixed, ctx names tagged)
python analysis/arm_a_science/e3_finalize.py --tag v1
python analysis/arm_a_science/e3_finalize.py --tag ctx1024
python analysis/arm_a_science/e3_finalize.py --tag ctx512

ls -la results/arm_a_science/e3_*.json results/arm_a_science/raw_frozen/meta_*.json
```

If Drive I/O is slow, run the three census commands in separate cells and
keep the machine awake; they are independent.

---

## Predeclared primary quantities (frozen before execution)

| id | quantity | E2 synthetic reference | outcome bands |
|---|---|---|---|
| A | `Delta_u(64)` per-cell mean | 0.421 (cells 0.19–0.52) | E3-A ≥0.30; E3-B 0.15–0.30; E3-C <0.15 |
| B | local/global top-N curves + delta(N) with doc-block CI | pooled u Δ: 0.220/0.416/0.374/0.112 at N=64/256/1024/4096 | qualitative shape |
| C | core mass ladder 1/3.125/6.25/12.5/25% | u core 0.294/0.480/0.614/0.753/0.879; 90% width min at 3.125–6.25% | E3-A core(6.25%) ≥0.50; B 0.35–0.50; C <0.35 |
| D | stability: same-doc lag 1–127 + long-range bins vs cross-doc | random-token lag1/cross = 1.37 (u); repeated-content 0.54/0.47/0.40 vs 0.09 shuffled | E3-A persistence signal; B weaker; C none |
| E | per-head RoPE-band statistic vs pair-preserving null (5000 draws) | x pooled z=+13.2, 32/32 cells z>3; init z=−0.2 | E3-A ≥24/32 cells z>3 and pooled z>5; B 12–23; C <12 |

Final label = primary band on A; B–E support bands and any contradicting
position/context stratum must be reported as measured. E3-A, E3-B and E3-C
are all acceptable outcomes.

## What to send back

- `results/arm_a_science/e3_topn_global_local.json`
- `results/arm_a_science/e3_population_stability.json`
- `results/arm_a_science/e3_core_tail.json`
- `results/arm_a_science/e3_frequency_bands.json`
- `results/arm_a_science/e3_frequency.json`
- the five `e3_summary_<tag>.json` files and the four
  `raw_frozen/meta_frozen_<tag>.json` files (checkpoint SHA-256, corpus
  verification mode, token counts, device)

## Notes

- `--fast-verify` skips per-file corpus SHA-256 (the contract/index digest is
  still checked). Drop it for the final authoritative run if Drive I/O allows.
- The model runs fp32 without autocast; do not compare bf16 activations to
  the E2 numbers.
- `e3_finalize.py` never clobbers the E2 artifacts and writes `_ctx1024` /
  `_ctx512` suffixes for strata runs.
- If `Delta_u(64)` lands in E3-C, freeze it and report the failure; do not
  iterate the metric.
