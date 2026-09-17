# E3 frozen-corpus census package (PREPARED — not executed)

Status: **prepared, not run.** The frozen corpus
(`phase_bdh_stage2_5b_v1`, see `context/frozen_corpus_contract.json`) is not
present on the local CPU-only analysis machine, so every Arm-A community
result in this campaign is E1/E2 (synthetic packed batches). This package is
the explicit upgrade path to **E3** evidence.

## What it does

`e3_census.py` streams the frozen packed corpus exactly like the production
trainer (same `FrozenPackedCorpus`, same hash verification) and runs the same
pass-1/pass-2 capture used by `collect.py`:

- pass 1: per-(batch, level, head) coordinate mass / positive counts over ALL
  streamed tokens, greedy per-token top-N ladder ratios, top-64 identity
  sets, RoPE-pair masses, band statistics;
- pass 2: cross-fitted global top-N, core/tail metrics (with residual
  quantiles 50/80/90%), Tier-B stability on the same streamed batches.

Outputs (per tag):

```
results/arm_a_science/raw_frozen/pass1_frozen_<tag>.npz
results/arm_a_science/raw_frozen/pass2_frozen_<tag>.npz
results/arm_a_science/raw_frozen/ranks_frozen_<tag>.npz
results/arm_a_science/raw_frozen/meta_frozen_<tag>.json
```

Analyze them with the wrappers (writes the mission-required E3 names without
clobbering the E2 artifacts):

```
python analysis/arm_a_science/e3_finalize.py --tag v1
```

which produces `e3_topn_global_local.json`, `e3_population_stability.json`,
`e3_core_tail.json`, `e3_frequency_bands.json`, `e3_frequency.json`.

Context-length robustness on the corpus: run once per T with distinct tags,

```
python analysis/arm_a_science/e3_census.py --ckpt latest --corpus-root ... \
    --context 256  --batches 8 --tag ctx256
python analysis/arm_a_science/e3_census.py --ckpt latest --corpus-root ... \
    --context 512  --batches 8 --tag ctx512
python analysis/arm_a_science/e3_census.py --ckpt latest --corpus-root ... \
    --context 1024 --batches 8 --tag ctx1024
```

then compare Δu(64), core mass and band z against
`results/arm_a_science/context_length.json` (E2 synthetic series).
T > 2048 is a separate length-extrapolation experiment and must be reported
separately, not as ordinary evaluation.

## Colab cell (for the GPU machine)

```python
# exact frozen corpus path from context/frozen_corpus_contract.json
!cd /content/iclr-oc && python analysis/arm_a_science/e3_census.py \
    --ckpt latest \
    --corpus-root "/content/drive/Shareddrives/ICLR PHASE BDH/phase_bdh/corpus/stage2/frozen_5b_v1" \
    --batches 8 --microbatch 8 --tag v1
```

Notes:

- `--batches` counts 64-row corpus batches; `--microbatch` controls the
  forward chunk (CPU memory). 8 batches x 64 rows = 512 packed windows
  (1,048,576 tokens) is the suggested minimum for E3; scale up if the
  Colab GPU has headroom.
- The model is run in **fp32 with no autocast** on the chosen device so the
  captured statistics match the fp32 CPU analysis; do not compare bf16
  activations to the local numbers.
- The package never modifies `opt/`, `training/`, or the corpus.
- Record the sequence range (`--start-sequence`) in the ledger when the run
  is performed; all batches after the start sequence are streamed in
  deterministic order.
- A local smoke check without the corpus: `python
  analysis/arm_a_science/e3_census.py --plan` (prints the plan only).
