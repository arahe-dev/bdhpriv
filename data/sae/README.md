# Local SAE assets

Downloaded on 2026-09-16 with the Hugging Face Hub CLI. The large parameter and
Parquet files are intentionally local-only; their source revisions and hashes
are recorded below.

## Gemma Scope 2 270M PT

- Hub repository: `google/gemma-scope-2-270m-pt`
- Resolved revision: `b218cd5d69dc2fa71cff448b68d625e6c9702d49`
- Model named by the configs: `google/gemma-3-270m-pt`

| Local bundle | Files | SAE metadata |
|---|---|---|
| `mlp_out/layer_12_width_16k_l0_medium/` | `config.json` (303 bytes; SHA-256 `4e5a88aa8f0ccebad5fc99e206392195ced4423085d2977a34665dcbad28d19d`), `params.safetensors` (84,020,088 bytes; SHA-256 `c35c352e528df7ae1a12694821b6ed240b4345faf4ce7338dce178181bfbe7bf`) | width 16,384; L0 60; `jump_relu`; MLP hook at layer 12 |
| `mlp_out/layer_12_width_16k_l0_small/` | `config.json` (303 bytes; SHA-256 `fa7fd9767389362fbb0f6478a39f0afd43f668ab6621e6dc3610c2cc7413b9c5`), `params.safetensors` (84,020,088 bytes; SHA-256 `89df587de818b73aec68c5c5d16245669b78cdf8a8759dc2eef9c2fea3b3698c`) | width 16,384; L0 20; `jump_relu`; MLP hook at layer 12 |

The README snapshot linked in the request (`a5861d55920cc76ec4ba1e6d36ba5ba2bbc8faf4`) contains the two requested parameter paths but not their config sidecars. Both requested parameter files have the same LFS SHA-256 at that snapshot and at the resolved revision above; the newer revision was used so each downloaded SAE bundle is self-describing.

## Anthropic public feature dataset

- Hub dataset: `hbe/neuronpedia-sae-concepts`
- Config: `anthropic`, split `train`
- Resolved revision: `15099f41edc73eb3ab09862bef47a71ae24c1c5c`

| Local file | Bytes | SHA-256 |
|---|---:|---|
| `anthropic/train/anthropic_concepts.parquet` | 170,708 | `322e4c0ae477f6cbce6e9089346b7188e1eb94b32ae6855f374a1a6fc72ae0b5` |
| `anthropic/train/monosemantic_2023.parquet` | 157,590,801 | `e7abb3b86ec0d740b9161982f07cdc56583eaf5a1c181e3134bd5bedebeb74ba` |

This is public feature data: the config combines the 2023 public features and
the 2,999 published Claude 3 Sonnet features. No Claude 3 Sonnet SAE weights
were downloaded.

## Verification

- The two Gemma SafeTensors files and two Anthropic Parquet files match the
  SHA-256/LFS hashes reported by the pinned Hub revisions.
- Both Gemma configs match the pinned Hub bytes exactly.
- SafeTensors headers parse successfully; each contains `b_dec`, `b_enc`,
  `threshold`, `w_dec`, and `w_enc` with the expected 640/16,384 dimensions.
- Both Parquet files have valid `PAR1` start and end markers.
