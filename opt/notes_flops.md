# Arm-A FLOP / traffic accounting (production shape, global batch 64)

Constants: T=2048, D=256, N=16384, H=4, K=4096, Dv=256, L=8, B=64.
Per-update input tokens = 131,072. All figures per full training update.

## Canonical (measured G4: 2.873 s/update, B32x2)

| Op (per level, x64 seqs, x8 levels) | FLOPs/update | Traffic note |
|---|---|---|
| project_x: [T,D]@[D,N] | 8.8 TF | dense, efficient |
| attention QK^T: H x 2T^2K | 70 TF | blocked T^2 scores materialized (33 MB/seq/level bf16), saved by SAC |
| attention scores@V: H x 2T^2Dv | 4.4 TF | reads scores back |
| decoder_y: H x 2KD per token | 8.8 TF | dense |
| encoder: [T,N]@[N,D] | 8.8 TF | dense |
| coordinator mask-BMM [T,T]@[T,D] | 1.1 TF | T^2 traffic per level |
| writer + LNs | ~1 TF | dense |
| **total fwd** | **~103 TF** | + backward ~2x + SAC recompute of all non-score ops |

Achieved ~26 TFLOP/s effective -> strongly memory/recompute-bound, not FLOP-bound.

## opt1: exact scan + prefix coordinator, no checkpoint

| Op | FLOPs/update | Traffic note |
|---|---|---|
| project_x / decoder_y / encoder (unchanged) | 26.4 TF | same dense matmuls (new roof) |
| scan: 4KDv per token-head (matvec + rank-1) | 17.6 TF | state [H,K,Dv] streamed; zero T^2 |
| prefix coordinator | 0.1 TF | O(TD) |
| writer + LNs | ~1 TF | |
| **total fwd** | **~45 TF** | no scores, no recompute, no T^2 traffic |

FLOP ratio ~2.3x plus removal of score save/reload traffic and (if memory
allows) removal of checkpoint recompute (~1.3-1.5x on the step). Combined
projection lands in the 2-3x "very good" band. >3x claims must be
correctness-audited, per contract.

## Memory sketch (bf16 activations, per microbatch seq, per level)

- x_bt / ypre / paper_y_flat: 3 x T*N x 2B = 3 x 64 MB = 192 MB (T=2048)
- canonical scores (blocked): H x 256 x T x 2B = 4 MB transient, but SAC-saved
  across levels: H*T^2 x 2B x L = 33 MB x 8 = 268 MB/seq global-batch... x32
  microbatch = 8.6 GB saved + everything else recomputed.
- opt1-nockpt per level ~200 MB/seq -> B32: 6.4 GB/level x 8 = ~51 GB
  + params/grads/AdamW (~2 GB) + headroom -> fits 95 GB G4. B64 (~100 GB)
  borderline: measure, do not assume.
- Local 4060 (8 GB): use T=256-1024, L=2-8 ladder; rank candidates, do not
  project absolute tok/s to G4.

## Next roofs after opt1 (do NOT pursue before opt1 is benchmarked)

1. torch.compile on opt1 (inductor fusion of chunked-scan elementwise chains).
2. Fused Triton scan kernel (state in SRAM tiles; rank-1 update, no A
   materialization at any block size) - the production form of scan_chunked.
3. Segment-start hoisting (derive once per microbatch, reuse across L).
4. Allocator / layout / B64 re-test (post-memory-win only).
