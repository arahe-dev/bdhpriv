# META MUSE 1.3 — OVERNIGHT ARM-A OPTIMIZATION MANDATE

You are the primary overnight optimizer in this session. Do not dispatch another model unless explicitly told to.

Read in this order:
1. `README_FIRST.md`
2. `context/optimization_contract.md`
3. `context/known_findings.md`
4. `context/g4_baseline.json`
5. `reference/scan_coordinator_oracle.py`
6. `reference/arm_a_timing_cell_colab.py`

Your job is systems optimization, not architecture research.

Start by running `scripts/report_env.py` and record the result.

The local repo may be empty. Create the optimization implementation under the workspace without altering the files in `reference/`.

First milestones:
1. Make the oracle tests systematic/randomized.
2. Extract a local canonical Arm-A integration reference from the Colab timing cell without changing semantics.
3. Implement an exact GPU-friendly score-free attention candidate.
4. Implement exact segmented-prefix coordinator.
5. Establish correctness.
6. Benchmark full training updates on the local RTX 4060 using shapes that fit.
7. Attack checkpoint recomputation and memory.
8. Preserve the top 2-3 distinct correctness-passing candidates for G4 confirmation.

Do not:
- change model dimensions or depth;
- change masks, RoPE, initialization, optimizer, loss, or global-batch semantics;
- reduce valid target counts and call it a speedup;
- invent MoE or Phase equations;
- use CPU timing to rank CUDA kernels;
- extrapolate local absolute throughput to G4;
- spend hours tuning tiny compiler flags before structural transformations are exhausted.

If full production B/T cannot fit in laptop VRAM, use production K=4096, D=256, H=4 where possible while reducing B/T/L for representative CUDA work, and keep the benchmark type clearly labeled.

Every performance claim must have a corresponding correctness PASS.

Finish with:
- fastest candidate;
- next-best fallback(s);
- exact changes/diffs;
- local relative speedup;
- peak VRAM;
- correctness evidence;
- failed-idea ledger;
- exact commands/files for tomorrow's G4 confirmation.
