# Arm-B / Arm-C semantics recovery (gate document)

Status: **BLOCKED — Arm-B and Arm-C are not defined anywhere in this
workspace.** Track B/C implementation is gated until their canonical sources
are supplied. This document records the exhaustive search and what little
adjacent evidence exists, so the gate can be lifted without re-inference.

## Search performed (2026-09-16)

Command:
`rg -in "arm[- ]?[abcd]\b|branch" C:\iclr-oc --glob '!runs/**' --glob '!*.jsonl'`

Files searched (entire repository, excluding the local `runs/` export):
- `README.md`, `README_FIRST.md`, `PROMPT_FOR_META_MUSE.md`, `AGENTS.md`
- `context/*` (`canonical_source_provenance.md`, `frozen_corpus_contract.json`,
  `g4_baseline.json`, `known_findings.md`, `optimization_contract.md`,
  `transformer_control_context.json`)
- `reference/*` (`arm_a_timing_cell_colab.py`, `scan_coordinator_oracle.py`)
- `opt/*`, `results/*`, `scripts/*`, `training/*`, `campaigns/*`

Result: every hit is **Arm-A** or prose about "branches" inside Arm-A
candidate families (e.g. "frequency-band / resonance branch"). There is no
Arm-B or Arm-C module, config, doc, checkpoint, or ledger entry.

## What does exist that could be mistaken for B/C

1. **Separate modern Transformer control** —
   `context/transformer_control_context.json`.
   - L=8, D=384, D_FF=960, T=2048, V=8192, 6 Q-heads / 2 KV-heads, head_dim 64.
   - ~15.1M params total; measured 1.097M input tok/s at B64 on G4.
   - Its own note: "context only; separate modern Transformer control, not
     Arm-A", and from `known_findings.md` #4/#6 it establishes the ~24x
     throughput gap and "is not an optimization target."
   - It is **not named Arm-B or Arm-C**, has no source file in the repo, and
     has no recurrent-state semantics to transfer. Treating it as Arm-B/C
     would be exactly the redefinition the master directive forbids.

2. **Arm-A execution variants** (`opt3c_*`) — execution-only forks of Arm-A,
   not separate arms.

3. **This campaign's candidate families** (`ExpertizedArmA`,
   `RoutedExpertArmA`) — Arm-A derivatives, not Arm-B/C.

## Consequence

The directive's rule applies:

> Do not infer branch semantics from names.
> Do not rename A/B/C/D.
> Do not redefine Arm-B or Arm-C.

There is nothing to infer from. `campaigns/ARM_BC_EVIDENCE_TRANSFER.md` cannot
be written honestly, `results/autoresearch_arm_b.jsonl` /
`autoresearch_arm_c.jsonl` cannot be seeded, and B0/C0 baselines cannot be
established, because:

- we do not know B/C's forward equations, dimensions, state semantics,
  Q=K status, RoPE placement, Dx/Dy/E analogues, coordinator behavior,
  parameter sharing, or persistence;
- we do not know their canonical source files or commit hashes;
- we cannot reproduce their correctness tests or training readiness.

Guessing would violate the master directive and contaminate the evidence
chain. No B/C line of work has been started beyond this recovery attempt.

## To lift the gate, provide any one of

1. the repository/path containing Arm-B and Arm-C canonical sources (plus
   pinned commit hashes), or
2. a written semantics spec per arm (equations, dimensions, state, sharing,
   RoPE placement, coordinator, known benchmarks, correctness tests), or
3. an explicit instruction that the `context/transformer_control_context.json`
   model is to be treated as Arm-B or Arm-C — which requires the owner to
   name it, since renaming/redefining is otherwise forbidden.

## Meanwhile

Track A (Arm-A paper-grade science) is fully unblocked and proceeding:
training trajectory, 4-checkpoint sparsity maturation, top-N ladder,
RoPE-frequency structure, block-sparsity falsification, external SAE
calibration, figure/claim scaffolds.
