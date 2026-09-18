# Akasha CPU diagnostics and SODA compatibility

Akasha's canonical runtime already provides the reference recurrent inference
engine in `akasha.runtime`. The commands below exercise it on a Linux worker
without CUDA.

## CPU/RAM diagnostics

Install a CPU-only PyTorch build, then run:

```bash
python -m scripts.akasha_ram_only --report
python -m scripts.akasha_ram_only --pytest tests/akasha --skip-slow -q
```

The launcher rejects a CUDA-enabled PyTorch build. It does not pretend that
`CUDA_VISIBLE_DEVICES` alone converts a CUDA installation into a CPU-only
installation.

## SODA checkpoint conversion

The SODA-BDH full run uses the `soda_bdh_ckpt_v1` format. Convert a completed
checkpoint only through the explicit compatibility adapter:

```bash
python -m scripts.akasha_soda_compat \
  --checkpoint /path/to/soda_bdh/latest.pt \
  --out-dir results/akasha/soda_compat_package
```

The adapter verifies the SODA format, implementation prefix, frozen dense
Arm-A architecture, tensor names, and tensor shapes. It writes a normal
Akasha package with provenance that records:

```text
adapter_status = EXPLICIT_COMPATIBILITY_ONLY
canonical_akasha_trainer_status = NOT_CANONICAL_AKASHA_TRAINER
```

Load the resulting package with `akasha.load_model` or
`akasha.runtime.AkashaModel`. SODA compatibility does not certify SODA as the
canonical Akasha trainer, and the adapter never accepts a CUDA execution path.
