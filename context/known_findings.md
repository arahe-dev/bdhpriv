# Known findings before the overnight run

1. Canonical Arm-A on G4 is ~45,614 input tok/s at B32x2, median ~2.873 s/update.
2. B64x1 genuinely OOMed; B32x2 is stable.
3. Peak measured memory at B32x2: ~75.73 GiB allocated / ~83.43 GiB reserved.
4. A separate optimized modern Transformer control ran ~1.097M tok/s on the same G4; the gap is ~24x.
5. The gap is not explained by stored parameter count alone. Arm-A repeatedly applies a huge N=16384 neuronal representation across 8 shared-depth iterations.
6. Weight sharing saves storage, not compute.
7. The current Arm-A attention explicitly materializes blocked T x T score matrices even though Q=K and there is no softmax/scale.
8. The exact state recurrence and segmented-prefix coordinator have already been proved equivalent in the included tiny float64 oracle, including gradients.
9. The Python recurrence is ONLY a readable oracle. A production kernel must use a GPU-efficient scan/fusion strategy; a Python token loop at T=2048 would be unacceptable.
10. The first objective is structural compute/memory reduction, not low-value compiler-flag sweeps.
