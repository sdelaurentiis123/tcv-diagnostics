# Codec-free operator smoke TF32 amendment

**Amended:** 2026-08-24, after failed engineering job 6933538 and before its
replacement launch

**Scope:** bounded 85604 engineering smoke only

**Scientific model-selection consequence:** none

## Evidence from the replacement metric

Job 6933538 retained the frozen maximum/RMS gates from the numerical
amendment. It again completed all four optimizer steps and online W&B logging,
then failed only the volume RMS gate:

| state view | normalized maximum | normalized RMS |
|---|---:|---:|
| C5P volume | 0.000783019 | 0.000100478 |
| E6B volume | 0.000624232 | 0.000112082 |
| E6B boundary | 0.00000475720 | 0.00000129808 |

The A100 environment reported:

```text
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = False
```

The convolutional model therefore used cuDNN TF32 by default. TF32's reduced
mantissa is consistent with the measured full-volume reduction discrepancy.

## Prospective correction

The maximum and RMS thresholds frozen in the prior amendment do not change.
The replacement engineering smoke disables both cuDNN and matrix-multiplication
TF32 and sets float32 matrix multiplication precision to `highest` before any
training or evaluation:

```text
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
```

This asks whether the architecture is roll-equivariant under full float32
execution. It does not choose the numerical precision for later scientific
training. A later training protocol must state and match its precision policy
across C5P and E6B.

No additional tolerance change is authorized by this amendment.

