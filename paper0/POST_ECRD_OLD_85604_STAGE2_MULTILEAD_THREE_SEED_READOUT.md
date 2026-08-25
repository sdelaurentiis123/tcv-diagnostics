# Old-85604 Stage-2 multi-lead confirmation: three-seed readout

**Result date:** 2026-08-25

**Development simulation:** 85604 only

**Held-out 85606 and newer NERSC data:** not read

**Training commit:** `50ed12bd3defdb2685603e6a69f1972587d5564d`

**Reduction commit:** `be44b511a0e4d330c675cd3ecf843449253626c5`

**Slurm array:** `6936641` (both tasks `COMPLETED`, exit `0:0`)

## Outcome

The direct multi-lead repair passed its frozen gate independently for seeds
1701, 1702, and 1703. Seed averaging was not used to rescue any failed run.
The prospectively authorized next experiment is therefore the bounded,
inference-only direct-versus-autoregressive comparison at terminal horizons
four and eight saved frames.

This establishes a reproducible direct-transition result. It does not yet
establish autonomous rollout stability, transport fidelity, stochastic
calibration, assimilation readiness, diagnostic ranking, or steering.

## Three-seed result

The selection score is the unweighted mean over leads `1,2,4,8,16` of the
shared `Ne`/`Pe`/`Pi` derivative MSE divided by persistence derivative MSE.
Lower is better and one is persistence.

| Seed | Selected epoch | Five-lead error / persistence | Lead-1 shared MSE | All frozen gates |
|---:|---:|---:|---:|:---:|
| 1701 | 4 | `0.488139` | `0.00529950` | pass |
| 1702 | 4 | `0.486474` | `0.00526557` | pass |
| 1703 | 4 | `0.490368` | `0.00526330` | pass |
| **Median** | **4** | **`0.488139`** | **`0.00526557`** | **pass** |

The range of the five-lead score is only `0.003894`, so the conclusion is
not driven by one favorable initialization.

## Direct forecast skill by lead

These are teacher-forced direct predictions. Each model receives the true
state at the forecast start and predicts one target at the stated gap.

| Lead (saved frames) | Minimum skill | Median skill | Maximum skill |
|---:|---:|---:|---:|
| 1 | `0.8628` | `0.8636` | `0.8637` |
| 2 | `0.5628` | `0.5686` | `0.5732` |
| 4 | `0.4209` | `0.4209` | `0.4255` |
| 8 | `0.4041` | `0.4044` | `0.4063` |
| 16 | `0.2962` | `0.2967` | `0.3055` |

Skill is `1 - model_error / persistence_error`; positive values beat
persistence. Every field—not only the shared-field average—has positive
skill at every tested lead in every seed.

## Scientific interpretation

The Stage-1 parent was trained only at lead one even though its architecture
contained a lead-time embedding. Across all three seeds, querying that parent
at unseen leads produced severe errors, while explicit supervision over the
same leads repaired the problem without sacrificing lead-one accuracy.

The defensible conclusion is narrow:

> A material part of the old model's finite-horizon failure came from asking
> an untrained lead-time conditioner to extrapolate, rather than from a proven
> inability of the codec-free local operator to represent finite-time C5P
> transitions.

The direct result cannot tell us whether errors compound when predictions are
fed back as inputs. The next bounded comparison isolates that question by
holding starts, targets, checkpoint selection, and terminal horizon fixed.

## Integrity checks

- Training used frames `[0,432)` and validation `[496,624)`; guard
  `[432,496)` was not read.
- Only old simulation 85604 was read.
- No physics-derived quantity entered the loss.
- Both confirmation runs executed exactly 2,132 optimizer updates.
- Exact checkpoint reload, finite-metric, loss-decrease, and integer
  toroidal-shift equivariance gates passed for every seed.
- TF32 was disabled.
- Every run-level and task-level SHA-256 inventory verified after completion.
- Both new W&B runs report remote state `finished`:
  [seed 1702](https://wandb.ai/sdelaurentiis123-columbia-university/tcv-diagnostics-paper0/runs/p0oldmultileadscale-s1702-j6936642)
  and
  [seed 1703](https://wandb.ai/sdelaurentiis123-columbia-university/tcv-diagnostics-paper0/runs/p0oldmultileadscale-s1703-j6936641).

## Provenance

- Frozen scaling manifest SHA-256:
  `321b96a002acfcf0da86690180c829c8a4a859447263a0fbc70e5592163eb63d`.
- Seed-1701 tracked result SHA-256:
  `6f4036ff6fd50a7090e60a351242f1a6ad00af6d3762322fe1075d22a9808c2a`.
- Seed-1702 tracked result SHA-256:
  `9ba99fc02ac8620dd1a1e917a7f5973f08925ada3389a3a65b913f941d410881`.
- Seed-1703 tracked result SHA-256:
  `ac17dc508c1fb9a64d9f7e6ef60eae3a54e87f66f437f2c37bbe4d57046e0585`.
- Three-seed reduction SHA-256:
  `33f3be426c6bf512ef32469737765900797d739bdb83742eeec61560c32d2ca3`.
- Selected checkpoint SHA-256 values:
  seed 1701 `35b82de5d16001dac99e1e97c5002b75d22fa9836dd5e83a90a68a4d6c64adaa`,
  seed 1702 `b9007f818eb35d82a1e4c21771dfc1ad870591feb777f5087f3b2a49847cd50d`,
  and seed 1703
  `3b81201aa0f8a4a8d73a3b87cfaed1b9b12815768b4f2bcd075815146d2d559b`.

## Decision

The frozen reduction records:

```text
three_seed_mechanism_confirmed: true
bounded_rollout_authorized: true
decision: freeze_bounded_direct_vs_autoregressive_validation
```

Proceed only with the already specified 85604 validation comparison at
horizons four and eight. Do not infer long-rollout, transport, probabilistic,
or held-out performance from this result.
