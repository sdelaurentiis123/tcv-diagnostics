# Post-ECRD old-85604 Stage-1 full codec-free protocol

**Frozen:** 2026-08-24, after paired pilot job `6933616`

**Development source:** existing 624-frame TCV/Hermes 85604 archive only

**Held-out 85606:** unopened and prohibited

## Question

Does a codec-free, toroidally equivariant one-step operator learn the frozen
85604 chronological validation transition more reliably from the reduced C5P
view or from the actual saved Hermes evolved state?

The pilot established that both implementations train and reload correctly.
It also exposed a numerical conditioning problem: in state-standardized units,
the NVe and Vort derivative persistence MSEs are roughly two orders of
magnitude larger than Ne. Consequently, nominally equal component weights do
not produce equal persistence-relative optimization pressure.

This full run changes only the training budget, seeds, output initialization,
and direct-field derivative scaling described below. It does not add a codec,
physics loss, stochastic generator, rollout loss, assimilation, or 85606.

## Immutable data contract

- training frames: `[0,432)`;
- guard frames: `[432,496)`, never read;
- chronological validation frames: `[496,624)`;
- existing state normalization fit on training frames only;
- one-frame context and one-frame lead;
- 431 training pairs and 127 validation pairs per arm;
- random shared circular toroidal rolls during training only.

State views:

- `C5P = [Ne,Pe,Pi,phi,Vi]`;
- `E6B = [Ne,Pe,Pi,NVe,NVi,Vort] + Bphi`.

The comparison metric remains the mean raw standardized derivative MSE over
the identical `[Ne,Pe,Pi]` validation targets. Numeric training losses are not
compared across state views.

## Architecture

Both arms use `CodecFreeIncrementOperator3D` with:

- base width 24;
- channel multipliers `[1,2,4]`;
- two residual blocks per level;
- 128-channel lead embedding;
- circular toroidal padding and no absolute toroidal coordinate;
- zero/wall padding on the two nonperiodic axes;
- downsampling only on the nonperiodic axes, never toroidally;
- no latent codec;
- joint output of all fields in the selected state view;
- a retained mixed-boundary head for E6B;
- zero-initialized final derivative projections, so the initial forecast is
  exactly persistence.

The two processors have matched capacity; only state-dependent stems and the
E6B boundary head differ.

## Direct-field derivative scaling

For each arm, fit one RMS value per volume field and per E6B boundary side
from all one-step derivatives in the training split only. Let `r_c` denote the
training RMS for component `c`. Optimize

```text
mean_c mean_elements ((predicted_derivative_c - target_derivative_c) / r_c)^2
```

Thus a zero-derivative persistence predictor has unit training loss in every
component. The prediction itself remains in the original state-standardized
derivative coordinates; the RMS values only condition the direct field MSE.

This is not a physics-derived loss. No flux, spectrum, cross-phase, coherence,
PDE residual, or conservation term is used.

## Frozen optimization

- seeds: `1701`, `1702`, `1703`;
- epochs: `12` per seed and state view;
- batch size: one transition;
- gradient accumulation: four transitions;
- AdamW, peak learning rate `2e-4`, weight decay `1e-4`;
- 5% linear warmup, cosine decay to `1e-5`;
- gradient clipping at norm `1.0`;
- bfloat16 autocast, TF32 disabled;
- deterministic seed-specific epoch permutations;
- one A100 allocation at a time through a concurrency-limited Slurm array;
- local authoritative artifacts and required online W&B tracking.

## Checkpoint selection and reporting

For each seed and state view, select the checkpoint minimizing validation mean
standardized derivative MSE over `[Ne,Pe,Pi]`. Report raw MSE, persistence MSE,
and persistence-relative skill for every predicted field and Bphi side.

Mechanical validity requires exact pair/update counts, finite metrics, exact
checkpoint reload, numerical toroidal equivariance, and explicit false flags
for 85606 access and physics-derived loss use.

## Prospective interpretation rule

Summarize median and seed range, not only the best seed.

- Advance E6B as the primary state for the multi-lead rung if its median
  shared-field MSE is no more than 10% above C5P and every evolved volume field
  has positive median persistence-relative skill.
- If E6B is more than 10% worse on shared fields or retains a nonpositive-skill
  evolved field, retain C5P as the performance control and treat E6B as an
  unresolved exact-state optimization/state ablation. Do not claim state
  completeness is disproved.
- A mechanical failure is repaired before any multi-lead or stochastic run.

The next rung is separately frozen multi-lead and short-unroll training. This
protocol does not authorize opening 85606.
