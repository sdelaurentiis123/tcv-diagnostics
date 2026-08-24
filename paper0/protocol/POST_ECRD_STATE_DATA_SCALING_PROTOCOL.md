# Post-ECRD state/data scaling and codec-free operator protocol

**Frozen:** 2026-08-24, before implementation or training under this protocol

**Development simulation:** TCV/Hermes 85604 only

**Held-out simulation 85606:** unopened and prohibited

**Status:** implementation and bounded 85604 engineering smoke tests are
authorized. Scientific training is not authorized until a dated dataset
amendment records the new 85604 paths, identities, splits, and hashes.

## 1. Motivation and narrow claim

The completed reduced-state model ladder did not produce a forecast prior that
passed the Paper 0 transport-covariance gate. That result does not establish
that diffusion is fundamentally unsuitable, that missing state is the sole
bottleneck, or that the available sample count is sufficient.

This protocol separates two hypotheses prospectively:

1. **data limitation:** the model family improves as genuinely additional
   chronological or independently initialized 85604 material is added;
2. **state limitation:** a codec-free operator trained on the exact saved
   Hermes state predicts transport-relevant transitions better than the same
   operator trained on C5P.

Architecture, data amount, and retained state must not be changed together in
the decisive comparison.

## 2. Inventory frozen before implementation

The only 85604 material visible at the previously audited paths on 2026-08-24
is the existing trajectory:

| property | observed value |
|---|---:|
| BOUT++ spatial shards | 256 |
| physical trajectories | 1 |
| saved frames | 624 |
| normalized time | 285000 through 471900 |
| normalized cadence | 300 |
| physical cadence | 3.131905426 microseconds |
| native volume grid | 64 by 32 by 81 |
| engineering grid | 64 by 32 by 88 |
| simulated toroidal fraction | one fifth |
| mode mapping | physical toroidal mode `n = 5 k` |
| run ID | `0385667a-1757-49a1-a163-c941978d657a` |
| restart parent | `f8b4d12f-2627-4686-93a3-df784107667e` |

The 256 files are MPI spatial partitions of one trajectory, not independent
runs. The audited raw state contains the evolved volumes `Ne`, `Pe`, `Pi`,
`NVe`, `NVi`, and `Vort`. The existing Paper 0 engineering artifact also
contains the retained radial potential-boundary record `Bphi`.

No path for a larger 85604 dataset was present in the repository or at the
known 85604 locations. Consequently, no `2x`, `4x`, or `all-data` scientific
run is authorized by this version of the protocol.

## 3. Immutable existing split

The current trajectory retains the existing split without reclassification:

| region | global frames | purpose |
|---|---:|---|
| training | `[0, 432)` | parameter fitting and normalization |
| guard | `[432, 496)` | leakage barrier |
| chronological validation | `[496, 624)` | model selection on 85604 |

Normalization is fit on `[0, 432)` only. Guard frames may not be consumed by
an input, target, normalization, or checkpoint-selection rule.

The prior stationarity screen failed. Validation is therefore interpreted as
later-background chronological extrapolation, not as an IID sample from the
training distribution.

## 4. Required amendment for additional data

Before reading values from a larger 85604 dataset, a dated, committed
amendment must record only the minimal inventory:

- exact paths and hashes or stable source identifiers;
- trajectory, restart, continuation, and equilibrium labels;
- frame counts, fields, grid, cadence, and time ranges;
- which units are independent runs and which are continuations;
- one whole-trajectory or whole-equilibrium blind split when the inventory
  permits it;
- chronological train, guard, and validation boundaries inside every
  development trajectory;
- attainable data budgets.

The nominal budgets are `1x`, `2x`, `4x`, and all available data. A budget is
omitted if the inventory does not contain that much unique material. Frames
must never be duplicated or oversampled to manufacture a nominal budget.
Window count is not an independence count.

The entire 85606 simulation remains outside inventory, preprocessing,
normalization, model selection, and path discovery.

## 5. State views

### 5.1 Reduced observed state

`C5P = [Ne, Pe, Pi, phi, Vi]`.

This is the reduced-state control. It is not called exact or Markov-complete.

### 5.2 Exact saved-state candidate

`E6B = [Ne, Pe, Pi, NVe, NVi, Vort] + Bphi`.

The six volumes are the evolved Hermes variables in this archive. `Bphi` is
the retained radial potential-boundary state established by the completed
potential/vorticity closure audit. This is called the **exact saved-state
candidate**, not the complete internal solver state: unsaved substep, source,
or solver memory may still exist.

The operator predicts the six evolved volume increments and the `Bphi`
boundary increment jointly. For evaluation, `Vi` is reconstructed from
`Ne,NVi` using the frozen Hermes floor convention, and `phi` is reconstructed
from `Vort`, pressure, geometry, and predicted `Bphi` using the previously
validated Hermes transformation. Derived `phi` and `Vi` are not training
targets for E6B.

## 6. Controlled model sequence

### 6.1 Existing reduced-state controls

Completed one-data-unit H1, B5, and ECRD results are reused; they are not
rerun unchanged. If the new inventory makes larger budgets attainable, the
same frozen implementations may be trained at those larger budgets with
matched optimization and evaluation.

### 6.2 Codec-free full-resolution increment operator

The first new model is a mixed-boundary 3D residual U-Net/operator with:

- direct field input and output, with no autoencoder or latent codec;
- circular padding only on the toroidal axis;
- zero/wall padding on the two nonperiodic axes;
- no toroidal striding or downsampling;
- optional weak multiresolution processing only on the nonperiodic axes;
- no absolute toroidal coordinate;
- one shared random circular toroidal roll for every state in a training pair;
- joint output across every field in the selected state view;
- variable-specific training-only normalization;
- normalized state-derivative prediction.

For a lead of `ell` saved frames, the target is

```text
(x[t + ell] - x[t]) / ell
```

and the reconstruction is `x[t] + ell * predicted_derivative`. Lead time is a
model condition. The model receives geometry and boundary information that is
available at inference. No future boundary value may enter the input.

C5P and E6B use the same processor width, depth, optimization budget, temporal
pair set, and seed bank. Input/output stems and the E6B boundary head are the
only state-dimension-dependent components.

### 6.3 GAOT-style operator

A GAOT adaptation is a second architecture, not the starting implementation.
It is authorized only after the codec-free one-step baseline passes its
engineering checks and shows nontrivial one-step skill. The public GAOT code
is a reference implementation; the 2026 plasma paper reports architectural
differences from that repository and ablates geometry features in its fully
periodic 2D experiment. Any port must therefore be provenance-tracked and
validated for this mixed-boundary 3D domain.

### 6.4 Persistent global-local stochastic operator

This model is not authorized yet. It becomes the single stochastic extension
only if one of the following is established on 85604 development data:

1. exact-state deterministic forecasts work but C5P futures remain
   conditionally dispersed; or
2. multiple independent trajectories or branched restarts support estimation
   of a conditional stochastic law.

Its persistent latent must represent coherent rollout-scale uncertainty, not
decorative independently resampled output noise.

## 7. Training curriculum

Stages are gated and sequential:

1. one-frame, one-step teacher-forced increment prediction;
2. lead-time-conditioned pairs at leads `1, 2, 4, 8, 16` frames;
3. four-frame autoregressive training/evaluation;
4. eight-frame autoregressive training/evaluation.

Stage 2 is not authorized until Stage 1 passes. Stage 3 is not authorized
until the selected Stage 2 checkpoint improves the one-step and multi-lead
validation gates without a material transport regression.

The training objective contains normalized field and saved-boundary errors
only. Flux, spectra, cross-phase, coherence, conservation, PDE residuals, and
blob labels are evaluation quantities and may not enter the loss.

Weights & Biases tracking is required for cluster training. Offline raw
artifacts remain authoritative; a W&B failure must not erase or invalidate a
completed local checkpoint.

## 8. Evaluation and uncertainty

Checkpoint selection uses every frozen chronological 85604 validation block,
not one favorable window. Required evaluation includes:

- field RMSE, MAE, bias, variance, and error versus horizon;
- directional toroidal spectra with `n = 5 k`;
- memberwise density/pressure-potential cross-phase and coherence;
- local radial transport maps and integrated separatrix transport;
- spatial transport-covariance error;
- profile and distribution stability during rollouts.

Physics metrics are computed only after prediction. Nonlinear metrics are
computed member by member whenever an ensemble exists.

For an exact-state deterministic model, the first ensemble mechanism is a
documented distribution over the inferred initial saved state plus independently
trained model seeds. Learned stochastic innovations are deferred until the
criteria in Section 6.4 are met.

## 9. Prospective decision logic

| result | supported interpretation | next action |
|---|---|---|
| performance improves steadily with attainable data budgets and C5P/E6B are similar | primarily sample-limited in the tested range | continue scaling the simplest passing operator |
| E6B materially improves transport at matched data and compute | saved-state sufficiency matters | use E6B teacher and develop partial-observation state inference |
| both scaling and E6B help | both mechanisms matter | retain the factorial design and scale E6B |
| E6B one-step works but C5P stays dispersed | uncertainty is substantially hidden-state uncertainty | build a sequential hidden-state student |
| neither scaling nor E6B repairs one-step transport | operator, cadence, transformation, or unsaved-state bottleneck remains | stop scaling and localize that failure |

No result from one continued 85604 trajectory authorizes a claim of broad
cross-shot or cross-equilibrium generalization.

## 10. Held-out boundary

85606 may be opened exactly once only after the architecture, state view,
training budget, lead-time curriculum, seed rule, checkpoint-selection rule,
rollout horizons, metric implementation, transport transformation, plots, and
acceptance thresholds are committed and frozen. Until then:

- no 85606 path discovery;
- no metadata access;
- no normalization or preprocessing;
- no forecast;
- no assimilation, sensor ranking, or steering.

