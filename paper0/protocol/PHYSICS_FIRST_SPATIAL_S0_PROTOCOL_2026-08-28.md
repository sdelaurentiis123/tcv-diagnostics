# Physics-first forecast readout and S0 spatial reconstruction protocol

**Frozen:** 2026-08-28, before inspecting the new NERSC inventory or S0
validation results.

## Scope and motivation

This prospective amendment creates a meeting-quality physics readout from
already frozen old-85604 forecast artifacts and runs one small simultaneous
spatial-reconstruction experiment. It does not select or train a new temporal
forecast architecture.

The readout asks four questions:

1. Which visible plasma structures are forecastable?
2. Where does local transport coherence disappear?
3. Which density--potential modes lose their coupled phase or coherence?
4. How much unobserved spatial structure is recoverable from two localized,
   simultaneous synthetic density measurements?

Simulation 85606 remains sequestered. This work must not enumerate, inspect,
open, hash, or otherwise touch any 85606 path or artifact. The newer NERSC
85604 segment is inventory-only until the existing post-ECRD amendment passes.

## Immutable old-85604 data protocol

- Development run: 85604 only.
- Stored cadence: 3.1319 microseconds per saved frame.
- Toroidal fraction: one fifth of the torus, so physical mode number is
  `n = 5 k`.
- Chronological training: frames `[0, 432)`.
- Unread guard: frames `[432, 496)`.
- Chronological validation: frames `[496, 624)`.
- Fields used by S0: standardized fluctuations of `Ne`, `Pe`, `Pi`, and `phi`.
- Normalization and per-cell climatology: fitted on `[0, 432)` only.

The guard remains excluded. Frames are not interpreted as independent physical
shots.

## Part A: NERSC 85604 read-only inventory

Run the frozen inventory in
`POST_ECRD_NERSC_85604_INVENTORY_AMENDMENT_2026-08-24.md` on the canonical
85604 path only. Verify hashes, dimensions, physical time, cadence, coordinate
semantics, the iteration reset, continuity of all six evolved variables,
settings/grid compatibility, and availability of radial `phi` guard cells.

No preprocessing or temporal pairs may be built in this task. In particular,
no transition may cross a source-file boundary or reset until continuity has
passed. The NERSC segment is described as a continuation unless evidence proves
an independent restart.

## Part B: forecast-physics figures

### Model labels and fair horizons

Use these descriptive labels in the figures:

| Artifact family | Figure label | Available future frames |
| --- | --- | --- |
| deterministic codec-free C5P operator | codec-free multi-lead operator | 1, 2, 4, 8, 16 |
| B5-Context | conditioned field-residual diffusion | 1 |
| ECRD | equivariant conditioned diffusion | 1 |
| persistent global--local model | persistent global--local generator | 1, 2, 3, 4 |

The primary across-model comparison is **future frame 1 only**, corresponding
to 3.1319 microseconds. No absent stochastic horizon may be synthesized or
implied. Longer deterministic evolution and the shared deterministic/persistent
four-frame comparison are separate panels.

### Representative cases

- For a deterministic horizon-eight field panel, select the validation start
  whose absolute confined-separatrix integrated particle-transport error is the
  median among all eligible starts. Ties use the earliest frame.
- For shared one-frame ensemble panels, select the validation target whose
  conditioned-diffusion ensemble-mean absolute confined-separatrix integrated
  particle-transport error is the median. Ties use the earliest frame.
- The same selected target is used for every model in a shared comparison.

This rule is mechanical and does not select visually attractive examples.

### Authoritative diagnostics

- Transport uses the frozen geometry-aware member-wise implementation and the
  confined separatrix at model face `x=15`, poloidal rows `y=8..23`.
- Positive transport is outward in the model `+x` direction.
- The primary integral is for the simulated one-fifth toroidal wedge.
- Nonlinear transport is evaluated for each ensemble member before reduction.
- Local transport is displayed by poloidal location along the confined
  separatrix after the authoritative toroidal and face weighting.
- The cumulative curve is the ordered cumulative sum of those signed local
  contributions. It is an explanatory decomposition of the same integrated
  wedge transport.
- The transport variogram is the first-order periodic toroidal structure
  function of the authoritative local separatrix contribution, evaluated at
  lags `1, 2, 4, 8, 16, 32, 40` cells and labeled by physical toroidal
  separation. Curves are averaged over eligible targets; uncertainty intervals
  use chronological moving-block bootstrap with block length at least the
  frozen decorrelation estimate.
- Density--potential spectra use train-split normalization, remove the toroidal
  mean, Fourier transform only along the periodic toroidal axis, pool the
  cross-spectrum over strict eligible spatial cells and validation targets, and
  report physical `n=5k`. Phase is `arg(S_Ne,phi)` and magnitude-squared
  coherence is `|S_Ne,phi|^2/(S_Ne,Ne S_phi,phi)`. The evaluated band
  `n=20..35` is shaded.

All curves must state whether they show truth, ensemble mean, individual
members, or an ensemble expectation. Truth is black in every panel. Geometry,
units, physical time, model label, horizon, and target frame must be visible.

## Part C: S0 synthetic leave-one-diagnostic-out reconstruction

S0 is simultaneous and contains no temporal forecast. The task is:

`geometry + localized diagnostics A and B -> full four-field fluctuation state -> held-out diagnostic C`.

### Synthetic observation footprints

Each channel is a boxcar average over a `3 x 3 x 5` neighborhood in model
`x, y, z`; toroidal indices wrap periodically and nonperiodic indices never
wrap. Cells outside the strict operator/wall mask are excluded, and a channel
is invalid if fewer than 60% of its nominal cells remain. All three families
observe `Ne` fluctuations only.

The preregistered channel centers are:

| Family | Physical description | `(x,y)` centers | toroidal center `z` |
| --- | --- | --- | --- |
| A | outboard-midplane localized patch | `(13,17)`, `(16,17)`, `(19,17)`, `(13,19)`, `(16,19)`, `(19,19)` | 0 |
| B | displaced outer/X-point-side patch | `(13,23)`, `(16,23)`, `(19,23)`, `(13,25)`, `(16,25)`, `(19,25)` | 22 |
| C | spatially separated inner-side holdout | `(13,8)`, `(16,8)`, `(19,8)`, `(13,10)`, `(16,10)`, `(19,10)` | 44 |

If a preregistered channel fails the 60% geometry rule, it is omitted and the
omission is reported; it is not moved after inspecting reconstruction skill.
These are **BES/GPI-like localized density diagnostics**, not faithful BES or
GPI forward models.

### Baselines and fitting

1. **Zero fluctuation:** predict the train-split per-cell climatology, which is
   zero after anomaly construction.
2. **Ridge/Wiener reconstruction:** map the concatenated A+B channels to all
   four fields jointly. Fit the equivalent dual ridge solution in streamed
   output chunks so the full target matrix need not remain in memory.

For ridge regularization only, use an internal chronological split contained
inside the official training region:

- fit `[0, 320)`;
- internal guard `[320, 336)`;
- tune `[336, 432)`.

Choose the smallest value in `10^[-5,-4,-3,-2,-1,0,1]` minimizing the mean of
(a) equal-field standardized full-state RMSE and (b) held-out-C standardized
RMSE. Refit with the chosen value on all `[0,432)` and evaluate once on
`[496,624)`.

A small masked convolutional reconstruction model is permitted only after the
linear pipeline and all leakage tests pass. It is not required for this scoped
readout and must not delay the linear result.

### S0 metrics and figures

- held-out-C standardized RMSE, normalized RMSE, bias, and Pearson correlation;
- equal-field full-state standardized RMSE;
- reconstruction skill versus cylindrical physical distance to A or B;
- per-mode error and retained power versus physical toroidal `n=5k`;
- separate skill in frozen outboard-midplane, X-point-stencil, confined-edge,
  private-flux, and scrape-off-layer masks;
- low modes `n<=15`, the evaluated band `n=20..35`, and higher represented
  modes reported separately.

The S0 hero target is the validation frame with median held-out-C ridge NRMSE;
ties use the earliest frame. The hero figure shows hidden truth, the A/B/C
footprints, ridge reconstruction, reconstruction error, and true versus
predicted held-out C channels. All model comparisons use the same frame and
truth-derived color limits.

S0 ridge predictive uncertainty, if shown, is the chronological calibration
residual covariance estimated on the internal tuning block; it is not a learned
stochastic plasma posterior. No coverage claim is required.

## Uncertainty and decision language

Chronological moving-block bootstrap is used for reported intervals. Adjacent
frames are never counted as IID. The readout is descriptive evidence from one
85604 realization and must not claim shot/device generalization.

The final HTML is concise and figure-first. It may conclude only what a plotted
quantity supports. A missing artifact is labeled unavailable rather than
silently replaced. Existing temporal emulator rankings remain unchanged by S0.

## Outputs

- `paper0/results/physics_first_spatial_s0_2026_08_28/`
- `paper0/figures/physics_first_spatial_s0_2026_08_28/`
- `paper0/reports/paper0-physics-spatial-readout-2026-08-28.html`
- a machine-readable manifest with code revision, dirty state, commands,
  source identifiers, selected frames, horizons, seeds, and hashes.

Any later change to masks, selection rules, metrics, or horizons requires a
dated amendment written before inspecting the affected result.
