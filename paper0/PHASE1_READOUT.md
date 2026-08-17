# Phase 1 data readout

**Status:** data characterization complete; steady-state learning gate closed

**Latest evidence job:** `6890606`

**Latest evidence commit:** `1d6e1fc962341ac07377c48cdca9274a7e3e7df8`

**Run accessed:** `85604` only

**Run not accessed:** `85606`

## Executive conclusion

The data are neither “uselessly incoherent” nor demonstrated stationary.
Instead, the 85604 trajectory contains two separable behaviors:

1. **Slow background evolution.** Whole-field means and fluctuation amplitudes
   change across the 624-frame, 1.951 ms record. All nine predeclared scalar
   stationarity series fail the block-range criterion, and several also have
   large fitted drift. The proposed chronological split is therefore not yet
   frozen as samples from one stationary distribution.
2. **Fast, translating non-axisymmetric structure.** Fixed-grid toroidal
   residuals cross `1/e` in less than one 3.132 microsecond frame, but a single
   oracle circular shift restores one-step correlation to approximately
   `0.48--0.66`. The best shift is consistently 9--12 of 88 toroidal cells.
   Complex Fourier phase advances coherently, particularly around stored
   `k=4--5`, or full-torus `n=20--25`.

The immediate architecture implication is not “make every future frame fresh
noise.” A credible emulator must represent a slowly evolving axisymmetric
background, periodic toroidal phase transport, and stochastic residual
uncertainty separately. The current evidence does not yet choose FGN,
diffusion, PDE-Refiner, or another model.

The learning gate remains closed because no statistically steady interval has
been established under the frozen rule. Metric-oracle implementation can
continue without opening that gate.

## Verified source facts

- One 85604 trajectory contains 624 frames at a physical cadence of
  `3.131905426352636 microseconds`, spanning `1951.1771 microseconds` from the
  first stored frame to the last.
- Raw BOUT arrays use `[time, x, y, z]`; the Well shards use
  `[trajectory, time, x, y, z]`. This agrees with the
  [BOUT++ output documentation](https://bout-dev.readthedocs.io/en/latest/user_docs/output_and_post.html).
- The Well shards concatenate exactly: the first contains global frames
  `[0,500)` and the second `[500,624)`. Those are storage names, not the new
  learning split.
- `zperiod=5` means the simulated periodic domain is one fifth of a torus. The
  [BOUT++ option documentation](https://bout-dev.readthedocs.io/en/stable/user_docs/bout_options.html)
  defines this domain as `0` to `2pi/zperiod`; stored Fourier index `k` maps to
  full-torus toroidal mode number `n=5k`.
- The 88-cell converted toroidal coordinate is an index-like Fourier resampling
  coordinate. It is not a physical angle vector.
- BOUT++ is version `5.2.1`, revision
  `7d28d67c3f12c24ec281c0982e870f5369c65a6f`.
- Raw field metadata restore the unit information dropped by the Well
  conversion:

| Field | Source component | Unit | Physical conversion |
|---|---|---|---:|
| `Ne` | `evolve_density` | `m^-3` | `1e19` |
| `Te` | `evolve_pressure` | `eV` | `50` |
| `Ti` | `evolve_pressure` | `eV` | `50` |
| `phi` | `vorticity` | `V` | `50` |
| `Vi` | `evolve_momentum` | `m / s` | `69205.61141651045` |
| `Ve` | `evolve_momentum` | `m / s` | `69205.61141651045` |
| `Vort` | `vorticity` | `C m^-3` | `1.602176634` |
| `Pe`, `Pi` | `evolve_pressure` | `Pa` | `80.1088317` |
| `NVe`, `NVi` | `evolve_momentum` | `kg / m^2 / s` | `0.001157548211197342` |

The run evolves electron momentum and vorticity. Hermes computes potential from
vorticity through an elliptic inversion that can include ion-pressure terms;
see the [Hermes-3 equation documentation](https://hermes3.readthedocs.io/en/latest/equations.html).
Therefore the inherited C5 set `Ne, Te, Ti, phi, Vi` is a useful observable
baseline but is not assumed to be a complete Markov state. C5-versus-augmented
state sufficiency must be tested at the teacher-forced one-step rung.

A subsequent exact source-state review sharpens this statement: the six
volumetric solver variables are `[Ne, Pe, Pi, NVe, NVi, Vort]`; `phi` is a
derived elliptic field with short radial-boundary memory. The direct independent
field most clearly absent from C5 is electron momentum. See
[`PHASE1_STATE_TIME_DECISION_MEMO.md`](PHASE1_STATE_TIME_DECISION_MEMO.md).

## Frozen candidate split and normalization

The predeclared candidate split was:

| Region | Indices | Frames | Status |
|---|---:|---:|---|
| train | `[0,432)` | 432 | candidate only |
| guard | `[432,496)` | 64 | unused |
| validation | `[496,624)` | 128 | candidate only |

The 64-frame guard remains safely longer than the maximum allowed 32-frame
training window. The split is chronologically leakage-safe, but it is **not yet
accepted as a stationary train/validation split**.

Candidate training-only normalization, fit over exactly 77,856,768 cells per
field, is:

| Field | Transform | Mean | Standard deviation |
|---|---|---:|---:|
| `Ne` | `ln(x + 1e-6)` | -1.936845 | 1.435363 |
| `Te` | identity | 0.929182 | 0.532093 |
| `Ti` | identity | 1.260845 | 0.470183 |
| `phi` | identity | 2.848521 | 1.279659 |
| `Vi` | identity | -0.176674 | 0.920958 |

These values are reproducible and training-only, but remain candidate
normalization until the split decision is resolved.

## Steady-state screen

The frozen screen divided the full sequence into eight blocks of 78 frames.
Every reported series exceeded the maximum normalized block-mean range of
`1.0`. The result is an operational engineering failure, not a mathematical
proof that no stationary stochastic description exists.

| Series | Fitted drift / temporal SD | First-half shift / pooled SD | Block-mean range / temporal SD |
|---|---:|---:|---:|
| `Ne` spatial mean | -0.442 | +0.446 | 2.407 |
| `Ne` fluctuation RMS | +2.629 | -1.821 | 3.102 |
| `Te` spatial mean | +1.697 | -0.887 | 2.694 |
| `Te` fluctuation RMS | +0.168 | -0.255 | 2.354 |
| `Ti` spatial mean | +1.888 | -0.884 | 2.911 |
| `Ti` fluctuation RMS | -1.350 | +0.704 | 2.417 |
| `phi` fluctuation RMS | +0.004 | -0.131 | 1.280 |
| `Vi` spatial mean | -1.157 | +0.875 | 2.852 |
| `Vi` fluctuation RMS | +0.893 | -0.438 | 2.399 |

This pattern is consistent with slow profile/amplitude evolution superposed on
fast fluctuations. It may be expected in a flux-driven transport simulation,
but that simulator interpretation needs confirmation from Ben or the run
producer. The failure cannot be fixed by randomly shuffling windows.

## Decorrelation ladder

All estimates below use candidate training frames only. They are diagnostic
under nonstationarity and cannot select the split.

### Full mean-removed pattern

Removing one scalar mean per frame leaves the slowly varying axisymmetric
profile. First `1/e` crossing times are:

| Field | Frames | Microseconds |
|---|---:|---:|
| `Ne` | 19.042 | 59.638 |
| `Te` | 0.819 | 2.565 |
| `Ti` | 2.244 | 7.029 |
| `phi` | 10.222 | 32.013 |
| `Vi` | 1.680 | 5.262 |

The frozen representative median is `2.244` frames or `7.029 microseconds`.
The curves are non-monotone, so the first crossing must be read together with
the full curve and integrated/zero-crossing times. `Ne` retains a long positive
tail and has no non-positive crossing within 108 frames.

### Toroidal residual

Define the model-coordinate toroidal residual at every frame and `(x,y)` by

```text
delta_z X = X - mean_z(X).
```

This removes stored `k=0`, equivalent to full-torus `n=0`. Its fixed-grid first
`1/e` crossings are all below one saved frame:

| Field | Crossing (frames) | Crossing (microseconds) | Non-axisymmetric fraction of temporal variability |
|---|---:|---:|---:|
| `Ne` | 0.952 | 2.980 | 0.400 |
| `Te` | 0.530 | 1.660 | 0.508 |
| `Ti` | 0.729 | 2.283 | 0.602 |
| `phi` | 0.566 | 1.772 | 0.149 |
| `Vi` | 0.910 | 2.851 | 0.656 |

Potential's temporal variability is dominated by its axisymmetric component in
this uniform sampled model coordinate; ion temperature and ion velocity have
larger non-axisymmetric fractions. These are not geometry-weighted energies.

### Toroidal translation oracle

The future frame can be circularly shifted only for verification. At one-frame
lag:

| Field | Fixed-grid correlation | Shift-aligned correlation | Best shift (cells) | Best shift (full-torus degrees) | Mean mode coherence `k=4..7` / `n=20..35` |
|---|---:|---:|---:|---:|---:|
| `Ne` | +0.336 | 0.516 | -10 | -8.182 | 0.393 |
| `Te` | -0.193 | 0.662 | -11 | -9.000 | 0.532 |
| `Ti` | +0.133 | 0.488 | -11 | -9.000 | 0.326 |
| `phi` | -0.118 | 0.638 | -12 | -9.818 | 0.542 |
| `Vi` | +0.306 | 0.482 | -9 | -7.364 | 0.173 |

The similar optimal shift across all fields and approximately linear Fourier
phase progression with `k` support a coherent translation interpretation. The
fixed-grid loss is therefore not equivalent to complete physical incoherence.

Within the stated target band, lag-one magnitude coherence is generally highest
at `k=4` (`n=20`) and declines toward `k=7` (`n=35`). Examples:

| Field | `k=4`, `n=20` | `k=5`, `n=25` | `k=6`, `n=30` | `k=7`, `n=35` |
|---|---:|---:|---:|---:|
| `Ne` | 0.677 | 0.476 | 0.221 | 0.197 |
| `Te` | 0.797 | 0.652 | 0.380 | 0.298 |
| `Ti` | 0.622 | 0.391 | 0.157 | 0.133 |
| `phi` | 0.833 | 0.681 | 0.362 | 0.294 |
| `Vi` | 0.378 | 0.158 | 0.077 | 0.079 |

Coherence is not power. Phase 2 must report the power carried by each mode
before deciding which loss of coherence is physically material.

## Consequences for emulator design

### 1. Do not use absolute time as a lookup key

The context fields already encode the evolving profiles. With one trajectory,
absolute frame number would invite memorization and would not generalize to a
second run. Keep physical cadence and relative lead explicit; add `delta_t`
conditioning only if multiple cadences are trained.

### 2. Separate background, phase transport, and stochastic residual

A promising representation is conceptually:

```text
future state = evolving n=0 background
             + translated/coherently evolved n>0 component
             + stochastic residual uncertainty.
```

This does not prescribe one architecture. It supplies a failure decomposition
that FGN, diffusion, PDE-Refiner, and joint-residual approaches must each pass.

### 3. Preserve periodic translation equivariance

Toroidal padding, tokenization, and positional encodings must respect the
periodic `z` symmetry. A model that treats the 88-cell toroidal axis as an
ordinary bounded image direction can turn coherent phase motion into blur.

### 4. Treat f8 spectral capacity as an oracle question

The legacy f8 codec has only 11 latent toroidal cells, whose directly resolved
Nyquist index is `k=5` (`n=25`). It cannot be presumed to carry independent
phase information at `k=6--7` (`n=30--35`). Whether that matters depends on
ground-truth power, cross-phase, and transport contribution. Phase 2 must test
codec reconstruction mode by mode before any new diffusion training.

### 5. Test state sufficiency before architecture complexity

The raw simulation advances `Ne`, both pressures, both parallel momenta, and
`Vort`; the C5 observables omit direct vorticity and electron momentum.
Potential and ion pressure contain enough information to make direct vorticity
partly redundant, but the exact elliptic relation also carries radial-boundary
memory. A matched one-step observed-state-versus-source-state test can reveal
an irreducible partial-observation error that no decoder or noise schedule will
repair.

### 6. Long rollouts have a different target than short forecasts

Before decorrelation, evaluate trajectory and phase accuracy. After
decorrelation, evaluate conditional distributions, spectra, cross-field phase,
transport, and calibration. Pixelwise agreement with one future realization is
not the correct long-horizon target.

## What is now established

- Exact chronology, cadence, axis order, toroidal fraction, and mode mapping.
- Raw units and conversion factors for C5 and augmented-state candidates.
- A leakage-safe candidate boundary with a 64-frame guard.
- Exact candidate training-only normalization.
- Failure of the predeclared whole-interval stationarity screen.
- Strong separation between axisymmetric memory and fixed-grid
  non-axisymmetric memory.
- Substantial recovery under a common toroidal translation.
- Meaningful one-step phase coherence at `n=20--25`, with weaker coherence by
  `n=30--35`.
- C5 is an observable baseline, not a verified complete dynamical state.

## What is not established

- A statistically steady learning interval.
- Preservation of every term in a complete heat-flux definition. Subsequent O1
  work has separately measured codec spectra, cross-field structure, and
  radial ExB particle/internal-energy transport.
- One-step or rollout superiority of any learned architecture.
- Probabilistic calibration of fields, modes, or flux.
- Correct particle or heat flux implementation.
- Any diagnostic ranking under a validated forecast prior.
- Any new result on 85606.
- Any experimental realism or steering/control claim.

## Questions requiring Ben or simulator-owner input

1. Is normalized time `285000` already after the intended burn-in, or is this
   stored interval expected to continue relaxing on a transport timescale?
2. Is there a known statistically steady suffix, a longer continuation, or a
   run-level diagnostic that should define steady state?
3. For this exact Hermes configuration, should the emulator use the six evolved
   variables `[Ne, Pe, Pi, NVe, NVi, Vort]`, or should `phi` also be predicted
   redundantly for fast transport evaluation?
4. Is the one-microsecond relaxed radial `phi` boundary state material at the
   3.132-microsecond saved cadence, and should it be retained explicitly?
5. Is the approximately 9--12-cell per-frame toroidal phase advance an expected
   physical propagation rate in these field-aligned coordinates?

## Execution ledger

Every attempt used a unique output directory; no failed artifact was
overwritten.

| Job | Outcome | Meaning |
|---|---|---|
| `6890522` | failed before profiling | raw conversion metadata was a one-element array, not a scalar |
| `6890531` | failed before profiling | literal Vi unit was `m / s`, not manifest spelling `m/s` |
| `6890544` | failed after calculation, before valid output | auxiliary HDF5 object references were not JSON-serializable; the partial file was rejected |
| `6890563` | completed | immutable steady-state failure and candidate normalization |
| `6890591` | completed | A002 full-pattern diagnostic decorrelation |
| `6890601` | completed | A003 axisymmetric/non-axisymmetric decomposition |
| `6890606` | completed | A004 shift-aligned and complex-mode coherence oracle |

The latest exact compact result is
`paper0/results/phase1_85604_profile_6890606.json`, SHA-256
`9ef0868a21ebbee883f154f13fe4068d50d47474017cf775ba3d5c3e51b7fc15`.

## Phase 1 decision

**Characterization passes; the learning split does not.** Proceed to Phase 2
metric and oracle validation, because those tests can be developed on synthetic
known-answer fields and 85604-only codec reconstructions. Do not launch a new
learned baseline until the steady-interval question is resolved or a
prospectively documented nonstationary conditional-training protocol replaces
that requirement.
