# Old-85604 matched state-view multi-lead protocol

**Frozen prospectively:** 2026-08-26, after the persistent global-local
physics result and before any saved-state multi-lead optimizer update

**Development simulation:** old 85604 only

**Held-out 85606, newer NERSC files, and guard interval:** unopened and
prohibited

## Scientific question

Does the saved Hermes evolved-state representation benefit from the same
multi-lead curriculum that repaired the reduced five-field transition, and
does that representation improve bounded forecasts of the derived fields and
transport?

This is a matched **state-view** experiment. It is not an architecture search,
a stochastic-model experiment, or a claim that the saved arrays contain every
piece of internal solver state.

## Why this experiment is not a duplicate

The existing matched one-step experiment used three seeds and found that the
saved-state arm had `1.4603` times the shared `Ne`/`Pe`/`Pi` validation MSE of
the reduced five-field arm. Adding current `phi` and `Vi` as causal auxiliary
inputs recovered only `2.12%` of the saved-state error in a one-seed screen.

Separately, five-lead supervision of the reduced five-field model improved its
mean persistence-normalized error to approximately `0.488` across three seeds,
and bounded autonomous one-frame updates preserved substantially more
spectral power and separatrix transport than direct coarse predictions.

No experiment has applied that successful temporal curriculum to the saved
Hermes state, reconstructed its derived fields without future truth, and
evaluated the resulting transport. This protocol fills exactly that gap.

## Immutable data contract

Use the already validated old-85604 model artifact only:

```text
/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525
```

- 624 saved frames;
- cadence `3.131905426352636 microseconds`;
- training frames `[0,432)`;
- unread guard frames `[432,496)`;
- chronological validation frames `[496,624)`;
- normalization fitted on training frames only;
- one-frame history;
- lead steps `[1,2,4,8,16]`;
- 2,129 training pairs and 609 validation pairs;
- shared random circular toroidal-roll augmentation in training only;
- `zperiod=5`, so stored toroidal Fourier index `k` maps to physical mode
  number `n=5k`.

Temporal pairs are correlated constraints from one trajectory. They are not
reported as independent physical samples.

The two state views are:

```text
C5P = [Ne, Pe, Pi, phi, Vi]
E6B = [Ne, Pe, Pi, NVe, NVi, Vort] + Bphi
```

`E6B` is called the **saved evolved-state view** in conclusions. It is not
silently promoted to a complete simulator restart state.

## Matched model and parent checkpoints

Both arms use the existing `CodecFreeIncrementOperator3D`:

- base width 24;
- channel multipliers `[1,2,4]`;
- two residual blocks per level;
- 128-channel lead embedding;
- circular padding and stride one along the toroidal axis;
- downsampling only along the two nonperiodic axes;
- joint output of every field in the selected state view;
- retained two-sided `Bphi` head for E6B;
- no codec, latent bottleneck, stochastic generator, or physics-derived loss.

Initialize seed 1701 strictly from the selected 12-epoch Stage-1 parents:

| State view | Parent metric | Checkpoint SHA-256 |
|---|---:|---|
| C5P | `0.005322341561633884` | `887dfcbe37e214f7217a5d4b900381cea370ca2e2c96687d2d6cd92c9e951c33` |
| E6B | `0.007772147896373167` | `020367a9a8a964c6ce9f2cdf0e9850fd7b8ee4ff82f936f6c6153bee7d269bea` |

Load model weights strictly and verify bitwise equality. Reset AdamW in both
arms; do not restore either parent optimizer.

## Prospective paired pilot budget

The earlier C5P four-epoch fine-tune selected the last epoch. To prevent an
unequal or obviously truncated duration comparison, this protocol reruns both
arms under one new matched budget rather than comparing a longer E6B run to
the historical four-epoch C5P result.

- seed: `1701` for both arms;
- epochs: 12;
- sample batch size: one transition;
- gradient accumulation: four transitions;
- expected optimizer updates: `ceil(2129/4) * 12 = 6,396` per arm;
- AdamW, peak learning rate `5e-5`, minimum `5e-6`, weight decay `1e-4`;
- 5% linear warmup followed by cosine decay;
- gradient clipping at norm `1.0`;
- bfloat16 autocast, TF32 disabled;
- identical pair order and augmentation seed across state views;
- required online W&B;
- one right-sized Rusty GPU per array task.

Fit derivative RMS scales independently for each state view using all 2,129
training pairs. Optimize the equal-component, persistence-normalized direct
state-derivative MSE. The E6B loss includes all six evolved volume variables
and the two `Bphi` sides. Flux, spectra, cross-phase, coherence, conservation,
and PDE quantities remain outside the loss.

## Checkpoint selection

Evaluate every epoch separately at each lead. For lead \(\ell\), define

\[
q_\ell =
\frac{
  \operatorname{MSE}_{\mathrm{model}}(Ne,Pe,Pi;\ell)
}{
  \operatorname{MSE}_{\mathrm{persistence}}(Ne,Pe,Pi;\ell)
}.
\]

Select the checkpoint minimizing the unweighted mean of \(q_\ell\) over
\(\ell\in\{1,2,4,8,16\}\). The shared-field selector gives the two state
views an identical comparison target. Report, but do not use for selection,
every predicted field and both boundary sides.

If the selected checkpoint is epoch 12, state explicitly that the duration
question remains censored at the budget boundary. Do not silently extend one
arm or tune the learning-rate schedule after seeing the result.

## Mechanical and transition gates

Each arm must pass:

1. exact pair and optimizer-update counts;
2. finite training and validation metrics;
3. decreasing epoch-mean training loss;
4. exact selected-checkpoint reload;
5. integer toroidal-shift equivariance within the existing tolerance;
6. positive persistence-relative skill for every predicted volume field at
   every lead;
7. positive persistence-relative skill for each E6B boundary side at every
   lead;
8. false flags for 85606 access, newer-data access, target-truth use during
   forecasting, and physics-derived training losses.

The paired pilot advances to derived-field physics evaluation only if both
arms are mechanically valid. A failed E6B mechanical gate is repaired without
changing the scientific budget; a scientific failure is not tuned away.

## Causal derived-field reconstruction

For E6B forecasts:

\[
V_i = \frac{N V_i}{2\,\operatorname{softFloor}(N_e,10^{-7})}.
\]

Reconstruct `phi` by applying the already validated, pinned Hermes/BOUT++
elliptic operator to the **predicted** `Ne`, `Pi`, `Vort`, and predicted
`Bphi`, using the frozen geometry and settings. No target-frame field or
boundary value may enter this solve. Truth replay must still pass before
candidate forecasts are scored.

C5P predicts `phi` and `Vi` directly. Both arms are then compared in the same
physical coordinate set `[Ne,Pe,Pi,phi,Vi]`.

## Frozen bounded evaluation

Using every eligible validation start, compare at terminal horizons four and
eight:

1. persistence;
2. one direct terminal prediction;
3. repeated autonomous lead-one predictions;
4. repeated autonomous lead-two predictions;
5. at horizon eight, two autonomous lead-four predictions.

Every composition feeds the complete predicted state view back into the next
call. E6B propagation does not require an intermediate truth `phi`; derived
coordinates are reconstructed only from predicted state. All methods at one
horizon use identical starts and targets.

Report:

- shared and per-field normalized RMSE and persistence-relative skill;
- physical toroidal spectra in `k=1–3`, `k=4–5`, and `k=6–7`, corresponding
  to `n=5–15`, `n=20–25`, and `n=30–35`;
- `Ne`–`phi`, `Pe`–`phi`, and `Pi`–`phi` complex cross-spectrum, phase, and
  coherence;
- local radial-face particle and heat-transport error;
- separatrix-integrated transport relative L2 and correlation;
- transport error by chronological validation block.

All nonlinear diagnostics are computed after inference. This deterministic
experiment has no ensemble calibration or covariance claim.

## Prospective state-view decision

Favor the E6B saved-state view only if, across horizons four and eight and the
three chronological validation blocks, it:

1. improves the median separatrix-transport relative L2 over the paired C5P
   arm by at least 10%;
2. improves the median complex pressure/density–potential cross-spectrum
   error;
3. does not worsen shared `[Ne,Pe,Pi]` state error or median spectral-power
   error by more than 10%; and
4. has no failed evolved-field persistence skill or causal reconstruction
   gate.

If E6B meets these conditions, freeze the same pair at three seeds before any
stochastic experiment. If it does not, retain C5P as the old-data performance
control and record that the tested saved-state parameterization did not repair
the transition under matched optimization. That outcome does not prove that
hidden state is irrelevant; it may instead identify state parameterization,
elliptic coupling, cadence, or data diversity as the next bottleneck.

## Prohibited scope

This protocol does not authorize:

- 85606 or the newer NERSC files;
- assimilation, ETKF/EnKF, diagnostic ranking, or steering;
- a new stochastic generator, diffusion schedule, codec, GAOT, FNO, or
  adversarial loss;
- independent-window sample-size claims;
- any physics quantity in the training objective;
- a state-completeness claim beyond the saved E6B+Bphi arrays.
