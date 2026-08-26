# Persistent global-local four-frame model: physics readout

**Result date:** 2026-08-26

**Development simulation:** old 85604 only

**Held-out 85606, newer NERSC data, and guard interval:** not read

**Forecast job:** `6938347` (`COMPLETED`, exit `0:0`)

**Physics-scoring job:** `6938348` (`COMPLETED`, exit `0:0`)

## Result in one sentence

The persistent global-local four-frame model produced the strongest
four-frame **ensemble mean** so far, but its stochastic members did not span
the domain-scale transport uncertainty; it passed four of seven frozen
physics families and failed the three covariance/calibration families.

`status: completed_failed` means that both jobs completed normally and the
prospective scientific gate failed. It is not an execution error.

## What improved

Relative to the deterministic four-step parent used by the frozen scorer:

| Evaluation quantity | Parent | Candidate | Change |
|---|---:|---:|---:|
| Median absolute log spectral-power error | `0.7563` | `0.3313` | `56.2%` lower |
| Normalized `Ne`–`phi` complex cross-spectrum error | `0.2355` | `0.2157` | `8.4%` lower |
| Amplitude-weighted `Ne`–`phi` phase error | `2.507°` | `1.943°` | `0.565°` lower |
| Median integrated-transport mean relative L2 | `0.4429` | `0.2813` | `36.5%` lower |

The fair field CRPS was also below the selected-mean MAE at both evaluated
horizons: `0.03395 < 0.04357` at one frame and `0.05535 < 0.07038` at four
frames. These are real gains in the conditional mean and marginal forecast
distribution.

## What failed

The ensemble remained underdispersed and its spatial covariance was nearly
uninformative:

| Frozen uncertainty gate | Required | Observed | Result |
|---|---:|---:|:---:|
| Local transport corrected spread/skill | at least 3 of 4 quantities in `[0.8,1.25]` | `0.592–0.602` | fail |
| Integrated transport median corrected spread/skill | `>= 0.60` | `0.2211` | fail |
| Median spatial transport-covariance relative Frobenius error | `<= 0.90` | `0.9958` | fail |

An error of approximately one for the covariance comparison means that the
estimated covariance is scarcely better than a zero-covariance reference in
the scorer's normalized Frobenius metric. Increasing the number of generated
members would estimate this wrong distribution more precisely; it would not
repair the distribution.

The integrated spread/skill by quantity was:

| Quantity | Corrected spread/skill |
|---|---:|
| Particle flux | `0.5143` |
| Electron internal-energy flux | `0.1484` |
| Ion internal-energy flux | `0.2664` |
| Total internal-energy flux | `0.1759` |

## Comparison with earlier stochastic models

The current integrated median spread/skill, `0.2211`, is below the previous
joint residual models: approximately `0.4255` for the unconditional model,
`0.4575` for the context-conditioned model, and `0.362–0.365` for the
equivariant variants. The new architecture therefore did not trade a better
mean for a better ensemble; it improved the mean while the stochastic branch
collapsed further.

This closes the narrow hypothesis that persistent low-mode noise plus a
global-local decoder, by itself on the reduced five-field trajectory, is
sufficient to recover integrated transport covariance.

## Training-budget caveat

The selected checkpoint was epoch 20, the last authorized epoch. The frozen
validation score decreased monotonically from `2.2213` at epoch 2 to `1.7888`
at epoch 20, although the epoch-18 to epoch-20 gain was only about `0.38%` and
the learning rate had reached `1e-6`.

Thus this was the best checkpoint inside the fixed pilot budget, not proof of
asymptotic convergence. A longer run might improve the denoising objective,
but the covariance miss is too large to claim that duration alone will repair
it. Any duration test must be a separately matched, prospective comparison.

## Scientific consequence

The result sharpens the distinction between two tasks:

1. The data and operator contain enough signal to improve short-horizon
   spectra, cross-field phase, and mean transport.
2. The reduced trajectory has not supported learning the correct conditional
   domain-scale covariance.

The next experiment therefore tests a different causal hypothesis: whether
the saved Hermes evolved coordinates support better finite-time and bounded
rollout dynamics than the derived five-field view when both receive the same
codec-free multi-lead curriculum. It does not add another stochastic branch.

## Provenance

Authoritative compact result:

```text
paper0/results/post_ecrd_old_85604_persistent_global_local_physics_6938348.json
SHA-256 ad9f9d4ac63fee3da2d6e0d1cf7844f72368c0233123169617c20eb5a2b598af
```

Authoritative forecast:

```text
/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_persistent_global_local_physics_evaluation/job_6938347/generation/forecast_M32_four_frame.h5
SHA-256 60a6926dde5c765081b7ac578966036c0769aff8c538317265e93673da10f4ae
```

The frozen 32-member evaluation used 36 starts, four future frames, and no
target truth during generation. Forecast generation took a median `73.73 s`
per start with 18 diffusion steps and 35 network evaluations per member.

W&B: [finished scoring run](https://wandb.ai/sdelaurentiis123-columbia-university/tcv-diagnostics-paper0/runs/p0oldpglscore-j6938348-g6938347).
