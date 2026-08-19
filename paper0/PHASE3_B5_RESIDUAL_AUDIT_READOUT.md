# Phase 3 B5 deterministic-mean residual audit readout

**Status:** completed architecture-sizing audit; no B5 model has been trained

**Development simulation:** 85604 training targets only

**Sequestered simulation:** 85606 remained closed

## Bottom line

The exact frozen C5P-H1 deterministic mean is already close to its 85604
training targets at one saved-frame lead, but its remaining error is not
independent white noise. It is:

1. strongly dependent on field and spatial region;
2. jointly coupled across fields, especially between `Pe` and `Pi`;
3. spatially coherent over several cells, including the periodic toroidal
   direction;
4. concentrated disproportionately in non-axisymmetric toroidal modes; and
5. rapidly changing in detailed pattern while its overall amplitude changes
   more slowly.

Therefore the primary B5 experiment should be one **joint, conditional,
field-coordinate residual generator** for all five fields. It should retain
the full 88-cell toroidal domain, draw a new innovation at every
autoregressive step, and condition the innovation amplitude and structure on
the current state and frozen H1 mean. Five independent residual models, a
single scalar noise amplitude, or another DCAE-latent corrector are not
supported as the primary design.

This is an architecture-sizing conclusion, not a validation result. The H1
parent was trained on these same targets, so the measured residual is
in-sample model error. It cannot be identified with irreducible aleatoric
uncertainty from one realized trajectory per context.

## Execution and provenance

Rocky 9 H100 job `6901393` completed with Slurm state `COMPLETED`, exit `0:0`,
and elapsed time `00:17:29`. It used exact Paper 0 commit
`88fdcc8fa9ce7e2ba24958cd873cb7c4c5a771ff`, passed the complete in-job suite
(`1011 passed, 1 skipped, 29 subtests passed`), verified all eight staged data
shards, and finished its online W&B run.

The job generated the complete context-only H1 forecast before constructing a
target reader. It then closed and hashed the forecast as
`d1342e53ac5f9077665f7a707ca035b9069a616d7db10371e1f6c1ae504d9c18`.
Only after that boundary did it open training truth and form

\[
r_t=x_t-\mu_{\mathrm{H1}}(x_{t-1}).
\]

The canonical residual tensor contains 430 targets and has shape
`[430,5,64,32,88]` in field order `[Ne,Pe,Pi,phi,Vi]`. Guard and validation
frames were not read. The full residual record has SHA-256
`d8f05af4d1619fbe7ad8774360c29aa38cd69494d80624d55e30fdb57361cf67`.

## Residual magnitude

All values below use the decoded training-standardized field coordinates.

| Field | RMS | MAE | signed bias | residual / target variance |
|---|---:|---:|---:|---:|
| `Ne` | 0.05503 | 0.03527 | +0.000369 | 0.00303 |
| `Pe` | 0.04826 | 0.02450 | -0.000040 | 0.00233 |
| `Pi` | 0.06097 | 0.03171 | -0.000507 | 0.00371 |
| `phi` | 0.04633 | 0.03212 | +0.000358 | 0.00214 |
| `Vi` | 0.10252 | 0.05887 | +0.000456 | 0.01051 |

These low values establish that B5 is correcting a small one-step residual
rather than relearning the whole field. They do **not** establish validation
skill or calibrated uncertainty. `Vi` is the clear field-level outlier: its
residual variance is roughly one percent of target variance, versus about
0.2--0.4 percent for the other fields.

The residual scale is strongly heteroscedastic. The ratio between the 95th
and 5th percentile of pointwise residual standard deviation is `6.00` for
`Ne`, `33.40` for `Pe`, `31.55` for `Pi`, `6.18` for `phi`, and `12.85` for
`Vi`. A global per-field scale is useful for numerical normalization, but it
is not an adequate model of conditional local spread.

## Geometry dependence

The same parent fails in different places for different fields:

| Authoritative region | `Ne` RMS | `Pe` RMS | `Pi` RMS | `phi` RMS | `Vi` RMS |
|---|---:|---:|---:|---:|---:|
| confined edge | 0.0317 | 0.1151 | 0.1339 | 0.0723 | 0.0393 |
| outboard midplane | 0.0409 | 0.0785 | 0.1009 | 0.0583 | 0.0430 |
| inner divertor leg | 0.0715 | 0.0128 | 0.0144 | 0.0349 | 0.1792 |
| outer divertor leg | 0.0573 | 0.0171 | 0.0177 | 0.0309 | 0.0688 |
| private flux | 0.1007 | 0.0087 | 0.0107 | 0.0336 | 0.1280 |
| scrape-off layer | 0.0434 | 0.0310 | 0.0462 | 0.0431 | 0.1050 |
| separatrix cell band | 0.0508 | 0.0609 | 0.0770 | 0.0660 | 0.1128 |
| X-point stencil | 0.0455 | 0.0387 | 0.0512 | 0.0465 | 0.0485 |

Pressure errors are largest in the confined edge and outboard midplane;
`Ne` is largest in private flux and the inner divertor; `Vi` is largest in
the inner divertor and separatrix band. This supports explicit spatial
position awareness and dynamic conditioning. It does not justify using the
evaluation-region masks as loss weights.

## Spatial dependence and model extent

The table reports the first `1/e` crossing and, in parentheses, the first
three-lag stable near-zero distance with `|rho| <= 0.10`.

| Axis | `Ne` | `Pe` | `Pi` | `phi` | `Vi` |
|---|---:|---:|---:|---:|---:|
| nonperiodic `x` | 2 (3) | 2 (7) | 3 (8) | 4 (7) | 1 (2) |
| nonperiodic `y` | 1 (1) | 1 (1) | 1 (1) | 1 (1) | 1 (1) |
| periodic stored `z` | 7 (11) | 3 (8) | 3 (7) | 4 (10) | 6 (8) |

Under the frozen CorrDiff-inspired sizing rule, any later nonperiodic patch
must span at least 17 cells in `x` with at least 8 cells of overlap, and at
least 3 cells in `y` with at least 1 cell of overlap. The primary smoke should
still try the full `64 x 32 x 88` field first. The entire periodic `z` extent
remains mandatory: a short local autocorrelation distance does not remove the
domain-wide phase information of the material `k=1` mode.

## Temporal dependence

The detailed residual pattern becomes anticorrelated immediately: every
field has its first nonpositive and `1/e` crossing at one saved frame
(`3.1319` microseconds). Stable near-zero pattern correlation occurs at frame
3 for `Ne` and frame 2 for every other field. Its positive-lobe integral scale
is only 0.5 frame.

The per-frame residual RMS changes more slowly:

| Field | RMS-amplitude `1/e` lag | stable near-zero lag |
|---|---:|---:|
| `Ne` | 11 frames | 22 frames |
| `Pe` | 4 frames | 16 frames |
| `Pi` | 4 frames | 6 frames |
| `phi` | 4 frames | 6 frames |
| `Vi` | 5 frames | 19 frames |

The defensible interpretation is that a rollout should draw a new detailed
innovation at each step, while the distribution of that innovation remains
conditioned on a more slowly varying context. It would be incorrect to reuse
one fixed noise draw for an entire trajectory. It would also be incorrect to
call the one-step pattern decorrelation proof of irreducible randomness.

## Cross-field dependence

The global centered residual correlation matrix, ordered
`[Ne,Pe,Pi,phi,Vi]`, is

\[
\begin{pmatrix}
1 & .394 & .356 & .079 & .169 \\
.394 & 1 & .804 & .116 & .048 \\
.356 & .804 & 1 & .219 & .014 \\
.079 & .116 & .219 & 1 & -.118 \\
.169 & .048 & .014 & -.118 & 1
\end{pmatrix}.
\]

Its eigenvalues are `[2.1282, 1.1531, 0.8514, 0.6789, 0.1884]`, with entropy
effective rank `4.047` and participation-ratio rank `3.531`. Regional entropy
effective ranks range from approximately `3.33` to `4.34`. The target is
therefore neither five independent fields nor one common scalar perturbation.
The `Pe`--`Pi` correlation of `0.804` is particularly important for a future
joint heat-transport distribution.

## Toroidal support

Because the simulation stores one fifth of the torus, the physical mapping is
`n=5k`. The table gives residual power divided by truth power in each band:

| Field | `k=1..3` (`n=5..15`) | `k=4..5` (`n=20..25`) | `k=6..7` (`n=30..35`) | `k>=8` (`n>=40`) |
|---|---:|---:|---:|---:|
| `Ne` | 0.237 | 0.310 | 0.857 | 1.056 |
| `Pe` | 0.183 | 0.0527 | 0.422 | 0.876 |
| `Pi` | 0.176 | 0.0574 | 0.423 | 0.885 |
| `phi` | 0.196 | 0.116 | 0.676 | 1.111 |
| `Vi` | 0.503 | 1.307 | 1.825 | 1.516 |

Truth power is dominated by `k=0` equilibrium structure (about 97.9--99.1
percent depending on field), so these ratios must not be confused with the
fraction of total plasma-state energy. They show where the small one-step
error lives: H1 predicts the axisymmetric bulk well but loses a much larger
fraction of non-axisymmetric content. `Vi` is already worse than zero in the
paper-relevant `n=20..35` band, while `Ne` and `phi` also degrade sharply by
`n=30..35`. A field-space residual corrector needs enough toroidal resolution
and capacity to address this without any spectral training loss.

## What this authorizes next

The result supports writing a separate implementation/smoke protocol with the
following primary design:

- frozen H1 deterministic mean;
- one joint five-field residual generator in decoded standardized field space;
- conditioning on `x_{t-1}` and the frozen H1 mean;
- per-field residual scaling without subtracting the learned mean structure;
- full toroidal domain with periodic operations;
- a new innovation at every autoregressive step;
- no physics-derived training loss;
- one bounded H100 memory/optimization smoke on training targets only.

The audit does not authorize full B5 training, validation scoring, O3,
assimilation, diagnostic ranking, or 85606 access. Those remain closed until
the implementation smoke is separately frozen, executed, and reviewed.

## Immutable artifacts

The scientific root is
`/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_h1_residual_audit/job_6901393`.
The W&B mirror is
<https://wandb.ai/sdelaurentiis123-columbia-university/tcv-diagnostics-paper0/runs/p0b5resid-6901393-s1701>.
Local Ceph artifacts, not W&B, are the scientific authority.
