# Phase 3 B3 full-training and one-step evaluation protocol

**Decision status:** frozen after the bounded B3 implementation smoke and
before full B3 training, checkpoint selection, scientific ensemble generation,
or B3 evaluation implementation

**Development simulation:** TCV/Hermes `85604`

**Sequestered simulation:** `85606`

**85606 access authorized:** no

**Newly authorized scope:** one full seed-1701 B3-FGN-H1 training run and one
one-saved-step probabilistic evaluation on the existing 85604 validation
interval

The machine-readable authority is
`paper0/manifests/phase3_b3_full_evaluation_85604.json`.

## 1. Decision boundary

Rocky 9 H100 job `6898604` established only that the deterministic H1 parent,
new global-noise path, frozen codec, decoded-field fair-CRPS loss, staged
optimizer, fixed validation noise, checkpoint reload, canonical forecast
interface, and online W&B tracking work together. Its compact result has
SHA-256 `dbac54c033917abbfec7e380d96a0c9be93667ae58240b4403400b57c76e2808`.
It used 16 training targets, four validation targets, two epochs, and two
optimizer steps. Those losses are not scientific results.

This protocol asks three narrower development questions:

1. Can the FGN retrofit turn the selected deterministic H1 transition into a
   useful one-step ensemble without materially sacrificing point skill?
2. Does improved marginal calibration extend to material toroidal modes and
   member-wise nonlinear transport?
3. Does one seed justify a separately frozen three-seed replication, or should
   Paper 0 move to the next model rung?

This remains an O2-style one-saved-step experiment. It does not authorize
autoregressive O3/O4 rollout, additional FGN seeds, ablations, assimilation,
diagnostic ranking, 85606 access, or control claims.

## 2. Hypothesis and failure interpretation

B3 injects one low-dimensional global random vector into every transformer
block and fine-tunes the complete deterministic transition with marginal fair
CRPS. The hypothesis is that this produces conditionally diverse forecasts
whose ensemble spread tracks one-step error while retaining the parent model's
joint field structure.

Marginal fair CRPS does not constrain density-potential cross-phase or radial
transport directly. Therefore:

- better field CRPS with wrong cross-phase or transport is a failed B3 physics
  gate and evidence that marginal CRPS is insufficient here;
- nonzero member variation without correct coverage is not calibration;
- good ensemble-mean RMSE without member-wise transport fidelity is not a
  transport-faithful probabilistic forecast;
- a one-seed pass is provisional development evidence, not an architecture-
  level or held-out claim.

## 3. Immutable full training

Full training uses exactly the architecture and objective accepted by the
bounded smoke. It starts from, rather than replaces, the selected deterministic
`C5P-H1` seed-1701 transition.

- fields: `[Ne,Pe,Pi,phi,Vi]`;
- context: frame `t-1` only;
- target: frame `t`;
- absolute or normalized frame time: absent;
- training targets: `[2,432)`, all 430 once per epoch;
- validation targets: `[498,624)`, all 126 chronologically every epoch;
- guard frames `[432,496)`: unread;
- codec: frozen seed-1701 `C5P-dcae_l10`;
- latent normalization: existing training-only seed-1701 H1 artifact, no
  refit;
- global raw noise: independent 32-component standard Gaussian vector per
  member;
- training members: two;
- training objective: equal-channel decoded standardized-field fair CRPS;
- epochs: 100;
- microbatch: one target;
- gradient accumulation: 16 targets;
- optimizer steps per epoch: 27;
- total optimizer steps: 2,700;
- optimizer: AdamW with betas `[0.9,0.99]` and zero weight decay;
- common-parameter peak learning rate: `3e-5`;
- new-parameter peak learning rate: `1e-4`;
- linear warmup: 270 optimizer steps;
- independent cosine decay to exactly zero after warmup;
- global gradient-norm clip: one;
- precision: bfloat16 autocast;
- early stopping: prohibited.

The final partial accumulation contains 14 targets and is divided by 14, not
16. No physics-derived quantity enters optimization or checkpoint selection.

## 4. Checkpoint selection

A NumPy `Generator(PCG64(31003))` creates one float32 validation bank of shape
`[126,2,32]` in chronological target order. Its bytes are saved and hashed.
The same two noise vectors per target are used at every epoch.

After all 100 epochs, select the earliest epoch attaining the numerically
lowest equal-channel decoded standardized-field fair CRPS over all 126 targets.
No RMSE, spread, spectrum, cross-phase, transport, block, W&B, or 85606 value
may alter selection. The selected transition and final optimizer state are
separate immutable artifacts. A fresh reconstruction must reproduce a fixed
selected-checkpoint probe bit for bit on the same allocated device.

The validation interval is both the development selection interval and the
scientific-development scoring interval. Results are therefore explicitly
model-selection results on 85604, not unbiased final-test estimates.

## 5. Independent scientific ensemble

Scientific evaluation does not reuse the M2 selection noise. A NumPy
`Generator(PCG64(31032))` creates a separate float32 bank with shape
`[126,32,32]`, ordered by target frames 498 through 623, then member, then raw
noise feature. The bank is persisted and hashed before forecast generation.

For each target, the model receives only standardized frame `t-1` and that
target's 32 stored noise vectors. It writes one float32 forecast with canonical
axes

~~~text
[target=126, member=32, future_time=1, channel=5, x=64, y=32, z=88].
~~~

Forecast generation and truth scoring are separate processes. The generator
may know target identity but may not open target field data. It closes and
hashes the complete forecast before the scorer opens validation truth.

Member order is immutable. Prefixes of the one stored ensemble define
`M in {4,8,16,32}`. Regeneration, recentering, variance inflation, clipping,
member sorting before storage, rejection sampling, and post-hoc calibration
are prohibited. The model runs in evaluation mode, so dropout is disabled.

## 6. Comparators

The primary matched comparator is the exact deterministic parent:

- arm `C5P-H1`, seed 1701;
- selected checkpoint SHA-256
  `5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e`;
- frozen O2 forecast SHA-256
  `a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed`;
- frozen O2 score SHA-256
  `ebdc707e2be500af7de492038ae8bfb4d126b81b271b340345b85a7fba1d5593`.

Also report persistence, training-only toroidal spectral AR(1), and the
completed B2-LDM-H2 seed-1701 M32 ensemble as secondary context. B2 is not a
matched-history comparator and cannot replace H1 in a B3 gate. Its frozen
forecast SHA-256 is
`0e3f1f2ea7dc733293dab526d0f7312d83f4d62fd9cd6708744900c5cbdb5e18`
and score SHA-256 is
`b9982b48d893865d82d76197fdba6c6b8a6c4886fa139f45c6654a94404fe53e`.

No comparator is retrained or reselected. The same truth, masks, potential
policy, reductions, and metric implementations are used for all comparable
quantities.

## 7. Fair scoring and field coordinates

For scalar truth `y` and exchangeable members `x_1,...,x_M`, lower is better:

\[
\operatorname{fCRPS}_M
=
\frac{1}{M}\sum_{m=1}^{M}|x_m-y|
-
\frac{1}{M(M-1)}\sum_{m<m'}|x_m-x_{m'}|.
\]

The empirical-distribution score is also reported:

\[
\operatorname{eCRPS}_M
=
\frac{1}{M}\sum_{m=1}^{M}|x_m-y|
-
\frac{1}{M^2}\sum_{m<m'}|x_m-x_{m'}|.
\]

Only fCRPS enters the primary gate. The sorted-ensemble identity is required
for voxel-scale scoring; materializing every pairwise 3D difference is
prohibited.

Primary equal-channel scores use the frozen standardized model coordinates.
Before every marginal `phi` error, CRPS, spread, coverage, or rank calculation,
subtract the full spatial mean separately from truth and from each member at
each target. Raw stored-gauge values are descriptive only. Toroidal `k>0`
cross-spectra and radial ExB transport are already invariant to a spatially
constant potential shift.

For every channel and region report ensemble-mean RMSE, MAE, bias, population
variance ratio, anomaly correlation, fCRPS, eCRPS, corrected spread-skill,
coverage, and rank diagnostics. Channels are reduced spatially first and then
given equal weight.

With unbiased within-ensemble variance `s^2`, ensemble mean `xbar`, and truth
`y`, corrected spread-skill is

\[
R_{\mathrm{ss}}
=
\frac{
\sqrt{\frac{M+1}{M}\langle s^2\rangle}
}{
\sqrt{\langle(\bar{x}-y)^2\rangle}
}.
\]

For M32, interval definitions remain `I17=[x_(8),x_(25)]`,
`I27=[x_(3),x_(30)]`, and `I31=[x_(1),x_(32)]`, with nominal exchangeable
coverages `17/33`, `27/33`, and `31/33`. Rank histograms contain 33 bins and
use the previously frozen deterministic tie-breaking rule. Voxel counts are
never treated as independent physical samples.

## 8. Geometry and chronological blocks

Primary disjoint masks are the authoritative strict wall-interior operator
cells in confined edge, private-flux region, and scrape-off layer. Mandatory
secondary masks are separatrix cell band, outboard-midplane row, X-point
topology stencil, inner divertor leg, and outer divertor leg. All masks come
from the hash-locked geometry implementation and artifacts; image coordinates
and legacy five-window probe labels are inadmissible.

Overall metrics use targets `[498,624)`. Temporal robustness uses six fixed,
non-overlapping 21-frame blocks:

~~~text
[498,519), [519,540), [540,561),
[561,582), [582,603), [603,624).
~~~

Each block spans approximately 65.77 microseconds. A temporal rule passes only
if it passes overall and in at least five of six blocks.

## 9. Spectra and cross-field structure

The toroidal FFT is along the 88-cell periodic wedge. The simulation has
`zperiod=5`, so stored Fourier index `k` maps to full-torus mode number
`n=5k`. The frozen training-only bands are

~~~text
k=1..3  <=> n=5..15
k=4..5  <=> n=20..25
k=6..7  <=> n=30..35.
~~~

A field band or cross-field band is material only if the already frozen 85604
training calculation assigns at least 1% of non-axisymmetric power or cross
amplitude. Validation truth cannot select bands.

Field power is computed for each member before ensemble averaging. For pairs
`(Ne,phi)`, `(Pe,phi)`, and `(Pi,phi)`, compute

\[
S_{ab}^{(m)}(k)=a_k^{(m)}b_k^{(m)*},
\qquad
\overline{S}_{ab}(k)=\frac{1}{M}\sum_m S_{ab}^{(m)}(k).
\]

Cross-phase is `arg(overline(S_ab))`; coherence uses member-averaged auto- and
cross-spectra. The product of ensemble-mean fields is prohibited as the
probabilistic cross-spectrum because it discards member covariance.

Report member-expected power, ensemble-mean-field realization coherence,
member-truth realization-coherence distributions, and calibration of band
power. Report cross amplitude, circular cross-phase error, coherence change,
and calibration of real and imaginary cross-spectrum projections. Directional
`x`, `y`, and toroidal spectra are mandatory reports; only frozen material
toroidal bands carry this gate's thresholds.

## 10. Member-wise authoritative transport

For truth and every member independently:

1. inverse-transform `[Ne,Pe,Pi,phi]` without clipping;
2. periodically resample 88 to the authoritative native 81 toroidal cells;
3. apply the frozen geometry-aware radial ExB face operator;
4. reduce strict-face contributions and the outward confined-separatrix wedge.

The four quantities are radial ExB particle, electron internal-energy, ion
internal-energy, and total internal-energy transport. They are not called the
complete experimental heat flux.

The admissible ensemble statistic is

\[
\mathbb{E}_m[\Gamma(x_m)],
\]

not `Gamma(E_m[x_m])`. Report relative L2, normalized bias, correlation,
truth-magnitude-weighted sign disagreement, fCRPS/eCRPS, spread-skill, rank,
coverage, quantiles, and training-threshold event-conditioned performance.
Truth-empty event scopes follow the already committed A016 rule: they are
explicitly N/A after integrity checks and are neither passes nor failures.

## 11. Monte Carlo and conditional uncertainty

All decisions use M32. Repeat fCRPS, eCRPS, spread-skill, and aggregate
transport calibration on stored prefixes M4, M8, and M16. A primary scalar is
Monte Carlo stable only when

\[
|q_{16}-q_{32}| \le 0.10|q_{32}|+10^{-8}.
\]

No regenerated ensemble may rescue a failure.

For temporal uncertainty conditional on 85604, use the frozen moving-block
bootstrap: block length 21, 2,000 replicates, six sampled blocks per replicate,
106 valid starts, seed 85604032. Nonlinear metrics are recomputed inside each
replicate. This is temporal sampling uncertainty within one simulation run,
not cross-shot uncertainty.

## 12. One-seed acceptance gate

Every required scalar must be finite and every required correlation defined.
The selected seed passes only if every applicable condition below passes.

### 12.1 Integrity

1. Exactly 100 epochs, 2,700 optimizer steps, complete fixed-target histories,
   earliest-minimum checkpoint selection, finite losses and gradients, and
   exact same-device checkpoint reload.
2. Exact frozen parent, codec, normalization, protocol, manifest, data, and
   noise banks; no clipping, future truth, absolute time, physics loss, guard
   read, or 85606 access.
3. Exactly 32 finite independently conditioned members, canonical axes, and
   nonzero spread in all five fields and all three primary regions.
4. Forecast generation finishes before target truth is opened.

### 12.2 Field skill and calibration

1. Equal-channel ensemble-mean RMSE and MAE are each at most 1.05 times the
   frozen deterministic H1 seed-1701 values.
2. Equal-channel fCRPS is strictly below both deterministic H1 MAE and the best
   applicable uncompressed-reference MAE.
3. At least four field fCRPS values are strictly below the corresponding H1
   field MAE; the fifth is at most 1.05 times H1.
4. At least four fields have corrected spread-skill in `[0.80,1.25]`; the
   fifth lies in `[0.67,1.50]`.
5. At least four fields have absolute coverage error at most 0.10 for I17,
   0.08 for I27, and 0.06 for I31; the fifth may use twice those tolerances.
   Every primary region has I31 coverage in `[0.75,0.995]`.

### 12.3 Spectra and joint structure

1. Every training-material field band has member-expected power ratio in
   `[0.75,1.30]`.
2. Every training-material field band has ensemble-mean-field realization
   coherence with truth at least 0.80.
3. Every training-material primary cross-field band has circular cross-phase
   error at most 20 degrees and absolute coherence change at most 0.15.
4. Material band-power and cross-spectrum-projection calibration has corrected
   spread-skill in `[0.67,1.50]` and I31 coverage in `[0.75,0.995]`.

### 12.4 Transport

For all four strict-face quantities: relative L2 at most 0.40, correlation at
least 0.70, and weighted sign disagreement at most 0.20.

For all four confined-separatrix quantities: relative L2 at most 0.30,
absolute normalized bias at most 0.15, correlation at least 0.80, and weighted
sign disagreement at most 0.15.

Additionally:

1. at least three separatrix transport fCRPS values are strictly below the
   corresponding H1 deterministic absolute-error score; the fourth is at most
   1.05 times H1;
2. at least three separatrix quantities have corrected spread-skill in
   `[0.67,1.50]`, I27 coverage within 0.12 of nominal, and I31 coverage within
   0.10 of nominal; the fourth remains finite and noncollapsed;
3. every eligible upper-decile event magnitude relative error is at most 0.50
   and weighted sign disagreement at most 0.25.

### 12.5 Monte Carlo stability

Every primary aggregate field, material-band, cross-field, and separatrix-
transport fCRPS component passes the M16-versus-M32 rule. No prefix rescues a
failure.

The numerical physics thresholds deliberately match the frozen B2 gate; only
the primary deterministic comparator changes from H2 to B3's actual H1 parent.

## 13. Stop/go rule

If all one-seed conditions pass, B3 is **provisionally promising**. That result
authorizes writing—but not silently executing—a separate protocol for seeds
1702 and 1703 with matched same-seed H1 parents/codecs. It does not authorize
O3 or 85606.

If marginal field calibration passes but joint spectra, cross-field, or
transport fails, record that marginal FGN-CRPS is insufficient and move to the
next predeclared representation rung, with PDE-Refiner versus joint stochastic
residual chosen by the oracle-localized failure. Do not tune noise dimension,
loss weights, learning rates, thresholds, or ensemble draws on these results.

If point skill or integrity fails, stop B3 immediately and diagnose the
retrofit/training failure before any replication. If Monte Carlo stability
fails, report it and evaluate a separately frozen larger-ensemble sensitivity;
do not resample M32.

No threshold may be relaxed after results are seen. A metric implementation
bug requires a documented amendment, retention of the original result, and a
consistent rerun of all affected forecasts or scores.

## 14. Execution, compute, and artifacts

Full training and forecast generation require one Rocky 9 H100/H200 and a
finished online W&B run. Scientific truth scoring may use allocated CPU or the
same Rusty job after the immutable forecast is closed. No GPU outside Rusty is
authorized.

The run directory stores configuration, copied normalization, both noise
banks, 100-line history, selected checkpoint, final optimizer state, generation
manifest, forecast, raw scores, block metrics, compute accounting, W&B record,
Slurm record, environment, exact command, and SHA-256 inventory. W&B is a
monitoring mirror; immutable Ceph artifacts are authoritative. Large files stay
out of Git, while compact results and their hashes are tracked.

Before full execution, the repository must contain and test:

1. a full-only entrypoint that cannot accept smoke or another seed;
2. fail-closed manifest, protocol, evidence, source, and artifact checks;
3. an M32 forecast writer whose loader cannot open truth fields;
4. a separate scorer reusing the frozen B2 metric definitions;
5. known-answer tests for the independent M32 noise bank, canonical axes,
   H1 comparator, member-wise nonlinear diagnostics, and the B3 gate;
6. one four-target evaluator smoke, explicitly non-scientific;
7. the complete local and Rocky 9 test suites.

## 15. Claims boundary

The strongest possible conclusion from a passing run is:

> On the later 85604 development interval, one seed of an FGN-retrofitted H1
> transition produced a useful one-step ensemble while preserving the frozen
> transport diagnostics well enough to merit independent-seed replication.

It would not establish autonomous rollout skill, held-out 85606 performance,
a validated architecture, transport-faithful emulation on a second simulation,
experimental diagnostic realism, assimilation value, diagnostic ranking,
cross-shot generalization, or steering.
