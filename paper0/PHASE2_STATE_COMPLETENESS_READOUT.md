# Phase 2 evolved-state and momentum-closure readout

**Status:** completed; exact velocity/momentum closure accepted for 85604

**Development run:** TCV/Hermes `85604` only

**Held-out run 85606 accessed:** no

**Training performed:** no

**Automatic model-channel change authorized:** no

## Executive conclusion

The complete raw 85604 archive confirms that the six volumetric variables
advanced by Hermes are

```text
Ne, Pe, Pi, NVe, NVi, Vort.
```

The saved electron and ion parallel velocities are algebraically equivalent to
the saved momenta when they are paired with density. Both exact source
relations pass all 624 frames and all 103,514,112 physical cells with zero
point discrepancies at the prospectively frozen `1e-12` absolute and
relative tolerances.

This does **not** make the historical five-channel state complete. Historical
`C5T=[Ne,Te,Ti,phi,Vi]` contains the ion velocity needed to recover
`NVi`, but contains neither `Ve` nor `NVe`. It also
substitutes derived `phi` for evolved `Vort`; that separate
potential/vorticity and boundary-state gate is not yet closed.

The practical conclusion is:

> Velocity versus momentum is not an information-loss issue on 85604 when
> density and the corresponding species velocity are both retained. Missing
> electron parallel state, pressure flooring, and potential reconstruction
> remain distinct issues.

No architecture or channel set is selected by this audit.

## 1. Exact question

At the executed Hermes revision, velocity is derived from solver momentum
using

\[
\operatorname{softFloor}(N,f)
=
\max(N,0)
+
f\exp\!\left[-\frac{\max(N,0)}{f}\right].
\]

For this run, \(f=10^{-7}\), electron atomic mass is \(1/1836\), and ion
atomic mass is \(2\). The exact saved-output hypotheses were therefore

\[
\widehat{NVe}
=
\frac{1}{1836}
\operatorname{softFloor}(Ne,10^{-7})\,Ve,
\]

\[
\widehat{NVi}
=
2\operatorname{softFloor}(Ne,10^{-7})\,Vi.
\]

The audit also evaluated deliberately naive versions using `Ne`
directly instead of `softFloor(Ne,1e-7)`. Those attribution checks ask
whether the floor is numerically active; they do not replace the source-exact
equations.

## 2. Frozen scope and execution integrity

The protocol and machine-readable manifest were committed before the first
all-frame raw-momentum read:

- protocol:
  `paper0/protocol/PHASE2_STATE_COMPLETENESS_PROTOCOL.md`;
- manifest:
  `paper0/manifests/phase2_85604_state_completeness.json`;
- executed Paper 0 commit:
  `4913361b4f1ee5f04f8fd3e95ac9240b3941c9fc`;
- Slurm job: `6891855`;
- platform: Rocky Linux 9.8 on Rusty worker `worker5594`;
- execution: CPU-only, 16 deterministic rank shards;
- terminal state: `COMPLETED`, exit `0:0`;
- elapsed time: `00:28:16`;
- started: `2026-08-17T08:09:16-04:00`;
- finished: `2026-08-17T08:37:32-04:00`.

Every raw rank `0..255`, processor coordinate, and saved time
`0..623` was verified. Eleven field metadata records were checked on
every rank:

```text
evolved: Ne, Pe, Pi, NVe, NVi, Vort
derived: Te, Ti, Ve, Vi, phi
```

After stripping two guards per decomposed axis, each of the eight value streams
`Ne, Pe, Pi, NVe, NVi, Vort, Ve, Vi` covered

```text
[time=624, x=64, y=32, z=81]
```

or 103,514,112 values. The native toroidal domain has `zperiod=5`; no
resampled or held-out data were used.

The immutable raw result is

```text
/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_85604_state_completeness/job_6891855/state_completeness_audit.json
```

with SHA-256

```text
9fec0426a97fab9e15b0029d80f1f6c6464d0d7e34aac4216ec4a76ceb3bda93
```

The job's own `artifact_sha256.txt` contains the same digest and
inventories the command, environment, all 16 shard results, and merged result.
The tracked compact result is
`paper0/results/phase2_state_completeness_6891855.json`. It was generated
by the prospectively committed compactor at
`54d2bba33cf4a5458bc8e61cb794024de0849d7f` and has SHA-256
`565a4e27e87d4f5a3e647daf77486020ac627f43ffb5cd30a8daf74b7199cf20`.

## 3. Numerical result

### Source-exact closure

| Relation | Full-domain relative L2 | Maximum absolute error | Discrepant points | Passing frames |
|---|---:|---:|---:|---:|
| `NVe` from `Ne,Ve` and source floor | `5.18518e-16` | `6.93889e-18` | 0 / 103,514,112 | 624 / 624 |
| `NVi` from `Ne,Vi` and source floor | `2.95087e-16` | `8.88178e-16` | 0 / 103,514,112 | 624 / 624 |

Both relations also pass all 624 frames separately in the predeclared
guard-independent transport interior `y=1..30` and target-dependent rows
`y in {0,31}`. Every one of the eight 78-frame blocks contains zero
point discrepancies.

The residuals are ordinary floating-point roundoff, many orders of magnitude
inside the frozen pointwise rule

\[
|r-c| \le 10^{-12}+10^{-12}|r|.
\]

### Density-floor attribution

The smallest stored physical-domain density is

```text
Ne_min = 4.190229129105658e-5
```

which is over 400 times the configured `1e-7` floor. Across the full
domain, transport interior, target rows, every frame, and every temporal block:

| Count | Value |
|---|---:|
| `Ne < 0` | 0 |
| `Ne < 1e-7` | 0 |
| `softFloor(Ne,1e-7) != Ne` at machine equality | 0 |

The naive direct-density relations therefore have the same numerical result
as the source-exact relations. This statement is specific to the saved 85604
interval; it is not a general license to remove the simulator floor.

### Finiteness

All 103,514,112 values in every streamed field are finite. In native normalized
storage units:

| Field | Minimum | Maximum | RMS |
|---|---:|---:|---:|
| `Ne` | `4.19023e-5` | `1.72949` | `0.434850` |
| `Pe` | `3.05721e-6` | `4.57997` | `0.850990` |
| `Pi` | `-0.0234715` | `4.11144` | `0.848103` |
| `NVe` | `-0.00627609` | `0.00715781` | `0.000469720` |
| `NVi` | `-0.745473` | `0.585852` | `0.271124` |
| `Vort` | `-2.03640` | `1.71530` | `0.0387116` |
| `Ve` | `-45.0364` | `58.8087` | `3.44333` |
| `Vi` | `-8.36569` | `6.79921` | `0.936976` |

These are raw normalized array values. The audit separately verified each
field's stored physical-unit label and conversion metadata; the table does not
silently relabel normalized values as SI.

## 4. What is now established

### Ion parallel state

Historical C5 contains `Ne` and `Vi`. On this 85604 output,

\[
NVi = 2\,Ne\,Vi
\]

to roundoff at every physical point. Thus replacing stored ion momentum by the
pair `(Ne,Vi)` loses no ion-momentum information here.

### Electron parallel state

The same equivalence holds between `(Ne,Ve)` and
`(Ne,NVe)`, but historical C5 contains neither `Ve` nor
`NVe`. Electron parallel state remains an independent omission from the
old emulator input.

### Representation versus learnability

Algebraic equivalence is not optimization equivalence. `Ve` and
`NVe`, for example, have very different native amplitudes and spatial
weighting. A future matched baseline may choose velocity or momentum for
conditioning and scaling reasons, but it must include the corresponding
electron information and apply an explicit, training-only normalization
policy.

### Relation to pressure

The earlier all-frame pressure audit remains unchanged:

- `Pe` is recoverable from `Ne,Te` throughout 85604;
- `Pi` is not exactly recoverable from `Ne,Ti` at the 3,412
  cells where evolved `Pi` is negative and temperature applies a floor.

Momentum closure therefore does not erase the pressure-state distinction.

## 5. What is not established

This job does not show:

- that electron momentum materially improves one-step or rollout forecasts;
- that a single frame of the six evolved volumes is a sufficient discrete
  Markov state;
- that `phi` can be reconstructed exactly from `Vort`,
  `Pi`, geometry, and the available boundary state;
- that the one-microsecond radial-potential boundary memory is negligible;
- that 85604 is statistically steady;
- that any codec preserves the newly identified state;
- that any deterministic, diffusion, FGN, refiner, or residual architecture is
  preferred;
- that the conclusions generalize to 85606 or another operating regime.

In particular, a deterministic algebraic identity is not a forecast result.
The predictive value of electron state must later be tested by a matched O2
one-step experiment, not assumed.

## 6. Consequence for candidate emulator states

| Candidate | Current status |
|---|---|
| `C5T=[Ne,Te,Ti,phi,Vi]` | Historical baseline only; ion momentum is retained, electron momentum is omitted, ion-pressure flooring loses state, and `phi/Vort` closure is unresolved. |
| `C5P=[Ne,Pe,Pi,phi,Vi]` | Repairs direct pressure representation but still omits electron parallel state and substitutes `phi` for `Vort`. |
| `S6=[Ne,Pe,Pi,NVe,NVi,Vort]` | Exact volumetric evolved state. It still requires an explicit policy for potential reconstruction and radial-boundary memory before transport scoring. |
| `S6+phi` | Pragmatic redundant-volume candidate. Easier for the validated transport evaluator, but consistency between `Vort`, `Pi`, and `phi` would be learned rather than enforced. |
| History-conditioned observed state | Legitimate baseline if delay history recovers omitted state predictively; this must be demonstrated with matched O2 tests and must not use absolute frame number. |

The audit supports carrying one exact source-state candidate and one pragmatic
observed-state/history baseline into later matched tests. It does not yet
choose between them.

## 7. Next gate

The next frozen audit measures the saved radial `phi` guard state and its
departure from the instantaneous boundary target across all 624 frames of
85604. That audit is already defined in
`paper0/protocol/PHASE2_PHI_BOUNDARY_STATE_PROTOCOL.md`.

Its role is deliberately narrow:

1. quantify whether the relaxed boundary state differs from the instantaneous
   target at saved times;
2. verify guard-copy and toroidal-constancy assumptions;
3. determine whether a paired exact elliptic solve is required to measure
   interior materiality.

Even a nonzero boundary departure will not by itself prove that the boundary
materially changes interior `phi`. The potential/vorticity gate closes
only after the exact operator and boundary policy are tested together.

Only after that evidence is frozen should Paper 0 select model state channels,
construct new conversion files, or begin matched codec/dynamics training.
