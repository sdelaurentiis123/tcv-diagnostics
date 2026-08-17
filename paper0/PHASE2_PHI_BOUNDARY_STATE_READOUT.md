# Phase 2 saved potential-boundary-state readout

**Status:** completed; a distinct saved radial-potential boundary state is
present in 85604

**Development run:** TCV/Hermes `85604` only

**Held-out run 85606 accessed:** no

**Training performed:** no

**Interior-potential materiality established:** no

**Automatic model-state change authorized:** no

## Executive conclusion

The radial `phi` guards saved by Hermes are finite and satisfy the two
source-defined structural identities:

1. the outermost guard copies the adjacent guard;
2. the midpoint between the adjacent guard and adjacent interior cell is
   constant around the toroidal direction.

Both checks pass on both radial sides with zero discrepant points. The stored
guards therefore encode a valid compact boundary value rather than arbitrary
guard noise.

That compact value is not equal to the instantaneous zero-gradient target at
any saved frame/y location on either side. The continuous departure has RMS
`1.073 V` at the inner boundary and `0.513 V` at the outer
boundary. Because the historical model tensors strip radial guards, they omit
this genuine saved state.

The correct conclusion is deliberately limited:

> Guard-stripped volumetric fields are not the exact saved discrete state.
> Whether the omitted boundary value materially changes interior potential or
> transport is still unknown and requires a paired exact elliptic solve.

This audit does not choose `S6+Bphi`, `S6+phi`, or a
history-conditioned state.

## 1. What the boundary value means

Hermes derives interior potential from vorticity through a
geometry-dependent elliptic solve. The executed input also enables a relaxed
radial boundary:

```text
phi_boundary_relax = true
phi_boundary_timescale = 1 microsecond
phi_core_averagey = false
```

At each internal solver update, the boundary midpoint is blended between its
old value and the current toroidal mean of adjacent interior `phi`. In
schematic form,

\[
b_{\mathrm{new}}
=
w\,b_{\mathrm{old}}
+
(1-w)\,\overline{\phi}_{\mathrm{interior}},
\]

where \(b\) is the midpoint between the adjacent guard and interior value.
The code then fills the guard so that every toroidal point has this same
midpoint.

At the saved cadence of `3.131905426 microseconds`, the homogeneous
one-frame coefficient for the configured one-microsecond timescale is

\[
\exp(-3.131905426)=0.0436345755.
\]

That number is not a predicted empirical lag-one correlation. The target
changes with the plasma and the solver takes many internal steps between saved
frames.

## 2. Gauge-invariant observable

For a side, frame, global y index, and toroidal index \(k\), the frozen audit
defines

\[
b(k)
=
\frac{1}{2}
\left[
\phi_{\mathrm{adjacent\ guard}}(k)
+
\phi_{\mathrm{adjacent\ interior}}(k)
\right],
\]

\[
\phi_{\mathrm{target}}
=
\left\langle
\phi_{\mathrm{adjacent\ interior}}(k)
\right\rangle_k,
\]

\[
d(k)=b(k)-\phi_{\mathrm{target}}.
\]

The departure \(d\) is unchanged if an arbitrary constant is added to
potential everywhere. The audit therefore measures boundary memory without
confusing it with the gauge freedom of electrostatic potential.

An instantaneous-Neumann boundary corresponds to \(d=0\). Failing that exact
classification does not by itself say that the departure is physically
important; the continuous amplitude and a paired solve are both required.

## 3. Frozen scope and execution identity

The protocol and manifest were committed before the first all-frame read of
raw radial `phi` guards:

- protocol:
  `paper0/protocol/PHASE2_PHI_BOUNDARY_STATE_PROTOCOL.md`;
- manifest:
  `paper0/manifests/phase2_85604_phi_boundary_state.json`;
- executed Paper 0 commit:
  `cee2264a88ae7a912f8a70a06086137bf16d4e76`;
- Slurm job: `6891890`;
- platform: Rocky Linux 9.8 on Rusty worker `worker5594`;
- execution: one CPU, no GPU;
- terminal state: `COMPLETED`, exit `0:0`;
- elapsed time: `00:05:17`;
- finished: `2026-08-17T08:55:42-04:00`.

The job verified all 256 rank filenames and every locked input, geometry,
Hermes source, manifest, protocol, and implementation hash. It read `phi`
values only from the 32 prospectively selected radial-boundary ranks:

```text
inner: PE_XIND=0,  PE_YIND=0..15
outer: PE_XIND=15, PE_YIND=0..15
```

All 624 frames, 32 global y locations, and 81 native toroidal values were
covered on each side. Run 85606 was not read.

The immutable result is

```text
/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_85604_phi_boundary_state/job_6891890/phi_boundary_state_audit.json
```

with SHA-256

```text
79c67709c921caa1ddf1ea3e4d8f431ce88e220adc70247527c7a8a5e5f637cc
```

The job's own `artifact_sha256.txt` records the same result hash plus
the command and environment hashes. The exact 127,860-byte JSON is tracked at
`paper0/results/phase2_phi_boundary_state_6891890.json`; its regression
test locks the complete file digest, execution identity, structural gates,
continuous amplitudes, and intentionally open materiality decision.

## 4. Exact structural checks

### Finiteness

The outermost guard, adjacent guard, and adjacent interior planes contain zero
non-finite values on both radial sides.

### Outermost-guard copy

Each side contains 1,617,408 comparisons:

```text
624 frames * 32 y cells * 81 toroidal cells.
```

| Side | Discrepant points | Maximum absolute roundoff |
|---|---:|---:|
| inner | 0 / 1,617,408 | `4.44089e-15` |
| outer | 0 / 1,617,408 | `6.21725e-15` |

### Toroidal midpoint constancy

| Side | Discrepant points | Maximum absolute roundoff |
|---|---:|---:|
| inner | 0 / 1,617,408 | `1.86517e-14` |
| outer | 0 / 1,617,408 | `1.42109e-14` |

Both relations pass the frozen
`1e-12 + 1e-12 * abs(reference)` rule. This establishes that the saved
guard planes faithfully carry the compact source-defined boundary midpoint.

## 5. Departure from the instantaneous target

There are

```text
624 frames * 32 y cells = 19,968
```

independent midpoint/target comparisons per side. Every one fails the exact
instantaneous-Neumann classification:

| Side | Nonzero locations | Locations per 78-frame block |
|---|---:|---:|
| inner | 19,968 / 19,968 | 2,496 in each of 8 blocks |
| outer | 19,968 / 19,968 | 2,496 in each of 8 blocks |

This count is an exact-state result, not an effect-size claim. The continuous
amplitudes provide the necessary scale:

| Metric | Inner | Outer |
|---|---:|---:|
| RMS departure | `1.07261 V` | `0.512986 V` |
| median absolute departure | `0.473574 V` | `0.391423 V` |
| 90th percentile absolute | `1.68234 V` | `0.775506 V` |
| 95th percentile absolute | `2.20548 V` | `0.919293 V` |
| 99th percentile absolute | `3.68757 V` | `1.48132 V` |
| maximum absolute departure | `8.11711 V` | `1.99169 V` |
| RMS adjacent-interior toroidal fluctuation | `1.44006 V` | `0.400332 V` |
| departure RMS / fluctuation RMS | `0.744835` | `1.28140` |

The inner maximum occurs at frame 591, global `y=4`. The outer maximum
occurs at frame 586, global `y=14`. These locations and all percentiles
were declared output fields before execution; no location was selected for a
new threshold after seeing the result.

The fluctuation ratio is only descriptive. It compares two boundary-local
scales and does not predict the elliptic response in the interior.

## 6. Time and topology dependence

The eight predeclared 78-frame blocks have the following departure RMS in
volts:

| Block | Frames | Inner | Outer |
|---:|---:|---:|---:|
| 0 | 0--77 | `0.8711` | `0.4447` |
| 1 | 78--155 | `0.8024` | `0.3518` |
| 2 | 156--233 | `0.6816` | `0.4267` |
| 3 | 234--311 | `0.7472` | `0.4815` |
| 4 | 312--389 | `0.7980` | `0.5331` |
| 5 | 390--467 | `1.0805` | `0.3750` |
| 6 | 468--545 | `1.3338` | `0.5488` |
| 7 | 546--623 | `1.7874` | `0.8023` |

The departure is therefore present throughout the run and changes in
amplitude, with the largest RMS in the final block on both sides. This is
another reason not to replace it by one global constant.

The per-y lag-one correlation of midpoint departure is:

- inner boundary: mean `0.4373`, range `-0.0453..0.8590`;
- outer boundary: mean `0.9709`, range `0.9602..0.9801`.

On the inner side, the high-correlation values occur at `y=8..23`,
matching the already frozen confined-core topology interval, while the
remaining rows are much less correlated. This is a descriptive spatial
alignment, not a causal interpretation.

The empirical correlations do not contradict the homogeneous
`0.04363` coefficient. The boundary is continuously driven by a moving
target, and many internal updates occur within one saved interval.

## 7. What is established

1. The saved radial guard planes contain a structurally valid compact boundary
   midpoint on both sides.
2. The midpoint is gauge-invariantly distinct from the instantaneous target at
   all saved frame/y locations.
3. The distinction persists in every temporal block and is not merely
   floating-point noise.
4. Guard-stripped `S6` is therefore not the exact saved discrete state.
5. A compact explicit boundary representation would require only two sides by
   32 y values per saved state, rather than another full 3D volume.

## 8. What is not established

This audit does not establish:

- how much the saved boundary changes interior `phi`;
- how much it changes particle or heat transport;
- whether short field history predicts the missing boundary value well enough;
- whether explicitly including 64 boundary scalars improves a learned model;
- exact forward closure from `Vort,Pi` to `phi`;
- stationarity, forecast skill, calibration, architecture choice, or
  held-out generalization.

The ratios above cannot be converted after the fact into a materiality
threshold.

## 9. Next exact oracle

The next source-state gate should pair two otherwise identical exact elliptic
solves for each predeclared 85604 state:

1. solve using the retained saved radial midpoint;
2. solve using the instantaneous-Neumann midpoint;
3. hold `Vort`, `Pi`, geometry, numerical options, and gauge policy
   fixed;
4. compare interior `phi` and the already validated native-81 radial ExB
   transport;
5. report all declared regions and time blocks without a post hoc acceptance
   cutoff.

The same compiled operator ladder should also test whether stored
`Vort` is reproduced from stored `phi` and `Pi` under the
exact boundary policy. Only that paired result can close the
potential/vorticity gate and support a final state choice.
