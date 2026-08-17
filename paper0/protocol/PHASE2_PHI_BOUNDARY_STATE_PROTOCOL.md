# Phase 2 saved potential-boundary-state protocol

**Status:** frozen before the first all-frame read of raw radial `phi` guards

**Development run:** TCV/Hermes `85604` only

**Sequestered run:** `85606`; prohibited from this protocol, implementation,
and launcher

**Purpose:** determine whether the saved radial potential guards contain a
nonzero, gauge-invariant boundary state that is absent from the guard-stripped
model tensors

The machine-readable authority is
`paper0/manifests/phase2_85604_phi_boundary_state.json`. This is a descriptive
source/data audit. It does not invert potential, train a model, select a state
representation, or authorize held-out access.

## 1. Source-defined boundary state

The executed vorticity component uses

```text
phi_boundary_relax = true
phi_boundary_timescale = 1 microsecond
phi_core_averagey = false  (default)
```

At each radial physical boundary and each y index, it calculates

```text
target = mean_z(phi_interior)
old_midpoint = 0.5 * (phi_adjacent_guard(k=0) + phi_interior(k=0))
new_midpoint = w * old_midpoint + (1 - w) * target
w = exp(-delta_t / tau)
```

then fills every adjacent guard value so that

```text
0.5 * (phi_adjacent_guard(k) + phi_interior(k)) = new_midpoint
```

for all toroidal k. The outermost radial guard is copied from the adjacent
guard after the elliptic solve. `phi` is a restart field so this guard state
survives restart even though potential is not a solver-evolved volume.

The saved-frame cadence is `3.131905426352636 microseconds`; the homogeneous
one-frame memory coefficient is therefore

```text
exp(-3.131905426352636) = 0.043634575521405435.
```

This coefficient is not an empirical autocorrelation prediction: the target
changes continuously and the solver can take many internal steps between
saved frames.

## 2. Frozen raw scope

Read `phi` for every saved frame `0..623` on the 32 ranks that touch a radial
physical boundary:

```text
inner: PE_XIND = 0,  all PE_YIND = 0..15
outer: PE_XIND = 15, all PE_YIND = 0..15
```

Verify all 256 rank filenames and control hashes first, but read field values
only from those 32 predeclared boundary ranks. Verify `zperiod=5`, native
`z=81`, the exact time vector, decomposition, processor coordinates, source
revision, `phi` metadata, and `[t,x,y,z]` axis order.

For each boundary rank, strip y guards but retain the explicitly declared x
indices:

| Side | outermost guard | adjacent guard | adjacent interior |
|---|---:|---:|---:|
| inner (`PE_XIND=0`) | 0 | 1 | 2 |
| outer (`PE_XIND=15`) | 7 | 6 | 5 |

The two physical local-y cells are indices `2:4`. Mapping by `PE_YIND` yields
all global y indices `0..31`. No interior-rank field values are selected after
looking at `phi`.

Previous work read guard-stripped C5 values and selected native frames. During
this freeze only source and metadata were inspected; no all-frame raw `phi`
guard values were read. This is a prospective boundary-state audit, not a
claim that interior potential is historically blind.

## 3. Frozen observables

For side, frame, global y, and toroidal index k define

```text
midpoint(k) = 0.5 * (adjacent_guard(k) + adjacent_interior(k))
target = mean_k(adjacent_interior(k))
departure(k) = midpoint(k) - target.
```

Both `departure` and `adjacent_guard - adjacent_interior` are invariant to an
additive potential gauge shift. Report:

- non-finite counts for all three x planes;
- exact outermost-versus-adjacent-guard closure;
- toroidal midpoint constancy within each `(side,frame,y)`;
- midpoint departure from the instantaneous Neumann target;
- RMS, mean, maximum absolute value and location, and absolute percentiles
  `[50, 90, 95, 99, 100]` for departure in normalized units and volts;
- RMS departure divided by RMS toroidal fluctuation of the adjacent interior
  field, with zero-denominator behavior explicit;
- summaries by side, y, and the eight frozen 78-frame blocks;
- lag-one temporal correlation of the per-frame/y midpoint departure,
  reported descriptively alongside, not equated to, the theoretical weight.

Record deterministic digests of the retained outermost, adjacent-guard, and
adjacent-interior streams for each side.

## 4. Frozen exact checks

Use

```text
atol = 1e-12
rtol = 1e-12
```

for the source-structural checks:

```text
abs(outermost_guard - adjacent_guard)
  <= atol + rtol * abs(adjacent_guard)

max_k(abs(midpoint(k) - mean_k(midpoint)))
  <= atol + rtol * max_k(abs(midpoint(k))).
```

An instantaneous-Neumann point passes when

```text
abs(mean_k(midpoint) - target)
  <= atol + rtol * abs(target).
```

This last check is an exact-state classification, not a materiality threshold.
The job must report every failure and the continuous departure magnitude.
Scientific pass/fail does not change the process exit status; provenance,
shape, coverage, and non-finite JSON violations do.

## 5. Frozen interpretation

- If outer-guard copy and midpoint constancy pass, the saved adjacent guard
  implements the source-defined compact boundary value at output time.
- If instantaneous Neumann passes everywhere, no distinct relaxed boundary
  value is visible at saved cadence, although source provenance remains noted.
- If it fails anywhere, guard-stripped S6 is not the exact saved discrete
  state: a compact radial boundary value or an observational history is needed
  for exact Markov closure.
- The amplitude ratio is descriptive. No post hoc cutoff may declare the
  boundary dynamically negligible.
- Material impact on interior `phi` requires a later paired exact elliptic
  solve with retained versus instantaneous-Neumann boundaries. This audit
  alone cannot choose `S6+Bphi`, `S6+phi`, or a history-based baseline.
- Potential/vorticity forward closure remains a separate compiled-operator
  gate.

## 6. Execution requirements

The eventual launcher must run from a clean committed Rocky 9 checkout, lock
the exact source and data-control hashes, refuse overwrites, record commands
and environment, require complete unique coverage of the 32 boundary ranks,
emit strict JSON and artifact hashes, and report
`held_out_85606_read = false`.

This protocol authorizes only the 85604 guard-state audit above.
