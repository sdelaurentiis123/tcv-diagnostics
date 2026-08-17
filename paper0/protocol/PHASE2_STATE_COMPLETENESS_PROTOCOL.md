# Phase 2 evolved-state inventory and momentum-closure protocol

**Status:** frozen before the first all-frame raw-momentum closure read

**Development run:** TCV/Hermes `85604` only

**Sequestered run:** `85606`; prohibited from this protocol, implementation,
and launcher

**Purpose:** establish exactly which fields form the saved Hermes state and
whether the saved parallel velocities are algebraically equivalent to the
saved evolved momenta under the executed Hermes source formula

The machine-readable authority is
`paper0/manifests/phase2_85604_state_completeness.json`. This is a deterministic
data/source audit. It does not select channels, train a model, evaluate a
forecast, or authorize access to the held-out simulation.

## 1. Question being tested

The exact 85604 input and Hermes revision register six volumetric solver
variables:

```text
Ne, Pe, Pi, NVe, NVi, Vort
```

The historical emulator instead used `Ne, Te, Ti, phi, Vi`. In the executed
`EvolveMomentum::transform_impl`, Hermes computes velocity from the solver
momentum as

```text
Nlim = softFloor(N, density_floor)
V = NV_solver / (AA * Nlim)
```

where

```text
softFloor(N, f) = max(N, 0) + f * exp(-max(N, 0) / f).
```

The solver momentum is restored before it is written. The frozen exact-output
hypotheses are therefore

```text
NVe == (1 / 1836) * softFloor(Ne, 1e-7) * Ve
NVi == 2 * softFloor(Ne, 1e-7) * Vi
```

Ion density is `Ne` because the ion component uses quasineutrality. These
relations test storage and derivation, not whether electron momentum is
predictable from the historical C5 channels: C5 contains neither `NVe` nor
`Ve`.

For attribution, the audit also evaluates the deliberately naive relations

```text
NVe == (1 / 1836) * Ne * Ve
NVi == 2 * Ne * Vi.
```

Those are not the executed source formula. Their discrepancy measures only
the materiality of the density-floor policy.

## 2. Frozen data and inventory scope

Read every saved time `0..623` from all rank indices `0..255` in the locked
85604 raw archive. Verify the exact time vector, MPI decomposition, processor
coordinates, Hermes revision, limiter, `zperiod=5`, and native axis order
`[t,x,y,z]` on every rank.

The inventory must find the following fields on every rank, each as float64
with local shape `[624,8,6,81]` and cell-centred axes:

```text
evolved:  Ne, Pe, Pi, NVe, NVi, Vort
derived:  Te, Ti, Ve, Vi, phi
```

Variable source, species where applicable, units, conversion, dimensions,
cell location, and time dimension must match the frozen manifest. Metadata
are compared across all ranks.

Strip two x and two y guards per rank before value statistics. The canonical
physical volume is

```text
[time=624, x=64, y=32, z=81]
```

or 103,514,112 points per field. Stream the six evolved fields and the two
derived velocities without writing a second canonical dataset. Record
deterministic stream digests for all eight value streams.

Prior work has already read all C5 values, all pressure values, and five
predeclared native frames. During this freeze only source text and
representative-file metadata were inspected. This is consequently a
prospective all-frame raw-momentum audit, not a claim that the dataset itself
has never been examined.

## 3. Frozen scopes and summaries

Report every closure over:

- the full physical domain, `y=0..31`;
- the guard-independent transport interior, `y=1..30`;
- the target-dependent rows, `y in {0,31}`.

Report counts by the eight predeclared 78-frame blocks and by frame, x, and y.
These are correlated summaries within one run, not independent experiments.

For all streamed fields report finite counts, minimum, maximum, sum of
squares, and RMS. For `Ne`, additionally report counts below zero, below the
`1e-7` density floor, and points where `softFloor(Ne,1e-7)` differs from `Ne`
by more than machine equality.

For each of the four closure relations report:

- pointwise discrepancy counts;
- maximum absolute error and its `(frame,x,y,z)` location;
- frame-wise maximum error and pass/fail;
- squared-error and squared-reference sums;
- relative L2 error, with zero-denominator behavior declared in code;
- discrepancy counts by frame, x, y, block, and spatial scope.

## 4. Frozen numerical gate

For reference `r` and candidate `c`, a point disagrees when

```text
abs(r - c) > atol + rtol * abs(r)
```

with

```text
atol = 1e-12
rtol = 1e-12.
```

A frame passes when

```text
frame_max_abs_error <= atol + rtol * frame_max_abs_reference
```

and contains no non-finite reference, candidate, or error. The scientific
result never changes the process exit status; structural or provenance
violations do.

## 5. Frozen interpretation

- If both soft-floor relations pass every full-domain frame and the relevant
  values are finite, `(Ne, Ve)` is algebraically equivalent to `(Ne, NVe)` and
  `(Ne, Vi)` is algebraically equivalent to `(Ne, NVi)` for this output.
- This would not make historical C5 complete: it contains `Vi` but not `Ve`,
  and it contains derived `phi` rather than evolved `Vort`.
- If either exact relation fails, do not silently replace momentum by velocity;
  first resolve dump ordering, boundary handling, metadata, or source mismatch.
- The naive direct-density relations are attribution checks only. Their pass or
  failure cannot override the exact source relation.
- No result automatically changes the training channels. Potential/vorticity
  closure and temporal-protocol decisions remain separate gates.
- No frequency, prevalence, or error magnitude discovered by this job may be
  converted post hoc into a new acceptance threshold.

## 6. Execution lock

The cluster implementation must:

1. run from a clean, committed Paper 0 checkout;
2. verify hashes for this manifest, the auditor, merger, locked input/settings,
   mesh, `evolve_momentum.cxx`, and `hermes_utils.hxx`;
3. verify the exact Hermes revision;
4. use 16 deterministic shards with rule `rank modulo 16 == s`;
5. merge only complete, unique coverage of all rank indices `0..255`;
6. refuse to overwrite an existing artifact directory;
7. emit strict JSON, commands, environment, log, and SHA-256 inventory;
8. report `held_out_85606_read = false`.

This protocol authorizes only the audit above on 85604.
