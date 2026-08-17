# Phase 2 all-frame pressure-closure audit protocol

**Status:** executed without changing the frozen rules; accepted job `6891583`
completed from clean commit `f5d4541`

**Development run:** TCV/Hermes `85604` only

**Sequestered run:** `85606`; prohibited from the implementation and launcher

**Purpose:** determine whether the one negative ion-pressure point found by the
native-frame oracle is an isolated target-row numerical undershoot or evidence
that the historical temperature channels materially fail to represent evolved
Hermes pressure throughout 85604

The machine-readable authority is
`paper0/manifests/phase2_85604_pressure_closure_audit.json`. This is a
descriptive data-quality audit. It does not tune a model, evaluate a learned
forecast, or alter the canonical channel set.

## 1. Why this audit precedes resampling and training

Hermes evolves `Ne`, `Pe`, and `Pi` directly. The historical emulator stores
`Ne`, `Te`, `Ti`, `phi`, and `Vi`. At the locked simulator revision,
`EvolvePressure` obtains temperature from pressure after applying a zero floor,
but the evolved pressure can retain a negative numerical undershoot. Therefore

```text
Pe = Ne * Te
Pi = Ni * Ti
```

need not hold at a point where evolved pressure is negative. The five-frame
oracle found one such `Pi` point on `y=31`; all pressure-flow operator
comparisons themselves passed. Before deciding whether Paper 0 should emulate
temperature, evolved pressure, or both, we must measure the full-run behavior
without selecting frames based on their values.

## 2. Frozen data scope

Read every frame `0..623` from all 256 distributed rank files under the locked
85604 archive. Verify the existing source, configuration, geometry,
decomposition, axis order, and processor-coordinate locks before aggregating
statistics. Strip two `x` and `y` guards per rank, giving exactly

```text
[time=624, x=64, y=32, z=81]
```

or 103,514,112 physical cells per field. The six audited fields are `Ne`,
`Ni`, `Te`, `Ti`, `Pe`, and `Pi`. No full canonical copy is needed: each rank
may be processed in memory, but its explicit `PE_XIND` and `PE_YIND` determine
global coordinates. The result records deterministic digests of each field's
guard-stripped rank stream.

Every rank must contain the exact time sequence from normalized time `285000`
through `471900` at cadence `300`. With the frozen cyclotron frequency this is
`3.131905426352636 us` per frame. This audit uses relative frame/time
coordinates only; it does not add absolute time as a model feature.

For temporal localization, report the eight predeclared contiguous 78-frame
blocks: `[0,77]`, `[78,155]`, `[156,233]`, `[234,311]`, `[312,389]`,
`[390,467]`, `[468,545]`, and `[546,623]`. These are summaries within one
simulation, not independent shots.

## 3. Frozen spatial scopes

Every statistic is reported for:

- the full physical domain, `y=0..31`;
- the guard-independent transport interior, `y=1..30` (Python slice `1:31`);
- the two target-dependent rows, `y in {0,31}`.

This separation was fixed before the all-frame read. It tests the specific
hypothesis suggested by frame 312 without deleting or relabeling the failing
point.

## 4. Value statistics

For every field and spatial scope, report:

- non-finite count;
- exact negative count (`value < 0`) and fraction;
- exact zero count (`value == 0`) and fraction;
- minimum and maximum with `(frame,x,y,z)` locations;
- negative counts by frame, global `x`, global `y`, and temporal block;
- the twenty most-negative points, or all such points if fewer exist.

There is no post-hoc magnitude threshold that converts a negative value into a
positive one. Magnitudes and locations are evidence for interpretation, not a
hidden acceptance rule.

## 5. Closure statistics

Evaluate four relations using the direct archived variable as reference:

```text
Ni == Ne
Pe == Ne * Te
Pi == Ni * Ti
Pi == Ne * Ti
```

The tolerance remains the one frozen for the native-frame oracle:

```text
atol = 1e-12
rtol = 1e-12
```

For each frame and spatial scope, the closure passes when

```text
max_abs_error <= atol + rtol * frame_max_abs_reference.
```

For prevalence counts, a point is discrepant when

```text
abs_error > atol + rtol * abs(reference_at_point).
```

These two named summaries answer different questions and must not be mixed.
Report frame-level passes, point-level discrepancy counts, maximum errors and
locations, and counts by frame, `x`, `y`, block, and spatial scope. For pressure
closures, additionally classify discrepancies by whether the direct pressure
is negative or nonnegative.

## 6. Completion versus scientific findings

The program exits nonzero for structural or provenance failures: wrong run,
source, controls, ranks, dimensions, coordinates, times, axes, incomplete
reads, overwrite attempts, or invalid JSON. Negative pressures, non-finite
state values, and closure discrepancies are scientific findings that must be
written completely; they do not by themselves make the audit process fail.

The result is complete only if all 256 ranks, all 624 times, all six fields,
and every physical cell were accounted for. The output must be strict JSON and
must record the Paper 0 commit, dirty-state gate, exact command, immutable
artifact path, and SHA-256 digests. `85606` access is forbidden.

## 7. Frozen interpretation rules

The historical five-channel state is exactly compatible with the audited
Hermes quantities only if all four relations pass every frame over the full
physical domain and all six fields are finite.

The temperature channels reproduce pressure for the currently accepted
guard-independent radial operator scope only if both pressure closures pass
every frame over `y=1..30`.

If either pressure closure fails inside `y=1..30`, Paper 0 must not use a
temperature-only state for exact pressure-based transport without explicitly
defining and validating a pressure-floor policy. If all negative `Pe` and `Pi`
values and all pressure-closure failures are confined to `y=0` or `y=31`, the
current temperature state may be adequate for the declared guard-independent
operator scope, but it remains an inexact representation of the full evolved
state.

No frequency or magnitude discovered by this job automatically authorizes a
channel change. The audit yields a recommendation to be reconciled with the
scientific target and simulator-owner guidance. It cannot establish anything
about 85606, learned forecasts, transport fidelity, or diagnostic ranking.

## 8. Execution history

The first implementation attempt, Rocky 9 job `6891417` from clean commit
`39bfb22`, used one serial scanner. It passed all provenance gates but was only
reading `BOUT.dmp.42.nc` of 256 after approximately 37 minutes. It was
cancelled at 38:46, before the 45-minute cap, and wrote no result JSON. No
partial statistics are accepted. The immutable no-result record is
`paper0/results/phase2_pressure_closure_6891417.json`.

This establishes an implementation-performance constraint only. A corrected
launcher may read disjoint rank shards concurrently and merge their sufficient
statistics after verifying complete, unique processor-coordinate coverage.
The six fields, all 624 frames, native cells, spatial scopes, time blocks,
closure formulas, tolerances, completion conditions, and interpretation rules
above remain unchanged.

The corrected execution is fixed to 16 shards before its first submission.
Shard `s` reads exactly the ranks satisfying `rank modulo 16 == s` and writes a
strict partial JSON containing no scientific conclusion. The reducer requires
all shard indices `0..15`, all rank indices `0..255`, and all 256 `(PE_XIND,
PE_YIND)` coordinates exactly once. It sums counts, takes maxima across shards,
recomputes every frame-level tolerance and pass decision from the merged
sufficient statistics, and only then derives the frozen interpretation. Any
missing or failed shard blocks the merge. Guard-stripped field streams are
locked by per-shard digests and a deterministic merged digest tree.

Parallel job `6891530` from clean commit `b672d69` was cancelled after 49
seconds because Slurm assigned the first `srun --exclusive` step all CPUs; only
shard step zero became active and no partial JSON was written. This is another
no-result execution finding. The corrected launcher adds `srun --exact`, so
each exclusive shard receives exactly the one CPU it requests. No scientific
choice above changes.

Job `6891570` from clean commit `347495f` showed that exact CPU allocation
still inherited the full 64 GB as step memory, again allowing only shard zero
to launch. It was cancelled after 48 seconds, with no partial JSON. The next
launcher adds `--mem=4G` per shard, partitioning the existing 64 GB request
across 16 exclusive steps. This changes no audit input, statistic, or rule.

Job `6891571` then started all 16 corrected shards from clean commit `f5d4541`
on the preemptible partition. Slurm preempted the healthy allocation after
11:39, before any shard finished; no partial JSON or scientific statistic was
accepted. The same clean commit and command were resubmitted on the
non-preemptible `gen` partition without changing an audit rule.

Job `6891583` completed all 16 shards and the strict reducer with exit code
`0:0` in 13:32. All 256 rank indices and processor coordinates were complete
and unique, all six fields were finite, and 85606 was not read. `Ne = Ni` and
`Pe = Ne * Te` passed all 624 frames. Both ion-pressure relations failed at
3,412 negative direct-`Pi` points over 72 frames, including 1,421 points over
47 frames in the fixed `y=1..30` interior. The frozen recommendation is
therefore to prefer direct evolved pressure or explicitly define and validate
a floor policy; an automatic channel change remains unauthorized. The compact
record and readable interpretation are
`paper0/results/phase2_pressure_closure_6891583.json` and
`paper0/PHASE2_PRESSURE_CLOSURE_READOUT.md`.
