# Phase 2 all-frame potential/vorticity closure protocol

Status: **frozen before the first all-624-frame forward-closure calculation**

This protocol extends the already accepted five-frame source-matched replay to
every saved frame of the 85604 development run. It is a deterministic data and
operator audit. It does not train a model, score a forecast, select an
architecture, or read 85606.

## 1. Question and decision boundary

The five frozen frames established both directions of the executed discrete
potential/vorticity relation:

1. runtime pressure plus stored vorticity and retained radial boundary state
   reconstruct stored potential; and
2. stored potential plus runtime pressure and the same boundary state generate
   stored vorticity.

The remaining question is narrower:

> Does that bidirectional source closure remain numerically valid at every one
> of the 624 saved 85604 frames?

A pass permits a separate, documented state-candidate decision. It does not
show that a candidate is predictively sufficient, that boundary memory is
negligible, that 85604 is stationary, or that any codec or forecast model
works.

## 2. Frozen evidence and scope

The audit reads only the raw 85604 archive at

```text
/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604
```

and uses the exact source and ABI revisions already validated by the
selected-frame replay. The following completed artifacts are immutable inputs:

- all-frame saved radial-potential boundary audit, job `6891890`;
- all-frame pressure-closure audit, job `6891583`;
- all-frame evolved-state and momentum audit, job `6891855`;
- runtime-pressure inverse replay, job `6892641`;
- selected-frame forward replay, job `6892764`.

Historical Paper 0 work has already inspected all-frame 85604 field values.
The prospective lock in this protocol is specifically before applying the
source-matched forward elliptic matrix to all 624 frames. It is not described
as a first read of the underlying values.

The exact frame scope is

```text
frame indices:             0, 1, ..., 623
normalized times:          285000, 285300, ..., 471900
normalized cadence:        300
physical cadence:          3.131905426352636 microseconds
physical shape per frame:  64 x 32 x 81
zperiod:                    5
```

All 103,514,112 physical points per volumetric field are in scope. No temporal
window is treated as an independent physical shot.

## 3. Executed source equation

For species pressure (P_s), density (N), and density floor
(f=10^{-7}), reproduce the executed `EvolvePressure` publication step:

\[
u_N = \max(N,0),
\]

\[
\operatorname{softFloor}(N,f)
=u_N+f\exp\!\left(-\frac{u_N}{f}\right),
\]

\[
P_s^{\mathrm{runtime}}
=N\,
\frac{\max(P_s^{\mathrm{raw}},0)}
{\operatorname{softFloor}(N,f)}.
\]

The pressure correction consumed by the vorticity component is

\[
\widehat{P}_i
=P_i^{\mathrm{runtime}}
-\frac{P_e^{\mathrm{runtime}}}{3672}.
\]

Define

\[
u=\phi+\widehat{P}_i,
\qquad
C=\frac{2}{B_{xy}^{2}}.
\]

The primary forward relation is

\[
\mathrm{Vort}_{\mathrm{forward}}
= C\,L_C(u),
\]

where (L_C) is not a newly written finite-difference approximation. It is
the exact matrix represented by the executed BOUT++ cyclic solver's public
`Laplacian::tridagCoefs` coefficients, applied through the same public
`rfft/irfft` convention over every native Fourier index (k=0,\ldots,40).

Because the simulated toroidal wedge is one fifth of a torus, reported mode
labels obey

\[
n=5k.
\]

The unexecuted `relax_potential` finite-volume operator is not an alternative
for this gate.

## 4. Retained radial boundary state

For each side, time, and global poloidal index, the saved boundary midpoint is
reconstructed from the raw potential guards as

\[
b_{\phi}
=\frac{\phi_{\mathrm{adjacent\ guard}}
+\phi_{\mathrm{adjacent\ physical}}}{2}.
\]

The midpoint must be toroidally constant and the outer guard must copy the
adjacent guard at the previously frozen (10^{-12}) absolute and relative
tolerances. The source-matched forward operator sets the two radial ghost rows
on a side to

\[
\phi_{\mathrm{ghost}}
=2b_{\phi}-\phi_{\mathrm{adjacent\ physical}}.
\]

All 64 stored radial rows remain operator rows. No physical row is replaced by
a boundary equation.

## 5. Immutable sharding and extraction

The eight shards are exactly the existing contiguous 78-frame temporal blocks:

| Shard | Half-open interval | Inclusive frames |
|---:|---:|---:|
| 0 | `[0, 78)` | `0..77` |
| 1 | `[78, 156)` | `78..155` |
| 2 | `[156, 234)` | `156..233` |
| 3 | `[234, 312)` | `234..311` |
| 4 | `[312, 390)` | `312..389` |
| 5 | `[390, 468)` | `390..467` |
| 6 | `[468, 546)` | `468..545` |
| 7 | `[546, 624)` | `546..623` |

Shard membership depends only on frame index. The extractor must:

1. verify the 256 rank filenames, decomposition, dimensions, metadata,
   complete affine time sequence, geometry hash, and source controls;
2. traverse each raw rank file once and write the eight canonical shard files
   without holding a complete 624-frame field in memory;
3. assemble `Ne`, raw `Pe`, raw `Pi`, stored `Vort`, and stored `phi` on the
   global physical mesh using `PE_XIND` and `PE_YIND`;
4. reconstruct both saved radial midpoint streams from the raw boundary ranks;
5. record shape-aware SHA-256 digests and exact frame/time lists for every
   canonical shard;
6. refuse an existing output path rather than overwrite it.

The raw-pressure inventory is also a frozen extraction identity. Across the
full domain it must reproduce:

```text
negative raw Pe count:  0
negative raw Pi count:  3412
negative Pi by block:   [0, 116, 1812, 86, 67, 69, 1262, 0]
minimum raw Pi:         -0.0234714551052543
minimum location:       (t,x,y,z) = (223,7,31,74)
```

These values came from the earlier independent pressure audit. They are not
acceptance thresholds discovered from the new forward result.

## 6. Ordered gates within each shard

The following order is mandatory. A later scientific comparison is blocked if
an earlier gate fails.

### G0: provenance and extraction

- all local files, completed inputs, raw controls, external source files,
  BOUT++ shared library, and `bout-config` match their frozen SHA-256 values;
- the Paper 0 worktree is clean at the exact submitted commit;
- the canonical shard contains exactly its 78 predeclared frames;
- every canonical array is finite and has the frozen axes and shape;
- processor and boundary coverage are complete and unique.

### G1: compiled known answers

The compiled forward path must pass:

- constant-null test;
- additive-gauge invariance;
- manufactured (k=0+k=3) forward/inverse round trip;
- presence of the manufactured (k=0) and (k=3) components.

The numerical rules remain identical to the accepted selected-frame protocol.

### G2: compiled input and runtime pressure

- the compiled stored-vorticity echo is bitwise equal to the canonical input;
- compiled runtime `Pe` and `Pi` reproduce the frozen scalar formula at every
  physical point using (10^{-12}) absolute and relative tolerances;
- all negative raw-pressure support is accounted for, including the frozen
  aggregate count after the eight shards are merged.

### G3: source forward closure

Only after G0--G2 pass may source vorticity be scored.

For each frame (t), let

\[
e_t
=\max_{x,y,z}
\left|
\mathrm{Vort}_{\mathrm{forward}}
-\mathrm{Vort}_{\mathrm{stored}}
\right|,
\]

and

\[
s_t
=\max_{x,y,z}
\left|
\mathrm{Vort}_{\mathrm{stored}}
\right|.
\]

Frame (t) passes only if all values are finite and

\[
e_t
\le
5\times10^{-10}
+5\times10^{-10}s_t.
\]

Every one of the 624 frames must pass. Correlation, pooled error, an additive
alignment, a regional subset, or temporal averaging cannot override a failed
frame.

## 7. Required reports

Each shard result must contain enough additive statistics for an exact merge
and must report:

- frame indices and normalized/physical times;
- per-frame maximum absolute error, scale-aware tolerance, relative L2, RMS,
  bias, correlation, non-finite count, and maximum-error location;
- pooled sufficient statistics without treating frames as independent shots;
- the same metrics for the authoritative named geometry regions;
- reference and residual power for every (k=0,\ldots,40), labeled by
  (n=5k);
- runtime-pressure discrepancies and negative raw-pressure counts;
- compiled known-answer metrics;
- canonical, BOUT output, binary, environment, and command hashes.

The final reducer must require exactly eight shard results, exact coverage of
`0..623` once each, matching source/protocol/code hashes, and successful
ordered gates. It reports temporal-block values descriptively. It may not use
624 as an independent-shot sample size.

## 8. Execution policy

Execution is CPU-only on Rocky 9. Use one four-rank BOUT++ replay at a time and
never run shards concurrently. The raw rank archive is traversed once during
canonical extraction, the oracle is compiled once per top-level run, and the
eight shard replays execute sequentially. The top-level job may request at
most 64 GiB and 60 minutes. No GPU is requested.

Every run uses a new job-specific result directory. Accepted prior artifacts
and failed attempts are immutable. The launcher must refuse a dirty worktree,
an unexpected commit, an existing result directory, a missing shard, or a
hash mismatch.

## 9. Decision rules

- **Pass:** all 624 frames pass every ordered gate. The retained-boundary,
  runtime-pressure potential/vorticity relation is then validated across the
  complete saved 85604 archive, and a separately committed state-candidate
  decision may proceed.
- **Fail:** retain the failed artifact, localize the first failing gate and
  frame, and do not select new state channels or train a codec.
- **Either outcome:** do not access 85606, change the Phase 1 split or
  normalization, select a forecast architecture, tune a model, or claim
  predictive sufficiency.

Passing this audit would establish deterministic source-state closure only.
Matched O1 reconstruction and O2 one-step experiments remain necessary to
learn whether an exact state or a pragmatic history-conditioned state is the
better emulator representation.
