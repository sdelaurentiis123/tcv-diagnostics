# Phase 2 shared model-dataset readout

**Decision:** pass

**Scientific scope:** verified 85604 engineering representation only

**Rocky 9 job:** `6893525`

**Paper 0 commit:** `929ed0cb2a861742bcab34101bc60fd53970d40c`

**Held-out 85606 read:** no

**Training performed or released:** no

## Result in one paragraph

Paper 0 now has one hash-locked dataset from which all three frozen state views
can be assembled. It contains all 624 frames of
`[Ne,Pe,Pi,NVe,NVi,Vort,phi,Vi]` on the common 88-cell periodic grid and
the two-sided `Bphi` boundary state. All source, coverage, time, resampling,
legacy-transform, HDF5 echo, boundary-cast, and training-only normalization
gates pass. The largest native-81 to model-88 to native-81 per-frame relative
\(L_2\) error is \(1.9675\times10^{-7}\), 10.2 times below the frozen
\(2\times10^{-6}\) limit. This releases the dataset as common engineering
evidence. It does not show that a codec or transition model works and does not
authorize model training until the matched O1/O2 protocol is committed.

## What was built

The immutable external root is:

~~~text
/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525
~~~

It contains eight non-overwritten HDF5 shards, each covering one fixed
78-frame interval. Together they contain:

| Quantity | Value |
|---|---:|
| frames | 624 |
| volume fields | 8 |
| spatial shape per field/frame | `[64,32,88]` |
| boundary shape per frame | `[2,32]` |
| HDF5 shard count | 8 |
| total HDF5 bytes | 3,599,761,472 |

The shared dataset supports:

- `E6B-H1 = [Ne,Pe,Pi,NVe,NVi,Vort] + Bphi`;
- `C5P-H2 = [Ne,Pe,Pi,phi,Vi]` over two ordered frames;
- `C5P-H1 = [Ne,Pe,Pi,phi,Vi]` over one frame.

Those labels describe data views, not accepted predictive models.

## Integrity results

Every prospectively frozen gate passes:

| Gate | Result |
|---|---|
| native and legacy source SHA-256 preflight | pass |
| boundary extraction-record and canonical-shard hashes | pass |
| frames `0..623` exactly once | pass |
| all required inputs and reopened outputs finite | pass |
| normalized time `285000..471900` in exact steps of `300` | pass |
| physical cadence conversion | `3.131905426352636 microseconds` |
| writer-to-reopened HDF5 array digests | bitwise exact |
| historical z88 `Ne`, `phi`, and `Vi`, all 624 frames | bitwise exact |
| `Bphi` versus explicit float32 source cast | bitwise exact |
| independent normalization recomputation | pass |
| every normalization scale finite and positive | pass |
| clean exact Git commit and Rocky major version 9 | pass |

The stored coordinate is normalized ion-cyclotron time,
\(\tau=\Omega_{ci}t\), not microseconds. The physical conversion uses

\[
\Omega_{ci}=95{,}788{,}333.03066081\ {\rm s}^{-1},
\qquad
\Delta t=\frac{300}{\Omega_{ci}}
=3.131905426352636\ {\rm \mu s}.
\]

## Toroidal round-trip result

For each native frame \(x\), the validator applies the frozen unwindowed
SciPy transform \(U:81\rightarrow88\), then
\(D:88\rightarrow81\), and measures

\[
\epsilon_{\mathrm{rel}\,2}
=
\sqrt{
\frac{\sum_i[D(U(x))_i-x_i]^2}
     {\sum_i x_i^2}
}.
\]

| Field | Maximum per-frame relative \(L_2\) | Frozen limit | Result |
|---|---:|---:|---|
| `Ne` | \(1.5420412\times10^{-7}\) | \(2\times10^{-6}\) | pass |
| `Pe` | \(1.6032685\times10^{-7}\) | \(2\times10^{-6}\) | pass |
| `Pi` | \(1.5968113\times10^{-7}\) | \(2\times10^{-6}\) | pass |
| `NVe` | \(1.9674526\times10^{-7}\) | \(2\times10^{-6}\) | pass |
| `NVi` | \(1.5760109\times10^{-7}\) | \(2\times10^{-6}\) | pass |
| `Vort` | \(1.9486528\times10^{-7}\) | \(2\times10^{-6}\) | pass |
| `phi` | \(1.5751611\times10^{-7}\) | \(2\times10^{-6}\) | pass |
| `Vi` | \(1.5237029\times10^{-7}\) | \(2\times10^{-6}\) | pass |

The new \(k=41,\ldots,44\) bins are still numerical padding. This result does
not claim new resolved physics above native \(k=40\), and the full-torus
mapping remains \(n=5k\).

## Frozen training-only normalization

Only frames `[0,432)` enter these statistics. `Ne` uses
\(\log(N_e+10^{-6})\); every other field uses the identity transform.
Direct negative `Pi` is preserved.

| Record | Count | Mean | Population standard deviation |
|---|---:|---:|---:|
| `Ne` after log transform | 77,856,768 | -1.93684496 | 1.43536251 |
| `Pe` | 77,856,768 | 0.41732773 | 0.73493589 |
| `Pi` | 77,856,768 | 0.47917411 | 0.69350059 |
| `NVe` | 77,856,768 | -0.0000358540 | 0.0004674003 |
| `NVi` | 77,856,768 | -0.03341741 | 0.26549971 |
| `Vort` | 77,856,768 | -0.0008084904 | 0.03870687 |
| `phi` | 77,856,768 | 2.84852124 | 1.27965871 |
| `Vi` | 77,856,768 | -0.17667432 | 0.92095822 |
| `Bphi/inner` | 13,824 | 0.66087010 | 0.64182751 |
| `Bphi/outer` | 13,824 | 2.07293524 | 0.46193680 |

The reducer recomputed these records by reopening the final HDF5 files and
streaming training frames independently of the shard partials. Counts matched
exactly; means, \(M_2\), variances, and standard deviations matched the frozen
\(10^{-12}\) relative and absolute tolerances.

## Execution and compute

The job ran on `worker5582`, Rocky Linux 9.8, and completed with Slurm state
`COMPLETED`, exit `0:0`, in `00:01:03`. Each of the eight conversion
steps used two CPU cores, took 11--13 seconds after source hashing, and peaked
near 1.23 GB RSS. The batch step peaked at 7,662,140 KB. No GPU was requested.

Before submission:

- the local complete suite passed with `368 passed, 1 skipped`;
- the exact Rocky 9 checkout passed with
  `368 passed, 1 skipped, 29 subtests passed`;
- the data environment imported the exact builder and reducer with
  h5py `3.16.0`, NetCDF4 `1.7.4`, NumPy `2.4.6`, and SciPy `1.17.1`;
- the 14 NetCDF-dependent implementation tests passed directly on Rusty.

The local skip is only the unavailable Mac NetCDF stack; it is covered by the
passing Rocky 9 run.

## Immutable artifacts

| Artifact | SHA-256 |
|---|---|
| full external result | `27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18` |
| normalization | `f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7` |
| source-hash preflight | `328dba064080b376f88137193afd07c0d56d96401b45a96fac29e4a0f0c0476b` |
| complete artifact index | `6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01` |

Every entry in the artifact index was rechecked with `sha256sum -c` after
job completion. The compact result and normalization are tracked unchanged at:

~~~text
paper0/results/phase2_model_dataset_6893525.json
paper0/results/phase2_model_dataset_normalization_6893525.json
~~~

## Exact launch and validation

The execution was submitted from the Rocky 9 login at exact clean commit
`929ed0cb2a861742bcab34101bc60fd53970d40c`:

~~~bash
sbatch \
  --export=ALL,PAPER0_EXPECTED_COMMIT=929ed0cb2a861742bcab34101bc60fd53970d40c \
  cluster/phase2_85604_model_dataset.sbatch
~~~

The launcher records every expanded builder and reducer command in
`job_6893525/commands.sh`. Artifact integrity is reproduced with:

~~~bash
cd /mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525
sha256sum -c artifact_sha256.txt
~~~

## What this changes, and what it does not

This result removes a major attribution hazard: future state and architecture
comparisons can no longer differ accidentally in source file, frame coverage,
toroidal transform, storage precision, or normalization fit.

It does not yet answer:

- whether the float32 exact-state volumes and boundary reconstruct the required
  field and transport metrics through a codec;
- whether `E6B-H1`, `C5P-H2`, or `C5P-H1` is predictively sufficient;
- whether one-step skill survives autonomous rollout;
- whether any stochastic representation is calibrated;
- whether assimilation or diagnostic ranking is defensible;
- anything about 85606.

The next required action is to freeze and commit the matched deterministic O1
codec and O2 one-step protocol. Only that later protocol may release training.
