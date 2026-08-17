# Phase 2 native-81 versus resampled-88 readout

**Status:** all prospectively frozen resampling and quantization gates pass

**Development data:** all 624 frames of TCV/Hermes run 85604

**Held-out data:** run 85606 was not read

**Execution:** Rocky 9 job `6891664`, clean Paper 0 commit `67abc70`

## Result in one paragraph

The existing unwindowed Fourier transform from 81 to 88 toroidal cells is
effectively reversible for the stored fields, but evaluating the nonlinear
Hermes-derived radial operator directly on 88 cells is not identical to its
native-81 evaluation. Every frozen gate passes when an 88-cell state is
downsampled back to 81 before scoring. Direct 88-cell evaluation changes total
radial face flow by approximately 1.6% and radial divergence by 3.5--3.8%,
which is the prospectively defined **small** range. Therefore Paper 0 may use
an 88-cell model grid, but its primary transport evaluator must downsample
every ensemble member to native 81 and apply the native operator there.

This is a numerical representation decision. It neither selects an
architecture nor establishes that any emulator preserves transport.

## What was compared

Let `U` be the frozen SciPy Fourier resampling from 81 to 88 toroidal cells,
`D` the corresponding 88-to-81 transform, and `Q_N` the validated partial
Hermes radial face-flow operator evaluated on `N` toroidal cells. Three paths
were evaluated for every 85604 frame:

```text
native:       Q_81(x_81)
round trip:   Q_81(D(U(x_81)))
direct 88:    Q_88(U(x_81))
```

The direct-88 result is aligned to 88 cells against `U(Q_81(x_81))`. This
measures the noncommutation of Fourier resampling with the nonlinear limiter,
positivity reconstruction, and finite-volume differentiation. It is not a
comparison between a forecast and truth.

The five source-faithful candidate fields were
`C5P = [Ne, Pe, Pi, phi, Vi]`. The evaluated primary quantities were particle,
electron internal-energy, ion internal-energy, and total internal-energy
radial face flow and their conservative radial divergences. Direct evolved
negative `Pi` values were retained.

## Frozen gates

The thresholds were committed before any round-trip field, transport,
float-quantization, or direct-88 value was calculated:

| Gate | Frozen maximum |
|---|---:|
| Maximum per-frame field round-trip relative L2 | `2e-6` |
| Primary transport round-trip aggregate relative L2 | `1e-4` |
| Primary transport round-trip per-frame relative-L2 p99 | `1e-3` |
| Five-frame raw-float64 versus stored-float32 transport aggregate relative L2 | `1e-5` |

Direct-88 aggregate relative L2 was not a pass/fail gate. Its frozen labels
were negligible below 1%, small from 1% to below 5%, material from 5% to below
10%, and severe at 10% or above.

## Structural checks

All structural checks passed:

- all 624 frames occurred exactly once across 17 chunk-aligned shards;
- all inputs and valid outputs were finite;
- on frames `[0, 156, 312, 467, 623]`, raw float64 `Ne`, `Pe`, `Pi`, and `phi`
  equaled the native Well fields after the declared float32 cast;
- on those same frames, the public SciPy call reproduced every legacy z88
  `C5T` field bit-for-bit;
- all source, geometry, converter-evidence, code, and manifest hashes matched;
- the run used `zperiod=5`, so stored wedge index `k` maps to full-torus mode
  number `n=5k`;
- run 85606 was not accessed.

The four extra nonnegative Fourier bins on the 88-cell grid, `k=41..44`, are
padding bandwidth. They do not represent additional simulator-resolved
physics.

## Field round trip

| Field | Aggregate relative L2 | Maximum frame | Frame p99 | Pass |
|---|---:|---:|---:|:---:|
| `Ne` | `1.4835e-7` | `1.5420e-7` | `1.5269e-7` | yes |
| `Pe` | `1.5207e-7` | `1.6033e-7` | `1.5804e-7` | yes |
| `Pi` | `1.5242e-7` | `1.5968e-7` | `1.5799e-7` | yes |
| `phi` | `1.5354e-7` | `1.5752e-7` | `1.5713e-7` | yes |
| `Vi` | `1.4864e-7` | `1.5237e-7` | `1.5139e-7` | yes |

The largest observed per-frame field error, `1.6033e-7` for `Pe`, is about
12.5 times below the frozen field threshold. These errors are at the expected
float32-transform scale; they are not evidence of a learned reconstruction.

## Native versus round-trip transport

### Total radial face flow

| Quantity | Aggregate relative L2 | Frame p99 | Maximum frame | Pass |
|---|---:|---:|---:|:---:|
| Particle | `4.9110e-6` | `7.3803e-6` | `7.6474e-6` | yes |
| Electron internal energy | `4.3816e-6` | `6.4540e-6` | `6.8951e-6` | yes |
| Ion internal energy | `4.5696e-6` | `6.7753e-6` | `7.1753e-6` | yes |
| Total internal energy | `4.4767e-6` | `6.6182e-6` | `7.0386e-6` | yes |

### Conservative radial divergence

| Quantity | Aggregate relative L2 | Frame p99 | Maximum frame | Pass |
|---|---:|---:|---:|:---:|
| Particle | `2.4436e-5` | `2.9782e-5` | `3.0664e-5` | yes |
| Electron internal energy | `2.0223e-5` | `2.5280e-5` | `2.6715e-5` | yes |
| Ion internal energy | `2.2640e-5` | `2.7658e-5` | `2.8757e-5` | yes |
| Total internal energy | `2.1541e-5` | `2.6667e-5` | `2.7871e-5` | yes |

The worst aggregate primary round-trip result is particle divergence at
`2.4436e-5`, about 4.1 times below its `1e-4` gate. Its frame p99 is
`2.9782e-5`, about 33.6 times below the separate `1e-3` gate.

## Raw float64 versus stored float32

The five value-independent raw-oracle frames isolate storage quantization
from resampling. Every primary aggregate passes. Total face-flow relative L2
ranges from `6.8666e-7` to `7.9441e-7`; divergence ranges from `3.6589e-6` to
`4.3766e-6`. The largest value is about 2.3 times below the frozen `1e-5`
threshold.

This is a five-frame quantization check, not an all-frame temporal result.
Accordingly, the compact record explicitly marks temporal-block summaries as
unavailable for this comparison.

## Direct 88-cell sensitivity

### Total radial face flow

| Quantity | Aggregate relative L2 | Frame p99 | RMS ratio | Correlation | Label |
|---|---:|---:|---:|---:|:---:|
| Particle | `0.01631` | `0.02361` | `1.00802` | `0.999900` | small |
| Electron internal energy | `0.01586` | `0.02181` | `1.00799` | `0.999907` | small |
| Ion internal energy | `0.01613` | `0.02328` | `1.00801` | `0.999903` | small |
| Total internal energy | `0.01596` | `0.02252` | `1.00799` | `0.999905` | small |

### Conservative radial divergence

| Quantity | Aggregate relative L2 | Frame p99 | RMS ratio | Correlation | Label |
|---|---:|---:|---:|---:|:---:|
| Particle | `0.03779` | `0.04341` | `1.01734` | `0.999446` | small |
| Electron internal energy | `0.03531` | `0.04112` | `1.01577` | `0.999508` | small |
| Ion internal energy | `0.03636` | `0.04229` | `1.01688` | `0.999490` | small |
| Total internal energy | `0.03543` | `0.04162` | `1.01638` | `0.999514` | small |

The direct 88-cell calculation is highly correlated with the aligned native
calculation and has little weighted sign disagreement, but it is systematically
slightly sharper: median absolute-tail ratios are about `1.007--1.008` for
face flow and `1.015--1.019` for divergence. The difference is larger after
the conservative radial difference, as expected for a derivative of a
resolution-dependent nonlinear face reconstruction.

This sensitivity is small rather than zero. It is large enough that mixing
native-81 truth scores with direct-88 forecast scores would introduce an
avoidable numerical definition change.

## Decision and consequences

The prospectively frozen decision rule now applies:

1. Models may retain an 88-cell computational grid.
2. Before primary transport evaluation, downsample every truth or forecast
   field from 88 to 81 with the frozen transform.
3. Apply `Q_81` separately to every ensemble member.
4. Reduce the member-wise transport distribution only after the nonlinear
   operator has been evaluated.
5. Report direct `Q_88` only as a named sensitivity.

This result does **not** authorize an architecture change, an automatic channel
change, use of transport as a training loss, or release of 85606. It also does
not make `C5P` a demonstrated Markov-complete state. One-step state-sufficiency
tests must still determine whether omitted evolved vorticity, electron
momentum, sources, or boundary information are required.

The transport validation ladder is still incomplete. Geometry-region surface
integrations, outward orientation, normalized-to-SI units, and an ensemble
known-answer test remain required before this partial operator can support a
paper-level physical particle- or heat-transport claim. Once those rungs pass,
O1 can evaluate transport through each codec and finish the representation
decision.

## Reproducibility

The Slurm job completed `0:0` on Rocky Linux 9.8 in `00:01:41`. It used 17
one-CPU shard steps; the largest shard peak RSS was 248,588 KiB, while the
batch step peak was 7,943,304 KiB.

- Full immutable result:
  `/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_85604_resampling/job_6891664/resampling_sensitivity.json`
- Full-result SHA-256:
  `4b903d27d303e7b5db086d4e1ea62856f65cac7aacc3e623ac98bab1706d2781`
- Artifact-digest-manifest SHA-256:
  `f64f9af13f7f8e7751d50cb384c14d41d98e13b7eb7fcd4878ea99747d8b661d`
- Job-log SHA-256:
  `3b761ad609173c86a9ff901464aad9e700bde0110342b876c7ebbdc1a50d49dc`
- Exact-commands SHA-256:
  `30eec1a2986dfcb4a0d871c07e0c50b73f7ad8e81af7c622643beb3bc6e9f792`
- Environment-record SHA-256:
  `8c515265cbfe39c76c75157062ef1281334e996702487822012f533efc54d4f0`
- Tracked compact result:
  `paper0/results/phase2_resampling_6891664.json`
- Compact-result SHA-256:
  `2d1ed6e7af5a1559e213590ed6315775400ebdcb1db849cfc826d77ef7d8b4a5`

The full result and all 17 partials passed `sha256sum -c` after completion.
The compact-result tool verifies the full result digest, exact job and commit,
85604-only scope, complete merge, grid shapes, `zperiod`, and comparison schema
before writing. Its first implementation attempt produced no output because it
incorrectly required temporal-block summaries for the five-frame float64
oracle. Commit `836417c` made that scope explicit and added a regression test;
the full scientific artifact was neither changed nor rerun.
