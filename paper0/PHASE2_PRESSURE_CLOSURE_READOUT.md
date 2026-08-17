# Phase 2 all-frame pressure-closure readout

## Bottom line

The inherited five observable channels are not an exact representation of the
evolved Hermes state for pressure-based transport. The discrepancy is narrow
and fully explained, but it is not confined to a disposable target boundary.

Across all 624 native 85604 frames, `Ne = Ni` and `Pe = Ne * Te` pass. Ion
pressure fails because Hermes retains occasional negative evolved `Pi` values
while deriving `Ti` from pressure floored at zero. Of 3,412 negative `Pi`
points, 1,421 lie in the previously fixed guard-independent transport interior
`y=1..30`. Those interior discrepancies occur in 47 of 624 frames.

Therefore Paper 0 must either forecast evolved `Pi` directly or define and
validate a pressure-floor target. It may not silently calculate exact ion
pressure transport as `Ne * Ti`.

## What was audited

Rocky 9 job `6891583` ran from clean commit `f5d4541` and completed on the
non-preemptible `gen` partition in 13 minutes 32 seconds. Sixteen independent
rank shards covered all 256 `(PE_XIND, PE_YIND)` coordinates exactly once,
then a strict reducer merged their sufficient statistics. The audit read:

- run 85604 only; 85606 remained sequestered;
- every stored frame `0..623`;
- native shape `[624, 64, 32, 81]` for each field;
- direct `Ne`, `Ni`, `Pe`, and `Pi`, plus derived `Te` and `Ti`;
- 103,514,112 physical cells per field;
- full `y=0..31`, interior `y=1..30`, and target rows `y in {0,31}`;
- the prospectively frozen `atol = rtol = 1e-12` closure rules.

All fields were finite. Artifact hashes, rank-stream digests, commands,
environment, and the exact reducer result are preserved under the immutable
job directory.

## Closure results

| Relation | Full-domain result | Interior result | Interpretation |
|---|---:|---:|---|
| `Ni = Ne` | 0 discrepant points; 624/624 frames pass | 0; 624/624 | quasineutral density is closed |
| `Pe = Ne * Te` | 0 discrepant points; 624/624 frames pass | 0; 624/624 | electron pressure is exactly recoverable to tolerance |
| `Pi = Ne * Ti` | 3,412 points; 72 frames fail | 1,421 points; 47 frames fail | temperature loses retained negative ion pressure |
| `Pi = Ni * Ti` | same support as above | same support as above | `Ni = Ne`, so the density choice is not the cause |

Every ion-pressure discrepancy has a negative direct `Pi` reference. There
are zero discrepancies where direct `Pi` is nonnegative. Thus this is not a
general inconsistency between density, pressure, and temperature; it is the
specific source-backed floor transformation already identified in the
five-frame oracle.

The largest full-domain discrepancy is `0.0234714551` at
`(frame,x,y,z) = (223,7,31,74)`. The largest interior discrepancy is
`0.00302343566` at `(223,7,30,54)`. These pressure values are normalized
Hermes outputs; multiply by `80.1088317 Pa` for the recorded physical
conversion.

## How rare, and where

Rarity must be stated two ways:

- by cell count, negative `Pi` occupies 3,412 of 103,514,112 full-domain
  values (`0.003296%`) and 1,421 of 97,044,480 interior values (`0.001464%`);
- by time, it appears in 72 of 624 full frames (`11.54%`) and 47 of 624
  interior frames (`7.53%`).

The points are strongly localized rather than uniformly scattered:

| Global `y` | Negative `Pi` points |
|---:|---:|
| 27 | 25 |
| 28 | 103 |
| 29 | 298 |
| 30 | 995 |
| 31 | 1,991 |

All lie at radial indices `x=5..14`; 3,316 of 3,412 lie at `x=5..10`. Counts
over the eight predeclared 78-frame blocks are
`[0, 116, 1812, 86, 67, 69, 1262, 0]`. The first and final blocks contain none,
while two temporal episodes dominate. These blocks localize behavior within
one trajectory; they are not independent physical shots.

`Ti` itself contains 1,074 formally negative values, but its minimum is only
`-4.87e-16`, consistent with roundoff around the zero floor. Those tiny signed
values cannot encode the finite negative evolved pressures.

## State consequence

The result separates three statements that had previously been tangled:

1. The conservative radial-flow implementation is correct on direct native
   `Ne`, `Pe`, and `Pi` fields for the selected real frames.
2. The electron branch closes through `Ne` and `Te` everywhere in 85604.
3. The ion temperature channel is not lossless with respect to evolved ion
   pressure, including within the accepted interior operator region.

The leading Paper 0 state candidate is therefore the evolved-pressure version
`[Ne, Pe, Pi, phi, Vi]`, with physical temperatures derived using the same
explicit simulator floor when temperature diagnostics are required. This is a
candidate to freeze prospectively, not a retroactive relabeling of historical
C5 models. The legacy `[Ne, Te, Ti, phi, Vi]` state remains the apples-to-apples
baseline. A state-sufficiency ablation must still test omitted evolved
variables such as vorticity and electron momentum, and `phi` metrics still
require gauge fixing.

## What this does not establish

This audit contains no learned forecast, codec reconstruction, resampled-88
field, diagnostic observation, ensemble, or held-out 85606 value. It does not
show that the rare negative pressures are physically important, nor does it
authorize clipping them. It shows only that direct evolved pressure and
floor-derived temperature are different targets and that the difference
reaches the operator interior often enough that Paper 0 must choose explicitly.

## Execution history and evidence

The corrected 16-shard launcher itself was proven healthy by job `6891571`,
but that preemptible job was externally preempted at 11:39 before any shard
finished. It wrote no partial JSON and contributes no scientific statistic.
The identical clean commit and command were resubmitted on `gen` as job
`6891583`; all 16 shards and the reducer completed with exit code `0:0`.

The compact accepted record is
`paper0/results/phase2_pressure_closure_6891583.json`; the preemption record is
`paper0/results/phase2_pressure_closure_6891571.json`. The immutable accepted
root is
`/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_85604_pressure_closure/job_6891583`.
The full strict JSON has SHA-256
`db340843ba77fe4d06da2842561ced77ac2814bfd084224baa85b4485ad840c2`.
