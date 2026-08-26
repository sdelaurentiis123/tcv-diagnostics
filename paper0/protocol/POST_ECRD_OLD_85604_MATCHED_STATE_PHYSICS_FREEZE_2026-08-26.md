# Old-85604 matched state-view physics freeze

**Frozen prospectively:** 2026-08-26, while both matched multi-lead training
jobs were still running and before any selected checkpoint, bounded forecast,
exact candidate potential, or paired physics result was inspected

**Parent protocol:**
`POST_ECRD_OLD_85604_MATCHED_STATE_MULTILEAD_PROTOCOL_2026-08-26.md`

This document removes ambiguity from the phrase “across horizons four and
eight and the three chronological validation blocks.” It does not change the
training budget, model, state views, data ranges, or scientific decision in
the parent protocol.

## Scope and authorization

The paired physics evaluation runs only if both C5P and E6B pass the frozen
mechanical and transition gates. It uses old simulation 85604 only. The guard
interval, 85606, and all newer NERSC files remain unread and prohibited.

No model is selected or trained during physics scoring. Spectra,
cross-spectra, and transport remain evaluation quantities only.

## Compared forecast paths

At horizon four, compare the following terminal predictions:

1. direct lead four;
2. four autonomous lead-one steps;
3. two autonomous lead-two steps.

At horizon eight, compare:

1. direct lead eight;
2. eight autonomous lead-one steps;
3. four autonomous lead-two steps;
4. two autonomous lead-four steps.

The state views use identical current and target frames for every shared
method. Each autonomous composition feeds its complete predicted state back
into the next call.

## Chronological blocks

Blocks are contiguous in target-frame order and fixed separately at each
horizon so their transition counts are as equal as possible:

| Horizon | Block | Target frames | Count |
|---:|---:|---:|---:|
| 4 | 1 | `[500,541)` | 41 |
| 4 | 2 | `[541,582)` | 41 |
| 4 | 3 | `[582,624)` | 42 |
| 8 | 1 | `[504,544)` | 40 |
| 8 | 2 | `[544,584)` | 40 |
| 8 | 3 | `[584,624)` | 40 |

Adjacent frames are not treated as independent physical samples. These
blocks expose chronological transfer; they are not three independent shots.

## Common physical view

C5P already predicts `[Ne,Pe,Pi,phi,Vi]`. E6B predicts
`[Ne,Pe,Pi,NVe,NVi,Vort]` and both radial sides of `Bphi`.

For E6B, reconstruct potential using the pinned Hermes/BOUT++ elliptic
operator with predicted `Ne`, `Pe`, `Pi`, `Vort`, and predicted `Bphi`. The
solver starts from zero interior potential and receives no target-frame
field, target potential, or target boundary.

Derive ion velocity using the executed Hermes density floor:

\[
u = \max(N_e,0),
\qquad
\operatorname{softFloor}(N_e,f)=u+f\exp(-u/f),
\qquad
V_i=\frac{NV_i}{2\,\operatorname{softFloor}(N_e,10^{-7})}.
\]

Do not clip the resulting velocity. Record the number and minimum of
nonpositive predicted density cells.

Common-view field errors use the training-only C5P scalar normalization for
both arms. Authoritative transport is evaluated on the native 81-cell
toroidal grid using the frozen geometry and masks.

## Frozen scalar summaries

For each state view, calculate metrics separately for every horizon, method,
and chronological block. The primary summaries pool those predeclared cells;
they do not select a favorable method after results are known.

### Separatrix transport error

For each of particle, electron internal-energy, ion internal-energy, and
total internal-energy transport, calculate separatrix time-series relative
L2. The state-view scalar is the median across all four quantities, seven
horizon/method combinations, and three blocks (`84` values).

### Complex cross-spectrum error

For each of `Ne-phi`, `Pe-phi`, and `Pi-phi`, and for stored-mode bands
`k=1-3`, `k=4-5`, and `k=6-7`, define

\[
E_S =
\frac{
\left(\sum_{k\in B}|S_{\rm pred}(k)-S_{\rm truth}(k)|^2\right)^{1/2}
}{
\left(\sum_{k\in B}|S_{\rm truth}(k)|^2\right)^{1/2}
}.
\]

The state-view scalar is the median across the three pairs, three bands,
seven horizon/method combinations, and three blocks (`189` values).
Phase and coherence remain separately reported.

### Shared state error

Use the median standardized RMSE across `Ne`, `Pe`, and `Pi`, all seven
horizon/method combinations, and three blocks (`63` values). Both arms use
the same normalization.

### Spectral-power error

For each common field and each of the three bands, define

\[
E_P=|\log(P_{\rm pred}/P_{\rm truth})|.
\]

The state-view scalar is the median across five fields, three bands, seven
horizon/method combinations, and three blocks (`315` values). A nonpositive
or nonfinite power ratio is a causal reconstruction failure rather than an
omitted value.

## Frozen decision

Favor E6B only if all of the following hold:

1. `E6B median separatrix transport error <= 0.90 * C5P`;
2. `E6B median complex cross-spectrum error < C5P`;
3. `E6B median shared-state error <= 1.10 * C5P`;
4. `E6B median spectral-power error <= 1.10 * C5P`;
5. every E6B evolved-field and boundary persistence-skill gate passed;
6. every exact-potential output passed its causal provenance checks; and
7. every primary scalar is finite.

If all conditions pass, authorize a matched three-seed confirmation of the
same deterministic pair. Otherwise retain C5P as the old-data control and
stop this saved-state branch. Neither outcome authorizes 85606, stochastic
training, assimilation, diagnostic ranking, or steering.
