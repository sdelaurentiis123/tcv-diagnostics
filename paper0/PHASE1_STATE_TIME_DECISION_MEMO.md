# Phase 1 state and time decision memo

**Status:** evidence-backed design memo; not a frozen training protocol

**Scope:** 85604 metadata, exact executed Hermes source, and already exposed
85604 fields only

**85606 accessed:** no

**Training authorized by this memo:** no

## Executive conclusion

The current five-channel LOLA state is not the six-field state integrated by
the simulator. The exact 85604 configuration advances

```text
Ne, Pe, Pi, NVe, NVi, Vort
```

with fixed geometry and fixed particle/heating sources. `Te`, `Ti`, `Ve`,
`Vi`, and interior `phi` are derived fields. In addition, the radial boundary
value of `phi` has a short relaxation memory that is retained in restart files
but removed when the converted model grid strips radial guards.

This sharpens two earlier questions:

1. The clearest independent instantaneous variable missing from C5 is electron
   parallel momentum, not simply “more pixels” or “more latent toroidal modes.”
2. The useful form of time information is a physical history window and the
   relative forecast lead. Absolute frame number is not a physical forcing
   variable and would invite memorization of this one trajectory.

The safest eventual primary state is the exact evolved state plus an explicit
policy for potential-boundary memory and potential reconstruction. The most
pragmatic baseline may instead predict the six evolved fields together with a
redundant `phi` channel from a short history. That choice must be frozen and
tested before new codec or dynamics training.

## 1. Exact source evidence

The executed input file contains

```text
[e]
type = evolve_density, evolve_pressure, evolve_momentum

[i]
type = quasineutral, evolve_pressure, evolve_momentum

[hermes]
components = (..., vorticity, ...)
```

At the hash-locked Hermes revision `920ba829`, the relevant registrations are:

```text
evolve_density:  solver->add(N, "N" + species)
evolve_pressure: solver->add(P, "P" + species)
evolve_momentum: solver->add(NV, "NV" + species)
vorticity:       solver->add(Vort, "Vort")
```

Ion density is set by quasineutrality rather than advanced independently. The
resulting six volumetric solver variables are therefore

| Variable | Meaning | Raw source component |
|---|---|---|
| `Ne` | electron density; also fixes ion density | `evolve_density` |
| `Pe` | electron pressure | `evolve_pressure` |
| `Pi` | ion pressure | `evolve_pressure` |
| `NVe` | electron parallel momentum | `evolve_momentum` |
| `NVi` | ion parallel momentum | `evolve_momentum` |
| `Vort` | generalized vorticity | `vorticity` |

The hash-locked representative raw rank file contains all six variables and all
five historical derived fields for 624 saved times. Its arrays are float64 with
local shape `[624, 8, 6, 81]`. An all-rank inventory remains a pre-conversion
gate. This is a six-field electrostatic model: the component list contains no
electromagnetic-potential evolution.

Current Hermes documentation independently describes density, parallel
momentum, and pressure/energy as the fluid state and potential as an elliptic
inversion from vorticity. That documentation is useful corroboration, but the
exact run revision and input above remain authoritative:

- [Hermes-3 equations](https://hermes3.readthedocs.io/en/latest/equations.html)
- [Hermes-3 numerical methods](https://hermes3.readthedocs.io/en/latest/solver_numerics.html)

The 2025 TCV-X21 Hermes validation paper also calls this model family
“six-field.” It must not be treated as the numerical manifest for our data:
the paper reports different aggregate source values from the local 85604
input. The [paper](https://arxiv.org/abs/2506.12180) is conceptual context, not
provenance for these files.

## 2. Evolved and derived fields are not interchangeable

At the executed revision, temperature is constructed as

```text
T = floor(P, 0) / softFloor(N, density_floor)
```

and velocity is constructed as

```text
V = NV / (A * softFloor(N, density_floor)).
```

Potential is calculated from `Vort` and the ion-pressure contribution through
the geometry-dependent elliptic inversion. Consequently:

| Historical channel | Status | Information retained or lost |
|---|---|---|
| `Ne` | evolved | direct solver variable |
| `Te` | derived | cannot recover negative evolved `Pe`, although none occurs in 85604 |
| `Ti` | derived | cannot recover the 3,412 negative `Pi` cells already measured |
| `phi` | derived with boundary memory | strongly informative but redundant with `Vort`, `Pi`, geometry, and boundary state |
| `Vi` | derived | usually recovers ion momentum with density, subject to the density-floor policy |

The completed pressure audit and O1 transport ladder now quantify one part of
this distinction. `Pe = Ne*Te` passes throughout 85604 at the audit tolerance;
`Pi = Ne*Ti` fails exactly at the negative evolved-pressure cells. Despite that
formal state difference, C5T-versus-direct-pressure transport differs by less
than `5.1e-7` relative L2 for the scored radial ExB quantities. Thus C5T's
pressure loss is real but does not explain the present O1 transport error.

No equivalent all-frame momentum-closure or potential/vorticity forward-
operator audit has yet been run. Those are the remaining deterministic state
oracles.

## 3. The potential boundary carries short memory

The input sets

```text
phi_boundary_relax = true
phi_boundary_timescale = 1e-6 seconds.
```

Hermes updates radial `phi` guard values by exponential relaxation toward the
current toroidal mean, then supplies those values to the elliptic inversion.
The field is stored in restart files specifically so the boundary state
survives a restart. The current converted model arrays strip both radial guard
cells, so this auxiliary state is not explicitly retained.

The saved-frame cadence is `3.131905426 microseconds`. Across one saved
interval, a one-microsecond memory has residual weight

```text
exp(-3.131905426) = 0.0436346.
```

That is short but not mathematically zero. It yields three defensible future
state definitions:

| Candidate | Contents | Advantage | Risk/cost |
|---|---|---|---|
| `S6+Bphi` | six evolved volumes plus compact radial `phi` boundary state | closest discrete Markov state | new mixed volumetric/boundary representation; elliptic solve required |
| `S6+phi` | six evolved volumes plus redundant interior potential | easy transport scoring and common grid interface | consistency between `Vort`, `Pi`, and `phi` is learned rather than guaranteed |
| historical C5/C5P with history | observed fields over several prior frames | reuses infrastructure; delay history can encode omitted state | remains partially observed and must prove one-step sufficiency |

Calling any of these “physically valid” requires saying which sense is meant:
source-state fidelity, diagnostic convenience, or predictive sufficiency.

## 4. What “include time” should mean

The record begins at normalized time `285000`, not zero. With the stored
cyclotron-frequency conversion, it spans approximately `2.9753--4.9265 ms`;
`BOUT.settings` records `restart=true`. This proves that the saved data begin
after earlier evolution. It does not prove that the retained interval is
statistically steady.

The local input has fixed spatial source expressions and no action or
time-varying control channel. Therefore the configured simulator is intended
to be autonomous:

\[
x_{t+\Delta t}=F_{\Delta t}(x_t),
\]

up to omitted/derived state and numerical boundary memory. Absolute time is
not an input to that transition law. Feeding normalized frame number to the
network would let it learn “where am I in 85604?” and gives no transferable
meaning on 85606.

The correct temporal information is:

1. chronological ordering;
2. physical cadence and requested relative lead `Delta t`;
3. a short state history when the chosen observed state is non-Markovian;
4. the current slow `n=0` background profile, already present in the fields.

This is consistent with the system-identification literature: when variables
are omitted, history-dependent or delay-embedded maps can represent the
resulting non-Markovian closure. See, for example,
[Uy and Peherstorfer (2021)](https://arxiv.org/abs/2103.01362) and
[Ouala et al. (2020)](https://arxiv.org/abs/1907.02452). These papers motivate
a test; they do not prove that five frames are sufficient for this plasma.

## 5. Nonstationarity does not have one universal consequence

The frozen eight-block screen failed for every scalar profile statistic. Two
claims must be separated:

- **Learning a local autonomous transition:** strict stationarity is not a
  mathematical prerequisite if the input state contains the variables that
  explain the changing background and train/validation coverage is explicit.
- **Claiming a stationary turbulent distribution after decorrelation:** this
  does require a defensible steady regime, or else the distribution must be
  conditioned on the evolving background/time block.

This suggests two possible protocols, neither yet selected.

### Option A: steady-suffix protocol

Ben or the run producer identifies a physically justified steady interval or
provides a continuation. Paper 0 then retains stationary long-run spectral and
transport claims within that interval. This is the cleanest interpretation.

### Option B: conditional-transient protocol

Keep the leakage-safe chronological train/guard/validation boundaries but
explicitly treat validation as later-background extrapolation. Train a
time-homogeneous transition model without absolute time. Report metrics by
slow-background bin and contiguous time block, and do not call the aggregated
validation distribution a stationary climatology.

Under Option B, short-horizon O2/O3 state-sufficiency experiments can proceed,
but stationary post-decorrelation transport-distribution claims remain closed.
This would be a prospective protocol replacement, not a silent relaxation of
the failed Phase 1 gate.

## 6. Recommended next deterministic ladder

Before choosing LOLA, FGN, diffusion, or PDE-Refiner:

1. **All-rank inventory:** verify that every rank and saved frame contains the
   same six evolved fields with identical metadata and complete coordinates.
2. **Momentum closure:** compare raw `NVe/NVi` with the exact source formula
   from `Ne`, `Ve/Vi`, masses, and the executed density floor over all 624
   frames.
3. **Potential closure:** apply the exact forward vorticity operator to stored
   `phi`, `Pi`, geometry, and radial boundary state, and compare with `Vort`.
4. **Boundary-memory materiality:** measure the stored midpoint boundary state,
   its departure from the instantaneous zero-gradient target, and its effect
   on the elliptic solution.
5. **Freeze state candidates:** select an exact source-state candidate and one
   pragmatic observed-state baseline.
6. **Matched O1 codecs:** only under an accepted temporal protocol, train the
   candidates from scratch with identical data, reconstruction loss, budget,
   and checkpoint rule.
7. **Matched O2 state sufficiency:** use the same deterministic backbone to
   compare single-frame full state and history-based observed state. This
   determines whether omitted electron momentum is actually a prediction
   bottleneck.

Only after these steps should stochastic architecture choice address residual
uncertainty. Noise injection cannot repair a systematically incomplete state
definition.

## 7. Questions now worth asking Ben

1. Was normalized time `285000` chosen because the run was considered past
   burn-in? What interval was used for published or internal time averages?
2. Is the continued drift over `2.975--4.926 ms` expected flux-driven profile
   evolution, or does it indicate that this restart had not equilibrated?
3. Does Ben agree that the exact volumetric solver state for this input is
   `[Ne, Pe, Pi, NVe, NVi, Vort]` and that C5's main missing independent field
   is electron momentum?
4. For a fast emulator, should `phi` be recomputed with the elliptic solve,
   predicted redundantly, or represented through a separate diagnostic head?
5. Should the relaxed radial potential boundary be retained explicitly, or is
   its one-microsecond memory negligible for the intended 3.132-microsecond
   cadence?
6. Is a conditional-transient training claim scientifically acceptable if no
   steady suffix exists, with stationary long-horizon claims withheld?
7. The local 85604 input specifies `4.5e21 s^-1` particle source and `200 kW`
   heating per species. Which simulation campaign and physics target do 85604
   and 85606 represent, and which transport balance should Paper 0 reproduce?

## Decision boundary

This memo changes no manifest, split, normalization, model, checkpoint, or
acceptance threshold. It authorizes no training and no 85606 access. Its
purpose is to replace the vague “C5 may be incomplete” concern with exact,
testable source-state questions.
