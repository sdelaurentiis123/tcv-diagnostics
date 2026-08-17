# Phase 2 geometry, units, and ensemble-transport protocol

**Protocol status:** frozen before implementation or release evaluation

**Development run:** TCV/Hermes `85604`

**Sequestered evaluation run:** `85606` (access remains forbidden)

**Purpose:** complete rungs 9--11 of the transport validation ladder without
changing the already validated Hermes face operator, the C5P state policy, or
the native-81 transport policy.

This protocol defines the spatial masks, sign convention, surface reduction,
normalized-to-SI conversion, and ensemble ordering used by Paper 0. Physics
quantities remain evaluation metrics only. Nothing here authorizes model
training, access to `85606`, or a learned-model transport claim.

## 1. Locked evidence

The geometry is
`/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc`,
SHA-256
`0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8`.
Execution must stop if the file does not match this value. The same exact hash
is stored in the machine-readable manifest.

The grid embeds Hypnotoad revision
`e4a1dff39b80e30aaa05eb6903a8dc72cf4ed832`. Its
`hypnotoad/core/mesh.py` has SHA-256
`3c4a3d8f5b94ab728650726fbf010af70f63ae6452a83e024460d34ab99336e3`.
That source establishes two semantics used below:

1. Hypnotoad chooses the sign of its radial coordinate so `x` increases
   radially across the grid.
2. `penalty_mask` is zero when both poloidal cell edges are inside the wall,
   one when both are outside, and a fractional **poloidal-edge length** when a
   cell crosses the wall. It is not a cell-area fraction.

The executed BOUT++ revision is
`7d28d67c3f12c24ec281c0982e870f5369c65a6f`. Its
`include/bout/constants.hxx`, SHA-256
`4a89ceb00a66799668b1b73d3598e2995d9e171680be0d5ce0d20fe6b33e63b2`,
defines

\[
e=1.602176634\times10^{-19}\ \mathrm{C},
\qquad
m_p=1.672621898\times10^{-27}\ \mathrm{kg}.
\]

The source and geometry hashes are evidence dependencies, not vendored code.

### Disclosed design observations

The following read-only structural probes were made before this protocol was
frozen. They selected definitions and are repeated as execution gates; they
are not post-hoc model results.

- The full grid is `68 x 32`; removing two radial guards on each side gives
  the model crop `64 x 32` and `model_x = grid_x - 2`.
- `ixseps1=18`, so the first local SOL cell is `x=16` and the exact
  separatrix x-low face has local left cell `x=15`.
- `jyseps1_1=7`, `jyseps1_2=16`, `jyseps2_1=16`, and
  `jyseps2_2=23`. The closed-field poloidal segment is `y=8..23`.
- The two separatrix branch corners coincide at
  `(R,Z)=(0.7942120340630372,-0.4420630317234957) m`.
- The maximum separatrix-face major radius on the closed-field segment is at
  `y=18`, `(R,Z)=(1.087493128717754,-0.017535153092040314) m`.
- Every one of the 2,016 radial `psixy` differences in the crop is positive;
  the smallest is `4.1698824270566684e-05`. The separatrix x-low `psi` agrees
  with `psi_bdry` to `1.64e-16` absolute.
- The crop spans approximately `psi_N=0.8991575..1.1084596`; consequently the
  closed portion of this crop is called the **confined edge**, not the whole
  plasma core.
- The strict wall-interior operator region contains 1,869 cells. The logical
  partition below contains 256 confined-edge, 219 private-flux, and 1,394 SOL
  cells.

No metric value from a learned forecast was inspected.

## 2. Index and topology convention

All cell masks have axes `[x,y]` on the `64 x 32` model crop. The physical
target rows `y=0` and `y=31` are excluded because the model tensor does not
contain the physical-boundary guard values required by the audited shifted
operator. The guard-independent operator region is therefore `y=1..30`.

The three primary topology masks are disjoint:

\[
\begin{aligned}
\mathrm{confined\ edge}:&\quad x<16,\;8\le y\le23,\\
\mathrm{private\ flux}:&\quad x<16,\;1\le y\le7\ \text{or}\ 24\le y\le30,\\
\mathrm{SOL}:&\quad x\ge16,\;1\le y\le30.
\end{aligned}
\]

Each is intersected with the strict wall-interior mask
`penalty_mask == 0`. Their union must equal every strict wall-interior cell in
the operator region exactly once.

Overlapping analysis masks are also defined:

- **separatrix cell band:** local `x in {15,16}`, `y=1..30`;
- **outboard-midplane row:** the closed-field `y` maximizing
  `Rxy_xlow` on the exact separatrix face; the expected rerun value is `y=18`;
- **X-point topology stencil:** local `x in {15,16}` and
  `y in {7,8,23,24}`;
- **inner divertor leg:** `y=1..7`;
- **outer divertor leg:** `y=24..30`.

All overlapping cell masks are intersected with the operator region and the
strict wall-interior mask. The inner/outer labels are checked against physical
major radius at the two target-side branches. These masks describe simulation
geometry; they are not experimental diagnostic response functions.

Fractional and unit `penalty_mask` cells are excluded from primary regional
statistics. A secondary sensitivity may include wall-crossing cells, but it
must be named explicitly and must not interpret `1-penalty_mask` as physical
cell area.

## 3. Exact separatrix surface and sign

The primary outward transport surface is only the closed-field separatrix:

\[
i_{\mathrm{left}}=15,\qquad y=8..23,\qquad z=0..80.
\]

A face is valid only if both adjacent cells are strict wall-interior cells and
the previously validated operator marks it valid. Private-flux/SOL branch
faces at the same radial index are not silently added to this confined-plasma
surface.

The executed Hermes face-flow sign follows from its finite-volume update: a
positive face flow leaves cell `i` and enters cell `i+1`. Because grid `psi`
and Hypnotoad `x` both increase radially, positive `+x` on the closed-field
separatrix is **outward from the confined region**. Elsewhere the universal
label remains `+x`; “outward” is not applied indiscriminately in the private
flux region.

For normalized face flow `F` on a declared face mask `S`, the simulated-wedge
surface flow is

\[
\mathcal F_{\mathrm{wedge}}(S)
=\sum_{(j,k)\in S} F_{j,k}\,\Delta y_j\,\Delta z,
\qquad
\Delta z=\frac{2\pi}{5N_z}.
\]

This is the primary convention. A full-torus-equivalent number may be shown
only as the explicitly named sensitivity

\[
\mathcal F_{\mathrm{full\ torus\ equivalent}}=5\mathcal F_{\mathrm{wedge}},
\]

which assumes five identical periodic wedges and is not a new independent
sample.

## 4. Physical-unit conversion

For the executed normalization,

\[
C_{s0}=\sqrt{\frac{eT_{\mathrm{norm}}}{m_p}},\qquad
\Omega_{ci}=\frac{eB_{\mathrm{norm}}}{m_p},\qquad
\rho_{s0}=\frac{C_{s0}}{\Omega_{ci}},
\]

with `Nnorm=1e19 m^-3`, `Tnorm=50 eV`, and `Bnorm=1 T`. The expected values
read from the run and independently reconstructed from source constants are

\[
C_{s0}=69205.61141651045\ \mathrm{m\,s^{-1}},
\]

\[
\Omega_{ci}=95788333.03066081\ \mathrm{s^{-1}},
\qquad
\rho_{s0}=7.224847664314034\times10^{-4}\ \mathrm{m}.
\]

The positive SI multipliers are

\[
C_N=\rho_{s0}^3N_{\mathrm{norm}}\Omega_{ci}
=3.612423832157018\times10^{17}\ \mathrm{s^{-1}},
\]

\[
C_P=\rho_{s0}^3(eT_{\mathrm{norm}}N_{\mathrm{norm}})\Omega_{ci}
=2.893870527993356\ \mathrm{W}.
\]

Thus `Q(Ne,phi)` is multiplied by `C_N`. Both `Q(Pe,phi)` and
`Q(Pi,phi)` are multiplied by `C_P`. The primary internal-energy quantities
already include the explicit factor `3/2` before conversion:

\[
F_{U_s}=\frac32 Q(P_s,\phi),
\qquad
F_{U_s}^{\mathrm{SI}}=C_P F_{U_s}.
\]

The unit converter must never apply the `3/2` factor a second time. The
positive multipliers preserve the signed direction.

## 5. Ensemble semantics

Forecast axes remain

```text
[batch, ensemble_member, future_time, channel, x, y, z]
```

Every 88-cell member is first Fourier-downsampled independently to native 81,
as frozen by the resampling protocol. State inversion, pressure selection,
face reconstruction, nonlinear transport, surface integration, and SI
conversion are then performed independently for every member:

\[
\mathcal F^{(m)}=\mathcal T(x^{(m)}).
\]

Only afterward may ensemble summaries, CRPS, coverage, or rank statistics be
computed. In general,

\[
\mathcal T\!\left(\frac1M\sum_m x^{(m)}\right)
\ne
\frac1M\sum_m\mathcal T(x^{(m)}),
\]

so transport from ensemble-mean fields is forbidden as a replacement.

## 6. Prospective acceptance gates

The implementation is released only if all checks pass unchanged:

1. The geometry and source hashes and embedded revisions match.
2. The crop, offset, topology indices, and native `N_z=81`, `zperiod=5`
   match this protocol.
3. `penalty_mask` is finite in `[0,1]`; strict, fractional, and exterior
   categories are mutually exclusive and exhaustive.
4. The three primary topology masks are pairwise disjoint and partition every
   strict wall-interior operator cell exactly once. Their rerun counts match
   the disclosed structural counts.
5. The separatrix x-low `psi` agrees with `psi_bdry` within `1e-12`; every
   radial `psixy` difference in the crop is positive.
6. The two X-point branch corners agree within `1e-12 m`; the OMP selection is
   unique and returns `y=18`; inner-target major radius is smaller than outer.
7. The exact confined separatrix face has left cell `15`, 16 valid poloidal
   rows, and no wall-penalized adjacent cell.
8. A signed manufactured face flow proves positive `+x` produces positive
   outward confined-separatrix flow and negative input reverses it.
9. A nonuniform-`dy` known answer proves the wedge integral equals a direct
   hand sum; explicit toroidal replication changes it by exactly five.
10. Source-constant reconstruction of `Cs0`, `Omega_ci`, `rho_s0`, `C_N`, and
    `C_P` agrees with the frozen values to relative tolerance `1e-14`.
11. Unit known answers preserve zero, sign, linear scaling, and the single
    explicit `3/2` internal-energy factor.
12. A nonlinear two-member transport case equals the two hand-computed member
    values and differs from transport computed from ensemble-mean fields.
13. All existing tests continue to pass, and `85606` remains untouched.

A failed gate blocks the transport release. It is not relaxed after observing
the result. The failure does not trigger architecture changes or training.
