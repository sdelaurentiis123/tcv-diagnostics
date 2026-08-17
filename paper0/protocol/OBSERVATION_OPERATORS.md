# Observation-operator acceptance ledger

**Status:** active from Phase 0

Paper 0 distinguishes a physically meaningful emulator state from a physically defensible diagnostic observation. A simulated field value at a convenient grid cell is not automatically a sensor signal.

## Terminology

- **State channel:** one field evolved by Hermes and modeled by the emulator.
- **Oracle observation:** direct access to a state component, used only to isolate observability or filter behavior.
- **Proxy observation:** a simplified, monotone, or geometry-inspired mapping that omits important instrument physics.
- **Accepted synthetic diagnostic:** an observation operator with declared geometry, response, cadence, resolution, units, noise, and known-answer tests.
- **Experimental diagnostic:** actual measured TCV data. Paper 0 does not contain experimental diagnostic data.

## Emulator state channels

The current C5 representation contains `Ne`, `Te`, `Ti`, `phi`, and `Vi`. These are candidate physically meaningful state channels because they are output fields from the Hermes simulation. They remain **provisionally accepted** until Phase 1 records:

1. the authoritative Hermes/BOUT variable definition for each field;
2. native normalization and conversion to physical units where possible;
3. sign and coordinate conventions;
4. whether `phi` has a fixed, reproducible gauge;
5. boundary and guard-cell semantics.

No channel will be relabeled as an SI quantity merely because a normalization scalar exists in a raw file.

## Diagnostic candidates

| ID | Candidate | Actual observable | Existing geometry evidence | Current status | Missing before scientific use |
|---|---|---|---|---|---|
| `CTRL-DIRECT` | Direct state samples | Selected standardized state values | Legacy point layouts | **Oracle/control only** | Never describe as hardware. Use only to diagnose forecast covariance and observability. |
| `LP-TARGET` | Target Langmuir probes | Ion-saturation current and floating potential; optionally derived Mach response where supported | Corrected target support places five logical points on each of the two target boundaries | **Candidate, blocked** | Implement `I_sat` and gauge-safe `V_float` response with declared sheath convention, units, collection area/scale treatment, cadence, transfer function, correlated noise, and known-answer tests. Direct `phi` and direct `Te` are not accepted substitutes. |
| `REFL-MID` | Midplane reflectometry | Reflected-wave delay/cutoff response governed by the density profile | Corrected outboard-midplane radial support with 12 logical channels | **Candidate, blocked** | Replace direct `Ne` points with a declared short-pulse reflectometry forward model or a carefully bounded measurement approximation; specify frequencies, resolution, cadence, dropout, and correlated errors. |
| `GPI-MID` | Outboard-midplane GPI | Line-integrated emissivity from neutral density and plasma-dependent collisional-radiative response | Corrected single-port 12-by-10, 5-by-4 cm field of view with finite field-aligned support | **Candidate, blocked** | Add neutral/emissivity model, line of sight, point-spread function, photon/read noise, cadence, and known-answer image tests. Direct density averages are a proxy only. Never replicate the camera around all toroidal planes. |
| `TS` | Thomson scattering | Local `Ne` and `Te` profiles at laser times | TCV diagnostic literature, geometry not yet encoded | **Background candidate** | Encode chord/volume geometry and slow cadence; likely useful for state initialization or low-frequency assimilation, not every turbulence frame. |
| `FIR` | FIR interferometry | Line-integrated density | TCV diagnostic literature, geometry not yet encoded | **Background candidate** | Encode physical chords, integration weights, cadence, and noise. |

## Acceptance gate

A diagnostic configuration may enter the Paper 0 ranking only after all of the following are frozen and tested on 85604:

1. Input state variables and units are declared.
2. Geometry is expressed in authoritative physical coordinates and maps unambiguously to the cropped simulation mesh.
3. The forward response equation is documented, including approximations and nuisance parameters.
4. Spatial and temporal resolution are explicit.
5. Noise, bias, missing-data behavior, and cross-channel correlations are explicit.
6. Gauge-dependent quantities are replaced by gauge-safe observables or use a documented gauge convention.
7. Synthetic known-answer tests pass.
8. Channel count and update cadence are included in the measurement budget.
9. The operator cannot read future truth or unmodeled fields.
10. Figures and tables label it `synthetic` and state whether it is an oracle, proxy, or accepted operator.

## Legacy reproduction exception

Phase 0 will rerun one legacy `layout=iter` ETKF experiment because reproducing an existing forecast and assimilation result is an explicit audit requirement. That layout contains 69 direct-state samples (`54 Ne + 15 phi`) on one toroidal plane. Its result tests code reproducibility only. It is excluded from final diagnostic-ranking claims and does not pass the acceptance gate above.

## Prior geometry findings retained as evidence

- The cropped 64-by-32 model state uses separatrix index `ixsep=16`; raw-grid index 18 must not be inserted directly into the cropped state.
- The corrected target support lies on the two BOUT target boundaries, not along the earlier `x=X-1` leg proxy.
- The physical midplane GPI is a localized single-toroidal-port camera, not five simultaneous regions and not an all-toroidal-plane array.
- TCV can access up to five GPI regions through hardware/configuration changes, but only two detectors operate simultaneously and several views are mutually exclusive.
- The current simulation fields can support simplified synthetic operators, but geometry alone does not make their response physically valid.

These statements are hypotheses and implementation constraints for Paper 0. Primary-source citations and exact operator equations will be frozen before Phase 5.
