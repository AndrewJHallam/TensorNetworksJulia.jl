# MATLAB-to-Julia porting map

This document records which MATLAB routines are treated as canonical for the
minimal Julia implementation.  The original MATLAB tree remains read-only.

## Scope and order

1. **Finite shared layer and DMRG** — implemented and covered by dense tests.
2. **Finite one-site and two-site TDVP** — implemented projector-splitting
   sweeps, checked against exact time evolution on small chains.
3. **Finite TEBD** — implemented reusable one/two-site gates, canonical
   alternating-bond sweeps, and first-, second-, and fourth-order formulas.
4. **Finite models and state utilities** — implemented generic
   nearest-neighbour, XYZ, Bose-Hubbard, PXP, and finite long-range MPOs;
   gauge-independent measurements and reduced states; MPS compression and
   compressed MPO application.
5. **Infinite fixed-bond-dimension layer** — implemented mixed-canonical state
   utilities, transfer fixed points, effective Hamiltonians, VUMPS, and
   uniform TDVP.
6. **Adaptive uniform bond growth** — implemented tangent-complement
   projection, scale-aware SVD rank selection, exact zero-block embedding,
   and adaptive nearest-neighbour/MPO/long-range VUMPS and TDVP drivers.
7. **Infinite triangular MPO layer** — implemented validated MPO structure,
   channel-by-channel projected environments, energy diagnostics, VUMPS, and
   TDVP.
8. **Exponential-sum long-range layer** — implemented direct geometric-tail
   environments, VUMPS, TDVP, an equivalent triangular-MPO constructor, and
   finite open-boundary termination.
9. **MATLAB parity fixtures** — implemented deterministic inputs and outputs
   exported by original finite DMRG/TDVP and infinite canonicalization,
   VUMPS, and TDVP routines. Normal tests consume the saved fixtures.
10. **Research workflows and validation** — implemented runnable quench
   examples, infinite measurements, timestep/bond convergence checks, and
   warmed runtime/allocation benchmarks. See `VALIDATION.md`.
11. **Small periodic unit cells** — implemented exact physical-site/MPO
   blocking for existing VUMPS/TDVP backends, with a staggered-cell example.
12. **Deferred extensions** — native large unit cells, symmetries, constrained and
   biorthogonal variants, excited-state solvers, and research-specific
   excitation/sign-MPO code.

## Tensor conventions

The first port deliberately retains the dense MATLAB layouts:

| object | Julia axes |
| --- | --- |
| finite MPS site | `(left bond, right bond, physical)` |
| finite MPO site | `(physical output, physical input, MPO left, MPO right)` |
| finite environment | `(ket bond, MPO bond, bra bond)` |
| finite two-site centre | `(left bond, right bond, physical 1, physical 2)` |
| uniform MPS tensor | `(left bond, right bond, physical)` |
| uniform centre matrix | `(left bond, right bond)` |
| nearest-neighbour uniform Hamiltonian | `(out 1, out 2, in 1, in 2)` |
| uniform triangular MPO | `(physical output, physical input, MPO left, MPO right)` |
| uniform MPO environment | `(ket bond, MPO channel, bra bond)` |

The physical MPO ordering is stated explicitly because the MATLAB `ncon`
labels can look transposed.  The Julia convention is tested with a complex
`Y` field, where an accidental transpose changes the answer.

## Canonical finite sources

Paths below are relative to `TDVP/mps-miscellany/Finite/Finite TDVP`.

| capability | canonical MATLAB source | Julia destination |
| --- | --- | --- |
| environments | `1_site_scheme/update_left_env.m`, `update_right_env.m` | `src/Finite/MPO.jl` |
| one-site effective action | `1_site_scheme/apply_H_to_A.m` | `src/Finite/DMRG.jl` |
| two-site effective action | `2_site_scheme/apply_HH_to_AA.m` | `src/Finite/DMRG.jl` |
| QR canonicalization | `1_site_scheme/l_orth.m`, `r_orth.m` | `src/Shared/MPSFactorizations.jl` |
| truncated SVD split | `2_site_scheme/l_orth_2site.m`, `r_orth_2site.m` | `src/Shared/MPSFactorizations.jl` |
| one-site DMRG | `1_site_scheme/dmrg_finite.m` | `src/Finite/DMRG.jl` |
| two-site DMRG | `2_site_scheme/dmrg_finite_2site.m` | `src/Finite/DMRG.jl` |
| one-site TDVP | `1_site_scheme/tdvp_finite.m` | `src/Finite/TDVP.jl` |
| two-site TDVP | `2_site_scheme/tdvp_finite_2site.m` | `src/Finite/TDVP.jl` |
| bond effective action | `1_site_scheme/apply_K_to_C.m` | `src/Finite/TDVP.jl` |
| local exponentials | `update_A.m`, `update_C.m`, `update_AA.m` | KrylovKit-backed helpers in `src/Finite/TDVP.jl` |
| two-site gate action | `2_site_scheme/update_AB.m` | `src/Finite/TEBD.jl` |
| canonical TEBD sweeps | `2_site_scheme/tebd_finite_2site.m` | `src/Finite/TEBD.jl` |
| second/fourth-order TEBD | `2_site_scheme/tebd2_finite_2site.m`, `tebd4_finite_2site.m` | `src/Finite/TEBD.jl` |
| model MPOs | `General/xyz_mpo.m`, `BH_mpo.m`, `pxp_mpo.m` | `src/Finite/Models.jl` |
| local/correlation measurements | `General/measure_o.m`, `measure_o2.m` | `src/Finite/Observables.jl` |
| reduced states/entanglement | `General/find_rho.m`, `measure_entanglement.m`, `measure_entspectrum.m` | `src/Finite/Observables.jl` |
| MPO application/compression | `1_site_scheme/applyMPO_finite.m`, `2_site_scheme/applyMPO_finite_zipup.m` | `src/Finite/Compression.jl` |
| construction/model checks | `General/finite_init.m`, `ising_mpo.m`, `ising_mpox.m`, `measure_mpo.m` | `src/Finite/State.jl`, `src/Finite/Models.jl`, and tests |

The Julia SVD API adds a real discarded-weight `cutoff`; `cutoff=0` preserves
the MATLAB max-dimension-only behaviour apart from removal of exact numerical
zeros.

The TEBD port combines the palindromic schedules with the gauge-propagating
sweeps from the canonical MATLAB routine.  It deliberately does not copy the
standalone `tebd2_finite_2site.m` behavior of independently splitting every
disjoint gate, because those truncations are not generally Schmidt-optimal.
The reported discarded weight is the conventional relative squared singular
weight, not the MATLAB routine's unused `1 - norm(Skept)/norm(S)` quantity.

Finite observables contract both norm environments and therefore do not rely
on an undocumented canonical gauge.  The Bose-Hubbard constructor generalizes
the MATLAB occupation cutoff, while XYZ retains its Pauli (not `S=σ/2`)
normalization.  The finite PXP constructor handles `N=1` and `N=2` explicitly;
the original boundary assignment assumes a longer chain.

## Canonical infinite sources

Paths below are relative to `TDVP/mps-miscellany/infiniteChain`.

| capability | canonical MATLAB source | Julia destination |
| --- | --- | --- |
| fixed-dimension VUMPS | `VUMPS/vumps.m` | `src/Infinite/State.jl`, `src/Infinite/NearestNeighbor.jl` |
| fixed-dimension uniform TDVP | `VUMPS/vtdvp.m` | `src/Infinite/NearestNeighbor.jl` |
| mixed-canonical manipulation | `stateManipulations/` | `src/Infinite/State.jl` |
| transfer and environment maps | local helpers in `VUMPS/vumps.m` and `vtdvp.m` | `src/Infinite/State.jl`, `src/Infinite/NearestNeighbor.jl` |
| tangent complements and adaptive growth | `TDVP/leftGaugeTangentBasis.m`, `TDVP/rightGaugeTangentBasis.m`, nested `increase_bond_dim` in `VUMPS/vumps.m` and `vtdvp.m` | `src/Infinite/BondExpansion.jl` |
| triangular-MPO environments | `MPO/find_MPO_fixedpoint.m`, `find_MPO_fixedpoints.m` | `src/Infinite/MPO.jl` |
| MPO VUMPS and TDVP | `VUMPS/vumps_mpo.m`, `TDVP/tdvp_mpo.m` | `src/Infinite/MPO.jl` |
| exponential-sum long range | `VUMPS/vumps_lr.m`, `VUMPS/vtdvp_lr.m` | `src/Infinite/LongRange.jl` |

The adaptive port uses the newer canonical algorithm: singular values rank the
new directions but do not multiply the tangent blocks.  The older standalone
`TDVP/increase_bond_dim.m` inserts `sqrt(S)` and does not preserve the canonical
isometries.  The Julia implementation also uses absolute/relative numerical
rank thresholds instead of exact array-zero comparisons or decimal rounding.

The MPO port treats the supported object explicitly as a lower-triangular
Hamiltonian MPO.  It does not copy the 2025 MATLAB call-signature mismatch
between `find_MPO_fixedpoint.m` and the adaptive/TDVP callers, and it reports
linear-solve residuals rather than ignoring GMRES flags.

The direct long-range port closes a left `O_L` tail with `O_R` and a right
`O_R` tail with `O_L`.  This follows `vumps_lr.m` and corrects the swapped
asymmetric closure in `vtdvp_lr.m`; the difference is invisible when
`O_L == O_R`.  Every decay is required to satisfy `abs(lambda) < 1`.

## Role of `Excitations Work`

`TDVP/Excitations Work` is treated as a regression/example collection rather
than the source of core algorithms.  The most useful later checks are:

- `Worksheet_OF_TCI_states.m`: nearest-neighbour VUMPS workflow;
- `Excitations_Worksheet_TLFIM.m`: VUMPS followed by a uniform-TDVP quench;
- `Diagram_magic_ising_alpha12_6site_exact_reduce.m`: canonical and transfer
  fixed-point residual checks.

Parameter-stamped worksheets, `.asv` files, saved data, plotting scripts, and
copied local transfer functions are not translated one-for-one.

## Validation gates

- Tensor contractions agree with independently constructed dense objects.
- QR/SVD factorizations preserve states and satisfy canonical identities.
- Finite DMRG agrees with exact diagonalization and does not increase energy.
- Finite real-time TDVP conserves norm/energy to integrator accuracy and
  agrees with dense evolution; imaginary time lowers energy.
- Finite TEBD gate tensors and order-1/2/4 product formulas agree with dense
  layer exponentials for odd and even chain lengths; truncation diagnostics
  are checked against a known Schmidt spectrum.
- Generic/XYZ/Bose-Hubbard/PXP MPOs and finite exponential-sum MPOs agree with
  independently assembled dense Hamiltonians, including short-chain edges.
- Local observables, correlations, reduced density matrices, Schmidt values,
  compression, and MPO application agree with dense arbitrary-gauge states.
- VUMPS reports left/right canonical residuals, centre consistency,
  fixed-point residuals, and energy density.
- Adaptive growth preserves the old state and both canonical identities
  exactly, respects the bond cap, and lowers the interacting Ising energy.
- Triangular-MPO channel equations are checked individually, and finite bulk
  contractions agree with independently built dense Hamiltonians.
- Short-range MPO and two-site-density solvers agree up to the expected common
  environment gauge shift.
- Direct exponential-tail energies agree with the algebraically equivalent
  triangular MPO, including complex/asymmetric endpoint tests.
- Uniform TDVP is checked for zero-Hamiltonian identity, small-step
  reversibility, norm/canonical preservation, and stationary-state phase
  evolution.

Dropbox may hydrate files lazily.  Missing files are never classified as
obsolete until their containing core directory has been explicitly opened and
re-scanned.
