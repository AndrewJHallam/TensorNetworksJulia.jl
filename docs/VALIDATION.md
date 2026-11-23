# Research workflows and validation

The initial algorithms are implemented. This document distinguishes measured
results from workflow limitations and future extensions.

## Runnable examples

Run from the package directory with your installed Julia:

```sh
julia --project=. examples/finite_quench.jl
julia --project=. examples/infinite_quench.jl
julia --project=. examples/staggered_unit_cell.jl
```

The finite example finds a 12-site Ising ground state using two-site DMRG,
refines it with one-site DMRG, then evolves independent copies after a field
quench with TDVP1, TDVP2, and TEBD2. CSVs record energy, norm, magnetization,
entropy, and actual bond dimension. The infinite example uses adaptive VUMPS
and TDVP, recording canonical/centre residuals and connected correlations.
Both examples warn if the ground-state or inner-solver budget is exhausted.

These short runs demonstrate workflows; they are not long-time error bounds.
For research runs, refine timestep and bond cap independently and monitor
discarded weights and inner solver convergence.

## Infinite measurements

`local_expectation(AL,C,O)`, `two_point_correlation(AL,C,O,P,r)`, and
`reduced_density_matrix(AL,C,n)` require consistent left-canonical `AL` and
centre `C`. Overall centre scaling cancels. `schmidt_values(C)` and
`entanglement_entropy(C)` report the bond spectrum and entropy. Measurements
retain complex values; operator indices are `(output,input)`.

Distance zero correlations use `O*P`. Connected correlations subtract the
product of the one-site means. Block density matrices use the usual `kron`
ordering, with the last site varying fastest. They grow exponentially with
block size, guarded by `max_blockdim`.

`transfer_spectrum(AL)` normalizes eigenvalues by the leading value.
`correlation_length(AL)` returns the longest transfer length in original
lattice sites for a one-site tensor. A product state has length zero;
degenerate leading magnitudes yield infinity. This longest length may not
couple to a particular operator. The large-space spectrum uses matrix-free
Arnoldi and raises an error when the requested eigenvalues do not converge.

## MATLAB parity

`validation/matlab/export_fixtures.m` calls the original MATLAB routines and
exports deterministic complex tensors and outputs into `test/fixtures/matlab`.
References come from MATLAB, rather than from the Julia implementation.
`provenance.txt` records the MATLAB version. Array values are column-major,
stored as real/imaginary CSV columns plus shape files. MATLAB's dropped
trailing singleton dimensions are restored when loading rank-four MPOs.

The comparisons cover finite MPOs/environments/effective actions, one- and
two-site DMRG sweeps and TDVP steps, infinite canonicalization, one fixed-bond
VUMPS iteration, and a uniform TDVP step. States are compared up to overall
phase; canonical gauges are compared through physical density matrices and
Schmidt values. Infinite VUMPS tolerances allow for the original MATLAB
eigensolver's default accuracy. This covers representative original routines,
not every old worksheet or research-specific variant.

The original `measure_o` contracts `O[input,output]`. Julia consistently uses
`O[output,input]`. The Y fixture compares MATLAB to Julia with the operator
transposed, exposing rather than concealing that convention difference.

To regenerate with MATLAB:

```matlab
addpath('validation/matlab')
export_fixtures
```

For a separately cloned package, use `export_fixtures([], '/path/to/TDVP')`
or set `TDVP_MATLAB_ROOT` to the directory containing `mps-miscellany`.
The original MATLAB sources are optional and remain outside this repository.

Normal Julia tests use the saved fixtures and do not need a MATLAB license.

## Accuracy and performance

```sh
julia --project=. validation/accuracy.jl
julia --project=. validation/benchmark.jl
```

Accuracy output:

- `validation/results/timestep.csv`: 8-site dense evolution at fixed final
  time 0.16, independently refined timesteps, TDVP1/2 and TEBD2/4. The full
  bond space removes truncation error for this small reference. State error
  is a phase-aligned vector norm, avoiding loss of precision near fidelity 1.
- `finite_bond.csv`: 32-site DMRG at bond caps 8/16/32, local residuals,
  discarded weights, and actual retained bond sizes. Matching results may
  mean the requested caps were not binding at the chosen cutoff.
- `infinite_bond.csv`: transverse Ising at field 1.3, bond caps 4/8/16,
  energy compared with the thermodynamic Ising integral. The reference uses
  65,536 periodic trapezoid points in this gapped regime.

The script fails on failed numerical checks or exhausted ground-state
convergence budgets, after writing diagnostics. `--full` raises the finite
chain to 64 sites and extends the bond caps. A dense Hilbert-space matrix is
constructed only for the eight-site reference.

The benchmark records warmed median runtimes and allocated bytes over three
samples for finite DMRG1/2, TDVP1/2, TEBD2, compressed MPO application, and
nearest-neighbour/MPO/long-range infinite VUMPS/TDVP updates. Mutating
workloads include copying the input. Allocated bytes count cumulative
allocations, not peak resident memory. BLAS uses one thread for repeatability;
`machine.txt` records Julia/BLAS versions and CPU/thread settings. Benchmark
timing does not certify convergence of a random-state solve. Run it without
other numerical workloads for useful comparisons.

The standard workloads were executed on 12 October 2026 with installed Julia
1.13.0 and MATLAB R2026a. The saved results show:

- TEBD2 state error falls by approximately four when halving the timestep;
  TEBD4 falls by approximately sixteen. Full-bond TDVP reproduces the
  eight-site dense reference to about 1e-14.
- Infinite Ising energy errors are approximately 1.4e-6, 4.5e-9, and 3.9e-13
  at bond caps 4, 8, and 16, with all ground-state runs converged.
- The original Julia right-environment contraction formed a chi^4 transfer
  tensor before absorbing the existing environment. At chi=32 one update
  allocated 841 MB. Restoring the original MATLAB ncon order reduces it to
  0.776 MB without changing the mathematical contraction. The analogous
  norm-environment order was also corrected. Allocation regression tests
  prevent reintroducing these intermediates.
- The subsequent package-wide contraction audit removed quartic
  intermediates in infinite right-transfer/source kernels and reduced the
  finite block-density temporary. Finite and infinite-MPO local Krylov solves
  now reuse scratch. See `CONTRACTION_AUDIT.md` for the source-family review.
- At finite bond dimension 32, current warmed DMRG1/2 sweeps take about
  0.60/1.33 seconds and allocate 328/571 MB cumulatively. After only the earlier
  environment fix, they took 0.73/1.84 seconds and allocated 2.29/6.26 GB.
  Before either improvement, they took 6.98/7.98 seconds and allocated
  40.14/44.11 GB. TDVP1/2 now take about 0.39/0.83 seconds and allocate
  268/434 MB. TEBD2 takes about 0.024 seconds and allocates 58 MB. These
  routines perform different work; this is not an algorithm-quality comparison.

`benchmarks.csv` contains current measurements;
`benchmarks_before_contraction_audit.csv` preserves the earlier corrected
baseline, and `benchmarks_before_env_fix.csv` preserves the original problematic
order. `allocations.csv` records the first per-kernel comparison, runnable
with `julia --project=. validation/allocations.jl`.
`contraction_audit.csv` records the later package-wide fixes and scratch reuse,
runnable with `julia --project=. validation/contraction_audit.jl`. Remaining
allocations include owned Krylov outputs, factorizations, environments, and
some infinite matvec scratch; cumulative bytes are not peak memory.

The extended `--full` workloads are supplied but were not executed in this
validation pass.

## Small periodic cells and remaining extensions

`block_mps([A1,A2,...])` contracts a periodic MPS cell into one uniform tensor.
`block_mpo([W1,W2,...])` contracts compatible lower-triangular bulk MPO tensors
into a super-site Hamiltonian. Each site must use the same auxiliary channel
meanings. Existing MPO VUMPS/TDVP and bond growth then work directly, including
site-dependent fields and outgoing couplings.

The staggered-cell example solves a two-site periodic Ising Hamiltonian and
measures each sublattice. Energies are per cell, so divide by cell length for
energy per physical site. Time is unchanged; transfer correlation lengths
are in cells, so multiply by cell length for physical-site units. A centre's
entropy is across cell boundaries. `kron(O,I)` and `kron(I,O)` measure the
first and second sites of a two-site cell.

Blocking is exact but the physical dimension grows as the product of the
site dimensions. A native multi-site implementation is still needed for
large cells: per-site canonical centres, cyclic environments, VUMPS updates,
and TDVP sweeps should avoid this growth. Symmetry sectors and excitation
solvers remain separate extensions; neither is claimed by these examples.
