# TensorNetworksJulia

Compact dense-array tensor-network algorithms in Julia, translated from
Andrew Hallam's MATLAB code built on `ncon`. The package uses
TensorOperations for contractions, KrylovKit for matrix-free solvers, and
FFTW for its dense-grid Vlasov reference routines.

## Setup and tests

Requires Julia 1.10 or later. From this repository's root:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Or, after changing into the repository directory in a Julia session:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using TensorNetworksJulia
```

The package is not currently registered in Julia's General registry. Use the
local project environment above. MATLAB and the original research datasets
are not needed for installation or tests. MATLAB comparisons use saved small
fixtures under `test/fixtures/`.

## Implemented algorithms

- Finite one-/two-site DMRG and projector-splitting TDVP.
- Finite first-, second-, and fourth-order nearest-neighbour TEBD.
- Ising, XYZ, Bose–Hubbard, PXP, custom nearest-neighbour, and exponential-sum
  long-range MPO constructors.
- Finite observables, correlations, reduced density matrices, entanglement,
  compression, and compressed MPO application.
- One-site uniform infinite-MPS VUMPS and TDVP with adaptive bond growth,
  using nearest-neighbour densities, lower-triangular MPOs, or direct
  exponential-sum interactions.
- Infinite observables, correlations, block reduced states, transfer spectra,
  correlation lengths, and exact blocking for small periodic unit cells.
- Dense-grid Vlasov–Poisson reference evolution with phase-space MPS filters,
  including a signed square-root-amplitude formulation.

The infinite code currently uses a one-site uniform representation. Small
periodic cells are supported through exact physical blocking, whose physical
dimension grows with cell length. Native large-cell algorithms, symmetry
sectors, and excitation solvers are not included.

## Finite example

```julia
using LinearAlgebra, Random, TensorNetworksJulia

N = 8
H = xyz_mpo(N; Jz=-1, hx=-0.7)
psi = random_mps(N, 2; maxdim=8, rng=MersenneTwister(1))
ground = dmrg!(psi, H; algorithm=:two_site, nsweeps=8, maxdim=32)
dmrg1_sweep!(psi, H)
tdvp2_step!(psi, H, 0.01; maxdim=32)

X = ComplexF64[0 1; 1 0]
Z = ComplexF64[1 0; 0 -1]
bonds = nearest_neighbor_bond_hamiltonians(N, -0.7X, [Z], [Z]; couplings=-1)
tebd_step!(psi, bonds, 0.01; order=2, maxdim=32)
magnetization = local_expectations(psi, Z)
entropy = entanglement_entropy(psi, N ÷ 2; base=2)
```

Finite sweep/evolution algorithms mutate `psi` and return diagnostics.
`apply_mpo` returns a new state in its `state` field. Real-time evolution
requires a complex-valued MPS; `random_mps` defaults to `ComplexF64`.
Spin model constructors use Pauli matrices with eigenvalues ±1.

## Infinite example

```julia
using Random, TensorNetworksJulia

h = ising_two_site(J=-1, hx=-0.7, interaction=:z)
ground = vumps(h, 1; rng=MersenneTwister(2), max_bond=8, maxiter=200, tol=1e-8)
evolved = infinite_tdvp_step(ground.AL, ground.AR, ground.C, h, 0.01)
Z = ComplexF64[1 0; 0 -1]
magnetization = local_expectation(ground.AL, ground.C, Z)
entropy = entanglement_entropy(ground.C; base=2)
xi = correlation_length(ground.AL)
```

Infinite routines return `AL`, `AR`, `C`, energies, residuals, and solver
diagnostics. Inspect convergence and refine bond dimension and timestep for
research calculations. `TriangularMPO` accepts the supported lower-triangular
Hamiltonian structure; arbitrary uniform MPOs are not supported.

## Layout and conventions

```text
src/
├── Shared/     tensor utilities, factorizations, and local solver scratch
├── Finite/     open-boundary states, MPOs, models, measurements, and solvers
├── Infinite/   uniform states, Hamiltonian backends, and measurements
├── PDE/        phase-space tensorization and dense Vlasov reference routines
└── TensorNetworksJulia.jl
```

The public API is available through `using TensorNetworksJulia`. MPS tensors
use `(left bond, right bond, physical)`; MPO tensors use
`(physical output, physical input, MPO left, MPO right)`. Tests mirror the
source organization. Solver-local scratch reduces repeated allocation while
preserving independently owned Krylov outputs.

## Runnable workflows and documentation

```sh
julia --project=. examples/finite_quench.jl
julia --project=. examples/infinite_quench.jl
julia --project=. examples/staggered_unit_cell.jl
julia --project=. examples/vlasov_scarfinder.jl
julia --project=. validation/accuracy.jl
julia --project=. validation/benchmark.jl
```

Examples generate ignored files under `examples/results/`; validation scripts
write to `validation/results/`. Selected small benchmark reports are included
as numerical baselines. Large simulation data, manuscript files, animations,
and local research search scripts are excluded from the upload.

- [Validation and MATLAB comparisons](docs/VALIDATION.md)
- [Contraction/allocation audit](docs/CONTRACTION_AUDIT.md)
- [Vlasov reference routines](docs/VLASOV.md)
- [MATLAB porting map](docs/PORTING_PLAN.md)
- [Contributing and strict tests](CONTRIBUTING.md)
- [Preparing and uploading a clean repository](docs/PUBLISHING.md)

## License

Copyright © 2026 Andrew Hallam. Released under the [MIT license](LICENSE).
Julia dependencies retain their respective licenses and are installed through
Julia's package manager; their source code is not bundled here.
