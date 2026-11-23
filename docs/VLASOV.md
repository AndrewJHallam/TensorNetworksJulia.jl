# Vlasov–Poisson examples

The PDE routines are dense-grid reference solvers with an MPS filter between
evolution intervals. They do not evolve exclusively in a compressed state.

Run a small two-stream example from the package root:

```sh
julia --project=. examples/vlasov_scarfinder.jl
```

For a shorter interactive run:

```julia
include("examples/vlasov_scarfinder.jl")
result = vlasov_example(levels=4, iterations=2, steps_per_iteration=5)
```

## Phase-space tensorization

An `Nx × Nv` array contains tensor amplitudes, not `Nx × Nv` MPS sites.
`phase_space_layout` specifies how spatial and velocity bits form sites.
Supported orderings are `:xfast`, `:vfast`, `:snake`, `:morton`, and `:quadtree`.
The quadtree representation pairs one x bit and one v bit in each site of
dimension four. The binary representations use sites of dimension two.
`significance=:coarse_to_fine` reverses the bit/site sequence.

```julia
using TensorNetworksJulia
layout = phase_space_layout(64, 64; ordering=:quadtree)
f = [exp(-v^2) * (1 + 0.1cos(x)) for x in range(0, 2pi; length=64),
                                             v in range(-6, 6; length=64)]
p = project_phase_space(f, layout; maxdim=4, target_sum=sum(f))
```

The sequential SVD is a tensor-train approximation with a per-cut rank bound.
It does not globally optimize the approximation. Raw density truncation need
not preserve positivity. Optional mass rescaling preserves the MPS bond bound
but does not restore energy or momentum. Inspect projection errors, negative
mass, velocity-edge values, and refinement with timestep, grid size, and rank.

## Evolution conventions

The equation is `∂t f + v∂x f + (E[f] + D(x,t))∂v f = 0`, with
`∂x E = ρ - mean(ρ)` and zero-mean electric field. The default integrator uses
Fourier translations in a symmetric space/velocity/space split. Driving is
evaluated at the step midpoint. FFTW supplies Fourier transforms for
translations and the Poisson solve.

The default Nyquist convention matches the original MATLAB implementation:
the Nyquist translation frequency is set to zero. This differs from a
continuous translation of a Nyquist oscillation. Resolve the distribution
and check spatial/velocity boundary effects before interpreting recurrence.

The square-root routines evolve a signed real amplitude `g` and reconstruct
`f = g²`. Their MPS filtering therefore guarantees nonnegative reconstructed
density, but is a different approximation from filtering density directly.
See the exported `vlasov_sqrt_scarfinder` and `vlasov_amplitude_step` docstrings.

## Validation

```sh
julia --project=. validation/vlasov_baseline.jl
julia --project=. validation/vlasov_sqrt_benchmark.jl
```

These are longer numerical experiments. They create ignored output under
`validation/results/`. Saved large MATLAB targets are optional: if absent,
the baseline still constructs its initial distribution analytically and runs
the Julia comparisons. The package tests always use the small checked-in
MATLAB fixtures and require no MATLAB installation or external datasets.

The upload includes these portable examples and validation entry points.
Extended parameter searches, manuscript experiments, animations, and their
large datasets remain in the original research workspace. The extended local
Vlasov research notes are preserved there; they are not dependencies of this
repository or its tests.
