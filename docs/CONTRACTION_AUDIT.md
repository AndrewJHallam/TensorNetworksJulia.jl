# Contraction and allocation audit — 12 October 2026

The audit covers every TensorOperations contraction in `src/Shared`,
`src/Finite`, and `src/Infinite`. The PDE additions contain no `@tensor`
networks; their dense evolution algorithms were preserved. Intermediate
sizes were checked as well as final sizes: numerical correctness alone does
not rule out an expensive contraction order.

## Findings and changes

| Area | Assessment and action |
| --- | --- |
| Shared MPS factorizations | Two-operand contractions; workspace/output sizes reflect the requested representation. No order change. |
| Finite state/overlap | Norm environment is absorbed first. No quartic bond intermediate. |
| Finite MPO environments | Left order is appropriate. The earlier right-environment fix is retained and guarded by tests. |
| Finite DMRG/TDVP actions | Existing reference contractions have appropriate orders. Repeated solver calls now use packed environments and reusable scratch. |
| Finite observables | Local/correlation contractions absorb their environments first. Norm-right order was already fixed. Block density formation now absorbs both boundaries before introducing the second physical block index. |
| Finite compression/MPO application | Single contractions. Enlarged MPO×MPS bonds and subsequent SVD are part of the algorithm. No order change. |
| Finite TEBD | Single gate contractions. Two-site physical/bond output sizes are intentional. No order change. |
| Infinite state/transfer maps | Identity maps absorb a matrix before joining ket and bra. No order change. |
| Infinite nearest-neighbour | Right Hamiltonian source formed a quartic bond intermediate; it now closes the right bond before inserting the other bra tensor. Left source and effective centre actions have appropriate orders. |
| Infinite MPO | Right operator-transfer block formed a quartic intermediate; it now absorbs the environment first. Left order is appropriate. Local VUMPS/TDVP solves also use reusable scratch. |
| Infinite long-range | Transfer maps and tail/centre closures already absorb matrices in an appropriate order. Temporary allocation remains, but no unnecessary quartic virtual-bond tensor was found. |
| Infinite bond expansion | Two-site Hamiltonian action and tangent projection have appropriate orders. Tangent/expanded tensors are the specified outputs. |
| Infinite observables | Matrix products have no hidden tensor tree. Physical block storage grows exponentially by design and is guarded. |
| Periodic unit-cell blocking | Two-operand contractions. Physical-dimension growth is intrinsic to exact blocking. |

## Reusable Krylov scratch

`src/Shared/MPOActions.jl` supplies private one-site, two-site, and bond action
contexts. A context is built once per local solve, packs fixed environment
matrices and MPO weights, and reuses its intermediate arrays across matvecs.
One-/two-site physical and auxiliary contractions use batched matrix
multiplication, avoiding a BLAS call per MPO coefficient. The public
`apply_heff1`, `apply_heff2`, and `apply_keff` reference APIs are unchanged.

Each operator call allocates its own returned output. KrylovKit may retain or
modify that output, so returning a single reused buffer would corrupt its
basis. Scratch does not alias the input or returned output. Separate solves
own separate contexts; a context is not shared between concurrent calls.
Tests cover ownership, input preservation, coefficient snapshots, complex
non-Hermitian MPOs, unequal dimensions, zero MPOs, and sparse models.

The scratch is quadratic in virtual bond sizes, with the physical and MPO
factors needed for each action. No full effective Hamiltonian or transfer
tensor is assembled. Infinite nearest-neighbour and direct long-range
matvecs retain their allocating implementations with appropriate orders;
the shared contexts currently serve finite and infinite MPO solvers.

## Measurements and reproduction

```sh
julia --project=. validation/contraction_audit.jl
julia --project=. validation/benchmark.jl
julia --project=. validation/accuracy.jl
julia --check-bounds=yes --depwarn=error --project=. test/runtests.jl
```

`validation/results/contraction_audit.csv` compares explicit old expressions
against current kernels at bond dimensions 8/16/32, checking complex outputs
as well as warmed allocation counts. It includes dense and model MPO matvecs.
The old expressions are confined to the validation script.

At bond dimension 32 the measured allocations per call are:

| Kernel | Before | After |
| --- | ---: | ---: |
| Infinite MPO right transfer | 33.6 MB | 150 KB |
| Infinite nearest-neighbour right source | 67.5 MB | 449 KB |
| Four-site block-density contraction | 8.66 MB | 1.06 MB |
| One-site effective-Hamiltonian action | 727 KB | 33 KB |
| Two-site effective-Hamiltonian action | 2.11 MB | 66 KB |
| Bond effective-Hamiltonian action | 264 KB | 16.5 KB |

Workspace construction is excluded from per-call measurements but included
in complete solver timings/allocations in `benchmarks.csv`.
`benchmarks_before_contraction_audit.csv` preserves the baseline after the
earlier finite-environment fix. The still earlier
`benchmarks_before_env_fix.csv` records that original problem separately.
Allocated bytes are cumulative traffic, not peak resident memory. Kernel
timings are short, machine-specific measurements and can be noisy; use the
full sweep measurements to assess workflow improvements.

The warmed 32-site, bond-dimension-32 measurements include context construction
and input copying. Compared with the baseline after the earlier environment
fix:

| Algorithm | Allocations before | Allocations after | Seconds before | Seconds after |
| --- | ---: | ---: | ---: | ---: |
| DMRG1 sweep | 2.29 GB | 328 MB | 0.734 | 0.599 |
| DMRG2 sweep | 6.26 GB | 571 MB | 1.841 | 1.335 |
| TDVP1 step | 1.61 GB | 268 MB | 0.480 | 0.393 |
| TDVP2 step | 4.22 GB | 434 MB | 0.989 | 0.826 |

These are measurements on the recorded machine/BLAS setup, not guarantees for
other Hamiltonians or devices. The reduction is principally in intermediate
array allocation; matrix multiplications, owned solver outputs, and SVD/QR
work still cost time and memory.

Validation includes original MATLAB fixtures and timestep/bond convergence
checks. Allocation bounds guard the three corrected orders and solver
workspaces. Remaining allocations include owned Krylov outputs,
factorizations, environment updates, and nearest-neighbour/long-range
infinite matvecs. The package is not claimed to be allocation-free.
