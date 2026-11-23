# Contributing

Use Julia 1.10 or later. From the repository root:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Keep MPS axes `(left bond, right bond, physical)` and MPO axes
`(physical output, physical input, MPO left, MPO right)`. See the porting map
and contraction audit in `docs/` before changing contractions. Numerical tests
should use independent dense references, physical invariants, or the saved
MATLAB fixtures. Krylov actions must return independently owned outputs.

Run strict checks for numerical changes:

```sh
julia --check-bounds=yes --depwarn=error --project=. test/runtests.jl
```

Examples and numerical experiments write to ignored `results/` directories.
Do not commit simulation grids, binary checkpoints, animations, local Julia
manifests, or manuscript build products. Keep the small fixture CSVs and shape
files: they are required by the tests. Update `packaging/upload.json` and
`.gitignore` deliberately when adding a portable validation entry point.

GitHub Actions tests Julia 1.10 and the latest stable Julia on Linux, plus
the latest stable Julia on macOS and Windows. Hosted CI runs after upload;
local test results do not substitute for that cross-platform check.

Contributions are made under the repository's MIT license.
