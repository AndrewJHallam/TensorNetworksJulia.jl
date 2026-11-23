# Preparing and uploading the repository

The package is MIT licensed. Julia and Python 3.10+ are sufficient for the
test/export workflow; MATLAB is needed only to regenerate reference fixtures.

## Clean export

From the original package working directory:

```sh
python3 scripts/prepare_github.py --check
python3 scripts/prepare_github.py --output ../TensorNetworksJulia-GitHub --archive
```

The exporter uses the explicit selection in `packaging/upload.json`. It
copies source, tests and small fixtures, documentation, examples, selected
validation scripts/reports, and GitHub configuration. A short portable
Vlasov guide replaces the extended local research notes in the upload.
The original long README is preserved locally as
`docs/LOCAL_RESEARCH_README.md` and excluded from the upload.

It excludes the local Manifest, research search directories, generated grids,
checkpoints, animations, manuscript, caches, and editor files. It validates
sizes, symlinks, common credential signatures, and machine-specific paths in
selected text files. Each selected file is limited to 5 MiB and the export
to 10 MiB. The exporter does not scan or copy the large simulation trees.

The destination and optional ZIP must not exist. Existing destinations are
never overwritten. Choose a new output directory to export a later snapshot.
`UPLOAD_CONTENTS.json` records copied paths, sizes, and SHA-256 hashes; it
excludes its own hash. The ZIP contains the selected package files, not Git
metadata, and can be extracted independently.

## Local verification

Inside the clean export:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Normal package tests do not require the original MATLAB tree or research
datasets. `Manifest.toml` is deliberately not committed so the Julia package
can resolve compatible dependencies on supported Julia versions.

## GitHub upload

The prepared folder has a local Git repository on `main`, with the selected
files staged for review. No remote, commit, or push is created automatically.
If starting from the ZIP instead, first run `git init -b main` and `git add .`.

Create an empty GitHub repository, for example `TensorNetworksJulia.jl`,
without adding an extra README or license. Then, inside the prepared folder:

```sh
git diff --cached --stat
git commit -m "Initial Julia tensor-network package"
git remote add origin https://github.com/YOUR-ACCOUNT/TensorNetworksJulia.jl.git
git push -u origin main
```

Replace `YOUR-ACCOUNT` with your GitHub account. Git uses your configured
author identity. GitHub's existing authentication setup handles the push.
The CI workflow runs on push and pull requests; it also supports manual runs.
Wait for the supported-version/platform jobs before tagging a release.

## MATLAB fixture regeneration

The MATLAB tree is optional and stays outside this repository. Point the
exporters at the directory containing `mps-miscellany`:

```matlab
addpath('validation/matlab')
export_fixtures([], '/path/to/TDVP')
export_vlasov_fixtures('/path/to/TDVP')
```

Both exporters check that the requested sources exist. Historical full-run
Vlasov data is optional; the small test fixtures can be regenerated without
that dataset. See `docs/VALIDATION.md` for the scientific comparisons.
