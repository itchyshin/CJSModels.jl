# Build the Julia documentation locally

From the package root, prepare a separate documentation environment and build:

```sh
JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=docs docs/make.jl
```

Documenter stays in `docs/Project.toml`, outside the runtime dependencies. Generated pages go to `docs/build/`, which is ignored by Git. Only `docs/src/` is rendered; archive and diagnostic files stay outside the published content. Per-line remote source links are disabled for the standalone bundle; the repository navbar link remains available. The reference check covers exported functions; private helper docstrings are outside the public manual.

The verified static bundle is published from `gh-pages`. No `deploydocs` call or custom workflow is installed. Publication and repository visibility require review and explicit authorization. The build does not establish scientific recovery or interval calibration.
