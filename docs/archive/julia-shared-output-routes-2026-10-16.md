# Julia CJS shared prediction and simulation output check

Status: verified for the focused shared site-effect route only. This is not a recovery or interval-calibration result.

Candidate: local CMR worktree `codex/cmr-finish-c944-20261016`, HEAD `c94437b64e4e24e95d72375503acc6d84e466019`. The test source `test/test_cmr_shared_laplace.jl` has SHA-256 `3f7c017ec3e0b27b0df20fa4a6ae948dea94acf829e26f8c4c87a64bb36df915`.

On Totoro, Julia 1.12.6, `JULIA_NUM_THREADS=2`, and `OPENBLAS_NUM_THREADS=1`, the focused shared Laplace test passed 816 assertions with no failures or errors. The new public-output section passed all 37 assertions. It compares integrated survival with a separate one-dimensional logistic-normal integral, checks interval survival using the integrated interval probability raised through the conditional unit-time process, and exercises seeded history simulation while preserving first-capture conditioning, masks, endpoints, and frequency counts.

Raw log: `julia-shared-output-routes-2026-10-16.log`, SHA-256 `574a1a94b97a6aeb095ddde13da6e643602bf89b14e023ffa0879e24d5c49622`.

This evidence does not qualify shared detection, paired process effects, all-year vectors, reference selection, individual heterogeneity, recovery, or interval coverage. The full package regression suite is a separate gate.
