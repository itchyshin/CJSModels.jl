# Tested development checkpoint — 24 October 2026

CJSModels (R) and CJSModels.jl (Julia) are separate development packages. Julia is independent of R, TMB and DRModels. R has its own native TMB DLL and an optional fixed-effects JuliaCall bridge. Retained R fit/API names include drm_cjs. No release or scientific-validation claim.

## Source preservation

R source: `/private/tmp/drmtmb-cmr-site-refresh-20261023`, commit `b3c2b6f4e29a26dcadf112c8e0f0106247a13221`.
Julia source: `/Users/z3437171/local-scratch/cmr-finish-20261016/DRM.jl-c944`, commit `c94437b64e4e24e95d72375503acc6d84e466019`, with recorded dirty CJS refinements preserved. All 29 copied source hashes were rechecked against the parent trees after extraction; neither parent tree was edited. See SOURCE-MANIFEST.json and inst/COPYRIGHTS. Shared numerical helper spans are explicitly attributed.

## Fresh evidence

- R 4.6.0, Apple clang 21: native source install/load succeeded. `R CMD build --no-build-vignettes` succeeded. The final tarball SHA-256 is `e7180613eda73c4820d05bb072a745fc4ef2d39c4c7ac89ad8e0be9c7e34d522`.
- Separate final installed-package testthat run: 17 test blocks, 2,280 passed assertions, zero failures, two finite-difference-gradient warnings in the prediction fixture case (parameters 2 and 5).
- `R CMD check --no-install --no-manual --no-build-vignettes --library=<isolated-library>`: Status OK. This command skips installation and tests; the independent installation and final test run above supply those checks. A fresh full check reached compiled-code checks but disk exhaustion prevented example output; it is not reported as passed. An earlier complete full check passed examples/tests with one source-packaging warning, subsequently fixed by header naming and temporary-object cleanup. Full current-source one-command check remains unconfirmed.
- Julia 1.10.12, four Julia threads and one BLAS thread: `Pkg.test()` passed 864 assertions across data contracts, the 719-history exact likelihood fixture, fixed-effects fits, terminal constraints, matrix bridge, derivatives/extreme links, and Gaussian quadrature moments. `testing-Manifest.toml` records the standalone test dependency resolution.
- Local isolated JuliaCall/RCall environment: a 44-history fixed-effects comparison gave R coefficients `(1.99417454, 0.02522045)` and Julia `(1.9941780, 0.0252176)`; maximum difference below 1e-5, loglikelihood difference below 1e-6 (both approximately -86.99874). Bridge environment dependencies remain outside the standalone Julia package.
- Documented R example passed. No workflow files or public/registry release were added. Both GitHub repositories were verified private.

Logs are in docs/evidence. The closeout report is in after-task.md. Workspace evidence also preserves interrupted-check logs and original dirty-source patches.

## Limits

This does NOT cover broad extraction-specific testing of all inherited shared/individual/selection/SD/profile combinations, repeated-sampling recovery, interval calibration, performance, cross-platform release checks, Bayesian models, or broad capture-recapture families. Random-effect and selection Julia bridges remain unsupported. Shared-effect reference selection remains unsupported. Scientific gates are preserved in VALIDATION_LEDGER.md.

The Mac experienced transient disk exhaustion. Only this task's redundant objects and Julia caches were cleaned. A proposed Totoro source transfer was denied by automatic approval review, was not retried, and was unnecessary after local Julia was found. No source left the Mac for Totoro. Private GitHub checkpoints are explicitly authorized by the user.
