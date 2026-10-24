# Tested development checkpoint, 24 October 2026

CJSModels (R) and CJSModels.jl (Julia) are separate development packages. Julia is independent of R, TMB and DRModels. R has its own native TMB DLL and an optional fixed-effects JuliaCall bridge. Retained R fit/API names include drm_cjs. No release or scientific-validation claim.

## Source preservation

R source: `/private/tmp/drmtmb-cmr-site-refresh-20261023`, commit `b3c2b6f4e29a26dcadf112c8e0f0106247a13221`.
Julia source: `/Users/z3437171/local-scratch/cmr-finish-20261016/DRM.jl-c944`, commit `c94437b64e4e24e95d72375503acc6d84e466019`, with recorded dirty CJS refinements preserved. All 29 copied source hashes were rechecked against the parent trees after extraction; neither parent tree was edited. See SOURCE-MANIFEST.json and inst/COPYRIGHTS. Shared numerical helper spans are explicitly attributed.

## Fresh evidence

- R 4.6.0, Apple clang 21: native source install/load succeeded. `R CMD build --no-build-vignettes` succeeded. The final tarball SHA-256 is `e7180613eda73c4820d05bb072a745fc4ef2d39c4c7ac89ad8e0be9c7e34d522`.
- Separate final installed-package testthat run: 17 test blocks, 2,280 passed assertions, zero failures, two finite-difference-gradient warnings in the prediction fixture case (parameters 2 and 5).
- Full `R CMD check --no-manual --no-build-vignettes` passed Status OK, exit 0, at R checkpoint `277dfe952ac3cb9445e0f2ad1bf3e46b8c016be8`. Installation, examples and tests ran; 2,280 assertions passed with two explained finite-difference warnings. This supersedes the earlier disk-limited check. See [full-check addendum](evidence/FULL-CHECK-ADDENDUM.md). PDF manual and vignette builds were omitted. This is development evidence, not a CRAN release check.
- Julia 1.10.12, four Julia threads and one BLAS thread: `Pkg.test()` passed 864 assertions across data contracts, the 719-history exact likelihood fixture, fixed-effects fits, terminal constraints, matrix bridge, derivatives/extreme links, and Gaussian quadrature moments. `testing-Manifest.toml` records the standalone test dependency resolution.
- Local isolated JuliaCall/RCall environment: a 44-history fixed-effects comparison gave R coefficients `(1.99417454, 0.02522045)` and Julia `(1.9941780, 0.0252176)`; maximum difference below 1e-5, loglikelihood difference below 1e-6 (both approximately -86.99874). Bridge environment dependencies remain outside the standalone Julia package.
- Documented R example passed. No workflow files or public/registry release were added. At the extraction checkpoint, both GitHub repositories were verified private; the later public-source approval is recorded below.

Logs are in docs/evidence. The closeout report is in after-task.md. Workspace evidence also preserves interrupted-check logs and original dirty-source patches.

## Limits

This does NOT cover broad extraction-specific testing of all inherited shared/individual/selection/SD/profile combinations, repeated-sampling recovery, interval calibration, performance, cross-platform release checks, Bayesian models, or broad capture-recapture families. Random-effect and selection Julia bridges remain unsupported. Shared-effect reference selection remains unsupported. Scientific gates are preserved in VALIDATION_LEDGER.md.

The Mac experienced transient disk exhaustion. Only this task's redundant objects and Julia caches were cleaned. A proposed Totoro source transfer was denied by automatic approval review, was not retried, and was unnecessary after local Julia was found. No source left the Mac for Totoro. The earlier private GitHub checkpoints are historical. The local cleanup stage initially had no commit/push authority; Shinichi explicitly approved public source, commits/pushes and both Pages sites later on 24 October 2026.

## Dropbox reconciliation, 24 October 2026

Canonical R root: `/Users/z3437171/Dropbox/Github Local/CJSModels`. Canonical Julia root: `/Users/z3437171/Dropbox/Github Local/CJSModels.jl`. The sibling `CJSModels-project` folder coordinates both. The earlier `Documents/Codex/2026-10-23/task-7` workspace remains an evidence archive.

The full-check receipt above belongs to the unchanged extraction checkpoint and its exact tarball, not automatically to later edits. [Route inventory](ROUTES.md) separates inherited implementation, tested fixed-effects behavior and deferred scientific qualification. [Import provenance](IMPORT-PROVENANCE.json) records the added documentation and evidence. Historical source manifests and reports remain unchanged.

The later local cleanup also passed a fresh full R check, Status OK, with 2,280 assertions and two finite-difference warnings. Both adapted fixed-effects tutorials ran, and a fresh bridge comparison loaded the Dropbox Julia source and met the earlier tolerances. Exact tarball identity and logs: [cleanup source receipt](evidence/cleanup-20261024/source.json). Julia source did not change, so its earlier full-suite result was retained without rerunning that suite.

The later website-preparation R archive also passed the full check, Status OK, with the same assertion/warning totals. [Final source receipt](evidence/siteprep-20261024/source.json) records its tarball hash and metadata scope. Local pkgdown and Documenter previews are prepared separately; their planned URLs do not establish public deployment. See [local build instructions](BUILD.md). No commits or pushes were made during that local-check stage.

## Publication approval, 24 October 2026

Shinichi approved making both existing repositories public, committing/pushing the reviewed changes and publishing both Pages sites. The original extraction commits remain ancestors. Static output is published from `gh-pages`; there is no new credential or registry release. Check the GitHub Pages deployment receipt for live status rather than inferring it from these source files.
