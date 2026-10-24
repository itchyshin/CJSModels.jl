# CJS dedicated extraction checkpoint — 2026-10-24

## 1. Goal
Prepare separate CJSModels R and standalone CJSModels.jl development packages for the requested morning checkpoint.

## 2. Implemented
CJS-only source extraction, independent package metadata, native R TMB dispatcher, standalone Julia module, optional fixed-effects bridge, tests, examples and provenance. Both requested GitHub repositories are private, with no Actions workflows.

## 3a. Decisions and Rejected Alternatives
Retained drm_cjs R API and fit class names for compatibility; used CJSModels as both Julia module and R DLL name. Kept R GPL and Julia MIT source separate. No destructive extraction, release, or broad feature additions. Assumption: existing experimental CJS routes remain experimental in the new packages; this checkpoint does not promote their scientific status.

## 4. Files Touched
All tracked files in this new repository are created by this extraction. SOURCE-MANIFEST.json lists copied source paths, pins and hashes. docs/FINAL-SOURCE-SHA256SUMS lists the final package files. Neither DRM source repository was edited.

## 5. Checks Run
See READINESS.md and evidence logs for fresh package checks, 864 standalone Julia assertions, focused R tests and fixed-effects cross-engine agreement.

## 6. Tests of the Tests
The first R test run caught an omitted prediction fixture. The first Julia test run caught an omitted offset helper. Both were restored and tests rerun. Exact likelihood fixtures include impossible histories, no first-capture emission, observation masks and administrative endpoints. No destructive mutation testing was performed.

## 7a. Issue Ledger
Fixed extraction fixture/helper omissions, package header naming, temporary object contamination, and JuliaCall environment selection. No GitHub issues were created. Scientific gates stay in VALIDATION_LEDGER.md.

## 8. Consistency Audit
Checked source hashes after extraction, namespace generation, absence of parent-package runtime dependencies, separate license provenance, private repository visibility, and absence of workflow files. Memory receipt: source AGENTS, hub ownership/private-Actions rules, R package engineer, verification and debugging skills shaped this work. Golden Set: not run; no global process policy was changed. Rose-style self-review only; no independent scientific review is claimed.

## 9. What Did Not Go Smoothly
Ownership acknowledgment wait was resolved by coordinator evidence and destination leases. Shared lease shell helper was slow; compatible registry entries were atomically written after all live leases were checked. Totoro source transfer was denied and was not retried. Local Julia was located and used. Disk exhaustion interrupted an R compile and two fresh full R checks; generated task artifacts were cleaned and affected checks rerun. JuliaCall needed a consistent bridge environment.

## 10. Known Residuals
Experimental inherited random-effect, selection and profile routes need broader focused validation in the new packages. Coverage, calibration, cross-platform checks, CRAN/General submission and public release are deferred. Two finite-difference warnings remain in the R prediction test case. No performance claim.

## 11. Team Learning
Locate Julia under juliaup application-bundle paths when PATH is empty. Select a bridge environment before embedding Julia; loading RCall first and activating a different environment can mix incompatible dependency versions. Keep a standalone package's optional bridge dependencies outside its own Project.toml.

## 12. Cross-Product Coverage
This checkpoint covers R TMB install/build, exact likelihood/data-contract/method tests, standalone Julia fixed-effects/matrix-bridge/derivative tests, Gaussian quadrature moments, and one R-to-Julia fixed-effects fit. It does NOT cover all inherited individual/shared/selection fit and interval combinations, repeated-sampling recovery or coverage, random-effect bridges, Bayesian models, broad capture-recapture families, or released-package readiness.
