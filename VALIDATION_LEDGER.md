# Development validation ledger — 2026-10-24

| Claim or route | Evidence | Status |
| --- | --- | --- |
| Exact fixed-effects conditional history likelihood | 719 independent probability fixtures in both engines | Numerically checked; includes masks, impossible captures and administrative suffixes |
| Standalone Julia API and numerical derivatives | 864 assertions in Pkg.test on Julia 1.10.12 | Passed development tests |
| Native R TMB build and load | R CMD INSTALL, R 4.6.0, Apple clang 21 | Passed |
| Fixed-effects R-to-Julia bridge | One 44-history fit; coefficients within 1e-5, loglikelihood within 1e-6 | Passed bounded integration check |
| Persistent individual effects, SD predictors, shared effects and reference selection | Existing CJS source preserved; not a new full validation campaign | Experimental; broad extraction-specific route tests remain |
| Parameter recovery and interval calibration | No repeated-sampling campaign in this lane | Deferred |
| Random-effect or selection R-to-Julia bridge | Unsupported | Not admitted |
| Public release, CRAN, Julia General, cross-platform acceptance | No submission, tag or publication performed | Not a release checkpoint |

Numerical fixture agreement does NOT establish parameter identifiability, recovery, confidence-interval coverage, or all-route parity. See docs/READINESS.md for package-check details and exact residuals.
