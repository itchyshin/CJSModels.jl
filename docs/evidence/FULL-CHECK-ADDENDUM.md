# Full R check and disk diagnosis — 24 October 2026

The fresh full check succeeded with **Status: OK**, exit code 0. It includes native installation, package/namespace loading and unloading, code and documentation analysis, compiled-code checks, examples, and tests. Test result: **FAIL 0 | WARN 2 | SKIP 0 | PASS 2280**. This supersedes the earlier disk-limited full-check status.

## Exact checkpoint and invocation

R source commit: `277dfe952ac3cb9445e0f2ad1bf3e46b8c016be8` (CJSModels 0.0.1.9000). Both twin worktrees remain clean. No package source or repository documentation was edited in this follow-up.
Tarball: `CJSModels_0.0.1.9000.tar.gz`, SHA-256 `e7180613eda73c4820d05bb072a745fc4ef2d39c4c7ac89ad8e0be9c7e34d522`. Archive comparison matches current committed files apart from standard build-added DESCRIPTION fields Packaged, Author and Maintainer.

```sh
TMPDIR="$PWD/evidence/fullcheck-20261024/tmp" \
_R_CHECK_CRAN_INCOMING_=false _R_CHECK_FORCE_SUGGESTS_=false \
OPENBLAS_NUM_THREADS=1 MAKEFLAGS=-j2 R_LIBS="$PWD/evidence/R-library" \
R CMD check --no-manual --no-build-vignettes \
  --output=evidence/fullcheck-20261024 CJSModels_0.0.1.9000.tar.gz
```

R 4.6.0, aarch64 macOS Tahoe 26.7.1, Apple clang 21. No --no-install or --no-tests flags. PDF manual and vignette building were omitted; this new package has no vignette sources. No CRAN submission or --as-cran claim.

Evidence: `R-check-full-20261024.log`; `fullcheck-20261024/CJSModels.Rcheck/00check.log`, `00install.out`, and `tests/testthat.Rout`.

## Disk diagnosis

Initial free space: 3.2 GiB; subsequent pre-run probe: 4,315,320 KiB. Final probe: about 8.2 GiB. Earlier failures were transient disk exhaustion. The fresh retry succeeded without any cleanup/deletion in this follow-up.

- Latest failed full-check stage: compiled code OK, then examples could not create `CJSModels-Ex.Rout`. Expected output path: `/Users/z3437171/Documents/Codex/2026-10-23/task-7/evidence/final/CJSModels.Rcheck/CJSModels-Ex.Rout`.
- Earlier attempt: Rd checks could not create `R_TempDir`; a named failure was `/var/folders/7x/ytfpq14s0v18frbm9v_w9f4c0000gq/T//Rtmpmy161z/Rf133516c1adb6`.
- Default TMPDIR, workspace and /tmp all use `/dev/disk3s1` (Data). A workspace temp directory avoids the shared default temp directory but is not a separate disk.
- Mounted Claude Science, Positron and RStudio volumes are read-only application images, so no supported alternative scratch volume was found.
- Fresh authorized local temp root: `evidence/fullcheck-20261024/tmp` (under the task workspace). No private source transfer to another machine and no retry of the denied Totoro transmission.
- No cleanup choice is required now because the complete retry passed.

## Two test warnings

Both arise during the fit at `tests/testthat/test-cmr-methods.R:102`, in the fixed-coefficient prediction fixture. `.cmr_fd_gradient()` at `R/cjs.R:1078` compares central finite differences at h and h/2 and warns when the difference exceeds `1e-5 * max(1, abs(FD_half))`. The warning is reproducible.

| Parameter | AD gradient | FD at h | FD at h/2 | Step-halving difference | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| phi:x (2) | -2.82391e-9 | -5.16762e-5 | -1.28604e-5 | 3.88158e-5 | 1e-5 |
| p:x (5) | -1.48366e-9 | -1.27590e-4 | -3.17849e-5 | 9.58052e-5 | 1e-5 |

The optimizer reports status 0, relative convergence (4), and AD gradient norm 2.82391e-9. The approximately fourfold reduction on step halving is consistent with central-difference truncation error in these predictor directions; that is an inference, not a general diagnostic validation. The fixture then sets explicit coefficients to check literal prediction outputs. Those assertions pass; no warning was suppressed in the package or its tests. The separate diagnostic script captured warnings for inspection only. See `two-warning-diagnostics.log` and `.csv`.

Additional messages: installed RcppEigen headers emit three unused-variable compiler warnings (`count`, `nsuper_et_post`, `nsuper_et`). The dependency check prints unreachable CRAN/Bioconductor index messages; installed dependencies satisfy the check. Neither becomes a formal R CMD check WARNING/NOTE; final Status is OK.

## Coverage limits remain unchanged

The R suite covers data/likelihood contracts, fixed-effects prediction and simulation, narrow fixed-effects Wald/profile/bootstrap behavior, and profile-optimizer helpers. Julia's 864 assertions cover fixed-effects contracts/fits, matrix bridge/derivatives, exact probability fixtures and quadrature moments. One fixed-effects R-to-Julia fit has been checked.

The bounded check does NOT cover extraction-specific end-to-end validation of inherited shared random effects (Laplace), persistent individual random effects (Gaussian quadrature and refinement), individual reference-selection correction, SD predictor formulas, or all associated prediction/simulation/profile/bootstrap combinations. Repeated-sampling recovery, interval coverage/calibration, cross-platform release checks and performance claims remain outside this checkpoint. Shared-effect selection, random slopes/shared-plus-individual combinations, and random-effect or selection R-to-Julia bridges remain unsupported.

No feature work, cleanup, deletion, release, new credentials or permission expansion was performed in this follow-up. The check process has completed.
