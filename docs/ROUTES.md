# CJS route inventory, 24 October 2026

This table describes the extracted twins. It reports no aggregate capability score. Source presence is distinct from a tested route and from scientific qualification.

| Route | R/native TMB | Standalone Julia | Evidence and remaining gate |
| --- | --- | --- | --- |
| Fixed effects, first-capture conditional likelihood | Implemented | Implemented | Exact 719-history probability fixtures; R test assertions and Julia 864-assertion suite at extraction checkpoint |
| Fixed-effect prediction and conditional simulation | Implemented and covered by focused R suite | Implemented; fixed-effects tests retained | Current tutorial smoke checks are recorded in cleanup report; no repeated-sample recovery or coverage claim |
| Fixed-effect Wald/profile/bootstrap | Narrow R tests | Methods inherited; fixed-effect checks only | Eligibility/status outputs matter; no interval calibration claim |
| Shared site/year intercepts, Laplace integration | Inherited experimental implementation | Inherited experimental implementation | Extraction-specific shared-route and output combinations unqualified; parent receipts remain historical |
| Persistent individual effects, quadrature/refinement | Inherited experimental implementation | Inherited experimental implementation | Broad extraction-specific fit/predict/simulate/inference combinations unqualified |
| Individual or shared process-specific SD predictors | Inherited experimental implementation | Inherited experimental implementation | No blanket SD/profile/selection qualification; parent status tables are archive evidence |
| Individual reference-cohort selection | Inherited experimental implementation | Inherited experimental implementation | Extraction-specific selection and associated output combinations unqualified |
| Optional R-to-Julia fixed-effects bridge | JuliaCall through separate RCall environment | Matrix entry point | One earlier 44-history fit agrees at coefficient 1e-5 and loglikelihood 1e-6 tolerances; new sibling-environment smoke recorded separately |
| Random-effect or selection R-to-Julia bridge | Rejected | No admitted bridge | Unsupported |
| Shared-effect reference selection, shared-plus-individual effects, random slopes, correlated or structured effects | Outside contract | Outside contract | Unsupported combinations; do not infer from adjacent implementation |
| Unknown entry, unseen animals/abundance, recruitment, tag loss, temporary emigration, behavioural trap dependence | Outside contract | Outside contract | Not estimated by this first-capture conditional model |
| Recovery, repeated-sample interval availability, coverage/calibration | Deferred | Deferred | No Phase B campaign authorized or run |
| Public release, registry, cross-platform acceptance | Not established | Not established | Local development cleanup only |

Evidence: [readiness](READINESS.md), [validation ledger](../VALIDATION_LEDGER.md), and [full R check addendum](evidence/FULL-CHECK-ADDENDUM.md). The full R check is tied to source checkpoint 277dfe9 and its tarball hash. Earlier broad parent-route evidence does NOT qualify either extracted twin. GPL R implementation remains outside the MIT Julia repository.
