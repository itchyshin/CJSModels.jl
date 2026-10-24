# Tested scope

The standalone Julia suite passed 864 assertions at its extraction checkpoint on Julia 1.10.12. It covered data contracts, exact conditional-likelihood fixtures, fixed-effects fits, matrix-bridge derivatives and quadrature moments. The numerical Julia source remains unchanged in this documentation cleanup. The current fixed-effects tutorial also executed successfully.

A fresh R/Julia fixed-effects fit agreed within coefficient 1e-5 and loglikelihood 1e-6 tolerances. This is one integration fixture, not broad engine parity.

| Route | Current interpretation |
| --- | --- |
| Fixed effects | Implemented and covered by focused development tests |
| Shared site/year intercepts with Laplace integration | Inherited experimental implementation; broad extraction-specific output and inference combinations unqualified |
| Persistent individual effects and quadrature refinement | Inherited experimental implementation; broad extraction-specific combinations unqualified |
| Process-specific SD predictors and individual reference selection | Inherited experimental implementation; no blanket fit/output/profile qualification |
| Random-effect or selection R-to-Julia bridge | Unsupported |
| Shared-effect selection, shared-plus-individual effects, random slopes and correlated/structured effects | Outside the current contract |
| Recovery and repeated-sample interval availability, coverage/calibration | Deferred; no repeated-sampling campaign in this cleanup |
| Registry release and cross-platform acceptance | Not established |

This first-capture conditional model does not estimate abundance or unseen animals. Unknown entry, recruitment, tag loss, temporary emigration and behavioural trap dependence are outside the contract. Apparent survival cannot separate death from permanent emigration.

## Related Julia work

Julia capture-recapture research code exists for population-size estimation, including [caprecap_survey](https://github.com/NurzhSapargali/caprecap_survey). Its target differs from CJS apparent-survival estimation. A bounded repository and registry-name search did not identify a dedicated Julia CJS fitter; this is not proof that none exists. No first-or-only claim is made here.
