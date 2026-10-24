# CMR mathematical and data contract, 2026-10-06

Status: frozen implementation contract; no empirical capability claim.

## Target
First-capture-conditional CJS: apparent survival, alive/available then absorbing dead/unavailable. No abundance, unknown entry, recruitment, tag loss, temporary emigration or behavioural trap dependence. Discrete states are summed exactly. ML only.

## Data and indexing
Y and observed are N by T. IDs are explicit unique nonmissing values. Occasions are strictly ordered, units declared, intervals positive. First[i] is first observed capture. End[i] is administrative follow-up, not last sighting. No all-zero histories. Capture marked unobserved is contradictory. Masked NA encounters may be represented internally by zero only with observed=false; this is not covariate imputation.
Detection rows have individual-major ordering: (i-1)*T+t; survival rows: (i-1)*(T-1)+t for t -> t+1. R and Julia indices start at one; convert explicitly at C++ boundary. Predictor tables use explicit id/occasion or id/interval_start/interval_end keys, never implicit shuffled row correspondence. Include all required post-capture predictors through End. Selection adds complete eligible pre-capture rows.
Retain/report final-first individuals with likelihood one. Never truncate at final sighting. Integer frequency multiplies individual marginal log likelihood, after integrating independent individual effects. Shared effects integrate the product jointly. No arbitrary likelihood weights.

## Formula and links
R: cmr_data(), drm_cjs(histories, phi=~1, p=~1, sd_phi=NULL, sd_p=NULL, selection=NULL). Julia: CMRData, cjs(data; phi=@formula(phi~1), p=@formula(p~1), sd_phi=nothing, sd_p=nothing, selection=nothing), CJSFit. Parameter LHSs are labels, not responses. Dedicated formula handling leaves bf grammar unchanged.
Initially logit unit-time survival s, interval survival phi=s^Delta, logit occasion p. Offset is a fixed link-scale contribution. Constant covariates within interval, or explicit segmentation. Store schemas/contrasts/levels/names/offsets/units. No missing predictor repair. Detection never receives Delta power.
Random terms are (1|group), one component/process; group can be site or year, or persistent id. Correlated components, slopes, structured effects and shared-plus-individual combinations are refused. sd_phi/sd_p formulas predict log SD using covariates constant within the unit.

## Exact forward likelihood
Initialize alive=1, dead=0 at first capture. For each later occasion, first transition: alive_new=alive*phi; dead_new=dead+alive*(1-phi). Multiply emission (p,0) for capture, (1-p,1) for noncapture, (1,1) for unobserved. Scale by sum; accumulate logs. Structural zero yields -Inf, never epsilon. Iterate through administrative End. First-capture emission and earlier zeros absent from ordinary CJS.
Known phi=(.8,.6), p2=.3,p3=.4: 100=.6256,101=.1344,110=.1824,111=.0576,1?1=.192; first2:010=.76,011=.24; first3:001=1. Normalize within first-capture cohorts, not across them.

## Continuous effects
Shared effects: integrate group products under properly normalized Gaussian densities; actual CJS Hessian and implicit mode derivatives, not independent-observation Z'WZ. Independent persistent individual effects: integrate one draw across each complete history; uncorrelated one/two-dimensional quadrature with fixed nodes during optimization, refine 25/41/65/97 between fits. Include log Gaussian normalization. Zero SD is an explicit reduced model.

## Reference selection
Eligibility time and explicit alive-and-available assumption are required, with complete survey and predictor history to first capture. The same phi/p coefficients govern pre/post periods. Unknown origins or freely fitted pre-only coefficients unsupported. Define w_i(b)=P(first observed f_i|alive/available at eligibility,b) including survival, surveyed nondetections and first-capture detection. Define H_i(b)=post-capture CJS likelihood.
L_i=integral(w_i H_i g_ref)/integral(w_i g_ref). Both integrals enter objective and derivatives. Final-first numerator equals denominator. No shared effects with reference selection. Averages over reference covariate distributions require that distribution; averaging marked covariates is not automatically a reference average.

## Identification and outputs
Reject unrestricted final time-specific phi/p before optimization; require a user constraint, never silently choose one. Rank tests alone are insufficient. Record objective, optimizer, gradients, curvature, boundaries, inner integration state and all attempts. Distinguish enrolled/informative individual and contributing-occasion counts.
Prediction labels: unit/interval apparent survival, detection conditional alive, zero-effect/conditional/integrated marked/reference targets. Simulation respects conditioning, intervals, masks and persistent effects. Profile/bootstrap preserve all failed attempts and denominators; unsupported intervals unavailable with explanation. AIC uses same conditional/integration target; AICc unsupported.

## Alignment
|Symbol|Data/formula|Generator|Recovery|
|---|---|---|---|
|Y,mask,f,end|history constructor|independent Bernoulli state/observation draws|retained conditioning counts|
|beta_phi|survival design phi|unit s=logistic(X beta+offset+b), phi=s^Delta|process-labelled coef|
|beta_p|occasion design p|detection=logistic(X beta+offset+b)|process-labelled coef|
|b_phi,b_p|one persistent/group draw|uncentred normal draw per unit|SD and unit metadata|
|gamma_phi,gamma_p|sd_phi/sd_p, constant within unit|SD=exp(W gamma)|log-SD coefficients|
|w_i|reference eligibility/prehistory|generate reference cohort forward, retain observed|selection integral diagnostics|

## Acceptance
Exact probabilities abs/rel 1e-10; cross-language aggregate loglik 1e-8; interior AD/step-checked FD gradients 1e-6; quadrature log-integral refinement 1e-8 separately for numerator/denominator. Enumeration and negative controls independent of fitting kernels. Local targeted tests precede broad regression. 10-fit pilots estimate throughput; 500 screening and prespecified 2000 qualification require the approved compute gates. No claims of calibrated inference from one fit.
