# Persistent individual effects in CJS

!!! note "Status: native routes implemented; M0–M6 qualification open"
    Individual effects, log-SD predictors, and individual reference selection
    have focused numerical checks. Prediction, simulation, and profile intervals
    are implemented for supported routes; route-specific recovery and variance
    identification remain Phase A qualification gates. Repeated-sample interval
    availability and coverage are deferred to Phase B.

Use this route when each animal may have a persistent latent tendency toward
survival or detection. The latent intercept is drawn once per individual and
is shared across that animal's full history. It is not redrawn by occasion.

```@example cjs-individual
using DRModels, StatsModels, StatsAPI

Y = [1 0 1 0; 1 1 0 0; 1 0 0 1; 1 1 1 0;
     1 0 1 1; 1 1 0 1; 1 0 0 0; 1 1 1 1]
data = CMRData(Y; ids=["bird-$i" for i in 1:size(Y, 1)], occasions=1:4,
               units="occasion")

# One persistent survival intercept per individual.
fit = cjs(data; phi=@formula(phi ~ 1 + (1 | id)), p=@formula(p ~ 1),
          terminal_constraint=(phi=0.7,), maxiters=60)
(route=fit.diagnostics.route,
 loglik=fit.loglik,
 quadrature=fit.diagnostics.integration_status,
 order=last(fit.diagnostics.refinement).order)
```

The profile method also accepts the constant survival-effect SD target. Its
status is part of the result. Phase A reports target-specific single-fit
endpoint capability and status only; repeated-sample interval availability
and calibration are deferred to Phase B.

```@example cjs-individual
individual_profile = confint(fit; method=:profile,
                             targets=["phi: (Intercept)", "sd:phi"])
(individual_profile.intervals, individual_profile.status)
```

The same interface supports detection heterogeneity and an independent pair:

```julia
survival_only = cjs(data; phi=@formula(phi ~ 1 + (1 | id)), p=@formula(p ~ 1))
detection_only = cjs(data; phi=@formula(phi ~ 1), p=@formula(p ~ 1 + (1 | id)))
paired = cjs(data; phi=@formula(phi ~ 1 + (1 | id)),
             p=@formula(p ~ 1 + (1 | id)))
```

Each route integrates complete histories with normalized Gaussian quadrature.
The fit checks orders 25, 41, 65, and 97 at the current outer parameter
values. It reports refinement only when every history's log integral changes
by at most `1e-8` and fitted parameters also stabilize; reaching order 97
without that result is reported as unrefined. A converged optimizer does not
establish variance identification or repeated-sample interval calibration;
that validation is deferred to Phase B.

When `start` is omitted, an individual-effect fit compares two deterministic
starts: log SD `log(0.5)` and log SD `0` (SD 1), with fixed coefficients at
zero. It selects the highest-likelihood finite candidate that converged and
passed quadrature refinement, when one exists; otherwise it returns the best
finite candidate with an explicit fallback status. The candidates may stop at
different refined quadrature orders, so the likelihood ranking is a start
diagnostic rather than a recovery or inferential guarantee.
`fit.diagnostics.start_candidates` records each requested and effective
gradient tolerance, gradient norm, quadrature order, curvature status, and
outcome. Positive curvature and
the stricter profile-gradient gate are reported separately; they are not
implied by optimizer convergence. Supplying `start` runs only that start,
which keeps explicit warm starts and profile refits single-start. The second
fit adds computation; in one 1,800-history screen case this policy took about
151 seconds versus 58 seconds for the default start alone. A better likelihood
at a near-zero SD boundary can still have weak curvature and does not establish
identified heterogeneity.

For ordinary individual-effect fits, `predict` requires
`target=:zero_effect` or `target=:integrated`; integrated predictions are
available on the response scale only. `simulate` draws fresh persistent
individual effects. For reference-selection fits, prediction requires
`target=:reference_selected`, and simulation uses bounded rejection to preserve
the fitted first-capture occasion.

`confint(...; method=:profile)` is implemented for ordinary and
reference-selection individual fits, including fitted log-SD coefficients and
supported constant-SD targets. Results report status for each target and set
`calibrated=false`; repeated-sample interval availability and coverage are
unqualified and deferred to Phase B. Wald and bootstrap
requests return `:guarded_unavailable` for individual-effect fits. For
fixed-effect fitting, prediction, and simulation, see the
[first-capture tutorial](cjs.md).

An integrated interval-survival prediction of length `Delta` averages the
conditional probability `s(b)^Delta` over the fitted effect distribution. It
does not raise the marginal unit-time survival to `Delta`. Under reference
selection, the average uses the selection-weighted effect distribution implied
by the supplied complete pre-capture history and survey mask.

## Predictor-dependent SDs and reference selection

The `sd_phi` and `sd_p` formulas predict the log SD of the persistent survival
and detection effects. Their predictors must be complete and constant within
individual. Reference selection is available when animals entered an alive-
and-available reference cohort at known eligibility occasions. It conditions
on each animal's realized first-capture occasion and uses the same survival and
detection coefficients before and after capture. Its selection weight includes
the complete observed survey and predictor history from eligibility through
first capture. It uses

```text
L_i = integral[w_i(b) H_i(b) g_ref(b) db] / integral[w_i(b) g_ref(b) db]
```

where `wᵢ` includes pre-capture survival, surveyed nondetections and detection
at first capture, and `Hᵢ` is the post-capture CJS history. Both integrals are
differentiated and included in the quadrature-refinement check. The scientific
motivation from survivorship selection is discussed by [Sarzo, King and McCrea
(2026)](https://doi.org/10.1093/biomtc/ujag086).

```@example cjs-selection
using DRModels, StatsModels, StatsAPI

Y = [1 0 1 0; 0 1 0 0; 1 1 0 0; 0 0 1 0; 0 0 0 1]
n, T = size(Y)
z = [-1.0, -0.5, 0.0, 0.5, 1.0]
phi_data = (id=repeat(1:n, inner=T-1), interval_start=repeat(1:T-1, n),
            interval_end=repeat(2:T, n), z=repeat(z, inner=T-1))
p_data = (id=repeat(1:n, inner=T), occasion=repeat(1:T, n), z=repeat(z, inner=T))
data = CMRData(Y; ids=1:n, occasions=1:T, units="years", phi_data, p_data)
fit = cjs(data; phi=@formula(phi ~ 1 + (1 | id)), p=@formula(p ~ 1 + (1 | id)),
          sd_phi=@formula(sd_phi ~ z), sd_p=@formula(sd_p ~ z),
          selection=(eligibility=ones(Int, n), assumption=:alive_available),
          maxiters=80)
(loglik=fit.loglik, status=fit.diagnostics.integration_status,
 order=fit.diagnostics.quadrature_order)

selected_profile = confint(fit; method=:profile,
                           targets=["phi: (Intercept)", "sd_phi: (Intercept)"])
(selected_profile.intervals, selected_profile.status)

selected_prediction = StatsAPI.predict(fit; target=:reference_selected)
selected_simulation = first(DRModels.simulate(fit; nsim=1, seed=7302))
(prediction_target=selected_prediction.target,
 first_capture_preserved=selected_simulation.first == data.first,
 simulated_histories=size(selected_simulation.Y, 1))
```

Final-first histories contribute one. The selection likelihood uses each
supplied animal’s reference-cohort covariate profile. Reference-selected
predictions use supplied covariate profiles; population-average predictions
would require a declared covariate distribution. Correlated random components,
random slopes, structured effects, shared-plus-individual effects,
shared-effect selection, and unknown entry are outside the current contract.
Passing numerical gates does not establish variance recovery or repeated-sample
interval coverage; the latter is deferred to Phase B.
See the [fixed-effects CJS tutorial](cjs.md) for that route’s methods and the
[shared-effects tutorial](cjs-shared-effects.md) for site and year effects.
