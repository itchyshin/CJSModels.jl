# First-capture-conditional capture–recapture

!!! note "Status: native CJS routes implemented; M0–M6 qualification open"
    Fixed-effects, shared site/year effects, persistent individual effects,
    individual-effect SD predictors, and reference-selection routes have native
    fitting and output methods. Route-specific recovery qualification remains
    open; repeated-sample interval availability and calibration are deferred to
    Phase B.

DRModels.jl fits a fixed-effects Cormack–Jolly–Seber (CJS) model to encounter
histories conditional on each individual's first capture. This estimates
apparent survival between occasions and detection conditional on being alive.
It does not estimate abundance or the number of unseen animals.
Apparent survival combines mortality and permanent emigration, which this model
cannot separate. The contract also excludes all-zero histories, unknown entry,
recruitment, tag loss, temporary emigration, and behavioural trap dependence.
A history whose first capture is its final observation contributes likelihood
one because there is no post-capture information.

## Fit separate survival and detection models

```@example cjs
using DRModels, StatsModels

Y = [1 0 0 0; 1 0 1 0; 1 1 0 0; 1 1 1 0;
     1 0 0 1; 1 0 1 1; 1 1 0 1; 1 1 1 1]
d = CMRData(Y; ids=1:8, occasions=1:4, units="years",
            frequency=[12, 4, 8, 4, 4, 4, 4, 4])
fit = cjs(d; phi=@formula(phi ~ 1), p=@formula(p ~ 1))
fit.diagnostics.converged
```

`phi` models logit unit-time survival. For an interval of length `Δ`, the
survival probability is `phi_unit^Δ`. `p` models logit detection at a surveyed
occasion, conditional on the animal still being alive. Interval lengths and
their unit label are part of the data contract.

## Check fitted probabilities and simulate histories

Training predictions include keyed interval and occasion rows that contribute
to the fit:

```@example cjs
pred = predict(fit)
first(pred.phi, 3)
first(pred.p, 3)
```

Conditional simulation keeps the observed first-capture times, survey masks,
and administrative endpoints. The first-capture emission is conditioned on,
and death is absorbing. A seed makes a reproducible draw without advancing the
caller's random-number generator.

```@example cjs
simulated = only(simulate(fit; nsim=1, seed=20261006))
refit = cjs(simulated; phi=@formula(phi ~ 1), p=@formula(p ~ 1))
refit.diagnostics.converged
```

## Uncertainty is a separate question

Wald and profile intervals report a status for each coefficient, including
when fit diagnostics guard an interval. Bootstrap requires the original fit to
pass its convergence and gradient checks; otherwise it throws an `ArgumentError`
before returning an interval result. When those checks pass, the result retains
each attempted refit and its status, with target-level availability based on
the usable replicates. Phase A reports target-specific single-fit endpoint
capability and status only. Repeated-sample interval availability and
coverage/calibration are deferred to Phase B; an available interval is not
evidence of nominal calibration.

```@example cjs
ci = confint(fit; method=:wald)
first(ci.intervals)
```

This page works through fixed-effects CJS. Persistent individual effects,
their predictor-dependent SD formulas, and reference-cohort selection
correction are described in the
[individual-effects tutorial](cjs-individual-effects.md); shared site/year
effects are described in the [shared-effects tutorial](cjs-shared-effects.md).
These native routes remain subject to Phase A recovery and single-fit interval
capability gates. Repeated-sample interval availability and calibration belong
to Phase B. Correlated random components, random slopes, structured effects,
shared-plus-individual effects, and shared-effect reference selection remain
outside the current contract; unknown entry is unsupported.

## Matching R worked example

The native R counterpart is the `drmTMB` vignette
`cjs-first-capture.Rmd`. Read the two package examples by scientific task:

| Task | R `drmTMB` vignette | Julia `DRModels.jl` pages |
|---|---|---|
| First-capture conditioning, fixed-effects fit, keyed prediction, conditional simulation, and fixed-effect intervals | “Fit a model”; “Predict and simulate conditionally”; “Inspect fixed-effect intervals” | This page |
| Shared site or year variation | “Allow survival to vary among sites” | [Shared site and year effects](cjs-shared-effects.md) |
| Persistent individual effects and process-specific SD predictors | “Allow persistent differences among animals” | [Persistent individual effects](cjs-individual-effects.md) |
| Reference-cohort selection correction | “Correct for first-capture selection” | [Persistent individual effects](cjs-individual-effects.md#Predictor-dependent-SDs-and-reference-selection) |

Both examples cover the same first-capture-conditional estimands and state
the same exclusions, but they are not a claim of route-wide numerical parity.
The site-survival example uses the same histories, formula, and terminal
constraint in both packages. Other demonstrations use package-specific data
or API details: R labels its selected integrated prediction `"integrated"`,
whereas Julia uses `:reference_selected`; the R individual-effects example
separates survival and detection SD predictors, while the Julia selection
example demonstrates both together. Interval methods also differ by route.
Use each package's status and eligibility output. Phase A does not establish
repeated-sample interval availability or calibrated coverage; those studies
are deferred to Phase B.
