# First-capture-conditional capture–recapture

!!! note "Fixed-effects tutorial; validation incomplete"
    This standalone tutorial covers fixed effects. Other inherited routes are
    experimental; consult [tested scope](status.md) for their unverified combinations.
    Repeated-sample recovery and interval calibration remain deferred.

CJSModels.jl fits a fixed-effects Cormack–Jolly–Seber (CJS) model to encounter
histories conditional on each individual's first capture. This estimates
apparent survival between occasions and detection conditional on being alive.
It does not estimate abundance or the number of unseen animals.
Apparent survival combines mortality and permanent emigration, which this model
cannot separate. The contract also excludes all-zero histories, unknown entry,
recruitment, tag loss, temporary emigration, and behavioural trap dependence.
A history whose first capture is its final observation contributes likelihood
one because there is no post-capture information.

## Fit separate survival and detection models

```julia
using CJSModels, StatsModels

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

```julia
pred = predict(fit)
first(pred.phi, 3)
first(pred.p, 3)
```

Conditional simulation keeps the observed first-capture times, survey masks,
and administrative endpoints. The first-capture emission is conditioned on,
and death is absorbing. A seed makes a reproducible draw without advancing the
caller's random-number generator.

```julia
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
the usable replicates. These checks report target-specific single-fit endpoint
capability and status only. Repeated-sample interval availability and
coverage/calibration remain deferred; an available interval is not
evidence of nominal calibration.

```julia
ci = confint(fit; method=:wald)
first(ci.intervals)
```



For other inherited routes and open checks, see [tested scope](status.md).
