# Shared site and year effects in CJS

!!! note "Status: native routes implemented; M0–M6 qualification open"
    The shared site and year likelihood routes have focused numerical and
    derivative checks. Prediction, simulation, and profile-interval methods are
    implemented for supported routes; route-specific recovery and broad variance
    identification remain Phase A qualification gates. Repeated-sample interval
    availability and coverage studies are deferred to Phase B.

Use this tutorial when animals share a site or survey year and you want that
shared variation represented in a first-capture-conditional
Cormack–Jolly–Seber (CJS) model. The executable example fits a shared site
intercept on apparent survival. A compact, nonexecuted snippet then shows the
three other implemented numerical routes: a site intercept on detection, independent site
intercepts on both processes, and a repeated-year vector on detection.

Every fit uses maximum likelihood (ML). The CJS likelihood conditions on each
animal's first observed capture and sums the possible alive/dead paths through
its follow-up. For a shared fit, histories in the same group contribute
together before the shared Gaussian effects are integrated. The history
calculation is exact; integration over shared effects uses a route-specific
Laplace approximation.

## Fit site-level survival variation

The small example has 24 four-occasion histories assigned to six sites. Each
site is represented by four histories. This call fits one shared survival
intercept as a worked example; it reports no speed comparison.

```@example cjs-shared-effects
using DRModels, StatsModels

n, nt = 24, 4
ids = ["animal-$(lpad(string(i), 2, '0'))" for i in 1:n]
sites = repeat(["north", "south", "east", "marsh", "ridge", "valley"], inner=4)
Y = zeros(Int, n, nt)
Y[:, 1] .= 1
for i in 1:n, t in 2:nt
    Y[i, t] = mod(i + 2t, 5) == 0 ? 1 : 0
end
phi_data = (id=repeat(ids, inner=nt-1),
            interval_start=repeat(1:nt-1, n),
            interval_end=repeat(2:nt, n),
            site=repeat(sites, inner=nt-1))
p_data = (id=repeat(ids, inner=nt),
          occasion=repeat(1:nt, n),
          site=repeat(sites, inner=nt))
data = CMRData(Y; ids, occasions=1:nt, units="years", phi_data, p_data)

fit = cjs(data; phi=@formula(phi ~ 1 + (1 | site)),
          p=@formula(p ~ 1), terminal_constraint=(phi=0.82,), maxiters=80)
(route=fit.diagnostics.route,
 loglik=fit.loglik,
 converged=fit.diagnostics.converged)
```

Shared-effect uncertainty is available through profile intervals for eligible
fixed coefficients and constant-SD targets. The returned status must be read
alongside the endpoints:

```@example cjs-shared-effects
site_profile = confint(fit; method=:profile,
                       targets=["phi: (Intercept)", "sd:phi"])
(site_profile.intervals, site_profile.status)
```

`site_phi` lets histories at the same site share apparent-survival variation;
detection has a fixed intercept. The CJS likelihood still conditions on each
animal's first observed capture and sums its possible alive/dead paths. The
history calculation is exact; the shared site effects are integrated jointly
across histories with a Laplace approximation.

## Three other shared-effect routes

Each formula below is a separate model fit. The first two reuse `data` from the
example. For the year-vector route, `occasion` supplies the repeated year label
on detection rows, and a new `CMRData` object carries that column.

```julia
# Site effect on detection
site_detection = cjs(data; phi=@formula(phi ~ 1),
                     p=@formula(p ~ 1 + (1 | site)))

# Independent shared site effects on survival and detection
paired_site = cjs(data; phi=@formula(phi ~ 1 + (1 | site)),
                  p=@formula(p ~ 1 + (1 | site)))

# Shared detection effects for each repeated occasion/year
p_year = merge(data.p_data, (year=data.p_data.occasion,))
year_data = CMRData(data.Y; ids=data.ids, occasions=data.occasions,
                    units=data.units, phi_data=data.phi_data, p_data=p_year)
year_detection = cjs(year_data; phi=@formula(phi ~ 1),
                      p=@formula(p ~ 1 + (1 | year)))
```

In the paired model, the survival and detection effects have independent
Gaussian distributions, while the same histories inform both. In the
year-vector model, histories spanning several years link the shared levels in
one joint fit. A group label must recur across histories to represent shared
variation. If a grouping column is absent from a process table or missing on a
contributing row, the fit stops with an error naming the process and column.
An individual ID is not a shared group: persistent individual effects and
shared-plus-individual models are outside this route.

## What the fit does and does not establish

The reported likelihood is an ML likelihood with a joint Laplace approximation
for the selected shared-effect route. The approximation applies to the
integration over group effects; it does not replace the exact CJS conditioning
or the alive/dead recursion. A converged optimizer is a numerical result for
this dataset, not evidence that a variance is well identified.

Shared prediction requires an explicit `target=:zero_effect` or
`target=:integrated`; integrated predictions are on the response scale. For an
interval of length `Delta`, integrated survival is `E_b[s(b)^Delta]`, not
`[E_b{s(b)}]^Delta`. The implementation also accepts process-specific
predictor-dependent log-SD formulas (`sd_phi` and `sd_p`); each SD predictor
must be constant within its grouping unit. These prediction paths are not yet
qualified across the shared-SD route. In particular, integrated prediction
fails closed with an unrefined-quadrature error when its probability tolerance
is not met at the maximum order. `simulate` returns conditional histories
with fresh shared group effects. `confint(...; method=:profile`) implements
eligible coefficient and supported SD profiles, with a status for each target
and `calibrated=false`. Wald and bootstrap requests remain
`:guarded_unavailable` for shared fits. The R-to-Julia CJS bridge remains
limited to fixed-effects fits; these native Julia methods do not expand its
scope.

The R and Julia parity fixtures cover site survival, site detection, a paired
site effect, and a survival-year vector. The detection-year route shown above
does not yet have a matched cross-language fixture. The tested negative log
likelihoods agree within 1e-8. This checks numerical parity on those fixtures;
it does not establish recovery, interval calibration, or broad variance
identification. Phase A reports target-specific single-fit endpoint capability
and status only; repeated-sample interval availability and coverage/calibration
are deferred to Phase B. Correlated random components, random slopes,
structured effects, shared-plus-individual effects, and shared-effect
reference-selection correction are outside the current contract. Apparent survival cannot
distinguish mortality from permanent emigration. All-zero histories, unknown
entry, recruitment, tag loss, temporary emigration, behavioural trap
dependence, and abundance estimation are outside the CJS contract.

For fixed-effects CJS, see [the first-capture tutorial](cjs.md); for individual
effects, see the [persistent individual-effects tutorial](cjs-individual-effects.md).
For formula syntax, see [model specification](../reference/model-specification.md);
for the current tested scope, see [what is tested](../capabilities.md).
