# CJSModels.jl

Experimental first-capture-conditional Cormack–Jolly–Seber models, extracted into a dedicated development package. Apparent survival is conditional on remaining alive and available; detection is conditional on being alive. The first capture conditions the likelihood and is not an ordinary detection contribution.

Requires Julia >=1.10. Run `julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'`. The Julia package runs independently of R, TMB and DRModels.

```julia
using CJSModels, StatsModels
y = [1 0 0 0; 1 0 1 0; 1 1 0 0; 1 1 1 0;
     1 0 0 1; 1 0 1 1; 1 1 0 1; 1 1 1 1]
d = CMRData(y; ids=1:8, occasions=1:4, units="years",
            frequency=[12,4,8,4,4,4,4,4])
f = cjs(d; phi=@formula(phi~1), p=@formula(p~1))
coef(f)
loglikelihood(f)
```

See `docs/READINESS.md` for exact checks and limits. Fixed effects, persistent individual effects with quadrature, and shared effects with Laplace approximation are inherited experimental routes; extraction does not establish route parity, parameter recovery or interval calibration. Shared-effect reference selection remains unsupported. No public release or registry submission has occurred.

`SOURCE-MANIFEST.json` records original source paths, commits, dirty state and file hashes. `inst/COPYRIGHTS` records provenance. R GPL source and Julia MIT source remain separate.
