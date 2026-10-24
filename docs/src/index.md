# CJSModels.jl

Fit first-capture-conditional Cormack-Jolly-Seber models directly in Julia. Estimate apparent survival and detection from encounter histories, independently of R, TMB and DRModels.

!!! warning "Experimental software"
    Results are provisional. Development checks do not establish parameter recovery, interval coverage or scientific maturity. Apparent survival combines mortality and permanent emigration.

Start with the [fixed-effects tutorial](tutorial.md), then consult the [API reference](reference.md) and [tested scope](status.md).

## Install from GitHub

Julia 1.10 or later is required:

```julia
using Pkg
Pkg.add(url="https://github.com/itchyshin/CJSModels.jl")
using CJSModels
```

The package is not registered in Julia General. No registry release is claimed.

## R companion

```@raw html
<p>Read the <a href="../CJSModels/index.html">R companion documentation</a>.</p>
```

[CJSModels](https://github.com/itchyshin/CJSModels) provides an R interface with native TMB and an optional fixed-effects Julia bridge. The bridge keeps RCall in a separate environment. Public development source is available on GitHub. The software remains experimental.
