"Experimental first-capture-conditional Cormack-Jolly-Seber models."
module CJSModels
using LinearAlgebra, Random, Statistics, Printf, Optim, ForwardDiff
using StatsModels: @formula, FormulaTerm, Term, ConstantTerm, FunctionTerm, schema, apply_schema, modelcols, coefnames
import StatsModels, Tables
import StatsAPI: coef, vcov, nobs, fitted, residuals, predict, aic, bic, dof, deviance, dof_residual, StatisticalModel, loglikelihood, stderror, confint
"A known additive predictor term on the linear-predictor scale."
offset(x) = x
export offset
include("quadrature.jl")
include("cmr_data.jl")
include("cjs.jl")
include("cjs_methods.jl")
include("cjs_selection_inference.jl")
export CMRData, CJSFit, cjs, coef, vcov, nobs, predict, aic, dof, loglikelihood, confint, simulate
end
