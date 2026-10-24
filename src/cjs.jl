# Native first-capture-conditional CJS. No ordinary-family likelihood reused.
using Optim, ForwardDiff
using LinearAlgebra: rank, qr, ColumnNorm, Symmetric, eigvals, inv, norm

_cjs_logadd(a,b) = a == -Inf ? b : b == -Inf ? a : max(a,b)+log1p(exp(-abs(a-b)))
_cjs_logistic_log(x) = x>=0 ? -log1p(exp(-x)) : x-log1p(exp(x))
_cjs_log1mexp(x) = x == 0 ? oftype(x,-Inf) : x < -log(2) ? log1p(-exp(x)) : log(-expm1(x))

function _cjs_history_loglik_logs(y, observed, first, last, logphi, logp, log1mphi=_cjs_log1mexp.(logphi), log1mp=_cjs_log1mexp.(logp))
    alive=zero(eltype(logphi)); dead=oftype(alive,-Inf)
    for t in first+1:last
        dead=_cjs_logadd(dead,alive+log1mphi[t-1])
        alive+=logphi[t-1]
        if observed[t]
            if y[t]==1
                alive+=logp[t]; dead=oftype(dead,-Inf)
            else
                alive+=log1mp[t]
            end
        end
    end
    return _cjs_logadd(alive,dead)
end

"Internal exact conditional history log likelihood, including administrative suffix."
function _cjs_history_loglik(y, observed, first, last, phi, p)
    length(y)==length(observed)==length(p)==length(phi)+1 || throw(ArgumentError("CJS probability/history dimensions differ"))
    all(x->isfinite(x)&&0<=x<=1,phi) && all(x->isfinite(x)&&0<=x<=1,p) || throw(ArgumentError("CJS probabilities must lie in [0,1]"))
    return _cjs_history_loglik_logs(y,observed,first,last,log.(phi),log.(p))
end

function _cjs_loglik(data::CMRData,phi::AbstractMatrix,p::AbstractMatrix)
    N,T=size(data.Y)
    size(phi)==(N,T-1) && size(p)==(N,T) || throw(ArgumentError("CJS probability matrices must be N by T-1 and N by T"))
    value=zero(promote_type(eltype(phi),eltype(p)))
    for i=1:N
        value+=data.frequency[i]*_cjs_history_loglik(view(data.Y,i,:),view(data.observed,i,:),data.first[i],data.end[i],view(phi,i,:),view(p,i,:))
    end
    return value
end

function _cjs_validate_term(t)
    if t isa FunctionTerm
        (t.f === (|) || nameof(t.f) in (:phylo,:spatial,:relmat,:animal,:temporal,:mi,:meta_V,:sd)) &&
            throw(ArgumentError("cjs currently supports fixed effects only; random, structured and missing-predictor terms are unsupported"))
        foreach(_cjs_validate_term,t.args)
    elseif t isa Tuple
        foreach(_cjs_validate_term,t)
    elseif hasproperty(t,:terms)
        foreach(_cjs_validate_term,getproperty(t,:terms))
    end
end

function _cjs_design(formula,table,rows,label;contrasts=Dict{Symbol,Any}())
    formula isa FormulaTerm && formula.lhs isa Term && formula.lhs.sym==label || throw(ArgumentError("cjs $label formula must have the label $label on its left side"))
    _cjs_validate_term(formula.rhs)
    terms=formula.rhs isa Tuple ? collect(formula.rhs) : Any[formula.rhs]
    fixed=Any[]; offsets=Any[]
    for term in terms
        if term isa FunctionTerm && nameof(term.f)==:offset
            length(term.args)==1 || throw(ArgumentError("cjs offset takes one numeric predictor"))
            push!(offsets,term.args[1])
        else
            push!(fixed,term)
        end
    end
    rhs=isempty(fixed) ? ConstantTerm(1) : length(fixed)==1 ? only(fixed) : Tuple(fixed)
    # Only the RHS participates in schema construction; labels need no response.
    use=isempty(rows) ? collect(eachindex(table.id)) : rows
    # Optional noncontributing keys can give a column a Missing union type.
    # Narrow COMPLETE selected columns before schema inference so a numeric
    # predictor remains continuous; no missing value is removed or repaired.
    selected=[v[use] for v in values(table)]
    sub=NamedTuple{keys(table)}(Tuple(any(ismissing,v) ? v : collect(skipmissing(v)) for v in selected))
    variables=StatsModels.termvars(rhs)
    for name in variables
        haskey(sub,name) || throw(ArgumentError("cjs $label predictor $name is absent"))
        any(ismissing,getproperty(sub,name)) && throw(ArgumentError("cjs $label predictor $name is missing on required post-capture rows"))
    end
    sch=schema(rhs,sub,Dict{Symbol,Any}(contrasts))
    applied=apply_schema(FormulaTerm(ConstantTerm(0),rhs),sch,StatisticalModel).rhs
    raw=modelcols(applied,sub)
    cols=raw isa Tuple ? hcat(raw...) : raw
    X=cols isa AbstractMatrix ? Matrix{Float64}(cols) : reshape(Float64.(collect(cols)),:,1)
    all(isfinite,X) || throw(ArgumentError("cjs $label predictors must produce finite model columns"))
    off=zeros(Float64,length(use))
    for term in offsets
        term isa Term || throw(ArgumentError("cjs offset must name a numeric link-scale column"))
        haskey(sub,term.sym) || throw(ArgumentError("cjs offset column $(term.sym) is absent"))
        v=getproperty(sub,term.sym)
        any(ismissing,v) && throw(ArgumentError("cjs offset cannot be missing"))
        off .+= Float64.(v)
    end
    all(isfinite,off) || throw(ArgumentError("cjs offsets must be finite"))
    isempty(rows) && (X=X[1:0,:];off=off[1:0])
    names=String.(vec(collect(coefnames(applied))))
    return (formula=formula,rhs=rhs,schema=sch,applied=applied,names=names,X=X,offset=off,rows=rows,columns=collect(1:size(X,2)),offset_terms=offsets,contrasts=contrasts,training_rows=rows)
end

function _cjs_constraint(constraint)
    constraint===nothing && return nothing
    constraint isa NamedTuple && length(constraint)==1 && only(keys(constraint)) in (:phi,:p) || throw(ArgumentError("terminal_constraint must be exactly (phi=value,) or (p=value,)"))
    v=only(values(constraint))
    v isa Real && isfinite(v) && 0<v<1 || throw(ArgumentError("Terminal constraint must be a finite, strictly interior probability"))
    return constraint
end

function _cjs_apply_design_rows(des, table, rows, label)
    isempty(rows) && return (X=zeros(0,length(des.names)),offset=Float64[])
    cols=Tables.columntable(table)
    for name in StatsModels.termvars(des.rhs)
        haskey(cols,name) || throw(ArgumentError("cjs $label predictor $name is absent in required prehistory rows"))
        any(ismissing, getproperty(cols,name)[rows]) && throw(ArgumentError("cjs $label predictor $name is missing on required prehistory rows"))
    end
    sub=NamedTuple{keys(cols)}(Tuple(getproperty(cols,k)[rows] for k in keys(cols)))
    raw=try
        modelcols(des.applied,sub)
    catch err
        err isa InterruptException && rethrow()
        throw(ArgumentError("cjs $label prehistory predictors do not match the fitted post-capture schema: $(sprint(showerror,err))"))
    end
    allcols=raw isa Tuple ? hcat(raw...) : raw
    allcols=allcols isa AbstractMatrix ? Matrix{Float64}(allcols) : reshape(Float64.(collect(allcols)),:,1)
    X=allcols[:,des.columns]
    off=zeros(Float64,length(rows))
    for term in des.offset_terms
        term isa Term && haskey(cols,term.sym) || throw(ArgumentError("cjs $label prehistory offset column is absent"))
        v=getproperty(cols,term.sym)[rows]
        any(ismissing,v) && throw(ArgumentError("cjs $label prehistory offset cannot be missing"))
        off .+= Float64.(v)
    end
    all(isfinite,X) && all(isfinite,off) || throw(ArgumentError("cjs $label prehistory design must be finite"))
    return (X=X,offset=off)
end

function _cjs_selection_prepare(prep, selection)
    selection isa NamedTuple && haskey(selection,:eligibility) || throw(ArgumentError("selection must be (eligibility=occasion_indices, assumption=:alive_available)"))
    get(selection,:assumption,nothing)==:alive_available || throw(ArgumentError("selection requires assumption=:alive_available at each eligibility occasion"))
    individual = hasproperty(prep, :individual) ? prep.individual : nothing
    no_random_effects = individual === nothing || (individual.phi === nothing && individual.p === nothing)
    (no_random_effects || _cjs_is_individual(prep)) ||
        throw(ArgumentError("selection requires random effects grouped by id; shared-effect normalization is unsupported"))
    # Post-capture grouping alone cannot establish a persistent individual
    # draw over eligibility prehistory. The selection route requires explicit id.
    for label in (:phi,:p)
        effect=individual === nothing ? nothing : getproperty(individual,label)
        effect===nothing || effect.group==:id ||
            throw(ArgumentError("selection requires $label random effects grouped by id; shared-effect normalization is unsupported"))
    end
    d=prep.data;N,T=size(d.Y);raw_eligible=collect(selection.eligibility)
    length(raw_eligible)==N || throw(ArgumentError("selection eligibility must give one valid occasion index at or before each first capture"))
    all(x->x isa Real && !(x isa Bool) && isfinite(x) && isinteger(x),raw_eligible) ||
        throw(ArgumentError("selection eligibility values must be finite whole-number occasion indices"))
    eligible=Int.(raw_eligible)
    length(eligible)==N && all(1 .<= eligible .<= d.first) || throw(ArgumentError("selection eligibility must give one valid occasion index at or before each first capture"))
    phi_rows=[(i-1)*(T-1)+t for i in 1:N for t in eligible[i]:d.first[i]-1]
    p_rows=[(i-1)*T+t for i in 1:N for t in eligible[i]:d.first[i] if d.observed[i,t]]
    ph=_cjs_apply_design_rows(prep.phi_design,d.phi_data,phi_rows,:phi)
    pd=_cjs_apply_design_rows(prep.p_design,d.p_data,p_rows,:p)
    phi_by_id=[Int[(i-1)*(T-1)+t for t in eligible[i]:d.first[i]-1] for i in 1:N]
    p_by_id=[Int[(i-1)*T+t for t in eligible[i]:d.first[i] if d.observed[i,t]] for i in 1:N]
    phi_X_by_id=[begin
        a=sum(length.(phi_by_id[1:i-1]))+1;b=a+length(phi_by_id[i])-1
        ph.X[a:b,:]
    end for i in 1:N]
    phi_off_by_id=[begin
        a=sum(length.(phi_by_id[1:i-1]))+1;b=a+length(phi_by_id[i])-1
        ph.offset[a:b]
    end for i in 1:N]
    p_X_by_id=[begin
        a=sum(length.(p_by_id[1:i-1]))+1;b=a+length(p_by_id[i])-1
        pd.X[a:b,:]
    end for i in 1:N]
    p_off_by_id=[begin
        a=sum(length.(p_by_id[1:i-1]))+1;b=a+length(p_by_id[i])-1
        pd.offset[a:b]
    end for i in 1:N]
    return merge(prep,(selection=(eligibility=eligible,phi_rows=phi_rows,p_rows=p_rows,
        phi_X=ph.X,phi_offset=ph.offset,p_X=pd.X,p_offset=pd.offset,
        phi_by_id=phi_by_id,p_by_id=p_by_id,phi_X_by_id=phi_X_by_id,
        phi_offset_by_id=phi_off_by_id,p_X_by_id=p_X_by_id,p_offset_by_id=p_off_by_id),))
end

function _cjs_prepare(data,phi,p;terminal_constraint=nothing,contrasts=Dict{Symbol,Any}())
    if (phi isa FormulaTerm && _cjs_has_shared(phi.rhs)) || (p isa FormulaTerm && _cjs_has_shared(p.rhs))
        return _cjs_prepare_random(data,phi,p;terminal_constraint,contrasts)
    end
    N,T=size(data.Y); last=T
    srows=[(i-1)*(T-1)+t for i=1:N for t=data.first[i]:data.end[i]-1]
    # A survival interval contributes only when a later surveyed occasion can
    # distinguish survival from death. Keep the full positional rows for the
    # forward kernel and for selection preparation, but rank-check this mask.
    sinformative=Bool[data.end[i]>t && any(view(data.observed,i,t+1:data.end[i]))
                      for i=1:N for t=data.first[i]:data.end[i]-1]
    prows=[(i-1)*T+t for i=1:N for t=data.first[i]+1:data.end[i]]
    sinformative_rows=srows[sinformative]
    sd_active=_cjs_design(phi,data.phi_data,sinformative_rows,:phi;contrasts=contrasts)
    # Keep positional rows for the forward kernel, but do not require predictors
    # or estimate design columns on survival intervals with no later survey.
    sX=zeros(length(srows),size(sd_active.X,2)); soff=zeros(length(srows))
    slocation=Dict(j=>k for (k,j) in enumerate(srows))
    for (k,j) in enumerate(sinformative_rows)
        sX[slocation[j],:]=sd_active.X[k,:]
        soff[slocation[j]]=sd_active.offset[k]
    end
    sd=merge(sd_active,(X=sX,offset=soff,rows=srows,training_rows=sinformative_rows,
                        informative=sinformative))
    surveyed_rows=[j for j in prows if data.observed[cld(j,T),mod1(j,T)]]
    pd=_cjs_design(p,data.p_data,surveyed_rows,:p;contrasts=contrasts)
    # Preserve individual-major post-capture positions for the state kernel,
    # while fitting schemas only on surveyed emissions. Omitted emissions use
    # neutral design/offset rows and never require detection predictors.
    pX=zeros(length(prows),size(pd.X,2)); poff=zeros(length(prows))
    location=Dict(j=>k for (k,j) in enumerate(prows))
    for (k,j) in enumerate(surveyed_rows)
        pX[location[j],:]=pd.X[k,:]; poff[location[j]]=pd.offset[k]
    end
    pactive=Bool[data.observed[cld(j,T),mod1(j,T)] for j in prows]
    pd=merge(pd,(X=pX,offset=poff,rows=prows,training_rows=surveyed_rows,
                 informative=pactive))
    st=Bool[mod1(j,T-1)==last-1 for j in srows]; pt=Bool[mod1(j,T)==last && data.observed[cld(j,T),mod1(j,T)] for j in prows]
    constraint=_cjs_constraint(terminal_constraint)
    # Structural terminal-product guard: can BOTH designs vary only their final
    # occasion? This checks the terminal direction, not merely matrix rank.
    can_terminal(X,v,active)=!isempty(v) && any(v[active]) && rank(hcat(X[active,:],Float64.(v[active])))==rank(X[active,:])
    if constraint===nothing && can_terminal(sd.X,st,sinformative) && can_terminal(pd.X,pt,pactive)
        throw(ArgumentError("Unrestricted terminal survival and detection are confounded; supply terminal_constraint=(phi=value,) (unit-time survival) or (p=value,)"))
    end
    if constraint!==nothing
        label=only(keys(constraint)); des=label==:phi ? sd : pd; terminal=label==:phi ? st : pt
        active=label==:phi ? sinformative : pactive
        Xfree=des.X[active .& .!terminal,:]
        # Remove constrained parameter directions instead of retaining flat
        # coefficients. A pivoted independent column subset preserves the
        # complete predictor span on every unconstrained row.
        r=rank(Xfree)
        keep=r==0 ? Int[] : sort(Vector(qr(Xfree,ColumnNorm()).p)[1:r])
        des=merge(des,(X=des.X[:,keep],names=des.names[keep],columns=keep,))
        label==:phi ? (sd=des) : (pd=des)
    end
    for (des,active) in ((sd,sinformative),(pd,pactive))
        isempty(des.rows) || rank(des.X[active,:])==size(des.X,2) || throw(ArgumentError("cjs design is rank deficient on contributing post-capture rows"))
    end
    return (data=data,phi_design=sd,p_design=pd,nphi=size(sd.X,2),np=size(pd.X,2),terminal_constraint=constraint,last=last,phi_terminal=st,p_terminal=pt)
end

function _cjs_process_logs(beta,prep;complements=false,random_phi=nothing,random_p=nothing)
    sd,pd=prep.phi_design,prep.p_design
    eta_s=sd.X*view(beta,1:prep.nphi)+sd.offset
    eta_p=pd.X*view(beta,prep.nphi+1:prep.nphi+prep.np)+pd.offset
    random_phi===nothing || (eta_s=eta_s+random_phi)
    random_p===nothing || (eta_p=eta_p+random_p)
    ls=_cjs_logistic_log.(eta_s); lp=_cjs_logistic_log.(eta_p)
    lcs=_cjs_logistic_log.(-eta_s); lcp=_cjs_logistic_log.(-eta_p)
    c=prep.terminal_constraint
    if c!==nothing
        label=only(keys(c)); v=log(only(values(c)))
        if label==:phi
            fixed_log=zero(eltype(eta_s))+v
            fixed_complement=zero(eltype(eta_s))+log1p(-only(values(c)))
            ls=ifelse.(prep.phi_terminal,fixed_log,ls); lcs=ifelse.(prep.phi_terminal,fixed_complement,lcs)
        else
            fixed_log=zero(eltype(eta_p))+v
            fixed_complement=zero(eltype(eta_p))+log1p(-only(values(c)))
            lp=ifelse.(prep.p_terminal,fixed_log,lp); lcp=ifelse.(prep.p_terminal,fixed_complement,lcp)
        end
    end
    return complements ? (ls,lcs,lp,lcp) : (ls,lp)
end

function _cjs_objective(beta,prep)
    _cjs_is_individual(prep) && return _cjs_individual_objective(beta,prep)
    _cjs_is_shared(prep) && return _cjs_shared_laplace(beta,prep)
    return _cjs_fixed_objective(beta,prep)
end

function _cjs_fixed_objective(beta,prep;random_phi=nothing,random_p=nothing)
    length(beta)==prep.nphi+prep.np || throw(ArgumentError("CJS coefficient dimension mismatch"))
    ls,lcs,lp,lcp=_cjs_process_logs(beta,prep;complements=true,random_phi,random_p)
    d=prep.data; N,T=size(d.Y)
    value_type=promote_type(eltype(beta),
        random_phi===nothing ? eltype(beta) : eltype(random_phi),
        random_p===nothing ? eltype(beta) : eltype(random_p))
    total=zero(value_type); s=1; p=1
    # Reuse invocation-local AD-typed history buffers. Keeping them local makes
    # nested ForwardDiff calls reentrant while avoiding four fresh vectors per
    # individual on every conditional objective evaluation.
    logphi=zeros(value_type,T-1); logp=zeros(value_type,T)
    log1mphi=fill(oftype(total,-Inf),T-1); log1mp=fill(oftype(total,-Inf),T)
    for i=1:N
        k=d.end[i]-d.first[i]
        # Layout is individual-major. Fill untouched pre-capture entries only
        # to reuse the exact history kernel; that kernel never reads them.
        fill!(logphi,zero(total)); fill!(logp,zero(total))
        fill!(log1mphi,oftype(total,-Inf)); fill!(log1mp,oftype(total,-Inf))
        for j=0:k-1
            t=d.first[i]+j
            logphi[t]=ls[s+j]*d.intervals[t]
            logp[t+1]=lp[p+j]
            log1mp[t+1]=lcp[p+j]
            log1mphi[t]=ls[s+j]==0 ? log(d.intervals[t])+lcs[s+j] : _cjs_log1mexp(logphi[t])
        end
        total+=d.frequency[i]*_cjs_history_loglik_logs(view(d.Y,i,:),view(d.observed,i,:),d.first[i],d.end[i],logphi,logp,log1mphi,log1mp)
        s+=k; p+=k
    end
    return -total
end

"""
    CJSFit

Maximum-likelihood CJS result. Contains process-labelled
coefficients, training schemas/contrasts, offsets, log likelihood, curvature,
optimizer attempts and identification/boundary diagnostics. A converged fit
alone is not evidence that uncertainty is calibrated.
"""
struct CJSFit{D,S,P,R,Q}
    data::D
    beta::Vector{Float64}
    coefficients::NamedTuple
    phi_design::S
    p_design::P
    loglik::Float64
    vcov::Matrix{Float64}
    diagnostics::R
    prepared::Q
end

"""
    cjs(data::CMRData; phi=@formula(phi~1), p=@formula(p~1),
        terminal_constraint=nothing, sd_phi=nothing, sd_p=nothing,
        selection=nothing, start=nothing, maxiters=1000, g_tol=1e-8)

Fit first-capture-conditional CJS by ML. Survival coefficients
model logit **unit-time apparent survival**; interval survival is `s^Delta`.
Detection coefficients model logit capture probability conditional on alive.
LHS symbols `phi`/`p` are labels, never response columns. `offset(x)` is a
fixed link-scale contribution. Training schemas, contrasts and levels are kept.

Unrestricted terminal time-specific survival/detection requires an explicit
`terminal_constraint=(phi=value,)` or `(p=value,)`, fixing one final unit-time
survival or detection probability globally. Constrained flat directions are
removed. One shared Gaussian intercept `(1 | group)` per process uses the joint
CJS Laplace approximation. Shared effects are independent across processes;
their conditional Hessian retains all cross-process and cross-year entries.
Persistent individual intercepts `(1 | id)` use normalized Gaussian quadrature
over each full history. `sd_phi` and `sd_p` optionally give process-specific
formulas for log SD; predictors must be constant within the individual or
shared grouping unit. Shared SD formulas use unit-specific scales in the
joint Laplace objective.
Orders 25, 41, 65 and 97 are fixed within each optimizer run and refined between
fits. `selection=(eligibility=indices, assumption=:alive_available)` applies an
individual first-capture correction, requiring each random effect to be grouped
by `id`, complete prehistory predictors,
and shared pre/post coefficients. It conditions on the realized first-capture
occasion. Shared-effect selection, random slopes, and shared-plus-individual
effects remain unavailable. Ordinary shared and unselected individual effects
support explicit prediction targets and persistent-draw simulation. Selected
individual effects support response-scale selected-cohort prediction and
rejection-sampled simulation for the fitted capture layout. Prediction accepts
new selected profiles only when eligibility, first capture, complete prehistory,
survey masks, and covariates are supplied explicitly. Shared SD formulas retain guarded
prediction/simulation. Heterogeneity uncertainty remains unqualified.
ML is the sole likelihood target.
"""
function cjs(data::CMRData;phi=@formula(phi~1),p=@formula(p~1),terminal_constraint=nothing,
             sd_phi=nothing,sd_p=nothing,selection=nothing,start=nothing,
             maxiters::Integer=1000,g_tol::Real=1e-8,contrasts=Dict{Symbol,Any}())
    any(data.observed[i,t] for i in eachindex(data.first) for t=data.first[i]+1:data.end[i]) || throw(ArgumentError("cjs has no surveyed post-capture occasions: final-first and fully unsurveyed histories are retained but provide no likelihood information"))
    prep=_cjs_prepare(data,phi,p;terminal_constraint=terminal_constraint,contrasts=contrasts)
    if selection!==nothing && !_cjs_is_individual(prep)
        throw(ArgumentError("reference selection correction is currently available only with persistent individual effects; shared-effect normalization is unsupported"))
    end
    selection!==nothing && (prep=_cjs_selection_prepare(prep,selection))
    if sd_phi!==nothing || sd_p!==nothing
        if _cjs_is_shared(prep)
            prep=_cjs_prepare_shared_sd(prep,sd_phi,sd_p;contrasts=contrasts)
        elseif !_cjs_is_individual(prep)
            throw(ArgumentError("sd_phi and sd_p require an individual or shared effect on the corresponding process"))
        end
    end
    _cjs_is_individual(prep) && (prep=_cjs_prepare_sd(prep,sd_phi,sd_p;
        selection=hasproperty(prep,:selection) ? prep.selection : nothing,
        contrasts=contrasts))
    prep=merge(prep,(prediction_contract=(version=1, selection_enabled=selection!==nothing,
        sd_formula_present=(phi=sd_phi!==nothing,p=sd_p!==nothing)),))
    if _cjs_is_individual(prep)
        start !== nothing && return _cjs_fit_individual(prep;start,maxiters,g_tol)
        nfixed=prep.nphi+prep.np
        nscale=prep.individual.nscale
        default_start=vcat(zeros(nfixed),fill(log(.5),nscale))
        high_sd_start=vcat(zeros(nfixed),zeros(nscale))
        candidates=[_cjs_fit_individual(prep;start=s,maxiters,g_tol)
                    for s in (default_start,high_sd_start)]
        function status(fit)
            finite_parameters=all(isfinite,fit.beta)
            finite_gradient=all(isfinite,fit.diagnostics.gradient) && isfinite(fit.diagnostics.gradient_norm)
            (!finite_parameters || !isfinite(fit.loglik) || !finite_gradient) && return :nonfinite
            !fit.diagnostics.converged && return :unconverged
            !fit.diagnostics.quadrature_refined && return :unrefined
            !fit.diagnostics.positive_curvature && return :converged_refined_nonpositive_curvature
            fit.diagnostics.gradient_norm>1e-4 && return :converged_refined_above_profile_gradient_gate
            return :converged_refined_positive_curvature
        end
        finite_candidates=findall(i->all(isfinite,candidates[i].beta) &&
            isfinite(candidates[i].loglik) && all(isfinite,candidates[i].diagnostics.gradient) &&
            isfinite(candidates[i].diagnostics.gradient_norm),eachindex(candidates))
        admissible=findall(i->candidates[i].diagnostics.converged &&
            candidates[i].diagnostics.quadrature_refined &&
            all(isfinite,candidates[i].beta) && isfinite(candidates[i].loglik) &&
            all(isfinite,candidates[i].diagnostics.gradient) &&
            isfinite(candidates[i].diagnostics.gradient_norm),eachindex(candidates))
        pool=isempty(admissible) ? finite_candidates : admissible
        isempty(pool) && (pool=collect(eachindex(candidates)))
        selected=pool[argmax([candidates[i].loglik for i in pool])]
        fit=candidates[selected]
        start_candidates=[(start=copy(s),loglik=candidates[i].loglik,
            converged=candidates[i].diagnostics.converged,
            quadrature_refined=candidates[i].diagnostics.quadrature_refined,
            gradient_norm=candidates[i].diagnostics.gradient_norm,
            requested_gradient_tolerance=g_tol,
            gradient_tolerance=max(1e-5,g_tol),
            finite_parameters=all(isfinite,candidates[i].beta),
            positive_curvature=candidates[i].diagnostics.positive_curvature,
            quadrature_order=candidates[i].diagnostics.quadrature_order,
            status=status(candidates[i])) for (i,s) in enumerate((default_start,high_sd_start))]
        diagnostics=merge(fit.diagnostics,(start_strategy=:default_and_high_sd,
            start_candidates=start_candidates,selected_start=selected,
            selection_basis=isempty(admissible) ? :best_finite_objective_fallback : :best_finite_refined_loglik,
            selection_gradient_tolerance=max(1e-5,g_tol),profile_gradient_tolerance=1e-4,
            candidate_order_comparison=:candidate_specific_refined_orders))
        return CJSFit(fit.data,fit.beta,fit.coefficients,fit.phi_design,fit.p_design,
            fit.loglik,fit.vcov,diagnostics,fit.prepared)
    end
    _cjs_is_shared(prep) && return _cjs_fit_shared(prep;start,maxiters,g_tol)
    q=prep.nphi+prep.np
    b=start===nothing ? zeros(q) : Float64.(collect(start))
    length(b)==q && all(isfinite,b) || throw(ArgumentError("cjs start must be finite with one value per free coefficient"))
    fun=x->_cjs_objective(x,prep)
    attempts=NamedTuple[]
    if q==0
        bhat=Float64[]; value=fun(bhat); grad=Float64[]; H=zeros(0,0)
        converged=isfinite(value); iterations=0
    else
        result=Optim.optimize(fun,b,Optim.BFGS(),Optim.Options(iterations=maxiters,g_tol=g_tol);autodiff=:forward)
        bhat=Vector{Float64}(Optim.minimizer(result)); value=fun(bhat)
        grad=ForwardDiff.gradient(fun,bhat); H=ForwardDiff.hessian(fun,bhat)
        converged=Optim.converged(result); iterations=Optim.iterations(result)
    end
    gnorm=isempty(grad) ? 0. : norm(grad,Inf)
    eigen=isempty(bhat) ? Float64[] : eigvals(Symmetric(H))
    pdcurvature=!isempty(eigen) && all(isfinite,eigen) && minimum(eigen)>maximum(eigen)*1e-10
    push!(attempts,(start=b,solution=bhat,objective=value,converged=converged,iterations=iterations,gradient_norm=gnorm))
    ls,lp=_cjs_process_logs(bhat,prep)
    _,T=size(data.Y)
    delta=[data.intervals[mod1(row,T-1)] for row in prep.phi_design.rows]
    probs=vcat(exp.(ls),exp.(ls .* delta),exp.(lp)); boundary=any(x->x<=1e-8||x>=1-1e-8,probs)
    uncertainty_available=pdcurvature&&!boundary&&converged&&gnorm<=1e-4
    V=uncertainty_available ? Matrix(inv(Symmetric(H))) : fill(NaN,q,q)
    diagnostics=(objective=value,optimizer=q==0 ? :fixed : :BFGS,method=:ML,converged=converged,gradient=grad,gradient_norm=gnorm,hessian=H,curvature=eigen,positive_curvature=pdcurvature,boundary=boundary,attempts=attempts,integration=:exact_discrete_states,terminal_constraint=prep.terminal_constraint,enrolled=sum(data.frequency),informative=sum(data.frequency[i] for i in eachindex(data.first) if any(view(data.observed,i,data.first[i]+1:data.end[i]))),individual_rows=length(data.ids),contributing_occasions=sum(data.frequency .* (data.end .- data.first)),surveyed_occasions=sum(data.frequency[i]*count(view(data.observed,i,data.first[i]+1:data.end[i])) for i in eachindex(data.first)),uncertainty_available=uncertainty_available,uncertainty_reason=uncertainty_available ? "local ML curvature only; calibration not established" : "unavailable: nonconvergence, boundary, large gradient or nonpositive curvature")
    co=(phi=bhat[1:prep.nphi],p=bhat[prep.nphi+1:end])
    return CJSFit(data,bhat,co,prep.phi_design,prep.p_design,-Float64(value),V,diagnostics,prep)
end

# R supplies designs with fixed coding and coefficient order. The bridge checks
# their positional layout and likelihood contract without reparsing R formulas.
function _cjs_matrix_bridge_prepare(Y,observed,first,ends,intervals,frequency,
                                    X_phi,offset_phi,X_p,offset_p,
                                    terminal_name,terminal_value)
    Y isa AbstractMatrix || throw(ArgumentError("CJS bridge histories must be a matrix"))
    n, nt = size(Y)
    data = CMRData(Y; ids=collect(1:n), occasions=collect(1:nt), units="bridge",
                   observed=observed, intervals=intervals, ends=ends,
                   frequency=frequency)
    first isa AbstractVector && length(first)==n &&
        all(x->x isa Real && !(x isa Bool) && isfinite(x) && isinteger(x),first) &&
        all(i->first[i]==data.first[i],1:n) ||
        throw(ArgumentError("CJS bridge first-capture indices must match the observed histories"))
    # Obtain only the exact history row layout from a fixed template. Supplied
    # designs and the actual terminal constraint are validated below.
    prep0 = _cjs_prepare(data, @formula(phi ~ 1), @formula(p ~ 1);
                         terminal_constraint=(phi=0.5,))
    pd = prep0.phi_design
    qd = prep0.p_design
    function supplied_design(des,X,offset,label)
        X isa AbstractMatrix && size(X,1)==length(des.rows) ||
            throw(ArgumentError("CJS bridge $label matrix row mismatch: supply complete individual-major post-capture rows"))
        offset isa AbstractVector && length(offset)==size(X,1) ||
            throw(ArgumentError("CJS bridge $label offset length mismatch"))
        all(x->x isa Real && isfinite(x),X) && all(x->x isa Real && isfinite(x),offset) ||
            throw(ArgumentError("CJS bridge $label design and offsets must be finite numbers"))
        matrix=Matrix{Float64}(X); offsets=Float64.(offset)
        all(isfinite,matrix) && all(isfinite,offsets) ||
            throw(ArgumentError("CJS bridge $label design and offsets must be representable as finite Float64 values"))
        return merge(des,(X=matrix,offset=offsets,names=["x$j" for j in axes(matrix,2)],
                          columns=collect(axes(matrix,2))))
    end
    phi_design=supplied_design(pd,X_phi,offset_phi,:phi)
    p_design=supplied_design(qd,X_p,offset_p,:p)
    constraint=if terminal_name===nothing || (terminal_name isa AbstractString && terminal_name=="")
        nothing
    elseif terminal_name isa AbstractString && terminal_name in ("phi","p")
        terminal_value isa Real && isfinite(terminal_value) && 0<terminal_value<1 ||
            throw(ArgumentError("CJS bridge terminal constraint must be a strictly interior probability"))
        _cjs_constraint(terminal_name=="phi" ? (phi=Float64(terminal_value),) : (p=Float64(terminal_value),))
    else
        throw(ArgumentError("CJS bridge terminal_name must be \"phi\", \"p\", empty, or nothing"))
    end
    surveyed=Bool[data.observed[cld(row,nt),mod1(row,nt)] for row in qd.rows]
    phi_active=map(pd.rows) do row
        i=cld(row,nt-1); t=mod1(row,nt-1)
        any(view(data.observed,i,t+1:data.end[i]))
    end
    terminal_direction(X,v,active)=!isempty(v) && any(v[active]) && rank(hcat(X[active,:],Float64.(v[active])))==rank(X[active,:])
    if constraint===nothing && terminal_direction(phi_design.X,prep0.phi_terminal,phi_active) &&
       terminal_direction(p_design.X[surveyed,:],prep0.p_terminal[surveyed],trues(count(surveyed)))
        throw(ArgumentError("CJS bridge unrestricted terminal survival and detection are confounded; supply an explicit terminal constraint"))
    end
    for (label,des,terminal,active) in ((:phi,phi_design,prep0.phi_terminal,phi_active),
                                       (:p,p_design,prep0.p_terminal,surveyed))
        if constraint!==nothing && only(keys(constraint))==label
            active=active .& .!terminal
        end
        rank(des.X[active,:])==size(des.X,2) ||
            throw(ArgumentError("CJS bridge $label design is rank deficient on unconstrained contributing rows; supply the reduced coefficient columns in their R order"))
    end
    prep = merge(prep0, (phi_design=phi_design, p_design=p_design,
                         nphi=size(X_phi,2), np=size(X_p,2),
                         terminal_constraint=constraint))
    return prep
end

"""
    cjs_matrix_bridge(Y, observed, first, ends, intervals, frequency,
                      X_phi, offset_phi, X_p, offset_p,
                      terminal_name, terminal_value, maxiters, g_tol)

Internal fixed-effects bridge entry point. Designs contain complete post-capture
rows in individual-major order: intervals `first[i]:ends[i]-1` for survival and
occasions `first[i]+1:ends[i]` for detection, including unsurveyed detection slots.
Supplied `first` indices must match the histories. Offsets are on the logit scale;
survival is raised to the supplied interval lengths. `terminal_name` is `"phi"`,
`"p"`, `""`, or `nothing`; a constraint fixes the probability at the globally
final occasion, not each individual's administrative endpoint. R must remove
constrained flat columns before supplying its designs; coefficient order is
preserved. Zero-column designs evaluate a fully fixed model without optimization.
Returned row indices expose the positional contract but cannot detect a silent
permutation of otherwise valid matrix rows without caller-provided keys.
"""
function cjs_matrix_bridge(Y, observed, first, ends, intervals, frequency,
                           X_phi, offset_phi, X_p, offset_p,
                           terminal_name, terminal_value, maxiters, g_tol)
    maxiters isa Real && !(maxiters isa Bool) && isfinite(maxiters) && isinteger(maxiters) && maxiters>=1 ||
        throw(ArgumentError("CJS bridge maxiters must be a positive integer"))
    iterations_limit=try Int(maxiters) catch
        throw(ArgumentError("CJS bridge maxiters is outside the supported integer range"))
    end
    g_tol isa Real && !(g_tol isa Bool) && isfinite(g_tol) && g_tol>0 ||
        throw(ArgumentError("CJS bridge g_tol must be finite and strictly positive"))
    tolerance=Float64(g_tol)
    isfinite(tolerance) && tolerance>0 || throw(ArgumentError("CJS bridge g_tol must be representable as a positive Float64 value"))
    prep=_cjs_matrix_bridge_prepare(Y,observed,first,ends,intervals,frequency,
        X_phi,offset_phi,X_p,offset_p,terminal_name,terminal_value)
    q = prep.nphi + prep.np
    b = zeros(q)
    fun = x -> _cjs_objective(x, prep)
    if q == 0
        bhat=b; value=fun(b); grad=Float64[]; H=zeros(0,0); converged=isfinite(value); iterations=0
    else
        result=Optim.optimize(fun,b,Optim.BFGS(),Optim.Options(iterations=iterations_limit,g_tol=tolerance);autodiff=:forward)
        bhat=Vector{Float64}(Optim.minimizer(result)); value=fun(bhat)
        grad=ForwardDiff.gradient(fun,bhat); H=ForwardDiff.hessian(fun,bhat)
        converged=Optim.converged(result); iterations=Optim.iterations(result)
    end
    data=prep.data
    layout=(phi_rows=copy(prep.phi_design.rows),p_rows=copy(prep.p_design.rows),
            p_surveyed=Bool[data.observed[cld(row,size(Y,2)),mod1(row,size(Y,2))] for row in prep.p_design.rows],
            first=copy(data.first),ends=copy(data.end),
            phi_terminal=copy(prep.phi_terminal),p_terminal=copy(prep.p_terminal))
    return (beta=bhat, loglik=-Float64(value), gradient=grad, hessian=H,
            converged=converged, iterations=iterations, objective=Float64(value),row_layout=layout)
end

coef(f::CJSFit)=f.coefficients
function vcov(f::CJSFit)
    f.diagnostics.uncertainty_available || throw(ArgumentError(f.diagnostics.uncertainty_reason))
    return f.vcov
end
loglikelihood(f::CJSFit)=f.loglik
nobs(f::CJSFit)=f.diagnostics.enrolled
dof(f::CJSFit)=length(f.beta)
aic(f::CJSFit)=2length(f.beta)-2f.loglik

include("cjs_shared_laplace.jl")

include("cjs_individual_quadrature.jl")

include("cjs_shared_sd.jl")
