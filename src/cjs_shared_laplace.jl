# Shared intercepts use the exact CJS history kernel and a joint Gaussian mode.
# This bounded, dense-vector route does not change the sparse DRM engine.
using LinearAlgebra: cholesky, logdet, dot

_cjs_is_shared(prep) = hasproperty(prep,:shared) && prep.shared !== nothing
function _cjs_has_shared(term)
    term isa FunctionTerm && return term.f === (|) || any(_cjs_has_shared,term.args)
    term isa Tuple && return any(_cjs_has_shared,term)
    hasproperty(term,:terms) && return any(_cjs_has_shared,term.terms)
    return false
end

function _cjs_split_shared(formula,label)
    formula isa FormulaTerm && formula.lhs isa Term && formula.lhs.sym==label ||
        throw(ArgumentError("cjs $label formula must have the label $label on its left side"))
    terms=formula.rhs isa Tuple ? collect(formula.rhs) : Any[formula.rhs]
    fixed=Any[]; group=nothing
    for term in terms
        if term isa FunctionTerm && term.f === (|)
            group===nothing || throw(ArgumentError("cjs permits only one shared intercept per process"))
            length(term.args)==2 && term.args[1] isa ConstantTerm && term.args[1].n==1 && term.args[2] isa Term ||
                throw(ArgumentError("cjs shared effects require (1 | group); slopes and correlated effects are unsupported"))
            group=term.args[2].sym
            group==:id && throw(ArgumentError("cjs persistent individual effects and shared-plus-individual effects are unsupported in M3"))
        else
            _cjs_validate_term(term)
            push!(fixed,term)
        end
    end
    rhs=isempty(fixed) ? ConstantTerm(1) : length(fixed)==1 ? only(fixed) : Tuple(fixed)
    return FormulaTerm(formula.lhs,rhs),group
end

function _cjs_shared_mapping(prep,group,label,next,scale_index)
    group===nothing && return nothing
    des=label==:phi ? prep.phi_design : prep.p_design
    table=label==:phi ? prep.data.phi_data : prep.data.p_data
    haskey(table,group) || throw(ArgumentError("cjs $label shared grouping column $group is absent"))
    values=getproperty(table,group); d=prep.data; _,T=size(d.Y)
    terminal=label==:phi ? prep.phi_terminal : prep.p_terminal
    constrained=prep.terminal_constraint!==nothing && only(keys(prep.terminal_constraint))==label
    active=Bool[]
    for (k,row) in enumerate(des.rows)
        i=cld(row,label==:phi ? T-1 : T); t=mod1(row,label==:phi ? T-1 : T)
        contributes=label==:phi ? any(view(d.observed,i,t+1:d.end[i])) : d.observed[i,t]
        push!(active,contributes && !(constrained && terminal[k]))
    end
    any(k->active[k] && ismissing(values[des.rows[k]]),eachindex(active)) &&
        throw(ArgumentError("cjs $label shared group cannot be missing on contributing rows"))
    levels=unique([values[des.rows[k]] for k in eachindex(active) if active[k]])
    isempty(levels) && throw(ArgumentError("cjs $label shared effect has no informative grouping levels after masks and terminal constraints"))
    lookup=Dict(v=>j for (j,v) in enumerate(levels))
    index=[active[k] ? lookup[values[row]] : 0 for (k,row) in enumerate(des.rows)]
    individuals=[unique([cld(des.rows[k],label==:phi ? T-1 : T) for k in eachindex(index) if index[k]==j]) for j in eachindex(levels)]
    counts=[sum(d.frequency[i] for i in ids) for ids in individuals]
    by_history=[unique([values[des.rows[k]] for k in eachindex(index) if index[k]>0 && cld(des.rows[k],label==:phi ? T-1 : T)==i]) for i in eachindex(d.ids)]
    varying=any(x->length(x)>1,by_history)
    # Labels are arbitrary. A row-varying grouping is the joint year-vector
    # route; it must actually recur across histories to represent shared years.
    varying && !any(x->length(x)>=2,individuals) && throw(ArgumentError("cjs row-varying shared levels must repeat across overlapping histories; individual effects are unsupported"))
    !varying && length(levels)==length(filter(!isempty,by_history)) && all(==(1),counts) &&
        throw(ArgumentError("cjs grouping is individual-specific; shared intercepts require repeated histories within levels"))
    alllevels=unique(collect(skipmissing(values)))
    excluded=[x for x in alllevels if !haskey(lookup,x)]
    return (group=group,levels=levels,index=index,range=next:next+length(levels)-1,
            scale_index=scale_index,varying=varying,by_history=by_history,
            histories_per_group=counts,distinct_histories_per_group=length.(individuals),
            excluded_levels=excluded,singleton_levels=levels[counts.==1])
end

function _cjs_prepare_shared(data,phi,p;terminal_constraint=nothing,contrasts=Dict{Symbol,Any}())
    fphi,gphi=_cjs_split_shared(phi,:phi); fp,gp=_cjs_split_shared(p,:p)
    gphi===nothing && gp===nothing && throw(ArgumentError("cjs shared preparation requires a shared intercept"))
    gphi!==nothing && gp!==nothing && gphi!=gp &&
        throw(ArgumentError("cjs shared effects on both processes must use the same grouping column; crossed group combinations are unsupported"))
    prep=_cjs_prepare(data,fphi,fp;terminal_constraint,contrasts)
    nfixed=prep.nphi+prep.np
    sm=_cjs_shared_mapping(prep,gphi,:phi,1,nfixed+1)
    pm=_cjs_shared_mapping(prep,gp,:p,sm===nothing ? 1 : last(sm.range)+1,nfixed+(sm===nothing ? 1 : 2))
    if sm!==nothing && pm!==nothing
        sm.varying==pm.varying ||
            throw(ArgumentError("mixing site and year grouping structures across processes is not supported"))
        # Independent processes have separate latent ranges and level maps.
        # Masks and a process-specific terminal constraint can legitimately
        # remove a level from just one process; no cross-process covariance
        # requires matching active levels or their encounter order.
    end
    if prep.terminal_constraint===nothing
        augmented(des,m)=m===nothing ? des.X : hcat(des.X,
            [Float64(j==k) for j in m.index,k in eachindex(m.levels)])
        terminal_direction(X,v,active)=any(active) && any(v[active]) &&
            rank(hcat(X[active,:],Float64.(v[active])))==rank(X[active,:])
        # Use only rows that contribute to the observed history likelihood.
        # Positional zero rows retained for the forward kernel must not obscure
        # a terminal direction shared by the active rows.
        if terminal_direction(augmented(prep.phi_design,sm),prep.phi_terminal,
                              prep.phi_design.informative) &&
           terminal_direction(augmented(prep.p_design,pm),prep.p_terminal,
                              prep.p_design.informative)
            throw(ArgumentError("Unrestricted terminal shared survival and detection are confounded; supply terminal_constraint=(phi=value,) or (p=value,)"))
        end
    end
    if sm!==nothing && pm!==nothing && !sm.varying && !pm.varying
        for i in eachindex(data.ids)
            isempty(sm.by_history[i]) || isempty(pm.by_history[i]) || sm.by_history[i]==pm.by_history[i] ||
                throw(ArgumentError("cjs survival and detection shared site labels disagree for history $(repr(data.ids[i]))"))
        end
    end
    mappings=filter(!isnothing,[sm,pm])
    dimension=sum(length(m.levels) for m in mappings)
    route=any(m->m.varying,mappings) ? :year_vector : sm===nothing ? :site_p : pm===nothing ? :site_phi : :site_pair
    names=["log_sigma_$(label): $(m.group)" for (label,m) in ((:phi,sm),(:p,pm)) if m!==nothing]
    shared=(phi=sm,p=pm,dimension=dimension,nscale=length(mappings),route=route,names=names,
            diagnostics=(groups=[(process=label,group=m.group,levels=m.levels,
                histories_per_group=m.histories_per_group,distinct_histories_per_group=m.distinct_histories_per_group,
                excluded_levels=m.excluded_levels,singleton_levels=m.singleton_levels,
                identification_status=length(m.levels)<3 ? :too_few_groups_for_variance_evidence : :recovery_unqualified)
                for (label,m) in ((:phi,sm),(:p,pm)) if m!==nothing],
                recaptured_histories=sum(data.frequency[i] for i in eachindex(data.ids) if sum(view(data.Y,i,data.first[i]+1:data.end[i]))>0),
                qualification=:numerical_route_only))
    return merge(prep,(phi_design=merge(prep.phi_design,(formula=phi,)),
                       p_design=merge(prep.p_design,(formula=p,)),shared=shared))
end

function _cjs_shared_sigmas(theta,mapping)
    sigma=if hasproperty(mapping,:sd_X)
        exp.(mapping.sd_X*theta[mapping.scale_range]+mapping.sd_offset)
    else
        fill(exp(theta[mapping.scale_index]),length(mapping.levels))
    end
    all(x->isfinite(_cjs_primal(x)) && _cjs_primal(x)>0,sigma) ||
        throw(ArgumentError("cjs shared SDs must be finite and strictly positive; zero variance requires the explicit reduced model"))
    return sigma
end

function _cjs_shared_rows(u,theta,mapping,nrows)
    mapping===nothing && return nothing
    length(mapping.index)==nrows || throw(ArgumentError("CJS shared row mapping dimension mismatch"))
    sigma=_cjs_shared_sigmas(theta,mapping)
    return [j==0 ? zero(first(sigma)*first(u)) : sigma[j]*u[first(mapping.range)+j-1] for j in mapping.index]
end

function _cjs_shared_start(prep)
    sh=prep.shared
    b=zeros(prep.nphi+prep.np+sh.nscale)
    for mapping in (sh.phi,sh.p)
        mapping===nothing && continue
        if hasproperty(mapping,:sd_X)
            intercept=findfirst(==("(Intercept)"),mapping.sd_names)
            intercept===nothing || (b[mapping.scale_range[intercept]]=log(.3))
        else
            b[mapping.scale_index]=log(.3)
        end
    end
    return b
end

"Internal normalized joint shared-effect objective, in standardized Gaussian coordinates."
function _cjs_shared_conditional_nll(u,theta,prep)
    sh=prep.shared; nfixed=prep.nphi+prep.np
    length(u)==sh.dimension && length(theta)==nfixed+sh.nscale || throw(ArgumentError("CJS shared-effect parameter dimension mismatch"))
    rs=_cjs_shared_rows(u,theta,sh.phi,length(prep.phi_design.rows))
    rp=_cjs_shared_rows(u,theta,sh.p,length(prep.p_design.rows))
    return _cjs_fixed_objective(view(theta,1:nfixed),prep;random_phi=rs,random_p=rp)+
           dot(u,u)/2+sh.dimension*log(2pi)/2
end

_cjs_primal(x)=x
_cjs_primal(x::ForwardDiff.Dual)=_cjs_primal(ForwardDiff.value(x))
_cjs_dual_depth(::Type{T}) where T=0
_cjs_dual_depth(::Type{ForwardDiff.Dual{Tag,V,N}}) where {Tag,V,N}=1+_cjs_dual_depth(V)

struct _CJSSharedModeError <: Exception
    reason::String
    attempts::Vector{NamedTuple}
end
Base.showerror(io::IO,e::_CJSSharedModeError)=print(io,e.reason)

function _cjs_shared_mode(theta,prep;start=nothing)
    fun=u->_cjs_shared_conditional_nll(u,theta,prep)
    dimension=prep.shared.dimension
    initial=if start===nothing
        zeros(dimension)
    else
        candidate=Float64.(collect(start))
        length(candidate)==dimension || throw(ArgumentError("CJS shared-mode warm start dimension mismatch"))
        all(isfinite,candidate) ? candidate : zeros(dimension)
    end
    starts=all(iszero,initial) ? [initial] : [initial,zeros(dimension)]
    attempts=NamedTuple[]
    if start!==nothing && all(isfinite,initial)
        try
            value=fun(initial)
            if isfinite(value)
                g=ForwardDiff.gradient(fun,initial)
                if all(isfinite,g) && norm(g,Inf)<=1e-9
                    A=ForwardDiff.hessian(fun,initial)
                    positive_curvature=all(isfinite,A) &&
                        issuccess(cholesky(Symmetric(A);check=false))
                    if positive_curvature
                        gradient_norm=norm(g,Inf)
                        push!(attempts,(optimizer=:warm_start_stationary,
                            start=copy(initial),mode=copy(initial),objective=value,
                            converged=true,optimizer_converged=true,
                            gradient_norm=gradient_norm,positive_curvature=true,
                            iterations=0,
                            reason="accepted supplied stationary positive-curvature mode"))
                        return (mode=copy(initial),hessian=A,gradient=g,
                            gradient_norm=gradient_norm,converged=true,attempts=attempts)
                    end
                end
            end
        catch err
            err isa InterruptException && rethrow()
            # A rejected or numerically invalid warm-start check must not change
            # the established warm-optimizer then zero-start fallback below.
        end
    end
    for initial_start in starts
        best=copy(initial_start); bestvalue=Inf
        for method in (Optim.BFGS(),Optim.NewtonTrustRegion())
            try
                result=Optim.optimize(fun,best,method,Optim.Options(iterations=200,g_tol=1e-10,f_reltol=1e-13);autodiff=:forward)
                candidate=Vector{Float64}(Optim.minimizer(result)); value=fun(candidate)
                g=ForwardDiff.gradient(fun,candidate); A=ForwardDiff.hessian(fun,candidate)
                # Polish the primal stationarity so numerical mode tolerance does
                # not contaminate outer implicit derivatives.
                for _ in 1:8
                    norm(g,Inf)<=1e-10 && break
                    fac=cholesky(Symmetric(A);check=false)
                    issuccess(fac) || break
                    proposed=candidate-fac\g
                    newvalue=fun(proposed)
                    isfinite(newvalue) && newvalue<=value+1e-12*max(1.,abs(value)) || break
                    candidate=proposed; value=newvalue
                    g=ForwardDiff.gradient(fun,candidate); A=ForwardDiff.hessian(fun,candidate)
                end
                curvature=isfinite(value) && all(isfinite,A) && issuccess(cholesky(Symmetric(A);check=false))
                converged=curvature && all(isfinite,g) && norm(g,Inf)<=1e-9
                push!(attempts,(optimizer=method isa Optim.BFGS ? :BFGS : :NewtonTrustRegion,
                    start=copy(best),mode=copy(candidate),objective=value,converged=converged,
                    optimizer_converged=Optim.converged(result),gradient_norm=norm(g,Inf),
                    positive_curvature=curvature,iterations=Optim.iterations(result),reason=converged ? "stationary positive-curvature joint mode" : "inner gradient or curvature failed"))
                if isfinite(value) && value<bestvalue
                    best=candidate; bestvalue=value
                end
                converged && return (mode=candidate,hessian=A,gradient=g,gradient_norm=norm(g,Inf),converged=true,attempts=attempts)
            catch err
                err isa InterruptException && rethrow()
                push!(attempts,(optimizer=method isa Optim.BFGS ? :BFGS : :NewtonTrustRegion,
                    start=copy(best),mode=copy(best),objective=Inf,converged=false,optimizer_converged=false,
                    gradient_norm=Inf,positive_curvature=false,iterations=0,reason=sprint(showerror,err)))
            end
        end
    end
    throw(_CJSSharedModeError("cjs joint shared-effect mode failed; no valid Laplace likelihood",attempts))
end

"Internal joint CJS Laplace likelihood with implicit derivatives of the converged mode."
_cjs_shared_laplace_hessian(fun, u, inner, depth) =
    depth == 0 ? inner.hessian : ForwardDiff.hessian(fun, u)

function _cjs_shared_mode_lift_count(depth::Integer)
    depth >= 0 || throw(ArgumentError("nested dual depth must be nonnegative"))
    target_order = depth + 1
    represented_order = 1
    lifts = 0
    while represented_order < target_order
        represented_order *= 2
        lifts += 1
    end
    return lifts
end

function _cjs_shared_laplace(theta,prep;details=false,mode_start=nothing,mode_sink=nothing)
    primal=Float64.(_cjs_primal.(theta))
    all(isfinite,primal) || throw(ArgumentError("CJS shared parameters must be finite"))
    length(theta)==prep.nphi+prep.np+prep.shared.nscale || throw(ArgumentError("CJS shared-effect parameter dimension mismatch"))
    for mapping in (prep.shared.phi,prep.shared.p)
        mapping===nothing && continue
        _cjs_shared_sigmas(primal,mapping)
    end
    inner=_cjs_shared_mode(primal,prep;start=mode_start)
    mode_sink===nothing || (mode_sink[]=copy(inner.mode))
    u=inner.mode
    fun=v->_cjs_shared_conditional_nll(v,theta,prep)
    depth=_cjs_dual_depth(eltype(theta))
    if depth>0
        u=[zero(first(theta))+v for v in u]
        # At a converged primal root, Newton has zero derivative with respect
        # to its starting mode. These lifts solve the implicit stationarity
        # equation through each nested dual order, including Hessian order.
        for _ in 1:_cjs_shared_mode_lift_count(depth)
            g=ForwardDiff.gradient(fun,u); A=ForwardDiff.hessian(fun,u)
            u=u-cholesky(Symmetric(A))\g
        end
    end
    A=_cjs_shared_laplace_hessian(fun,u,inner,depth)
    fac=cholesky(Symmetric(A))
    value=fun(u)-prep.shared.dimension*log(2pi)/2+logdet(fac)/2
    return details ? (loglik=-value,objective=value,mode=u,hessian=A,
        gradient_norm=inner.gradient_norm,converged=inner.converged,attempts=inner.attempts,
        integration=:joint_CJS_Laplace,route=prep.shared.route) : value
end

function _cjs_shared_gradient(theta,prep;mode_cache=nothing)
    mode_cache===nothing && return ForwardDiff.gradient(x->_cjs_shared_laplace(x,prep),theta)
    start=mode_cache[]===nothing ? nothing : Float64.(collect(mode_cache[]))
    next_mode=Ref(start===nothing ? nothing : copy(start))
    gradient=ForwardDiff.gradient(x->_cjs_shared_laplace(x,prep;mode_start=start,mode_sink=next_mode),theta)
    mode_cache[]=next_mode[]===nothing ? nothing : copy(next_mode[])
    return gradient
end

"Internal numerical covariance sensitivity; it does not establish interval calibration."
function _cjs_curvature_sensitivity(coarse,fine,selected;gradient=nothing,
        perturbation_tol=.01,se_tol=.01,correlation_tol=.01,stationarity_tol=1e-4)
    all(x->isfinite(x)&&x>=0,(perturbation_tol,se_tol,correlation_tol,stationarity_tol)) &&
        perturbation_tol<1 || throw(ArgumentError("CJS curvature sensitivity tolerances must be finite and nonnegative; relative perturbation must be below one"))
    q=size(selected,1)
    q>0 && size(selected)==(q,q) && size(coarse)==(q,q) && size(fine)==(q,q) ||
        throw(ArgumentError("CJS curvature sensitivity requires equally sized square matrices"))
    gradient===nothing || length(gradient)==q ||
        throw(ArgumentError("CJS curvature stationarity gradient dimension mismatch"))
    thresholds=(relative_perturbation=perturbation_tol,relative_se=se_tol,
        correlation=correlation_tol,stationarity=stationarity_tol)
    failure(reason)=(available=false,numerically_stable=false,numerical_covariance_eligible=false,
        reason=reason,relative_perturbation=Inf,maximum_relative_se_change=Inf,
        maximum_correlation_change=Inf,stationarity=Inf,
        stationarity_available=gradient!==nothing,thresholds=thresholds)
    all(A->all(isfinite,A),(coarse,fine,selected)) || return failure(:nonfinite_curvature)
    matrices=map(A->Matrix(Symmetric((A+transpose(A))/2)),(coarse,fine,selected))
    factors=try map(A->cholesky(Symmetric(A)),matrices) catch err
        err isa PosDefException || rethrow()
        return failure(:nonpositive_curvature)
    end
    C=last(factors)
    # Congruence whitening measures perturbations in the candidate's curvature
    # units, including weak directions and the raw finite-difference asymmetry.
    relative=max(opnorm(C.L\(coarse-selected)/C.U),
        opnorm(C.L\(fine-selected)/C.U),opnorm(C.L\(coarse-fine)/C.U))
    covariances=map(f->Matrix(f\Matrix{Float64}(I,q,q)),factors)
    standard_errors=map(V->sqrt.(diag(V)),covariances)
    correlations=map((V,se)->V./(se*transpose(se)),covariances,standard_errors)
    pairs=((1,2),(1,3),(2,3))
    se_change=maximum(maximum(abs.(standard_errors[a]-standard_errors[b])./
        min.(standard_errors[a],standard_errors[b])) for (a,b) in pairs)
    cor_change=maximum(maximum(abs.(correlations[a]-correlations[b])) for (a,b) in pairs)
    matrix_stable=isfinite(relative)&&relative<=perturbation_tol &&
        isfinite(se_change)&&se_change<=se_tol && isfinite(cor_change)&&cor_change<=correlation_tol
    stationarity=gradient===nothing || !all(isfinite,gradient) ? Inf :
        norm(C.L\gradient)
    stationary=gradient!==nothing && isfinite(stationarity) && stationarity<=stationarity_tol
    return (available=true,numerically_stable=matrix_stable,
        numerical_covariance_eligible=matrix_stable&&stationary,reason=:numerical_diagnostic_only,
        relative_perturbation=relative,maximum_relative_se_change=se_change,
        maximum_correlation_change=cor_change,stationarity=stationarity,
        stationarity_available=gradient!==nothing,thresholds=thresholds)
end

"Internal diagnostic curvature from two checked differences of the exact mode gradient."
function _cjs_refined_curvature(gradient,theta;rtol=1e-4,atol=1e-5,
        gradient_at_point=nothing,progress=nothing)
    x=Float64.(theta);q=length(x)
    steps=eps(Float64)^(1/3).*max.(1.,abs.(x))
    coarse=zeros(q,q);fine=zeros(q,q)
    for j in 1:q
        delta=zeros(q);delta[j]=steps[j]
        coarse_plus=gradient(x+delta)
        progress===nothing || progress((axis=j,step=:coarse_plus))
        coarse_minus=gradient(x-delta)
        progress===nothing || progress((axis=j,step=:coarse_minus))
        fine_plus=gradient(x+delta/2)
        progress===nothing || progress((axis=j,step=:fine_plus))
        fine_minus=gradient(x-delta/2)
        progress===nothing || progress((axis=j,step=:fine_minus))
        coarse[:,j]=(coarse_plus-coarse_minus)/(2steps[j])
        fine[:,j]=(fine_plus-fine_minus)/steps[j]
    end
    finite=all(isfinite,coarse)&&all(isfinite,fine)
    tolerance=atol .+ rtol.*max.(1.,abs.(fine),abs.(coarse))
    symmetry_tolerance=atol .+ rtol.*max.(1.,abs.(fine),abs.(transpose(fine)))
    refinement=finite ? maximum(abs,fine-coarse) : Inf
    asymmetry=finite ? maximum(abs,fine-transpose(fine)) : Inf
    refinement_ratio=finite ? maximum(abs.(fine-coarse)./tolerance) : Inf
    symmetry_ratio=finite ? maximum(abs.(fine-transpose(fine))./symmetry_tolerance) : Inf
    available=finite && refinement_ratio<=1 && symmetry_ratio<=1
    extrapolated=(4fine-coarse)/3
    H=available ? Matrix(Symmetric((extrapolated+transpose(extrapolated))/2)) : fill(NaN,q,q)
    sensitivity=_cjs_curvature_sensitivity(coarse,fine,H;gradient=gradient_at_point)
    return (hessian=H,available=available,method=:refined_gradient_differences,
        coarse_hessian=coarse,fine_hessian=fine,covariance_sensitivity=sensitivity,
        refinement_error=refinement,asymmetry=asymmetry,
        refinement_ratio=refinement_ratio,symmetry_ratio=symmetry_ratio,
        steps=steps,evaluations=4q,
        absolute_tolerance=atol,relative_tolerance=rtol)
end

function _cjs_trace_timing(phase;details...)
    path=get(ENV,"DRMODELS_CMR_TRACE_TIMING_FILE","")
    isempty(path) && return nothing
    open(path,"a") do io
        print(io,time_ns(),'\t',phase)
        for (key,value) in pairs(details)
            print(io,'\t',key,'=',value)
        end
        println(io)
    end
    return nothing
end

function _cjs_fit_shared(prep;start=nothing,maxiters=1000,g_tol=1e-8)
    sh=prep.shared; q=prep.nphi+prep.np+sh.nscale
    b=start===nothing ? _cjs_shared_start(prep) : Float64.(collect(start))
    length(b)==q && all(isfinite,b) || throw(ArgumentError("cjs shared start must contain fixed coefficients followed by the log-SD design coefficients in prepared process order"))
    maxiters>=1 && isfinite(g_tol) && g_tol>0 || throw(ArgumentError("cjs optimizer requires positive maxiters and g_tol"))
    _cjs_trace_timing("shared_fit_start";dimension=q,maxiters=maxiters)
    attempts=NamedTuple[]; inner_failures=NamedTuple[]
    mode_cache=Ref{Union{Nothing,Vector{Float64}}}(nothing)
    fun=x->try
        _cjs_shared_laplace(x,prep;mode_start=mode_cache[],mode_sink=mode_cache)
    catch err
        if err isa _CJSSharedModeError
            mode_cache[]=nothing
            push!(inner_failures,(parameters=Float64.(_cjs_primal.(x)),reason=err.reason,attempts=err.attempts))
        end
        rethrow()
    end
    bhat=copy(b); value=Inf; converged=false; iterations=0; optimizer=:BFGS
    for method in (Optim.BFGS(),Optim.NelderMead(),Optim.BFGS())
        optimizer=method isa Optim.BFGS ? :BFGS : :NelderMead
        try
            result=method isa Optim.BFGS ?
                Optim.optimize(fun,bhat,method,Optim.Options(iterations=maxiters,g_tol=g_tol,f_reltol=0.0);autodiff=:forward) :
                Optim.optimize(fun,bhat,method,Optim.Options(iterations=maxiters,f_reltol=1e-10))
            candidate=Vector{Float64}(Optim.minimizer(result)); v=fun(candidate)
            g=_cjs_shared_gradient(candidate,prep)
            success=Optim.converged(result) && norm(g,Inf)<=max(1e-4,g_tol)
            push!(attempts,(optimizer=optimizer,start=copy(bhat),solution=candidate,objective=v,
                converged=success,iterations=Optim.iterations(result),gradient_norm=norm(g,Inf),reason="outer ML optimization"))
            if v<value || (success && !converged && v<=value+1e-10*max(1.,abs(value)))
                bhat=candidate; value=v; converged=success; iterations=Optim.iterations(result)
            end
            success && break
        catch err
            err isa InterruptException && rethrow()
            push!(attempts,(optimizer=optimizer,start=copy(bhat),solution=copy(bhat),objective=Inf,
                converged=false,iterations=0,gradient_norm=Inf,reason=sprint(showerror,err)))
        end
    end
    inner=try _cjs_shared_laplace(bhat,prep;details=true,mode_start=mode_cache[]) catch err
        err isa InterruptException && rethrow()
        nothing
    end
    inner===nothing || (mode_cache[]=Float64.(_cjs_primal.(inner.mode)))
    _cjs_trace_timing("shared_point_fit_complete";dimension=q,optimizer=optimizer,
        iterations=iterations,converged=converged&&inner!==nothing)
    grad=inner===nothing ? fill(NaN,q) : _cjs_shared_gradient(bhat,prep;mode_cache=mode_cache)
    _cjs_trace_timing("shared_final_gradient_complete";dimension=q,finite=all(isfinite,grad))
    trace_path=get(ENV,"DRMODELS_CMR_TRACE_TIMING_FILE","")
    progress=isempty(trace_path) ? nothing :
        event->_cjs_trace_timing("curvature_gradient_complete";dimension=q,
            axis=event.axis,step=event.step)
    _cjs_trace_timing("shared_curvature_start";dimension=q)
    curvature_check=inner===nothing ? nothing : try
        _cjs_refined_curvature(t->_cjs_shared_gradient(t,prep;mode_cache=mode_cache),bhat;
            gradient_at_point=grad,progress=progress)
    catch err
        err isa InterruptException && rethrow()
        (hessian=fill(NaN,q,q),available=false,method=:refined_gradient_differences,
         reason=sprint(showerror,err))
    end
    _cjs_trace_timing("shared_curvature_complete";dimension=q,
        available=curvature_check!==nothing && curvature_check.available)
    H=curvature_check===nothing ? fill(NaN,q,q) : curvature_check.hessian
    eigen=all(isfinite,H) ? eigvals(Symmetric(H)) : fill(NaN,q)
    positive=all(isfinite,eigen) && minimum(eigen)>max(0.,maximum(eigen)*1e-10)
    gnorm=all(isfinite,grad) ? norm(grad,Inf) : Inf
    value=inner===nothing ? Inf : Float64(inner.objective)
    sd_formulas=any(m->m!==nothing && hasproperty(m,:sd_formula) && m.sd_formula!==nothing,(sh.phi,sh.p))
    scales=sd_formulas ?
        [(process=label,group=m.group,level=m.levels[j],log_sigma=log(sigma),sigma=sigma)
         for (label,m) in ((:phi,sh.phi),(:p,sh.p)) if m!==nothing
         for (j,sigma) in enumerate(_cjs_shared_sigmas(bhat,m))] :
        [(process=label,group=m.group,log_sigma=bhat[m.scale_index],sigma=exp(bhat[m.scale_index]))
         for (label,m) in ((:phi,sh.phi),(:p,sh.p)) if m!==nothing]
    ls,lp=_cjs_process_logs(bhat,prep)
    _,T=size(prep.data.Y); delta=[prep.data.intervals[mod1(row,T-1)] for row in prep.phi_design.rows]
    boundary=any(x->x.sigma<=1e-6,scales) || any(x->x<=1e-8||x>=1-1e-8,vcat(exp.(ls),exp.(ls.*delta),exp.(lp)))
    d=prep.data
    reason="unavailable: shared-effect Wald and bootstrap intervals remain guarded; eligible interior coefficient profiles use confint(method=:profile), with calibration unqualified and exact-zero SD limits guarded"
    diagnostics=(objective=value,optimizer=optimizer,method=:ML,converged=converged&&inner!==nothing,
        gradient=grad,gradient_norm=gnorm,hessian=H,curvature=eigen,positive_curvature=positive,
        curvature_check=curvature_check,
        boundary=boundary,variance_boundary=any(x->x.sigma<=1e-6,scales),attempts=attempts,
        integration=:joint_CJS_Laplace,inner=inner,inner_failures=inner_failures,
        shared=sh.diagnostics,shared_scales=scales,route=sh.route,
        terminal_constraint=prep.terminal_constraint,enrolled=sum(d.frequency),
        informative=sum(d.frequency[i] for i in eachindex(d.first) if any(view(d.observed,i,d.first[i]+1:d.end[i]))),
        individual_rows=length(d.ids),contributing_occasions=sum(d.frequency.*(d.end.-d.first)),
        surveyed_occasions=sum(d.frequency[i]*count(view(d.observed,i,d.first[i]+1:d.end[i])) for i in eachindex(d.first)),
        uncertainty_available=false,uncertainty_reason=reason,
        qualification=:numerical_route_only,profile_shape=:unavailable,
        variance_recovery=:unqualified,group_repetition_design=:not_run)
    co=sd_formulas ?
        (phi=bhat[1:prep.nphi],p=bhat[prep.nphi+1:prep.nphi+prep.np],
         sd_phi=sh.phi===nothing ? nothing : bhat[sh.phi.scale_range],
         sd_p=sh.p===nothing ? nothing : bhat[sh.p.scale_range]) :
        (phi=bhat[1:prep.nphi],p=bhat[prep.nphi+1:prep.nphi+prep.np],
         log_sigma_phi=sh.phi===nothing ? nothing : bhat[sh.phi.scale_index],
         log_sigma_p=sh.p===nothing ? nothing : bhat[sh.p.scale_index])
    return CJSFit(d,bhat,co,prep.phi_design,prep.p_design,-value,fill(NaN,q,q),diagnostics,prep)
end
