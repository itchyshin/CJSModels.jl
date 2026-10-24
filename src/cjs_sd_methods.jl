# SD formulas for persistent independent individual effects. Public dispatch
# stays in cjs_methods.jl; shared effects and reference selection are not routed here.

_cjs_has_sd_individual(prep) = _cjs_is_individual(prep) &&
    any(m -> m !== nothing && hasproperty(m,:sd_formula) && m.sd_formula !== nothing,
        (prep.individual.phi,prep.individual.p))

"Validate individual SD coefficient ranges, retained schemas and fit provenance."
function _cjs_sd_methods_guard(fit)
    prep=fit.prepared
    _cjs_is_individual(prep) && !_cjs_is_shared(prep) ||
        throw(ArgumentError("cjs SD methods require persistent individual effects; shared SD effects are unsupported"))
    re=prep.individual
    actual=re.phi===nothing ? :individual_p : re.p===nothing ? :individual_phi : :individual_pair
    (re.phi!==nothing || re.p!==nothing) && re.route==actual ||
        throw(ArgumentError("cjs individual SD route and active processes disagree"))
    hasproperty(prep,:prediction_contract) ||
        throw(ArgumentError("cjs individual SD methods require prediction provenance; refit the model"))
    c=prep.prediction_contract
    c isa NamedTuple && keys(c)==(:version,:selection_enabled,:sd_formula_present) &&
        c.version isa Integer && !(c.version isa Bool) && c.version==1 &&
        c.selection_enabled isa Bool && c.sd_formula_present isa NamedTuple &&
        keys(c.sd_formula_present)==(:phi,:p) && all(x->x isa Bool,values(c.sd_formula_present)) ||
        throw(ArgumentError("cjs individual SD prediction provenance is malformed; refit the model"))
    selected=hasproperty(prep,:selection) && prep.selection!==nothing
    present=(phi=re.phi!==nothing && hasproperty(re.phi,:sd_formula) && re.phi.sd_formula!==nothing,
             p=re.p!==nothing && hasproperty(re.p,:sd_formula) && re.p.sd_formula!==nothing)
    c.selection_enabled==selected && c.sd_formula_present==present ||
        throw(ArgumentError("cjs individual SD prediction provenance disagrees with the prepared model; refit the model"))
    selected && throw(ArgumentError("cjs reference-selection prediction and simulation are unsupported"))
    hasproperty(prep,:correlation) && prep.correlation!==nothing &&
        throw(ArgumentError("cjs correlated process effects are unsupported"))
    nfixed=prep.nphi+prep.np;N=length(fit.data.ids)
    length(fit.beta)==nfixed+re.nscale && all(isfinite,fit.beta) && isfinite(fit.loglik) ||
        throw(ArgumentError("cjs individual SD methods require finite coefficients, likelihood and matching dimensions"))
    diag=fit.diagnostics
    hasproperty(diag,:converged) && diag.converged===true &&
        hasproperty(diag,:gradient_norm) && diag.gradient_norm isa Real && !(diag.gradient_norm isa Bool) &&
        isfinite(diag.gradient_norm) && 0<=diag.gradient_norm<=1e-4 &&
        hasproperty(diag,:quadrature_refined) && diag.quadrature_refined===true ||
        throw(ArgumentError("cjs individual SD methods require a converged fit, finite gradient norm at most 1e-4, and refined fitting quadrature"))
    hasproperty(diag,:quadrature_order) && !(diag.quadrature_order in _cjs_individual_integration_orders(prep)) &&
        throw(ArgumentError("cjs individual SD methods require a supported quadrature order for the active effect dimension"))
    scales=Vector{Float64}[];cursor=nfixed+1
    for (label,m) in ((:phi,re.phi),(:p,re.p))
        m===nothing && (push!(scales,zeros(N));continue)
        hasproperty(m,:group) && m.group==:id &&
            all(name->hasproperty(m,name),(:scale_range,:scale_index,:sd_X,:sd_offset,:sd_names,:sd_formula)) ||
            throw(ArgumentError("cjs sd_$label fitted design provenance is absent; refit the model"))
        r=m.scale_range
        r isa UnitRange{<:Integer} && !isempty(r) && first(r)==cursor && last(r)<=length(fit.beta) &&
            m.scale_index==first(r) && size(m.sd_X)==(N,length(r)) && length(m.sd_offset)==N &&
            length(m.sd_names)==length(r) && all(isfinite,m.sd_X) && all(isfinite,m.sd_offset) ||
            throw(ArgumentError("cjs sd_$label coefficient range or fitted individual design is invalid; refit the model"))
        cursor=last(r)+1
        if m.sd_formula!==nothing
            hasproperty(m,:sd_design) && m.sd_design!==nothing ||
                throw(ArgumentError("cjs sd_$label fitted factor, contrast and offset schema is absent; refit the model"))
            des=m.sd_design
            all(name->hasproperty(des,name),(:formula,:rhs,:applied,:columns,:names,:offset_terms,:rows,:X,:offset)) &&
                isequal(des.formula,m.sd_formula) && des.names==m.sd_names &&
                des.rows==findall(fit.data.first .< fit.data.end) &&
                des.X==m.sd_X[des.rows,:] && des.offset==m.sd_offset[des.rows] ||
                throw(ArgumentError("cjs sd_$label retained fitted schema and individual design disagree; refit the model"))
        else
            length(r)==1 && all(==(1),m.sd_X) && all(iszero,m.sd_offset) ||
                throw(ArgumentError("cjs default sd_$label must retain the ordinary intercept design"))
        end
        sigma=exp.(m.sd_X*view(fit.beta,r)+m.sd_offset)
        all(x->isfinite(x) && x>=0,sigma) ||
            throw(ArgumentError("cjs sd_$label values must be finite and nonnegative"))
        push!(scales,sigma)
    end
    cursor==length(fit.beta)+1 || throw(ArgumentError("cjs individual SD coefficient ranges do not cover the fitted parameters"))
    return (phi=scales[1],p=scales[2])
end

_cjs_sd_predictor_names(des) = unique(vcat(StatsModels.termvars(des.rhs),
    [t.sym for t in des.offset_terms if t isa Term]))

"Reuse the retained SD schema on one constant supplied profile per individual."
function _cjs_sd_prediction_rows(fit,table,des,label,keynames;training=false)
    cols,n=_cjs_table_columns(table,label)
    m=getproperty(fit.prepared.individual,label)
    mean_names=_cjs_sd_predictor_names(des)
    sd_names=m===nothing || !hasproperty(m,:sd_formula) || m.sd_formula===nothing ?
        Symbol[] : _cjs_sd_predictor_names(m.sd_design)
    mean_allowed=unique(vcat(collect(keynames),mean_names,label==:phi ? [:interval_length] : Symbol[]))
    allowed=training ? mean_allowed : unique(vcat(mean_allowed,sd_names))
    Set(keys(cols))==Set(allowed) ||
        throw(ArgumentError("cjs $label prediction table must match the fitted mean and SD predictor schema; missing=$(collect(setdiff(Set(allowed),Set(keys(cols))))), extra=$(collect(setdiff(Set(keys(cols)),Set(allowed))))"))
    mean_table=NamedTuple{Tuple(mean_allowed)}(Tuple(getproperty(cols,k) for k in mean_allowed))
    rows=_cjs_prediction_rows(fit,des,mean_table,label,keynames)
    m===nothing && return (rows=rows,table=mean_table,sigma=zeros(n))
    if training
        index=Dict(id=>i for (i,id) in enumerate(fit.data.ids))
        return (rows=rows,table=mean_table,sigma=[_cjs_sd_value(m,fit.beta,index[id]) for id in cols.id])
    elseif !hasproperty(m,:sd_formula) || m.sd_formula===nothing
        return (rows=rows,table=mean_table,sigma=fill(exp(fit.beta[m.scale_index]),n))
    end
    ids=unique(collect(cols.id));id_index=Dict(id=>i for (i,id) in enumerate(ids))
    pairs=Pair{Symbol,Any}[:id=>ids]
    for name in sd_names
        v=getproperty(cols,name)
        any(ismissing,v) && throw(ArgumentError("cjs sd_$label prediction predictor $name cannot be missing"))
        profile=eltype(v)[]
        for id in ids
            values=v[findall(x->isequal(x,id),cols.id)]
            all(x->isequal(x,first(values)),values) ||
                throw(ArgumentError("cjs sd_$label prediction predictor $name must be constant within each individual"))
            push!(profile,first(values))
        end
        push!(pairs,name=>profile)
    end
    sdtable=(;pairs...);sdes=m.sd_design
    raw=try
        modelcols(sdes.applied,sdtable)
    catch err
        err isa InterruptException && rethrow()
        throw(ArgumentError("cjs sd_$label prediction predictors do not match the fitted factor/contrast schema; use known fitted levels: $(sprint(showerror,err))"))
    end
    raw=raw isa Tuple ? hcat(raw...) : raw
    X=raw isa AbstractMatrix ? Matrix{Float64}(raw) : reshape(Float64.(collect(raw)),:,1)
    X=X[:,sdes.columns]
    size(X)==(length(ids),length(m.scale_range)) && all(isfinite,X) ||
        throw(ArgumentError("cjs sd_$label prediction design is invalid"))
    off=zeros(length(ids))
    for term in sdes.offset_terms
        term isa Term || throw(ArgumentError("cjs sd_$label offset must name a numeric log-SD column"))
        v=getproperty(sdtable,term.sym)
        eltype(v)<:Real || throw(ArgumentError("cjs sd_$label offset must be numeric on the log-SD scale"))
        off .+= Float64.(v)
    end
    sigma=exp.(X*view(fit.beta,m.scale_range)+off)
    all(isfinite,off) && all(x->isfinite(x) && x>=0,sigma) ||
        throw(ArgumentError("cjs sd_$label prediction offsets and SDs must be finite"))
    return (rows=rows,table=mean_table,sigma=[sigma[id_index[id]] for id in cols.id])
end

function _cjs_sd_prediction_table(fit,des,profile,label,keynames,beta,response_type;rule=nothing)
    base=_cjs_prediction_table(fit,des,profile.table,label,keynames,beta,response_type)
    (!response_type || rule===nothing || all(iszero,profile.sigma)) && return base
    link=_cjs_prediction_table(fit,des,profile.table,label,keynames,beta,false)
    eta=link.eta;q=copy(label==:phi ? base.phi_unit : base.p)
    interval=label==:phi ? copy(base.phi_interval) : nothing
    constraint=fit.prepared.terminal_constraint
    terminal=constraint!==nothing && only(keys(constraint))==label ?
        (label==:phi ? profile.rows.cols.interval_end .==last(fit.data.occasions) :
                      profile.rows.cols.occasion .==last(fit.data.occasions)) : falses(length(eta))
    weights=exp.(rule.logweights)
    for i in eachindex(eta)
        (terminal[i] || profile.sigma[i]==0) && continue
        values=exp.(_cjs_logistic_log.(eta[i].+profile.sigma[i].*rule.nodes))
        q[i]=sum(weights.*values)
        # Same fixed normal quadrature rule evaluates all supplied profiles.
        label==:phi && (interval[i]=sum(weights.*values.^base.interval_length[i]))
    end
    return label==:phi ? merge(base,(phi_unit=q,phi_interval=interval)) : merge(base,(p=q,))
end

"SD-dependent individual process prediction with the ordinary explicit targets."
function _cjs_predict_sd_individual(fit,newdata;type=:response,target=nothing)
    _cjs_sd_methods_guard(fit)
    type in (:response,:link,"response","link") || throw(ArgumentError("cjs prediction type must be :response or :link"))
    response_type=type in (:response,"response")
    target in (:zero_effect,:integrated) ||
        throw(ArgumentError("cjs individual SD prediction requires explicit target=:zero_effect or :integrated"))
    target==:integrated && !response_type &&
        throw(ArgumentError("cjs integrated prediction is available on the response scale only"))
    training=newdata===nothing
    if training
        phi,p,phi_rows,p_rows=_cjs_training_prediction_tables(fit)
        phi=_cjs_append_predictors(phi,phi_rows,fit.data.phi_data,fit.phi_design)
        p=_cjs_append_predictors(p,p_rows,fit.data.p_data,fit.p_design)
        newdata=(;phi,p,units=fit.data.units)
    else
        newdata isa NamedTuple && all(k->haskey(newdata,k),(:phi,:p,:units)) ||
            throw(ArgumentError("cjs newdata must be (phi=<keyed table>, p=<keyed table>, units=<fit units>)"))
        newdata.units==fit.data.units || throw(ArgumentError("cjs prediction units must match the fitted interval units $(repr(fit.data.units))"))
    end
    phi_keys=(:id,:interval_start,:interval_end);p_keys=(:id,:occasion)
    ph=_cjs_sd_prediction_rows(fit,newdata.phi,fit.phi_design,:phi,phi_keys;training)
    pp=_cjs_sd_prediction_rows(fit,newdata.p,fit.p_design,:p,p_keys;training)
    function tables(rule=nothing)
        phi=_cjs_sd_prediction_table(fit,fit.phi_design,ph,:phi,phi_keys,view(fit.beta,1:fit.prepared.nphi),response_type;rule)
        p=_cjs_sd_prediction_table(fit,fit.p_design,pp,:p,p_keys,view(fit.beta,fit.prepared.nphi+1:fit.prepared.nphi+fit.prepared.np),response_type;rule)
        return (;phi,p)
    end
    result=tables()
    integration=(order=0,refined=true,tolerance=1e-8,maximum_probability_change=0.)
    if target==:integrated && (any(>(0),ph.sigma) || any(>(0),pp.sigma))
        previous=nothing
        for order in _CJS_INDIVIDUAL_1D_ORDERS
            result=tables(_cjs_individual_rule(order))
            values=vcat(result.phi.phi_unit,result.phi.phi_interval,result.p.p)
            change=previous===nothing ? Inf : maximum(abs.(values.-previous);init=0.)
            integration=(order=order,refined=change<=1e-8,tolerance=1e-8,maximum_probability_change=change)
            integration.refined && break
            previous=values
        end
        integration.refined || throw(ArgumentError("cjs integrated SD prediction quadrature is unrefined at order $(last(_CJS_INDIVIDUAL_1D_ORDERS)); returned-probability tolerance 1e-8 was not met"))
    end
    return merge(result,(target=target,mixing=target==:integrated ? :fitted_marked_cohort_normal : :zero_effect,
                         units=fit.data.units,integration=integration))
end

"Draw independent SD-dependent persistent individual effects for each frequency copy."
function _cjs_simulate_sd_individual(fit;nsim::Integer=1,seed=nothing,rng=Random.default_rng())
    scales=_cjs_sd_methods_guard(fit)
    !(nsim isa Bool) && nsim>=1 || throw(ArgumentError("nsim must be a positive integer"))
    rng isa Random.AbstractRNG || throw(ArgumentError("rng must be an AbstractRNG"))
    drawrng=_cjs_random(rng,seed);d=fit.data;N,T=size(d.Y)
    ep=fit.phi_design.X*view(fit.beta,1:fit.prepared.nphi)+fit.phi_design.offset
    eq=fit.p_design.X*view(fit.beta,fit.prepared.nphi+1:fit.prepared.nphi+fit.prepared.np)+fit.p_design.offset
    all(isfinite,ep) && all(isfinite,eq) || throw(ArgumentError("cjs simulation link values must be finite"))
    pi=Dict(row=>k for (k,row) in enumerate(fit.phi_design.rows))
    qi=Dict(row=>k for (k,row) in enumerate(fit.p_design.rows))
    expanded=[(i,r) for i in 1:N for r in 1:d.frequency[i]]
    ids=[(d.ids[i],r) for (i,r) in expanded]
    function expand_keyed(table)
        cols=Tables.columntable(table);block=length(first(values(cols)))÷N
        pairs=Pair{Symbol,Any}[]
        for key in keys(cols)
            v=key==:id ? [(d.ids[i],r) for (i,r) in expanded for _ in 1:block] :
                [getproperty(cols,key)[(i-1)*block+k] for (i,_) in expanded for k in 1:block]
            push!(pairs,key=>v)
        end
        return (;pairs...)
    end
    out=CMRData[]
    for _ in 1:nsim
        Y=zeros(Int,length(expanded),T);observed=BitMatrix(undef,length(expanded),T)
        for (j,(i,_)) in enumerate(expanded)
            observed[j,:].=d.observed[i,:]
            Y[j,1:d.first[i]].=d.Y[i,1:d.first[i]]
            up=_cjs_individual_draw(drawrng,scales.phi[i]);uq=_cjs_individual_draw(drawrng,scales.p[i])
            alive=true
            for t in d.first[i]:d.end[i]-1
                kp=pi[(i-1)*(T-1)+t]
                phi=_cjs_individual_probability(ep[kp],scales.phi[i],up,
                    fit.prepared.phi_terminal[kp],fit.prepared.terminal_constraint,:phi)^d.intervals[t]
                alive=alive && rand(drawrng)<phi
                if alive && d.observed[i,t+1]
                    kq=qi[(i-1)*T+t+1]
                    p=_cjs_individual_probability(eq[kq],scales.p[i],uq,
                        fit.prepared.p_terminal[kq],fit.prepared.terminal_constraint,:p)
                    Y[j,t+1]=rand(drawrng)<p
                end
            end
        end
        # Preserve normalized keyed blocks, including noncontributing rows.
        phi_data=expand_keyed(d.phi_data);p_data=expand_keyed(d.p_data)
        push!(out,CMRData(Y,observed,copy(ids),copy(d.occasions),copy(d.intervals),
            [d.first[i] for (i,_) in expanded],[d.end[i] for (i,_) in expanded],
            ones(Int,length(expanded)),d.units,phi_data,p_data))
    end
    return out
end
