include("cjs_shared_methods.jl")
include("cjs_sd_methods.jl")

# Reader-facing methods for the native first-capture-conditional CJS fit.

using Random: Random
using Distributions: Chisq, Normal, quantile

function _cjs_table_columns(table, label)
    cols = try
        Tables.columntable(table)
    catch
        throw(ArgumentError("cjs $label prediction data must be a named table"))
    end
    isempty(cols) && throw(ArgumentError("cjs $label prediction data cannot be empty"))
    n = length(first(values(cols)))
    all(v -> length(v) == n, values(cols)) || throw(ArgumentError("cjs $label prediction columns have unequal lengths"))
    return cols, n
end

function _cjs_prediction_rows(fit, des, table, label, keynames)
    cols, n = _cjs_table_columns(table, label)
    all(k -> haskey(cols, k), keynames) || throw(ArgumentError("cjs $label prediction table requires keys $(join(keynames, ", "))"))
    indices = Dict{Tuple,Int}()
    for i in 1:n
        key = Tuple(getproperty(cols, k)[i] for k in keynames)
        any(ismissing, key) && throw(ArgumentError("cjs $label prediction keys cannot be missing"))
        haskey(indices, key) && throw(ArgumentError("cjs $label prediction keys must be unique"))
        indices[key] = i
    end
    predictor_names = StatsModels.termvars(des.rhs)
    offset_names = [term.sym for term in des.offset_terms if term isa Term]
    allowed_names = unique(vcat(collect(keynames), predictor_names, offset_names,
                                label == :phi ? [:interval_length] : Symbol[]))
    Set(keys(cols)) == Set(allowed_names) ||
        throw(ArgumentError("cjs $label prediction table does not match the fitted predictor schema; missing=$(collect(setdiff(Set(allowed_names), Set(keys(cols))))), extra=$(collect(setdiff(Set(keys(cols)), Set(allowed_names))))"))
    for name in predictor_names
        haskey(cols, name) || throw(ArgumentError("cjs $label prediction predictor $name is absent"))
        any(ismissing, getproperty(cols, name)) && throw(ArgumentError("cjs $label prediction predictor $name cannot be missing"))
    end
    sub = NamedTuple{keys(cols)}(Tuple(collect(getproperty(cols, k)) for k in keys(cols)))
    applied = des.applied
    raw = try
        modelcols(applied, sub)
    catch err
        err isa InterruptException && rethrow()
        throw(ArgumentError("cjs $label prediction predictors do not match the fitted schema: $(sprint(showerror, err))"))
    end
    allcols = raw isa Tuple ? hcat(raw...) : raw
    allcols = allcols isa AbstractMatrix ? Matrix{Float64}(allcols) : reshape(Float64.(collect(allcols)), :, 1)
    selected = allcols[:, des.columns]
    size(selected, 2) == length(des.names) || throw(ArgumentError("cjs $label prediction coefficient columns do not match the fitted design"))
    all(isfinite, selected) || throw(ArgumentError("cjs $label prediction design must be finite"))
    offset = zeros(Float64, n)
    for term in des.offset_terms
        term isa Term && haskey(cols, term.sym) || throw(ArgumentError("cjs $label prediction offset column is absent"))
        values = getproperty(cols, term.sym)
        any(ismissing, values) && throw(ArgumentError("cjs $label prediction offset cannot be missing"))
        offset .+= Float64.(values)
    end
    all(isfinite, offset) || throw(ArgumentError("cjs $label prediction offsets must be finite"))
    return (cols=cols, n=n, index=indices, X=selected, offset=offset)
end

function _cjs_training_prediction_tables(fit)
    d = fit.data
    phi_rows = [((i - 1) * (length(d.occasions) - 1) + t, i, t)
                for i in eachindex(d.ids) for t in d.first[i]:d.end[i]-1]
    p_rows = [((i - 1) * length(d.occasions) + t, i, t)
              for i in eachindex(d.ids) for t in d.first[i]+1:d.end[i] if d.observed[i, t]]
    phi = (id=[d.ids[i] for (_, i, _) in phi_rows],
           interval_start=[d.occasions[t] for (_, _, t) in phi_rows],
           interval_end=[d.occasions[t+1] for (_, _, t) in phi_rows],
           interval_length=[d.intervals[t] for (_, _, t) in phi_rows])
    p = (id=[d.ids[i] for (_, i, _) in p_rows],
         occasion=[d.occasions[t] for (_, _, t) in p_rows])
    return phi, p, phi_rows, p_rows
end

function _cjs_append_predictors(table, rows, data_table, des)
    offset_names = [term.sym for term in des.offset_terms if term isa Term]
    names = unique(vcat(StatsModels.termvars(des.rhs), offset_names))
    pairs = Pair{Symbol,Any}[k => getproperty(table, k) for k in keys(table)]
    for name in names
        push!(pairs, name => [getproperty(data_table, name)[j] for (j, _, _) in rows])
    end
    return (; pairs...)
end

# Methods admission uses CJS convergence and integration evidence, not the
# curvature/latent Hessian tests used for a different model class.
function _cjs_individual_methods_guard(fit)
    prep = fit.prepared
    _cjs_is_shared(prep) && throw(ArgumentError("cjs shared-plus-individual methods are unsupported"))
    re = prep.individual
    re.route in (:individual_phi, :individual_p, :individual_pair) ||
        throw(ArgumentError("cjs methods support ordinary individual_phi, individual_p or individual_pair only"))
    actual_route = re.phi === nothing ? :individual_p : re.p === nothing ? :individual_phi : :individual_pair
    (re.phi !== nothing || re.p !== nothing) && re.route == actual_route ||
        throw(ArgumentError("cjs individual route and active processes disagree"))
    hasproperty(prep, :prediction_contract) ||
        throw(ArgumentError("cjs individual methods require version-one prediction provenance; refit the model"))
    provenance = prep.prediction_contract
    provenance isa NamedTuple && keys(provenance) == (:version, :selection_enabled, :sd_formula_present) ||
        throw(ArgumentError("cjs individual prediction provenance is malformed; refit the model"))
    provenance.version isa Integer && !(provenance.version isa Bool) && provenance.version == 1 &&
        provenance.selection_enabled isa Bool && provenance.sd_formula_present isa NamedTuple &&
        keys(provenance.sd_formula_present) == (:phi, :p) &&
        all(x -> x isa Bool, values(provenance.sd_formula_present)) ||
        throw(ArgumentError("cjs individual prediction provenance is malformed; refit the model"))
    actual_selection = hasproperty(prep, :selection) && prep.selection !== nothing
    actual_sd = (phi=re.phi !== nothing && hasproperty(re.phi, :sd_formula) && re.phi.sd_formula !== nothing,
                 p=re.p !== nothing && hasproperty(re.p, :sd_formula) && re.p.sd_formula !== nothing)
    provenance.selection_enabled == actual_selection && provenance.sd_formula_present == actual_sd ||
        throw(ArgumentError("cjs individual prediction provenance disagrees with the prepared model; refit the model"))
    actual_selection && throw(ArgumentError("cjs reference-selection prediction and simulation are unsupported"))
    any(values(actual_sd)) && throw(ArgumentError("cjs prediction and simulation with supplied SD formulas (including ~1) are unsupported"))
    nfixed = prep.nphi + prep.np
    length(fit.beta) == nfixed + re.nscale && re.nscale == count(!isnothing, (re.phi, re.p)) ||
        throw(ArgumentError("cjs individual methods parameter dimensions disagree with the ordinary route"))
    all(isfinite, fit.beta) && isfinite(fit.loglik) ||
        throw(ArgumentError("cjs individual methods require finite coefficients and log likelihood"))
    diag = fit.diagnostics
    hasproperty(diag, :converged) && diag.converged === true &&
        hasproperty(diag, :gradient_norm) && diag.gradient_norm isa Real && !(diag.gradient_norm isa Bool) &&
        isfinite(diag.gradient_norm) && 0 <= diag.gradient_norm <= 1e-4 &&
        hasproperty(diag, :quadrature_refined) && diag.quadrature_refined === true ||
        throw(ArgumentError("cjs individual methods require a converged fit, finite gradient norm at most 1e-4, and refined fitting quadrature"))
    hasproperty(diag,:quadrature_order) && !(diag.quadrature_order in _cjs_individual_integration_orders(prep)) &&
        throw(ArgumentError("cjs individual methods require a supported quadrature order for the active effect dimension"))
    active_indices = [m.scale_index for m in (re.phi,re.p) if m !== nothing && hasproperty(m,:scale_index)]
    length(active_indices) == re.nscale && length(unique(active_indices)) == re.nscale ||
        throw(ArgumentError("cjs individual scale indices are missing or duplicated"))
    scales = Float64[]
    for m in (re.phi, re.p)
        if m === nothing
            push!(scales, 0.)
            continue
        end
        hasproperty(m, :scale_index) && m.scale_index isa Integer &&
            nfixed < m.scale_index <= length(fit.beta) ||
            throw(ArgumentError("cjs ordinary individual scale provenance is malformed"))
        if hasproperty(m, :scale_range)
            length(m.scale_range) == 1 && only(m.scale_range) == m.scale_index ||
                throw(ArgumentError("cjs supplied SD designs are unsupported by these methods"))
        end
        if hasproperty(m, :sd_X)
            size(m.sd_X) == (length(fit.data.ids),1) && all(==(1),m.sd_X) &&
                hasproperty(m,:sd_offset) && all(iszero,m.sd_offset) ||
                throw(ArgumentError("cjs ordinary individual SD design provenance is malformed"))
        end
        sigma = exp(fit.beta[m.scale_index])
        isfinite(sigma) && sigma >= 0 || throw(ArgumentError("cjs individual SDs must be finite and nonnegative"))
        push!(scales, sigma)
    end
    return (phi=scales[1], p=scales[2])
end

function _cjs_reference_selection_guard(fit)
    prep=fit.prepared
    _cjs_is_shared(prep) && throw(ArgumentError("cjs shared-effect reference-selection prediction is unsupported"))
    hasproperty(prep,:selection) && prep.selection!==nothing ||
        throw(ArgumentError("cjs reference-selected prediction requires a fitted reference-selection model"))
    re=prep.individual
    re.route in (:individual_phi,:individual_p,:individual_pair) ||
        throw(ArgumentError("cjs reference-selected prediction requires an individual-effect route"))
    actual_route=re.phi===nothing ? :individual_p : re.p===nothing ? :individual_phi : :individual_pair
    re.route==actual_route ||
        throw(ArgumentError("cjs reference-selected prediction route and active processes disagree"))
    hasproperty(prep,:prediction_contract) ||
        throw(ArgumentError("cjs reference-selected prediction requires version-one selection provenance; refit the model"))
    provenance=prep.prediction_contract
    provenance isa NamedTuple && keys(provenance)==(:version,:selection_enabled,:sd_formula_present) &&
        provenance.version isa Integer && !(provenance.version isa Bool) && provenance.version==1 &&
        provenance.selection_enabled isa Bool && provenance.selection_enabled===true &&
        provenance.sd_formula_present isa NamedTuple && keys(provenance.sd_formula_present)==(:phi,:p) &&
        all(x->x isa Bool,values(provenance.sd_formula_present)) ||
        throw(ArgumentError("cjs reference-selected prediction provenance is malformed; refit the model"))
    actual_sd=(phi=re.phi!==nothing && hasproperty(re.phi,:sd_formula) && re.phi.sd_formula!==nothing,
               p=re.p!==nothing && hasproperty(re.p,:sd_formula) && re.p.sd_formula!==nothing)
    provenance.sd_formula_present==actual_sd ||
        throw(ArgumentError("cjs reference-selected prediction provenance disagrees with the prepared model; refit the model"))
    nfixed=prep.nphi+prep.np
    length(fit.beta)==nfixed+re.nscale && all(isfinite,fit.beta) && isfinite(fit.loglik) ||
        throw(ArgumentError("cjs reference-selected prediction parameter dimensions or values are invalid"))
    scale_indices=Int[]
    for m in (re.phi,re.p)
        m===nothing && continue
        if hasproperty(m,:scale_range)
            append!(scale_indices,m.scale_range)
        elseif hasproperty(m,:scale_index)
            push!(scale_indices,m.scale_index)
        else
            throw(ArgumentError("cjs reference-selected prediction SD parameter provenance is missing"))
        end
    end
    sort(scale_indices)==collect(nfixed+1:length(fit.beta)) ||
        throw(ArgumentError("cjs reference-selected prediction SD parameter indices are malformed"))
    diag=fit.diagnostics
    hasproperty(diag,:converged) && diag.converged===true && hasproperty(diag,:gradient_norm) &&
        diag.gradient_norm isa Real && !(diag.gradient_norm isa Bool) &&
        isfinite(diag.gradient_norm) && 0<=diag.gradient_norm<=1e-4 &&
        hasproperty(diag,:quadrature_refined) && diag.quadrature_refined===true ||
        throw(ArgumentError("cjs reference-selected prediction requires a converged fit and refined fitting quadrature"))
    return nothing
end

function _cjs_selection_profile_mean_table(fit,table,label,keynames,extra_names)
    cols,n=_cjs_table_columns(table,"selection $label")
    mean_names=unique(vcat(collect(keynames),StatsModels.termvars(getproperty(fit,Symbol(label,:_design)).rhs),
        [term.sym for term in getproperty(fit,Symbol(label,:_design)).offset_terms if term isa Term],
        label==:phi ? [:interval_length] : Symbol[]))
    allowed=unique(vcat(mean_names,extra_names))
    Set(keys(cols))==Set(allowed) ||
        throw(ArgumentError("cjs selection $label table must match its mean and selection fields; missing=$(collect(setdiff(Set(allowed),Set(keys(cols))))), extra=$(collect(setdiff(Set(keys(cols)),Set(allowed))))"))
    mean=NamedTuple{Tuple(mean_names)}(Tuple(getproperty(cols,k) for k in mean_names))
    if n==0
        X=zeros(Float64,0,size(getproperty(fit,Symbol(label,:_design)).X,2))
        rows=(cols=mean,n=0,index=Dict{Tuple,Int}(),X,offset=Float64[])
    else
        rows=_cjs_prediction_rows(fit,getproperty(fit,Symbol(label,:_design)),mean,label,keynames)
    end
    return rows,cols
end

function _cjs_reference_profile_logweight(fit,profile,beta,zphi,zp)
    d=fit.data;constraint=fit.prepared.terminal_constraint
    logw=zero(eltype(beta))
    for j in eachindex(profile.phi_eta)
        eta=profile.phi_eta[j]+profile.sigma_phi*zphi
        if constraint!==nothing && only(keys(constraint))==:phi &&
           profile.phi_end[j]==last(d.occasions)
            logw+=profile.phi_delta[j]*log(only(values(constraint)))
        else
            logw+=profile.phi_delta[j]*_cjs_logistic_log(eta)
        end
    end
    for j in eachindex(profile.p_eta)
        profile.p_observed[j] || continue
        eta=profile.p_eta[j]+profile.sigma_p*zp
        captured=profile.p_capture[j]==1
        occasion=profile.p_occasion[j]
        if constraint!==nothing && only(keys(constraint))==:p && occasion==last(d.occasions)
            c=only(values(constraint))
            logw+=captured ? log(c) : log1p(-c)
        else
            logw+=captured ? _cjs_logistic_log(eta) : _cjs_logistic_log(-eta)
        end
    end
    return logw
end

function _cjs_reference_selection_profiles(fit,newdata,phi_data,p_data,training)
    ids=unique(vcat(collect(phi_data.rows.cols.id),collect(p_data.rows.cols.id)))
    isempty(ids) && throw(ArgumentError("cjs reference-selected prediction requires at least one keyed individual"))
    supplied=!training && haskey(newdata,:selection)
    if !supplied
        profiles=Dict{Any,Any}();re=fit.prepared.individual
        for id in ids
            i=findfirst(isequal(id),fit.data.ids)
            i===nothing && throw(ArgumentError("cjs reference-selected prediction for new ID $(repr(id)) requires selection=(first, phi, p) with complete prehistory"))
            profiles[id]=(fit_index=i,
                sigma_phi=re.phi===nothing ? 0. : _cjs_sd_value(re.phi,fit.beta,i),
                sigma_p=re.p===nothing ? 0. : _cjs_sd_value(re.p,fit.beta,i))
        end
        return profiles
    end
    sigma_phi=Dict(id=>phi_data.sigma[findfirst(x->isequal(x,id),phi_data.rows.cols.id)] for id in ids)
    sigma_p=Dict(id=>p_data.sigma[findfirst(x->isequal(x,id),p_data.rows.cols.id)] for id in ids)

    selection=newdata.selection
    selection isa NamedTuple && all(k->haskey(selection,k),(:first,:phi,:p)) ||
        throw(ArgumentError("cjs newdata.selection must be (first=(id, first_index, eligibility_index), phi=<prehistory table>, p=<survey mask and capture table>)"))
    first_cols,nfirst=_cjs_table_columns(selection.first,"selection first")
    Set(keys(first_cols))==Set((:id,:first_index,:eligibility_index)) ||
        throw(ArgumentError("cjs selection first table must contain exactly id, first_index, and eligibility_index"))
    first_rows=Dict{Any,Tuple{Int,Int}}()
    for r in 1:nfirst
        id=first_cols.id[r];id===missing && throw(ArgumentError("cjs selection first IDs cannot be missing"))
        haskey(first_rows,id) && throw(ArgumentError("cjs selection first IDs must be unique"))
        f=first_cols.first_index[r];e=first_cols.eligibility_index[r]
        all(x->x isa Real && !(x isa Bool) && isfinite(x) && isinteger(x),(f,e)) ||
            throw(ArgumentError("cjs selection first_index and eligibility_index must be finite whole-number occasion indices"))
        f=Int(f);e=Int(e)
        1<=e<=f<=length(fit.data.occasions) ||
            throw(ArgumentError("cjs selection eligibility and first capture must satisfy 1 <= eligibility_index <= first_index <= number of fitted occasions"))
        first_rows[id]=(f,e)
    end
    all(id->haskey(first_rows,id),ids) ||
        throw(ArgumentError("cjs newdata.selection.first must declare an eligibility and first-capture occasion for every predicted ID"))

    re=fit.prepared.individual
    phi_sd_names=re.phi===nothing || !hasproperty(re.phi,:sd_formula) || re.phi.sd_formula===nothing ?
        Symbol[] : _cjs_sd_predictor_names(re.phi.sd_design)
    p_sd_names=re.p===nothing || !hasproperty(re.p,:sd_formula) || re.p.sd_formula===nothing ?
        Symbol[] : _cjs_sd_predictor_names(re.p.sd_design)
    phi_rows,phi_cols=_cjs_selection_profile_mean_table(fit,selection.phi,:phi,
        (:id,:interval_start,:interval_end),phi_sd_names)
    p_rows,p_cols=_cjs_selection_profile_mean_table(fit,selection.p,:p,
        (:id,:occasion),vcat([:observed,:capture],p_sd_names))
    for (label,cols,names) in ((:phi,phi_cols,phi_sd_names),(:p,p_cols,p_sd_names))
        for name in names
            haskey(cols,name) || throw(ArgumentError("cjs selection $label prehistory must supply SD predictor $name"))
            for id in ids
                rowids=getproperty(cols,:id)
                ix=findall(x->isequal(x,id),rowids)
                isempty(ix) && continue
                v=getproperty(cols,name)[ix]
                all(x->isequal(x,first(v)),v) || throw(ArgumentError("cjs selection sd_$label predictor $name must be constant within each ID"))
            end
        end
    end
    for (label,cols,names,future) in ((:phi,phi_cols,phi_sd_names,phi_data),(:p,p_cols,p_sd_names,p_data))
        for name in names, id in ids
            pre=findall(x->isequal(x,id),getproperty(cols,:id))
            isempty(pre) && continue
            fut=findfirst(x->isequal(x,id),future.rows.cols.id)
            expected=getproperty(future.rows.cols,name)[fut]
            all(r->isequal(getproperty(cols,name)[r],expected),pre) ||
                throw(ArgumentError("cjs selection and prediction sd_$label profiles disagree for ID $(repr(id))"))
        end
    end

    T=length(fit.data.occasions);occasion=fit.data.occasions
    beta=view(fit.beta,1:fit.prepared.nphi+fit.prepared.np)
    profiles=Dict{Any,Any}()
    for id in ids
        f,e=first_rows[id]
        expected_phi=[(id,occasion[t],occasion[t+1]) for t in e:f-1]
        expected_p=[(id,occasion[t]) for t in e:f]
        phi_keys=[(phi_cols.id[r],phi_cols.interval_start[r],phi_cols.interval_end[r]) for r in eachindex(phi_cols.id) if isequal(phi_cols.id[r],id)]
        p_ids=collect(p_cols.id);p_occasions=collect(p_cols.occasion)
        p_keys=[(p_ids[r],p_occasions[r]) for r in eachindex(p_ids) if isequal(p_ids[r],id)]
        Set(phi_keys)==Set(expected_phi) || throw(ArgumentError("cjs selection phi prehistory for ID $(repr(id)) must contain exactly one row for every interval from eligibility through first capture"))
        Set(p_keys)==Set(expected_p) || throw(ArgumentError("cjs selection p prehistory for ID $(repr(id)) must contain exactly one surveyed/missed row for every occasion from eligibility through first capture"))
        phi_eta=Float64[];phi_delta=Float64[];phi_end=Any[]
        for key in expected_phi
            r=phi_rows.index[key]
            delta=phi_cols.interval_length[r]
            delta isa Real && !(delta isa Bool) && isfinite(delta) && delta>0 ||
                throw(ArgumentError("cjs selection phi interval_length must be finite and positive"))
            k=e+length(phi_eta)
            isapprox(Float64(delta),Float64(fit.data.intervals[k]);rtol=1e-10,atol=1e-12) ||
                throw(ArgumentError("cjs selection phi interval_length must match the fitted occasion schedule"))
            push!(phi_eta,dot(view(phi_rows.X,r,:),view(beta,1:fit.prepared.nphi))+phi_rows.offset[r])
            push!(phi_delta,Float64(delta));push!(phi_end,phi_cols.interval_end[r])
        end
        p_eta=Float64[];p_observed=Bool[];p_capture=Union{Missing,Int}[];p_occasion=Any[]
        for (k,key) in enumerate(expected_p)
            r=p_rows.index[key];observed=p_cols.observed[r];captured=p_cols.capture[r]
            observed isa Bool || throw(ArgumentError("cjs selection p observed values must be true or false"))
            if observed
                captured isa Integer && !(captured isa Bool) && captured in (0,1) ||
                    throw(ArgumentError("cjs surveyed selection captures must be 0 before first capture and 1 at first capture"))
                captured==(k==length(expected_p) ? 1 : 0) ||
                    throw(ArgumentError("cjs selection history must contain only nondetections before the declared first capture and a detection at that occasion"))
            else
                ismissing(captured) || throw(ArgumentError("cjs unsurveyed selection occasions must have missing capture values"))
                k==length(expected_p) && throw(ArgumentError("cjs first-capture occasion must be surveyed"))
            end
            push!(p_eta,dot(view(p_rows.X,r,:),view(beta,fit.prepared.nphi+1:fit.prepared.nphi+fit.prepared.np))+p_rows.offset[r])
            push!(p_observed,observed);push!(p_capture,ismissing(captured) ? missing : Int(captured));push!(p_occasion,p_cols.occasion[r])
        end
        profiles[id]=(fit_index=nothing,sigma_phi=sigma_phi[id],sigma_p=sigma_p[id],
            phi_eta,phi_delta,phi_end,p_eta,p_observed,p_capture,p_occasion)
    end
    return profiles
end

function _cjs_reference_selected_table(fit,rows,label,keynames,beta,rule,profiles,prediction_sigma)
    eta=rows.X*beta+rows.offset
    T=length(fit.data.occasions);constraint=fit.prepared.terminal_constraint
    terminal=constraint!==nothing && only(keys(constraint))==label ?
        (label==:phi ? [last(fit.data.occasions)==x for x in rows.cols.interval_end] :
                       [last(fit.data.occasions)==x for x in rows.cols.occasion]) : falses(rows.n)
    if any(terminal)
        c=only(values(constraint));eta[terminal].=log(c/(1-c))
    end
    delta=label==:phi ? Float64.(rows.cols.interval_length) : ones(rows.n)
    out=Pair{Symbol,Any}[k=>getproperty(rows.cols,k) for k in keynames]
    label==:phi && push!(out,:interval_length=>delta)
    probabilities=similar(eta);unit=label==:phi ? similar(eta) : nothing
    re=fit.prepared.individual;m=label==:phi ? re.phi : re.p
    for r in 1:rows.n
        id=getproperty(rows.cols,:id)[r]
        haskey(profiles,id) || throw(ArgumentError("cjs selection profile is absent for prediction ID $(repr(id))"))
        profile=profiles[id]
        if !terminal[r] && profile.fit_index!==nothing && m!==nothing &&
           hasproperty(m,:sd_formula) && m.sd_formula!==nothing
            hasproperty(m,:sd_design) && m.sd_design!==nothing && profile.fit_index in m.sd_design.rows ||
                throw(ArgumentError("cjs reference-selected $label prediction for ID $(repr(id)) has no fitted SD profile; omit this row or supply complete SD predictors"))
        end
        logweights=fill(-Inf,length(rule.nodes),length(rule.nodes))
        for a in eachindex(rule.nodes),b in eachindex(rule.nodes)
            logselection=if profile.fit_index===nothing
                _cjs_reference_profile_logweight(fit,profile,view(fit.beta,1:fit.prepared.nphi+fit.prepared.np),rule.nodes[a],rule.nodes[b])
            else
                _cjs_individual_selection_logweight(profile.fit_index,a,b,
                    view(fit.beta,1:fit.prepared.nphi+fit.prepared.np),profile.sigma_phi,profile.sigma_p,
                    fit.prepared,rule.nodes,rule.nodes)
            end
            logweights[a,b]=rule.logweights[a]+rule.logweights[b]+logselection
        end
        normalizer=foldl(_cjs_logadd,vec(logweights);init=oftype(first(logweights),-Inf))
        isfinite(normalizer) || throw(ArgumentError("cjs selection event has zero or non-finite prediction weight for ID $(repr(id))"))
        total=zero(eltype(eta));unit_total=zero(eltype(eta));sigma=prediction_sigma[r]
        for a in eachindex(rule.nodes),b in eachindex(rule.nodes)
            weight=exp(logweights[a,b]-normalizer)
            u=m===nothing ? 0. : label==:phi ? rule.nodes[a] : rule.nodes[b]
            q=terminal[r] ? only(values(constraint)) : exp(_cjs_logistic_log(eta[r]+sigma*u))
            total+=weight*(label==:phi ? q^delta[r] : q)
            label==:phi && (unit_total+=weight*q)
        end
        probabilities[r]=total
        label==:phi && (unit[r]=unit_total)
    end
    if label==:phi
        push!(out,:phi_unit=>unit);push!(out,:phi_interval=>probabilities)
    else
        push!(out,:p=>probabilities)
    end
    return (;out...)
end

function _cjs_predict_reference_selected(fit,newdata)
    _cjs_reference_selection_guard(fit)
    training=newdata===nothing
    if newdata===nothing
        phi,p,phi_rows,p_rows=_cjs_training_prediction_tables(fit)
        phi=_cjs_append_predictors(phi,phi_rows,fit.data.phi_data,fit.phi_design)
        p=_cjs_append_predictors(p,p_rows,fit.data.p_data,fit.p_design)
        newdata=(;phi,p,units=fit.data.units)
    else
        newdata isa NamedTuple && all(k->haskey(newdata,k),(:phi,:p,:units)) ||
            throw(ArgumentError("cjs newdata must be (phi=<keyed table>, p=<keyed table>, units=<fit units>)"))
        newdata.units==fit.data.units || throw(ArgumentError("cjs prediction units must match the fitted interval units $(repr(fit.data.units))"))
    end
    beta_phi=view(fit.beta,1:fit.prepared.nphi)
    beta_p=view(fit.beta,fit.prepared.nphi+1:fit.prepared.nphi+fit.prepared.np)
    supplied=!training && haskey(newdata,:selection)
    if supplied
        phi_data=_cjs_sd_prediction_rows(fit,newdata.phi,fit.phi_design,:phi,
            (:id,:interval_start,:interval_end);training=false)
        p_data=_cjs_sd_prediction_rows(fit,newdata.p,fit.p_design,:p,
            (:id,:occasion);training=false)
    else
        phi_rows=_cjs_prediction_rows(fit,fit.phi_design,newdata.phi,:phi,
            (:id,:interval_start,:interval_end))
        p_rows=_cjs_prediction_rows(fit,fit.p_design,newdata.p,:p,(:id,:occasion))
        phi_data=(rows=phi_rows,sigma=[begin
            i=findfirst(isequal(id),fit.data.ids)
            i===nothing ? 0. : fit.prepared.individual.phi===nothing ? 0. : _cjs_sd_value(fit.prepared.individual.phi,fit.beta,i)
        end for id in phi_rows.cols.id])
        p_data=(rows=p_rows,sigma=[begin
            i=findfirst(isequal(id),fit.data.ids)
            i===nothing ? 0. : fit.prepared.individual.p===nothing ? 0. : _cjs_sd_value(fit.prepared.individual.p,fit.beta,i)
        end for id in p_rows.cols.id])
    end
    profiles=_cjs_reference_selection_profiles(fit,newdata,phi_data,p_data,training)
    function tables(rule)
        phi=_cjs_reference_selected_table(fit,phi_data.rows,:phi,
            (:id,:interval_start,:interval_end),beta_phi,rule,profiles,phi_data.sigma)
        p=_cjs_reference_selected_table(fit,p_data.rows,:p,
            (:id,:occasion),beta_p,rule,profiles,p_data.sigma)
        return (;phi,p)
    end
    previous=nothing;result=nothing
    integration=(order=0,refined=true,tolerance=1e-8,maximum_probability_change=0.)
    for order in _CJS_INDIVIDUAL_PAIRED_ORDERS
        result=tables(_cjs_individual_rule(order))
        values=vcat(result.phi.phi_unit,result.phi.phi_interval,result.p.p)
        change=previous===nothing ? Inf : maximum(abs.(values.-previous);init=0.)
        integration=(order=order,refined=change<=1e-8,tolerance=1e-8,
                     maximum_probability_change=change)
        integration.refined && break
        previous=values
    end
    integration.refined || throw(ArgumentError("cjs reference-selected prediction quadrature is unrefined at order 97; returned-probability tolerance 1e-8 was not met"))
    return merge(result,(target=:reference_selected,mixing=:first_capture_selection_weighted_normal,
        units=fit.data.units,integration))
end

# A common-terminal constraint replaces the unit probability before powers.
_cjs_individual_probability(eta, sigma, u, terminal, constraint, label) =
    terminal && constraint !== nothing && only(keys(constraint)) == label ?
        only(values(constraint)) : exp(_cjs_logistic_log(eta + sigma*u))

"Internal pure probabilities for injected persistent independent process draws."
function _cjs_individual_probabilities(eta_phi, eta_p, delta, sigma_phi, sigma_p, u_phi, u_p;
                                     terminal_phi=falses(length(eta_phi)),
                                     terminal_p=falses(length(eta_p)), constraint=nothing)
    length(delta) == length(eta_phi) == length(terminal_phi) && length(eta_p) == length(terminal_p) ||
        throw(ArgumentError("cjs individual probability rows do not align"))
    return (phi_interval=[_cjs_individual_probability(eta_phi[k],sigma_phi,u_phi,
                terminal_phi[k],constraint,:phi)^delta[k] for k in eachindex(delta)],
            p=[_cjs_individual_probability(eta_p[k],sigma_p,u_p,
                terminal_p[k],constraint,:p) for k in eachindex(eta_p)])
end

# Zero SD and inactive processes consume no normal RNG draw.
_cjs_individual_draw(rng, sigma) = sigma == 0 ? 0. : Random.randn(rng)

function _cjs_prediction_table(fit, des, table, label, keynames, beta, response_type;
                               sigma=0., rule=nothing)
    rows = _cjs_prediction_rows(fit, des, table, label, keynames)
    η = rows.X * beta + rows.offset
    all(isfinite, η) || throw(ArgumentError("cjs $label prediction link values must be finite"))
    constraint = fit.prepared.terminal_constraint
    terminal = falses(rows.n)
    if constraint !== nothing && only(keys(constraint)) == label
        terminal = label == :phi ?
            [last(fit.data.occasions) == x for x in rows.cols.interval_end] :
            [last(fit.data.occasions) == x for x in rows.cols.occasion]
        c = only(values(constraint))
        η[terminal] .= log(c / (1 - c))
    end
    base = Pair{Symbol,Any}[k => getproperty(rows.cols, k) for k in keynames]
    Δ = nothing
    if label == :phi
        haskey(rows.cols, :interval_length) || throw(ArgumentError("cjs phi prediction table requires interval_length"))
        Δ = Float64.(rows.cols.interval_length)
        all(x -> isfinite(x) && x > 0, Δ) || throw(ArgumentError("cjs interval_length values must be finite and positive"))
    end
    label == :phi && push!(base, :interval_length => Δ)
    if response_type
        q = 1.0 ./ (1.0 .+ exp.(-η))
        interval = label == :phi ? q .^ Δ : nothing
        if rule !== nothing && sigma > 0
            weights = exp.(rule.logweights)
            for i in eachindex(q)
                terminal[i] && continue
                values = exp.(_cjs_logistic_log.(η[i] .+ sigma .* rule.nodes))
                q[i] = sum(weights .* values)
                # Integrate interval survival separately: E[s^Delta].
                label == :phi && (interval[i] = sum(weights .* values .^ Δ[i]))
            end
        end
        push!(base, (label == :phi ? :phi_unit : :p) => q)
        label == :phi && push!(base, :phi_interval => interval)
    else
        push!(base, :eta => η)
    end
    return (; base...)
end

"""
Predict CJS processes on keyed rows, preserving fitted schemas, offsets and units.
Ordinary persistent individual fits require `target=:zero_effect` (latent effects
set to zero) or `target=:integrated` (response probabilities averaged over the
fitted marked-cohort normal law). Interval survival averages `s^Delta` directly.
The parameters are held fixed. These are supplied-profile process averages,
not history-conditioned effects or later-survivor averages. Integrated link
prediction remains unsupported. For a fitted reference-selection individual
model, `target=:reference_selected` returns response-scale process means under
the first-capture-weighted reference effect law and does not condition on later
histories. Fitted IDs reuse their fitted SD profiles. New IDs require an
explicit selection profile containing the eligibility index, first-capture
index, complete prehistory interval rows, and survey mask/captures. SD-formula
predictors must be supplied consistently in the prehistory and future process
tables. This predicts supplied profiles; a population average still requires
a declared reference covariate distribution.
Ordinary shared fits use fresh reference-group means with the same explicit targets.
"""
function predict(fit::CJSFit, newdata; type=:response, target=nothing)
    individual = _cjs_is_individual(fit.prepared)
    if hasproperty(fit.prepared,:selection) && fit.prepared.selection!==nothing
        type in (:response,"response") || throw(ArgumentError("cjs reference-selected prediction is available on the response scale only"))
        target==:reference_selected || throw(ArgumentError("cjs selected individual prediction requires target=:reference_selected"))
        return _cjs_predict_reference_selected(fit,newdata)
    end
    _cjs_is_shared(fit.prepared) && return _cjs_predict_shared(fit,newdata;type,target)
    _cjs_has_sd_individual(fit.prepared) && return _cjs_predict_sd_individual(fit,newdata;type,target)
    type in (:response, :link, "response", "link") || throw(ArgumentError("cjs prediction type must be :response or :link"))
    response_type = type in (:response, "response")
    scales = individual ? _cjs_individual_methods_guard(fit) : (phi=0., p=0.)
    if individual
        target in (:zero_effect, :integrated) || throw(ArgumentError("cjs individual prediction requires explicit target=:zero_effect or :integrated"))
        target == :integrated && !response_type && throw(ArgumentError("cjs integrated prediction is available on the response scale only"))
    else
        target === nothing || throw(ArgumentError("cjs fixed-effect prediction does not require a target"))
    end
    if newdata === nothing
        phi, p, phi_rows, p_rows = _cjs_training_prediction_tables(fit)
        phi = _cjs_append_predictors(phi, phi_rows, fit.data.phi_data, fit.phi_design)
        p = _cjs_append_predictors(p, p_rows, fit.data.p_data, fit.p_design)
        newdata = (; phi, p, units=fit.data.units)
    else
        newdata isa NamedTuple && haskey(newdata, :phi) && haskey(newdata, :p) && haskey(newdata, :units) ||
            throw(ArgumentError("cjs newdata must be (phi=<keyed table>, p=<keyed table>, units=<fit units>)"))
        newdata.units == fit.data.units || throw(ArgumentError("cjs prediction units must match the fitted interval units $(repr(fit.data.units))"))
    end
    function tables(rule=nothing)
        phi = _cjs_prediction_table(fit,fit.phi_design,newdata.phi,:phi,
            (:id,:interval_start,:interval_end),view(fit.beta,1:fit.prepared.nphi),response_type;
            sigma=scales.phi,rule)
        p = _cjs_prediction_table(fit,fit.p_design,newdata.p,:p,(:id,:occasion),
            view(fit.beta,fit.prepared.nphi+1:fit.prepared.nphi+fit.prepared.np),response_type;
            sigma=scales.p,rule)
        return (;phi,p)
    end
    result = tables()
    individual || return result
    integration = (order=0,refined=true,tolerance=1e-8,maximum_probability_change=0.)
    if target == :integrated && (scales.phi > 0 || scales.p > 0)
        previous = nothing
        for order in _CJS_INDIVIDUAL_1D_ORDERS
            result = tables(_cjs_individual_rule(order))
            probabilities = vcat(result.phi.phi_unit,result.phi.phi_interval,result.p.p)
            change = previous === nothing ? Inf : maximum(abs.(probabilities .- previous);init=0.)
            integration = (order=order,refined=change <= 1e-8,tolerance=1e-8,
                           maximum_probability_change=change)
            integration.refined && break
            previous = probabilities
        end
        integration.refined || throw(ArgumentError("cjs integrated prediction quadrature is unrefined at order $(last(_CJS_INDIVIDUAL_1D_ORDERS)); returned-probability tolerance 1e-8 was not met"))
    end
    return merge(result,(target=target,
        mixing=target == :integrated ? :fitted_marked_cohort_normal : :zero_effect,
        units=fit.data.units,integration=integration))
end

predict(fit::CJSFit; newdata=nothing, type=:response, target=nothing) = predict(fit, newdata; type,target)

function _cjs_random(rng, seed)
    seed === nothing && return rng
    seed isa Integer && seed >= 0 || throw(ArgumentError("seed must be a nonnegative integer or nothing"))
    return Random.MersenneTwister(seed)
end

function _cjs_simulate_reference_selected(fit;nsim=1,seed=nothing,rng=Random.default_rng(),max_selection_attempts=100_000)
    _cjs_reference_selection_guard(fit)
    !(nsim isa Bool) && nsim isa Integer && nsim>=1 || throw(ArgumentError("nsim must be a positive integer"))
    max_selection_attempts isa Integer && !(max_selection_attempts isa Bool) && max_selection_attempts>=1 ||
        throw(ArgumentError("max_selection_attempts must be a positive integer"))
    rng isa Random.AbstractRNG || throw(ArgumentError("rng must be an AbstractRNG"))
    drawrng=_cjs_random(rng,seed);d=fit.data;prep=fit.prepared;sel=prep.selection
    N,T=size(d.Y);re=prep.individual
    beta_phi=view(fit.beta,1:prep.nphi);beta_p=view(fit.beta,prep.nphi+1:prep.nphi+prep.np)
    eta_phi=fit.phi_design.X*beta_phi+fit.phi_design.offset
    eta_p=fit.p_design.X*beta_p+fit.p_design.offset
    phi_index=Dict(row=>i for (i,row) in enumerate(fit.phi_design.rows))
    p_index=Dict(row=>i for (i,row) in enumerate(fit.p_design.rows))
    phi_by_time=[Dict(mod1(row,T-1)=>j for (j,row) in enumerate(sel.phi_by_id[i])) for i in 1:N]
    p_by_time=[Dict(mod1(row,T)=>j for (j,row) in enumerate(sel.p_by_id[i])) for i in 1:N]
    constraint=prep.terminal_constraint
    function phi_pre(i,t,u,sigma)
        j=get(phi_by_time[i],t,0);j>0 || throw(ArgumentError("cjs selection survival profile is missing interval $t for ID $(repr(d.ids[i]))"))
        row=sel.phi_by_id[i][j]
        eta=dot(view(sel.phi_X_by_id[i],j,:),beta_phi)+sel.phi_offset_by_id[i][j]
        terminal=constraint!==nothing && only(keys(constraint))==:phi && t==T-1
        _cjs_individual_probability(eta,sigma,u,terminal,constraint,:phi)^d.intervals[t]
    end
    function p_pre(i,t,u,sigma)
        j=get(p_by_time[i],t,0);j>0 || return nothing
        eta=dot(view(sel.p_X_by_id[i],j,:),beta_p)+sel.p_offset_by_id[i][j]
        terminal=constraint!==nothing && only(keys(constraint))==:p && t==T
        _cjs_individual_probability(eta,sigma,u,terminal,constraint,:p)
    end
    output=CMRData[]
    for _ in 1:nsim
        expanded=[(i,r) for i in 1:N for r in 1:d.frequency[i]]
        ids=[(d.ids[i],r) for (i,r) in expanded]
        Y=zeros(Int,length(expanded),T);observed=BitMatrix(undef,length(expanded),T)
        ends=Int[]
        for (j,(i,_)) in enumerate(expanded)
            observed[j,:].=d.observed[i,:];Y[j,1:sel.eligibility[i]-1].=d.Y[i,1:sel.eligibility[i]-1]
            e=sel.eligibility[i];f=d.first[i];last=d.end[i]
            sigma_phi=re.phi===nothing ? 0. : _cjs_sd_value(re.phi,fit.beta,i)
            sigma_p=re.p===nothing ? 0. : _cjs_sd_value(re.p,fit.beta,i)
            accepted=false;u_phi=0.;u_p=0.
            for _attempt in 1:max_selection_attempts
                Y[j,e:f].=0
                u_phi=re.phi===nothing ? 0. : Random.randn(drawrng)
                u_p=re.p===nothing ? 0. : Random.randn(drawrng)
                alive=true;captured=false
                for t in e:f
                    if t>e
                        alive=alive && rand(drawrng)<phi_pre(i,t-1,u_phi,sigma_phi)
                    end
                    if !alive
                        break
                    end
                    if d.observed[i,t]
                        p=p_pre(i,t,u_p,sigma_p)
                        detected=rand(drawrng)<p
                        if detected
                            if t==f
                                captured=true;Y[j,t]=1;break
                            else
                                break
                            end
                        elseif t<f
                            Y[j,t]=0
                        end
                    end
                end
                if captured
                    accepted=true;break
                end
            end
            accepted || throw(ArgumentError("cjs reference-selection simulation did not realize the requested first-capture event for ID $(repr(d.ids[i])) after $max_selection_attempts attempts"))
            alive=true
            for t in f:last-1
                row=(i-1)*(T-1)+t;kphi=get(phi_index,row,0)
                kphi>0 || throw(ArgumentError("cjs post-capture survival profile is missing interval $t for ID $(repr(d.ids[i]))"))
                phi=_cjs_individual_probability(eta_phi[kphi],sigma_phi,u_phi,
                    prep.phi_terminal[kphi],constraint,:phi)^d.intervals[t]
                alive=alive && rand(drawrng)<phi
                next=t+1
                if alive && d.observed[i,next]
                    prow=(i-1)*T+next;kp=get(p_index,prow,0)
                    kp>0 || throw(ArgumentError("cjs post-capture detection profile is missing occasion $next for ID $(repr(d.ids[i]))"))
                    p=_cjs_individual_probability(eta_p[kp],sigma_p,u_p,
                        prep.p_terminal[kp],constraint,:p)
                    Y[j,next]=rand(drawrng)<p
                end
            end
            push!(ends,last)
        end
        function expand_keyed(table)
            cols=Tables.columntable(table);block=length(first(values(cols)))÷N
            pairs=Pair{Symbol,Any}[]
            for key in keys(cols)
                vals=key==:id ? [(d.ids[i],r) for (i,r) in expanded for _ in 1:block] :
                    [getproperty(cols,key)[(i-1)*block+k] for (i,_) in expanded for k in 1:block]
                push!(pairs,key=>vals)
            end
            return (;pairs...)
        end
        push!(output,CMRData(Y,observed,ids,copy(d.occasions),copy(d.intervals),
            [d.first[i] for (i,_) in expanded],ends,ones(Int,length(expanded)),d.units,
            expand_keyed(d.phi_data),expand_keyed(d.p_data)))
    end
    return output
end

"""
Simulate fresh first-capture-conditional marked-cohort histories with fitted
parameters held fixed. Every expanded frequency copy receives fresh independent
normal effects for active processes, persistent over its whole history. First
capture, survey masks, administrative ends, keyed predictors and units remain
fixed. Returns `Vector{CMRData}` suitable for refitting. This is not simulation
from existing individuals' posterior effects or a full population model.
An explicit seed leaves the caller RNG unchanged. Individual SD formulas draw
from the fitted SD for each supplied training profile. Reference-selection
fits simulate from the reference law and reject draws until each fitted first-
capture time is realized. The `max_selection_attempts` keyword bounds attempts
per individual. Ordinary shared fits draw one effect per group and process for
each replicate.
Individual confidence intervals remain guarded.
"""
function simulate(fit::CJSFit; nsim::Integer=1, seed=nothing, rng=Random.default_rng(), max_selection_attempts=100_000)
    hasproperty(fit.prepared,:selection) && fit.prepared.selection!==nothing &&
        return _cjs_simulate_reference_selected(fit;nsim,seed,rng,max_selection_attempts)
    _cjs_has_sd_individual(fit.prepared) && return _cjs_simulate_sd_individual(fit;nsim,seed,rng)
    individual = _cjs_is_individual(fit.prepared)
    scales = individual ? _cjs_individual_methods_guard(fit) : (phi=0.,p=0.)
    _cjs_is_shared(fit.prepared) && return _cjs_simulate_shared(fit;nsim,seed,rng)
    !(nsim isa Bool) && nsim >= 1 || throw(ArgumentError("nsim must be a positive integer"))
    rng isa Random.AbstractRNG || throw(ArgumentError("rng must be an AbstractRNG"))
    drawrng = _cjs_random(rng, seed)
    d = fit.data
    N, T = size(d.Y)
    ls, lp = _cjs_process_logs(fit.beta, fit.prepared)
    eta_phi = individual ? fit.phi_design.X * view(fit.beta,1:fit.prepared.nphi) + fit.phi_design.offset : nothing
    eta_p = individual ? fit.p_design.X * view(fit.beta,fit.prepared.nphi+1:fit.prepared.nphi+fit.prepared.np) + fit.p_design.offset : nothing
    phi_index = Dict(row => i for (i, row) in enumerate(fit.phi_design.rows))
    p_index = Dict(row => i for (i, row) in enumerate(fit.p_design.rows))
    out = CMRData[]
    for _ in 1:nsim
        expanded = [(i, r) for i in 1:N for r in 1:d.frequency[i]]
        ids = [(d.ids[i], r) for (i, r) in expanded]
        Y = zeros(Int, length(expanded), T)
        observed = BitMatrix(undef, length(expanded), T)
        ends = Int[]
        for (j, (i, _)) in enumerate(expanded)
            observed[j, :] .= d.observed[i, :]
            Y[j, 1:d.first[i]] .= d.Y[i, 1:d.first[i]]
            u_phi = individual ? _cjs_individual_draw(drawrng,scales.phi) : 0.
            u_p = individual ? _cjs_individual_draw(drawrng,scales.p) : 0.
            alive = true
            for t in d.first[i]:d.end[i]-1
                phi_row = (i-1)*(T-1)+t
                p_row = (i-1)*T+t+1
                kphi = phi_index[phi_row]
                phi = individual ? _cjs_individual_probability(eta_phi[kphi],scales.phi,u_phi,
                    fit.prepared.phi_terminal[kphi],fit.prepared.terminal_constraint,:phi)^d.intervals[t] :
                    exp(ls[kphi])^d.intervals[t]
                alive = alive && rand(drawrng) < phi
                next = t + 1
                if alive && d.observed[i, next]
                    kp = p_index[p_row]
                    p = individual ? _cjs_individual_probability(eta_p[kp],scales.p,u_p,
                        fit.prepared.p_terminal[kp],fit.prepared.terminal_constraint,:p) : exp(lp[kp])
                    Y[j, next] = rand(drawrng) < p
                end
            end
            push!(ends, d.end[i])
        end
        # Rebuild keyed tables directly from each expanded individual block.
        function expand_keyed(table)
            cols = Tables.columntable(table)
            block = length(first(values(cols))) ÷ N
            pairs = Pair{Symbol,Any}[]
            for key in keys(cols)
                vals = key == :id ? [(d.ids[i], r) for (i, r) in expanded for _ in 1:block] :
                       [getproperty(cols, key)[(i-1)*block+k] for (i, _) in expanded for k in 1:block]
                push!(pairs, key => vals)
            end
            return (; pairs...)
        end
        phi_data = expand_keyed(d.phi_data)
        p_data = expand_keyed(d.p_data)
        # These tables are expanded from normalized CMRData blocks, in canonical
        # individual-major order. Avoid re-matching every known key against all
        # keys for large frequency expansion; preserve the fixed-method path.
        sim = individual ? CMRData(Y,observed,ids,copy(d.occasions),copy(d.intervals),
                    [d.first[i] for (i,_) in expanded],ends,ones(Int,length(expanded)),
                    d.units,phi_data,p_data) :
            CMRData(Y; ids, occasions=d.occasions, units=d.units,
                    observed, intervals=d.intervals, ends, phi_data, p_data)
        push!(out, sim)
    end
    return out
end

_cjs_coef_names(fit) = vcat(["phi: $x" for x in fit.phi_design.names], ["p: $x" for x in fit.p_design.names],
    _cjs_is_shared(fit.prepared) ? fit.prepared.shared.names :
    _cjs_is_individual(fit.prepared) ? fit.prepared.individual.names : String[])

function _cjs_target_indices(fit, targets)
    names = _cjs_coef_names(fit)
    requested = targets === nothing ? names : String.(collect(targets))
    isempty(requested) && throw(ArgumentError("at least one CJS coefficient target is required"))
    all(n -> n in names, requested) || throw(ArgumentError("unknown CJS coefficient target; use one of $(join(names, ", "))"))
    return [(name, findfirst(==(name), names)) for name in requested]
end

_cjs_record(name, estimate, lower, upper, status; usable=missing, requested=missing) =
    (target=name, estimate=estimate, lower=lower, upper=upper, status=status,
     usable=usable, requested=requested)

function _cjs_wald(fit; level, targets)
    indices = _cjs_target_indices(fit, targets)
    α = 1 - level
    z = quantile(Normal(), 1 - α/2)
    eligible = fit.diagnostics.converged && fit.diagnostics.gradient_norm <= 1e-4 &&
               fit.diagnostics.positive_curvature && !fit.diagnostics.boundary &&
               all(isfinite, fit.vcov)
    names = _cjs_coef_names(fit)
    records = [_cjs_record(name, fit.beta[i],
        eligible ? fit.beta[i] - z*sqrt(fit.vcov[i,i]) : missing,
        eligible ? fit.beta[i] + z*sqrt(fit.vcov[i,i]) : missing,
        eligible ? :available : :guarded_unavailable) for (name, i) in indices]
    return (method=:wald, level=level, intervals=records, attempts=NamedTuple[], requested=0,
            status=eligible ? :available : :guarded_unavailable,
            reason=eligible ? "local asymptotic ML interval; calibration not established" : fit.diagnostics.uncertainty_reason)
end

function _cjs_profile_value(fit, i, value)
    free = [j for j in eachindex(fit.beta) if j != i]
    full = copy(fit.beta)
    full[i] = value
    if isempty(free)
        value = try
            _cjs_objective(full, fit.prepared)
        catch
            return missing
        end
        return isfinite(value) ? value : missing
    end
    objective = x -> begin
        candidate = Vector{eltype(x)}(undef, length(full))
        candidate[i] = value
        candidate[free] = x
        _cjs_objective(candidate, fit.prepared)
    end
    result = try
        Optim.optimize(objective, fit.beta[free], Optim.BFGS(),
                        Optim.Options(iterations=1000, g_tol=1e-7); autodiff=:forward)
    catch
        return missing
    end
    Optim.converged(result) || return missing
    gradient = try
        ForwardDiff.gradient(objective, Optim.minimizer(result))
    catch
        return missing
    end
    (all(isfinite, gradient) && norm(gradient, Inf) <= 1e-5) || return missing
    value = Optim.minimum(result)
    return isfinite(value) ? value : missing
end

function _cjs_profile_lr(profiled, loglik)
    (profiled === missing || !isfinite(profiled) || !isfinite(loglik)) &&
        return (missing, :failed_reoptimization)
    lr = 2 * (profiled + loglik)
    isfinite(lr) || return (missing, :failed_reoptimization)
    # Fixed-route roundoff convention; integrated-profile tolerances are separate.
    tolerance = 200eps(Float64) * max(1., abs(loglik))
    lr < -2tolerance && return (missing, :profile_improves_on_fit)
    return (max(0., lr), :available)
end

function _cjs_profile_endpoint(fit, i, direction, cutoff, center;
                               profile_value=_cjs_profile_value)
    step = 0.5
    inner = center
    outer = center + direction*step
    for _ in 1:12
        profiled = profile_value(fit, i, outer)
        value, point_status = _cjs_profile_lr(profiled, fit.loglik)
        point_status != :available && return (missing, point_status)
        value >= cutoff && break
        inner = outer
        step *= 2
        outer = center + direction*step
    end
    profiled = profile_value(fit, i, outer)
    value, point_status = _cjs_profile_lr(profiled, fit.loglik)
    point_status != :available && return (missing, point_status)
    value < cutoff && return (missing, :not_crossed)
    lo, hi = min(inner, outer), max(inner, outer)
    for _ in 1:45
        mid = (lo + hi)/2
        profiled = profile_value(fit, i, mid)
        lr, point_status = _cjs_profile_lr(profiled, fit.loglik)
        point_status != :available && return (missing, point_status)
        if direction < 0
            lr >= cutoff ? (lo = mid) : (hi = mid)
        else
            lr >= cutoff ? (hi = mid) : (lo = mid)
        end
    end
    return ((lo + hi)/2, :available)
end

function _cjs_profile_invalidate_records!(records)
    for i in eachindex(records)
        status = records[i].status == :not_attempted_invalid_fit ?
            :not_attempted_invalid_fit : :profile_improves_on_fit
        records[i] = merge(records[i], (lower=missing, upper=missing, status))
    end
    return records
end

_cjs_profile_unattempted(target,index,scale) =
    (target=target,index=index,scale=scale,status=:not_attempted_invalid_fit,
        center=missing,lower=missing,upper=missing,points=NamedTuple[],
        reason="a previous constrained point improved the fitted optimum; this target was not evaluated")

function _cjs_profile(fit; level, targets)
    indices = _cjs_target_indices(fit, targets)
    cutoff = quantile(Chisq(1), level)
    records = NamedTuple[]; attempts = NamedTuple[]; invalid_fit = false
    for (name, i) in indices
        if invalid_fit
            push!(records,_cjs_record(name,fit.beta[i],missing,missing,:not_attempted_invalid_fit))
            push!(attempts,_cjs_profile_unattempted(name,i,:logit))
            continue
        end
        if !fit.diagnostics.converged || fit.diagnostics.gradient_norm > 1e-4
            push!(records, _cjs_record(name, fit.beta[i], missing, missing, :guarded_unavailable))
            push!(attempts,(target=name,index=i,status=:guarded_unavailable,
                reason="fit convergence or gradient guard",points=NamedTuple[]))
            continue
        end
        points = NamedTuple[]
        traced_value = function(f,index,value)
            objective = _cjs_profile_value(f,index,value)
            lr,status = _cjs_profile_lr(objective,f.loglik)
            push!(points,(;value,objective,lr,status))
            return objective
        end
        lower, lower_status = _cjs_profile_endpoint(fit, i, -1, cutoff, fit.beta[i];profile_value=traced_value)
        upper, upper_status = lower_status == :profile_improves_on_fit ?
            (missing,:not_attempted_invalid_fit) :
            _cjs_profile_endpoint(fit, i, 1, cutoff, fit.beta[i];profile_value=traced_value)
        invalid_fit = lower_status == :profile_improves_on_fit || upper_status == :profile_improves_on_fit
        status = invalid_fit ? :profile_improves_on_fit :
                 lower_status == :failed_reoptimization || upper_status == :failed_reoptimization ? :failed_reoptimization :
                 lower_status == :available && upper_status == :available ? :available :
                 lower_status == :not_crossed && upper_status == :not_crossed ? :both_endpoints_not_crossed :
                 lower_status == :not_crossed ? :lower_endpoint_not_crossed : :upper_endpoint_not_crossed
        push!(records, _cjs_record(name, fit.beta[i], invalid_fit ? missing : lower,
            invalid_fit ? missing : upper, status))
        push!(attempts,(target=name,index=i,status,
            lower=(estimate=lower,status=lower_status),
            upper=upper_status==:not_attempted_invalid_fit ? missing : (estimate=upper,status=upper_status),points))
    end
    invalid_fit && _cjs_profile_invalidate_records!(records)
    completed=count(r->r.status==:available,records)
    return (method=:profile, level=level, intervals=records, attempts, requested=length(indices),completed,
            status=invalid_fit ? :profile_improves_on_fit :
                all(r -> r.status == :available, records) ? :available : :partial,
            cutoff=cutoff,calibrated=false,reason="profile likelihood ratio; a constrained improvement invalidates all intervals; calibration not established")
end

function _cjs_individual_profile_objective(template, free, prep; order, rule, zero_process=nothing)
    return x -> begin
        full=eltype(x).(template)
        full[free]=x
        _cjs_individual_objective(full,prep;order,rule,zero_process)
    end
end

function _cjs_individual_profile_optimizer_status(value, converged, gradient)
    gradient_norm=all(isfinite,gradient) ? (isempty(gradient) ? 0. : norm(gradient,Inf)) : Inf
    accepted=converged && isfinite(value) && isfinite(gradient_norm) && gradient_norm<=1e-5
    reason=!converged ? "constrained nuisance optimizer did not converge" :
        !isfinite(value) ? "constrained nuisance objective is non-finite" :
        !isfinite(gradient_norm) ? "constrained nuisance gradient is non-finite" :
        gradient_norm>1e-5 ? "constrained nuisance gradient exceeded the existing 1e-5 limit" : nothing
    return (;accepted,gradient_norm,reason)
end

function _cjs_individual_profile_termination(stopped_by)
    flag(name) = hasproperty(stopped_by, name) ? getproperty(stopped_by, name) : missing
    return (x_converged=flag(:x_converged), f_converged=flag(:f_converged),
        g_converged=flag(:g_converged), iteration_limit=flag(:iterations),
        time_limit=flag(:time_limit), line_search_failed=flag(:ls_failed),
        function_increased=flag(:f_increased),
        small_trustregion_radius=flag(:small_trustregion_radius),
        f_limit_reached=flag(:f_limit_reached), g_limit_reached=flag(:g_limit_reached),
        h_limit_reached=flag(:h_limit_reached), callback=flag(:callback))
end

function _cjs_individual_profile_nuisance_curvature(objective, point)
    isempty(point) && return (status=:empty, dimension=0, finite=true,
        max_asymmetry=0., min_eigenvalue=missing, max_eigenvalue=missing,
        positive_definite=missing, error=nothing)
    try
        hessian = Matrix{Float64}(ForwardDiff.hessian(objective, Float64.(point)))
        finite = all(isfinite, hessian)
        finite || return (status=:nonfinite, dimension=length(point), finite=false,
            max_asymmetry=Inf, min_eigenvalue=missing, max_eigenvalue=missing,
            positive_definite=missing, error=nothing)
        asymmetry = maximum(abs, hessian .- transpose(hessian); init=0.)
        eigenvalues = eigvals(Symmetric((hessian .+ transpose(hessian)) ./ 2))
        return (status=:available, dimension=length(point), finite=true,
            max_asymmetry=asymmetry, min_eigenvalue=minimum(eigenvalues),
            max_eigenvalue=maximum(eigenvalues), positive_definite=minimum(eigenvalues) > 0.,
            error=nothing)
    catch err
        err isa InterruptException && rethrow()
        return (status=:error, dimension=length(point), finite=false,
            max_asymmetry=missing, min_eigenvalue=missing, max_eigenvalue=missing,
            positive_definite=missing, error=sprint(showerror, err))
    end
end

function _cjs_individual_profile_objective_not_worse(value, reference)
    return isfinite(value) && isfinite(reference) &&
        value<=reference+1e-12*max(1.,abs(reference))
end

function _cjs_individual_profile_newton_polish(objective, start, reference_objective)
    try
        point=Float64.(start)
        isempty(point) && return nothing
        isfinite(reference_objective) || return nothing
        value=Float64(objective(point))
        isfinite(value) && _cjs_individual_profile_objective_not_worse(value,reference_objective) ||
            return nothing
        gradient=ForwardDiff.gradient(objective,point)
        hessian=Matrix{Float64}(ForwardDiff.hessian(objective,point))
        all(isfinite,gradient) && all(isfinite,hessian) || return nothing
        scale=max(1.,maximum(abs,hessian;init=0.))
        maximum(abs,hessian.-transpose(hessian);init=0.)<=1e-10*scale || return nothing
        symmetric=Symmetric((hessian.+transpose(hessian))./2)
        factor=cholesky(symmetric;check=false)
        issuccess(factor) || return nothing
        direction=-(factor\gradient)
        all(isfinite,direction) && dot(gradient,direction)<0 || return nothing
        for backtrack in 0:20
            candidate=point.+ldexp(1.,-backtrack).*direction
            candidate_value=Float64(objective(candidate))
            isfinite(candidate_value) || continue
            _cjs_individual_profile_objective_not_worse(candidate_value,value) &&
                _cjs_individual_profile_objective_not_worse(candidate_value,reference_objective) || continue
            candidate_gradient=ForwardDiff.gradient(objective,candidate)
            all(isfinite,candidate_gradient) || continue
            gradient_norm=norm(candidate_gradient,Inf)
            gradient_norm<=1e-5 || continue
            return (;solution=candidate,objective=candidate_value,gradient_norm)
        end
        return nothing
    catch err
        err isa InterruptException && rethrow()
        return nothing
    end
end

function _cjs_individual_profile_fallback_status(attempt, reference)
    no_worse=_cjs_individual_profile_objective_not_worse(attempt.objective,reference)
    accepted=attempt.check.accepted && no_worse
    reason=!attempt.check.accepted ? attempt.check.reason :
        !no_worse ? "Newton fallback objective was worse than the BFGS candidate" : nothing
    return (;accepted,reason)
end

function _cjs_individual_profile_restart_eligible(newton, bfgs)
    gradient_norm=newton.check.gradient_norm
    return isfinite(newton.objective) &&
        _cjs_individual_profile_objective_not_worse(newton.objective,bfgs.objective) &&
        isfinite(gradient_norm) && gradient_norm<=1e-5
end

function _cjs_individual_profile_optimizer_attempt(objective, start, method, optimizer, order)
    start=Float64.(start)
    try
        result=Optim.optimize(objective,start,method,
            Optim.Options(iterations=1000,g_tol=1e-7,f_reltol=1e-12);autodiff=:forward)
        solution=Float64.(Optim.minimizer(result))
        value=Float64(objective(solution))
        gradient=ForwardDiff.gradient(objective,solution)
        check=_cjs_individual_profile_optimizer_status(value,Optim.converged(result),gradient)
        termination=_cjs_individual_profile_termination(result.stopped_by)
        return (;order,optimizer,start,solution,objective=value,converged=Optim.converged(result),
            iterations=Optim.iterations(result),termination,check,error=nothing)
    catch err
        err isa InterruptException && rethrow()
        return (;order,optimizer,start,solution=copy(start),objective=Inf,converged=false,
            iterations=0,termination=missing,
            check=(accepted=false,gradient_norm=Inf,reason=sprint(showerror,err)),
            error=sprint(showerror,err))
    end
end

function _cjs_individual_profile_optimize(objective,start,order;
        run_attempt=_cjs_individual_profile_optimizer_attempt)
    bfgs=run_attempt(objective,start,Optim.BFGS(),:BFGS,order)
    trials=NamedTuple[bfgs]
    # Optim.converged may be true because only the objective-change criterion
    # fired. Do not let that short-circuit the trust-region comparison unless
    # Optim explicitly confirms gradient convergence; missing flags fail closed.
    bfgs_gradient_converged=hasproperty(bfgs,:termination) &&
        hasproperty(bfgs.termination,:g_converged) ? bfgs.termination.g_converged : missing
    bfgs.check.accepted && bfgs_gradient_converged===true &&
        return (;selected=bfgs,trials)
    if !isfinite(bfgs.objective)
        skipped=(;order,optimizer=:NewtonTrustRegion,status=:fallback_skipped,
            start=copy(bfgs.solution),solution=copy(bfgs.solution),objective=Inf,
            converged=false,iterations=0,termination=missing,nuisance_curvature=missing,
            check=(accepted=false,gradient_norm=Inf,
                reason="BFGS objective is non-finite; Newton fallback requires a finite comparison objective"),
            error=nothing)
        push!(trials,skipped)
        return (;selected=nothing,trials)
    end
    newton=run_attempt(objective,bfgs.solution,Optim.NewtonTrustRegion(),:NewtonTrustRegion,order)
    fallback=_cjs_individual_profile_fallback_status(newton,bfgs.objective)
    newton=(;newton...,check=(;newton.check...,accepted=fallback.accepted,reason=fallback.reason))
    if !fallback.accepted
        curvature=_cjs_individual_profile_nuisance_curvature(objective,newton.solution)
        newton=(;newton...,nuisance_curvature=curvature)
    end
    push!(trials,newton)
    fallback.accepted && return (;selected=newton,trials)

    # If the trust-region result is stationary to within a small Newton step,
    # try one curvature-checked correction, then require the usual optimizer
    # and gradient admission checks on a BFGS restart from that point.
    polished=_cjs_individual_profile_newton_polish(
        objective,newton.solution,bfgs.objective)
    if polished!==nothing
        restart=run_attempt(objective,polished.solution,Optim.BFGS(),:BFGSNewtonPolish,order)
        restart_no_worse=_cjs_individual_profile_objective_not_worse(
            restart.objective,newton.objective) &&
            _cjs_individual_profile_objective_not_worse(restart.objective,bfgs.objective)
        restart=(;restart...,check=(;restart.check...,
            accepted=restart.check.accepted && restart_no_worse,
            reason=!restart.check.accepted ? restart.check.reason :
                !restart_no_worse ? "BFGS Newton-polish objective was worse than a prior candidate" : nothing,
            restart_origin=:PositiveCurvatureNewtonPolish,
            polish_gradient_norm=polished.gradient_norm))
        push!(trials,restart)
        restart.check.accepted && return (;selected=restart,trials)
    end

    # A stationary Newton candidate can be rejected only because Optim did not
    # report convergence. Polish that point with BFGS. The restart still needs
    # Optim convergence, the existing numeric gradient bound, and a no-worse
    # objective; its stop flag need not be g_converged because the trust-region
    # comparison has already run and the independently computed gradient is
    # checked against the explicit acceptance bound.
    if _cjs_individual_profile_restart_eligible(newton,bfgs)
        restart=run_attempt(objective,newton.solution,Optim.BFGS(),:BFGSRestart,order)
        restart_no_worse=_cjs_individual_profile_objective_not_worse(
            restart.objective,newton.objective)
        restart=(;restart...,check=(;restart.check...,
            accepted=restart.check.accepted && restart_no_worse,
            reason=!restart.check.accepted ? restart.check.reason :
                !restart_no_worse ? "BFGS restart objective was worse than the Newton candidate" : nothing,
            restart_origin=:NewtonTrustRegion))
        push!(trials,restart)
        return (;selected=restart.check.accepted ? restart : nothing,trials)
    end
    return (;selected=nothing,trials)
end

function _cjs_individual_profile_attempt_record(template,free,index,target,trial;
        zero_process=nothing,zero_indices=Int[])
    full_start=copy(template); full_start[free]=trial.start
    full_solution=copy(template); full_solution[free]=trial.solution
    return (trial...,free_indices=copy(free),fixed_index=index,fixed_value=target,
        target_removed=index in zero_indices,zero_process,zero_indices=copy(zero_indices),
        full_start,full_solution)
end

function _cjs_individual_scale_profile_start(mapping,index,target,start;zero_process=nothing)
    adjusted=Float64.(start)
    zero_process===nothing || return adjusted
    hasproperty(mapping,:scale_range) && hasproperty(mapping,:sd_X) || return adjusted
    position=findfirst(==(index),mapping.scale_range)
    position===nothing && return adjusted
    X=mapping.sd_X
    size(X,1)>0 && position<=size(X,2) || return adjusted
    nuisance=setdiff(collect(1:size(X,2)),[position])
    isempty(nuisance) && return adjusted
    # When a scale coefficient moves, keep a supported high-leverage row's
    # log-SD fixed in the warm start. This preserves a useful continuation
    # branch for SD intercept/slope trade-offs; the optimizer still determines
    # the constrained profile value and can reject the candidate normally.
    best=nothing
    for row in axes(X,1),column in nuisance
        denom=X[row,column]
        isfinite(denom) && abs(denom)>sqrt(eps(Float64)) || continue
        score=X[row,position]*denom
        candidate_rank=(abs(score),score)
        best_rank=best===nothing ? nothing : (abs(best[3]),best[3])
        if isfinite(score) && (best===nothing || candidate_rank>best_rank)
            best=(row,column,score)
        end
    end
    best===nothing && return adjusted
    row,column,score=best
    abs(score)>sqrt(eps(Float64)) || return adjusted
    old=adjusted[index]
    isfinite(target) && isfinite(old) || return adjusted
    candidate=adjusted[mapping.scale_range[column]]+
        X[row,position]*(old-target)/X[row,column]
    isfinite(candidate) && abs(candidate)<=1e4 || return adjusted
    proposed=copy(adjusted)
    proposed[mapping.scale_range[column]]=candidate
    proposed[index]=target
    eta=mapping.sd_X*proposed[mapping.scale_range]+mapping.sd_offset
    # Keep a numerically evaluable start. Extreme underflow is useful for a
    # boundary profile, but overflow or non-finite predictors are not.
    all(isfinite,eta) && all(x->x<log(floatmax(Float64))-2,eta) || return adjusted
    adjusted[mapping.scale_range[column]]=candidate
    adjusted[index]=target
    return adjusted
end

function _cjs_individual_profile_path(initial, point_function;
        start_transform=(value,start,zero_process)->start)
    start=Float64.(initial)
    function point_at(value;zero_process=nothing)
        transformed=start_transform(value,copy(start),zero_process)
        point=point_function(value,transformed,zero_process)
        if hasproperty(point,:status) && point.status===:available &&
                hasproperty(point,:theta) && length(point.theta)==length(start) &&
                all(isfinite,point.theta)
            start=Float64.(point.theta)
        end
        return point
    end
    reset! = value -> (start=Float64.(value); nothing)
    return (;point_at,reset!)
end

function _cjs_individual_profile_point(fit, index, target; zero_process=nothing,
        start=fit.beta, integral_tol=1e-8, objective_tol=1e-8, parameter_tol=1e-5,
        run_attempt=_cjs_individual_profile_optimizer_attempt)
    prep=fit.prepared
    1 <= index <= length(fit.beta) || throw(ArgumentError("CJS profile coefficient index is invalid"))
    isfinite(target) || throw(ArgumentError("CJS profile value must be finite"))
    all(x->isfinite(x)&&x>0,(integral_tol,objective_tol,parameter_tol)) ||
        throw(ArgumentError("CJS profile tolerances must be finite and positive"))
    zero_process===nothing || zero_process in (:phi,:p) ||
        throw(ArgumentError("zero_process must be :phi or :p"))
    length(start)==length(fit.beta) && all(isfinite,start) ||
        throw(ArgumentError("CJS profile start must be finite and match the fitted parameter dimension"))
    zero_indices=zero_process===nothing ? Int[] : begin
        mapping=getproperty(prep.individual,zero_process)
        mapping===nothing ? Int[] : collect(mapping.scale_range)
    end
    template=Float64.(start)
    if !(index in zero_indices)
        template[index]=target
    elseif zero_process===nothing
        throw(ArgumentError("fixed CJS profile target cannot be removed by a zero-process constraint"))
    end
    free=setdiff(collect(eachindex(template)),unique(vcat([index],zero_indices)))
    frequencies=prep.data.frequency
    previous=nothing
    refinement=NamedTuple[]
    attempts=NamedTuple[]
    orders=_cjs_individual_integration_orders(prep;zero_process)
    profile_result=(value,status,theta,order)->(;objective=value,status,theta=copy(theta),order,
        index,target,fixed_index=index,fixed_value=target,free_indices=copy(free),
        zero_process,zero_indices=copy(zero_indices),refinement,attempts)
    # Keep the last constrained candidate visible after the loop's local scope.
    # The final unrefined status must retain the actual vector, not raise an
    # UndefVarError when no quadrature order met the convergence criteria.
    candidate=copy(template)
    for order in orders
        split=hasproperty(prep.individual,:integration) &&
            prep.individual.integration===:threshold_split_gauss_legendre
        rule=split ? nothing : _cjs_individual_rule(order)
        candidate=copy(template)
        objective=_cjs_individual_profile_objective(candidate,free,prep;order,rule,zero_process)
        optimizer_run=_cjs_individual_profile_optimize(objective,candidate[free],order;run_attempt)
        for trial in optimizer_run.trials
            push!(attempts,_cjs_individual_profile_attempt_record(candidate,free,index,target,trial;
                zero_process,zero_indices))
        end
        candidate=last(attempts).full_solution
        if optimizer_run.selected===nothing
            return profile_result(missing,:failed_reoptimization,candidate,order)
        end
        selected=optimizer_run.selected
        candidate[free]=selected.solution
        iterations=selected.iterations
        gradient_norm=selected.check.gradient_norm

        idx=findfirst(==(order),orders)
        comparison_order=orders[idx<length(orders) ? idx+1 : idx-1]
        parts,comparison=try
            current=_cjs_individual_history_logliks(candidate,prep;order,rule,components=true,zero_process)
            comparison_rule=split ? nothing : _cjs_individual_rule(comparison_order)
            compare=_cjs_individual_history_logliks(candidate,prep;
                order=comparison_order,rule=comparison_rule,components=true,zero_process)
            (current,compare)
        catch err
            err isa InterruptException && rethrow()
            push!(attempts,(order,status=:failed_refinement,iterations,gradient_norm,
                comparison_order,error=sprint(showerror,err)))
            return profile_result(missing,:failed_refinement,candidate,order)
        end
        objective_change=abs(sum(frequencies .* (comparison.conditional.-parts.conditional)))
        parameter_change=previous===nothing ? Inf : norm(candidate-previous,Inf)
        conditional_change=maximum(abs,comparison.conditional.-parts.conditional;init=0.)
        numerator_change=maximum(abs,comparison.numerator.-parts.numerator;init=0.)
        denominator_change=maximum(abs,comparison.denominator.-parts.denominator;init=0.)
        objective_limit=objective_tol*max(1,sum(frequencies))
        refined=previous!==nothing && all(isfinite,(objective_change,parameter_change,
            conditional_change,numerator_change,denominator_change)) &&
            objective_change<=objective_limit && parameter_change<=parameter_tol &&
            conditional_change<=integral_tol && numerator_change<=integral_tol &&
            denominator_change<=integral_tol
        push!(refinement,(order,comparison_order,objective_change,parameter_change,
            max_conditional_change=conditional_change,max_numerator_change=numerator_change,
            max_denominator_change=denominator_change,integral_tol,objective_tol=objective_limit,
            parameter_tol,refined))
        value=try
            _cjs_individual_objective(candidate,prep;order,rule,zero_process)
        catch err
            err isa InterruptException && rethrow()
            push!(attempts,(order,status=:failed_objective,iterations,gradient_norm,
                error=sprint(showerror,err)))
            return profile_result(missing,:failed_reoptimization,candidate,order)
        end
        push!(attempts,(order,status=refined ? :available : :unrefined,
            iterations,gradient_norm,objective=Float64(value),comparison_order,
            objective_change,parameter_change,conditional_change,numerator_change,
            denominator_change,objective_limit,integral_tol,parameter_tol,refined,error=nothing))
        if refined
            return profile_result(Float64(value),:available,candidate,order)
        end
        previous=copy(candidate)
        template=copy(candidate)
    end
    return profile_result(missing,:unrefined_at_cap,candidate,last(orders))
end

function _cjs_individual_profile_target(fit,name)
    name isa AbstractString || throw(ArgumentError("CJS profile target names must be strings"))
    names=_cjs_coef_names(fit)
    index=findfirst(==(name),names)
    if index!==nothing
        nfixed=fit.prepared.nphi+fit.prepared.np
        if index<=nfixed
            return (name,index,process=nothing,scale=:logit,alias=false,status=:ready,reason=nothing)
        end
        individual=fit.prepared.individual
        for process in (:phi,:p)
            mapping=getproperty(individual,process)
            mapping===nothing && continue
            if name in ("sd_$(process): "*n for n in mapping.sd_names)
                scale_index=findfirst(==(name),names)
                scale_index in mapping.scale_range ||
                    throw(ArgumentError("CJS SD coefficient label does not match its prepared parameter index"))
                return (name,index=scale_index,process,scale=:log_sd,alias=false,status=:ready,reason=nothing)
            end
        end
        throw(ArgumentError("CJS individual coefficient label has no prepared parameter mapping"))
    end
    process=name=="sd:phi" ? :phi : name=="sd:p" ? :p : nothing
    process===nothing && throw(ArgumentError("unknown CJS profile target; use an exact fitted coefficient label or sd:phi / sd:p"))
    mapping=getproperty(fit.prepared.individual,process)
    mapping===nothing && throw(ArgumentError("CJS profile target $name is inactive in this fit"))
    constant=length(mapping.scale_range)==1 &&
        (!hasproperty(mapping,:sd_X) || (size(mapping.sd_X,2)==1 && all(==(1),mapping.sd_X))) &&
        (!hasproperty(mapping,:sd_offset) || all(iszero,mapping.sd_offset))
    constant || return (name,index=mapping.scale_index,process,scale=:positive_sd,alias=true,
        status=:guarded_unavailable,reason="sd:$process is defined only for a constant/intercept-only SD model; request an sd_$process formula coefficient")
    return (name,index=mapping.scale_index,process,scale=:positive_sd,alias=true,status=:ready,reason=nothing)
end

function _cjs_individual_profile_guard(fit)
    prep=fit.prepared
    _cjs_is_individual(prep) && !_cjs_is_shared(prep) ||
        throw(ArgumentError("CJS individual profiles require an individual-only fit"))
    prep.individual.route in (:individual_phi,:individual_p,:individual_pair) ||
        throw(ArgumentError("CJS individual profile route is unsupported"))
    hasproperty(prep,:selection) && prep.selection!==nothing &&
        throw(ArgumentError("ordinary CJS individual profiles do not include reference-selection conditioning"))
    nfixed=prep.nphi+prep.np
    length(fit.beta)==nfixed+prep.individual.nscale && all(isfinite,fit.beta) && isfinite(fit.loglik) ||
        throw(ArgumentError("CJS individual profile parameter dimensions or values are invalid"))
    diag=fit.diagnostics
    admitted=hasproperty(diag,:converged) && diag.converged===true &&
        hasproperty(diag,:gradient_norm) && isfinite(diag.gradient_norm) &&
        0<=diag.gradient_norm<=1e-4 && hasproperty(diag,:quadrature_refined) && diag.quadrature_refined===true
    if !admitted
        reason=hasproperty(diag,:uncertainty_reason) ? diag.uncertainty_reason : nothing
        reason isa AbstractString && !isempty(reason) && return String(reason)
        return "CJS individual profiles require a converged fit, small gradient and refined quadrature"
    end
    hasproperty(diag,:quadrature_order) && !(diag.quadrature_order in _cjs_individual_integration_orders(prep)) &&
        throw(ArgumentError("CJS individual profile fit has an unsupported quadrature order"))
    return nothing
end

function _cjs_individual_profile_lr(point,fit)
    point.status==:available || return (missing,point.status,Inf)
    baseline=-fit.loglik
    tolerance=isempty(point.refinement) ? 200eps(Float64)*max(1,abs(baseline)) :
        2last(point.refinement).objective_tol
    lr=2*(point.objective-baseline)
    isfinite(lr) || return (missing,:failed_reoptimization,tolerance)
    lr < -tolerance && return (missing,:profile_improves_on_fit,tolerance)
    return (max(0.,lr),:available,tolerance)
end

function _cjs_profile_midpoint(lo,hi)
    isfinite(lo) && isfinite(hi) && lo<=hi ||
        throw(ArgumentError("CJS profile bracket must have finite ordered endpoints"))
    return signbit(lo)==signbit(hi) ? lo+(hi-lo)/2 : lo/2+hi/2
end

function _cjs_individual_profile_cached_point(path,cache,value)
    key=Float64(value)
    point=get!(cache,key) do
        path.point_at(key)
    end
    hasproperty(point,:status) && point.status===:available &&
        hasproperty(point,:theta) && path.reset!(point.theta)
    return point
end

function _cjs_individual_profile_target_endpoint(fit,path,cache,center,direction,cutoff;
        phase_ref=Ref(:expansion))
    function evaluate(value)
        phase_ref[] = (direction=direction < 0 ? :lower : :upper, phase=phase_ref[].phase)
        point=_cjs_individual_profile_cached_point(path,cache,value)
        lr,status,tolerance=_cjs_individual_profile_lr(point,fit)
        return (lr,status,tolerance,point)
    end
    inner=center
    step=.5
    outer=center+direction*step
    phase_ref[]=(direction=direction < 0 ? :lower : :upper,phase=:expansion)
    lr,status,tolerance,point=evaluate(outer)
    for _ in 1:12
        status==:available || return (estimate=missing,status,lr,tolerance,point)
        lr>=cutoff && break
        inner=outer
        step*=2
        outer=center+direction*step
        phase_ref[]=(direction=direction < 0 ? :lower : :upper,phase=:expansion)
        lr,status,tolerance,point=evaluate(outer)
    end
    status==:available || return (estimate=missing,status,lr,tolerance,point)
    lr>=cutoff || return (estimate=missing,status=:endpoint_not_crossed,lr,tolerance,point)
    lo,hi=min(inner,outer),max(inner,outer)
    for _ in 1:45
        mid=_cjs_profile_midpoint(lo,hi)
        phase_ref[]=(direction=direction < 0 ? :lower : :upper,phase=:bisection)
        value,point_status,tol,_=evaluate(mid)
        point_status==:available || return (estimate=missing,status=point_status,lr=value,tolerance=tol,point=cache[mid])
        if direction<0
            value>=cutoff ? (lo=mid) : (hi=mid)
        else
            value>=cutoff ? (hi=mid) : (lo=mid)
        end
        if abs(value-cutoff)<=tol
            point=cache[mid]
            return (estimate=mid,status=:available,lr=value,tolerance=tol,point)
        end
    end
    estimate=_cjs_profile_midpoint(lo,hi)
    phase_ref[]=(direction=direction < 0 ? :lower : :upper,phase=:endpoint_check)
    value,status,tolerance,point=evaluate(estimate)
    status==:available || return (estimate=missing,status,lr=value,tolerance,point)
    abs(value-cutoff)<=tolerance ||
        return (estimate=missing,status=:endpoint_lr_not_converged,lr=value,tolerance,point)
    return (estimate,status=:available,lr=value,tolerance,point)
end

function _cjs_individual_coefficient_profile(fit,target;level,name,
        profile_point=_cjs_individual_profile_point,observe_point=nothing)
    estimate=fit.beta[target.index]
    baseline=-fit.loglik
    cutoff=quantile(Chisq(1),level)
    cache=Dict{Float64,NamedTuple}()
    mapping=nothing
    if hasproperty(fit,:prepared) && hasproperty(fit.prepared,:individual)
        for process in (:phi,:p)
            candidate=getproperty(fit.prepared.individual,process)
            candidate===nothing && continue
            target.index in candidate.scale_range && (mapping=candidate;break)
        end
    end
    transform=mapping===nothing ? (value,start,zero_process)->start :
        (value,start,zero_process)->_cjs_individual_scale_profile_start(
            mapping,target.index,value,start;zero_process)
    context=Ref((direction=:center,phase=:center))
    function traced_profile_point(f,index,value;start,zero_process=nothing)
        point=profile_point(f,index,value;start,zero_process)
        if observe_point!==nothing
            lr,status,tolerance=_cjs_individual_profile_lr(point,fit)
            trials=hasproperty(point,:attempts) ? deepcopy(point.attempts) : NamedTuple[]
            converged=[hasproperty(trial,:converged) ? trial.converged : missing for trial in trials]
            gradients=[hasproperty(trial,:check) && hasproperty(trial.check,:gradient_norm) ?
                trial.check.gradient_norm : missing for trial in trials]
            curvature=[hasproperty(trial,:nuisance_curvature) ?
                deepcopy(trial.nuisance_curvature) : missing for trial in trials]
            order=hasproperty(point,:order) ? point.order : missing
            refinement=hasproperty(point,:refinement) ? deepcopy(point.refinement) : NamedTuple[]
            observe_point((target=name,process=target.process,scale=target.scale,
                direction=context[].direction,phase=context[].phase,
                attempted_value=Float64(value),internal_log_sd=target.scale==:log_sd ? Float64(value) : missing,
                zero_boundary=false,
                objective=hasproperty(point,:objective) ? point.objective : missing,
                likelihood_ratio=lr,tolerance,status,optimizer_trials=trials,converged,
                gradient_norms=gradients,curvature,quadrature_order=order,
                quadrature_refinement=refinement))
        end
        return point
    end
    point_function=observe_point===nothing ?
        ((f,index,value;start,zero_process=nothing)->
            profile_point(f,index,value;start,zero_process)) : traced_profile_point
    path_factory=()->_cjs_individual_profile_path(fit.beta,(value,start,zero_process)->
        point_function(fit,target.index,value;start,zero_process);start_transform=transform)
    context[]=(direction=:center,phase=:center)
    center=path_factory().point_at(estimate)
    center_lr,center_status,center_tolerance=_cjs_individual_profile_lr(center,fit)
    attempt_base=(target=name,index=target.index,scale=target.scale,estimate,center,
        center_lr,center_tolerance)
    if center_status!=:available
        return (_cjs_record(name,estimate,missing,missing,center_status),
            merge(attempt_base,(lower=missing,upper=missing,status=center_status,points=cache)))
    end
    abs(center_lr)<=center_tolerance ||
        return (_cjs_record(name,estimate,missing,missing,:center_inconsistent_with_fit),
            merge(attempt_base,(lower=missing,upper=missing,status=:center_inconsistent_with_fit,points=cache)))
    cache[estimate]=center
    lower=_cjs_individual_profile_target_endpoint(fit,path_factory(),cache,estimate,-1,cutoff;
        phase_ref=context)
    if lower.status==:profile_improves_on_fit
        return (_cjs_record(name,estimate,missing,missing,lower.status),
            merge(attempt_base,(lower,upper=missing,status=lower.status,
                points=sort!(collect(cache);by=first))))
    end
    upper=_cjs_individual_profile_target_endpoint(fit,path_factory(),cache,estimate,1,cutoff;
        phase_ref=context)
    invalid_fit=upper.status==:profile_improves_on_fit
    refinement_failure=lower.status in (:failed_refinement,:unrefined_at_cap) ? lower.status :
        upper.status in (:failed_refinement,:unrefined_at_cap) ? upper.status : nothing
    status=invalid_fit ? :profile_improves_on_fit :
        lower.status==:available && upper.status==:available ? :available :
        lower.status==:failed_reoptimization || upper.status==:failed_reoptimization ? :failed_reoptimization :
        refinement_failure!==nothing ? refinement_failure :
        lower.status!=:available ? lower.status : upper.status
    record=_cjs_record(name,estimate,invalid_fit ? missing : lower.estimate,
        invalid_fit ? missing : upper.estimate,status)
    attempt=merge(attempt_base,(lower,upper,status,points=sort!(collect(cache);by=first)))
    return (record,attempt)
end

function _cjs_individual_sd_profile(fit,target;level,name,
        profile_point=_cjs_individual_profile_point,observe_point=nothing)
    mapping=getproperty(fit.prepared.individual,target.process)
    estimate=exp(fit.beta[target.index])
    isfinite(estimate) && estimate>0 ||
        return (_cjs_record(name,estimate,missing,missing,:guarded_unavailable),
            (target=name,index=target.index,scale=:positive_sd,status=:guarded_unavailable,reason="non-finite or non-positive SD"))
    estimate<=1e-6 &&
        return (_cjs_record(name,estimate,missing,missing,:boundary_estimate),
            (target=name,index=target.index,scale=:positive_sd,status=:boundary_estimate,reason="boundary interval calibration remains unqualified"))
    baseline=-fit.loglik
    cutoff=quantile(Chisq(1),level)
    cache=Dict{Float64,NamedTuple}()
    context=Ref((direction=:center,phase=:center))
    function emit_point!(point,sigma;zero_boundary=false)
        observe_point===nothing && return point
        lr,status,tolerance=_cjs_individual_profile_lr(point,fit)
        trials=hasproperty(point,:attempts) ? deepcopy(point.attempts) : NamedTuple[]
        converged=[hasproperty(trial,:converged) ? trial.converged : missing for trial in trials]
        gradients=[hasproperty(trial,:check) && hasproperty(trial.check,:gradient_norm) ?
            trial.check.gradient_norm : missing for trial in trials]
        curvature=[hasproperty(trial,:nuisance_curvature) ?
            deepcopy(trial.nuisance_curvature) : missing for trial in trials]
        order=hasproperty(point,:order) ? point.order : missing
        refinement=hasproperty(point,:refinement) ? deepcopy(point.refinement) : NamedTuple[]
        observe_point((target=name,process=target.process,scale=:positive_sd,
            direction=context[].direction,phase=context[].phase,attempted_value=Float64(sigma),
            internal_log_sd=zero_boundary ? missing : log(Float64(sigma)),zero_boundary,
            objective=hasproperty(point,:objective) ? point.objective : missing,
            likelihood_ratio=lr,tolerance,status,optimizer_trials=trials,converged,
            gradient_norms=gradients,curvature,quadrature_order=order,
            quadrature_refinement=refinement))
        return point
    end
    path_factory=()->_cjs_individual_profile_path(fit.beta,(sigma,start,zero_process)->begin
        point=sigma==0 ? profile_point(fit,target.index,0.;zero_process=target.process,start) :
            profile_point(fit,target.index,log(sigma);start)
        return emit_point!(point,sigma;zero_boundary=zero_process!==nothing)
    end)
    current_path=path_factory()
    point_at=function(sigma)
        _cjs_individual_profile_cached_point(current_path,cache,sigma)
    end
    context[]=(direction=:center,phase=:center)
    center=point_at(estimate)
    center_lr,center_status,center_tolerance=_cjs_individual_profile_lr(center,fit)
    attempt_base=(target=name,index=target.index,scale=:positive_sd,estimate,center,
        center_lr,center_tolerance)
    if center_status!=:available
        return (_cjs_record(name,estimate,missing,missing,center_status),
            merge(attempt_base,(lower=missing,upper=missing,status=center_status,points=sort!(collect(cache);by=first))))
    end
    abs(center_lr)<=center_tolerance ||
        return (_cjs_record(name,estimate,missing,missing,:center_inconsistent_with_fit),
            merge(attempt_base,(lower=missing,upper=missing,status=:center_inconsistent_with_fit,points=sort!(collect(cache);by=first))))
    context[]=(direction=:lower,phase=:zero_boundary)
    zero_point=emit_point!(profile_point(fit,target.index,0.;zero_process=target.process,start=fit.beta),
        0.;zero_boundary=true)
    cache[0.]=zero_point
    zero_lr,zero_status,zero_tolerance=_cjs_individual_profile_lr(zero_point,fit)
    if zero_status!=:available
        return (_cjs_record(name,estimate,missing,missing,zero_status),
            merge(attempt_base,(lower=missing,upper=missing,status=zero_status,zero=(point=zero_point,lr=zero_lr,status=zero_status),points=sort!(collect(cache);by=first))))
    end
    current_path=path_factory()
    current_path.reset!(zero_point.theta)
    lower=if zero_lr<=cutoff
        (estimate=0.,status=:boundary_included,lr=zero_lr,tolerance=zero_tolerance,point=zero_point)
    else
        lo=0.;hi=estimate;lastpoint=zero_point;lastlr=zero_lr;lasttol=zero_tolerance;failed=nothing
        for _ in 1:45
            mid=_cjs_profile_midpoint(lo,hi)
            context[]=(direction=:lower,phase=:bisection)
            point=point_at(mid)
            lr,status,tolerance=_cjs_individual_profile_lr(point,fit)
            if status!=:available
                failed=(status=status,lr,tolerance,point);break
            elseif lr>=cutoff
                lo=mid
            else
                hi=mid
            end
            lastpoint=point;lastlr=lr;lasttol=tolerance
            if abs(lr-cutoff)<=tolerance
                lo=hi=mid
                break
            end
        end
        if failed!==nothing
            (estimate=missing,status=failed.status,lr=failed.lr,tolerance=failed.tolerance,point=failed.point)
        else
            value=_cjs_profile_midpoint(lo,hi)
            context[]=(direction=:lower,phase=:endpoint_check)
            point=point_at(value)
            lr,status,tolerance=_cjs_individual_profile_lr(point,fit)
            status==:available && abs(lr-cutoff)<=tolerance ?
                (estimate=value,status=:available,lr,tolerance,point) :
                (estimate=missing,status=status==:available ? :endpoint_lr_not_converged : status,lr,tolerance,point)
        end
    end
    if lower.status==:profile_improves_on_fit
        return (_cjs_record(name,estimate,missing,missing,lower.status),
            merge(attempt_base,(zero=(point=zero_point,lr=zero_lr,status=zero_status),
                lower,upper=missing,status=lower.status,points=sort!(collect(cache);by=first))))
    end
    current_path=path_factory()
    hi=max(estimate*1.5,estimate+.1);lo=estimate
    isfinite(hi) ||
        return (_cjs_record(name,estimate,lower.estimate,missing,:guarded_unavailable),
            merge(attempt_base,(zero=(point=zero_point,lr=zero_lr,status=zero_status),lower,
                upper=(estimate=missing,status=:guarded_unavailable,reason="positive-SD expansion overflowed"),
                status=:guarded_unavailable,points=sort!(collect(cache);by=first))))
    context[]=(direction=:upper,phase=:expansion)
    point=point_at(hi)
    lr,status,tolerance=_cjs_individual_profile_lr(point,fit)
    overflowed=false
    for _ in 1:12
        status==:available || break
        lr>=cutoff && break
        # Grow the distance from the MLE instead of doubling the SD itself.
        # A large jump can skip a refined likelihood-ratio crossing and land
        # beyond the quadrature refinement cap, falsely withholding the bound.
        next_hi=estimate+1.5*(hi-estimate)
        if !isfinite(next_hi)
            overflowed=true
            break
        end
        lo=hi;hi=next_hi
        context[]=(direction=:upper,phase=:expansion)
        point=point_at(hi)
        lr,status,tolerance=_cjs_individual_profile_lr(point,fit)
    end
    upper=if overflowed
        (estimate=missing,status=:guarded_unavailable,lr,tolerance,point,
            reason="positive-SD expansion overflowed")
    elseif status!=:available
        (estimate=missing,status,lr,tolerance,point)
    elseif lr<cutoff
        (estimate=missing,status=:endpoint_not_crossed,lr,tolerance,point)
    else
        failed=nothing
        for _ in 1:45
            mid=_cjs_profile_midpoint(lo,hi)
            context[]=(direction=:upper,phase=:bisection)
            point_mid=point_at(mid)
            value,point_status,tol=_cjs_individual_profile_lr(point_mid,fit)
            if point_status!=:available
                failed=(status=point_status,lr=value,tolerance=tol,point=point_mid);break
            elseif value>=cutoff
                hi=mid
            else
                lo=mid
            end
            if abs(value-cutoff)<=tol
                lo=hi=mid
                break
            end
        end
        if failed!==nothing
            (estimate=missing,status=failed.status,lr=failed.lr,tolerance=failed.tolerance,point=failed.point)
        else
            value=_cjs_profile_midpoint(lo,hi)
            context[]=(direction=:upper,phase=:endpoint_check)
            value_point=point_at(value)
            value_lr,value_status,value_tolerance=_cjs_individual_profile_lr(value_point,fit)
            value_status==:available && abs(value_lr-cutoff)<=value_tolerance ?
                (estimate=value,status=:available,lr=value_lr,tolerance=value_tolerance,point=value_point) :
                (estimate=missing,status=value_status==:available ? :endpoint_lr_not_converged : value_status,
                    lr=value_lr,tolerance=value_tolerance,point=value_point)
        end
    end
    invalid_fit=upper.status==:profile_improves_on_fit
    refinement_failure=lower.status in (:failed_refinement,:unrefined_at_cap) ? lower.status :
        upper.status in (:failed_refinement,:unrefined_at_cap) ? upper.status : nothing
    result_status=invalid_fit ? :profile_improves_on_fit :
        refinement_failure!==nothing ? refinement_failure :
        lower.status==:boundary_included && upper.status==:available ? :boundary_included :
        lower.status==:available && upper.status==:available ? :available :
        lower.status!=:available && lower.status!=:boundary_included ? lower.status : upper.status
    record=_cjs_record(name,estimate,invalid_fit ? missing : lower.estimate,
        invalid_fit ? missing : upper.estimate,result_status)
    attempt=merge(attempt_base,(zero=(point=zero_point,lr=zero_lr,status=zero_status),lower,upper,
        status=result_status,points=sort!(collect(cache);by=first)))
    return (record,attempt)
end

function _cjs_individual_profile(fit; level, targets,
        profile_point=_cjs_individual_profile_point,observe_point=nothing)
    names=_cjs_coef_names(fit)
    raw_targets=targets===nothing ? names : collect(targets)
    all(name->name isa AbstractString,raw_targets) ||
        throw(ArgumentError("CJS profile targets must be exact fitted coefficient labels supplied as strings"))
    requested=String.(raw_targets)
    isempty(requested) && throw(ArgumentError("at least one CJS profile target is required"))
    guard_reason=_cjs_individual_profile_guard(fit)
    if guard_reason!==nothing
        records=NamedTuple[]
        attempts=NamedTuple[]
        nfixed=fit.prepared.nphi+fit.prepared.np
        for name in requested
            index=findfirst(==(name),names)
            if index!==nothing
                target=(index,scale=index<=nfixed ? :logit : :log_sd,alias=false,
                    status=:ready,reason=nothing)
            elseif name in ("sd:phi","sd:p")
                process=name=="sd:phi" ? :phi : :p
                mapping=getproperty(fit.prepared.individual,process)
                mapping===nothing && throw(ArgumentError("CJS profile target $name is inactive in this fit"))
                target=(index=mapping.scale_index,scale=:positive_sd,alias=true,
                    status=:ready,reason=nothing)
            else
                _cjs_individual_profile_target(fit,name) # throws for unknown labels
                error("unreachable profile target validation state")
            end
            estimate=target.index===nothing ? missing : target.alias ? exp(fit.beta[target.index]) : fit.beta[target.index]
            status=target.status==:ready ? :guarded_unavailable : target.status
            reason=target.status==:ready ? guard_reason : target.reason
            push!(records,_cjs_record(name,estimate,missing,missing,status))
            push!(attempts,(target=name,index=target.index,scale=target.scale,status,reason))
        end
        return (method=:profile,level,intervals=records,attempts,requested=length(requested),
            status=:guarded_unavailable,reason=guard_reason,calibrated=false)
    end
    resolved=[_cjs_individual_profile_target(fit,name) for name in requested]
    records=NamedTuple[]
    attempts=NamedTuple[];invalid_fit=false
    for (name,target) in zip(requested,resolved)
        if invalid_fit
            estimate=target.index===nothing ? missing : target.alias ? exp(fit.beta[target.index]) : fit.beta[target.index]
            push!(records,_cjs_record(name,estimate,missing,missing,:not_attempted_invalid_fit))
            push!(attempts,_cjs_profile_unattempted(name,target.index,target.scale))
            continue
        end
        if target.status!=:ready
            push!(records,_cjs_record(name,missing,missing,missing,target.status))
            push!(attempts,(target=name,index=target.index,scale=target.scale,status=target.status,reason=target.reason))
            continue
        end
        record,attempt=target.alias ? _cjs_individual_sd_profile(fit,target;level,name,
            profile_point,observe_point) :
            _cjs_individual_coefficient_profile(fit,target;level,name,
                profile_point,observe_point)
        push!(records,record)
        push!(attempts,attempt)
        invalid_fit=record.status==:profile_improves_on_fit
    end
    invalid_fit && _cjs_profile_invalidate_records!(records)
    complete=count(r->r.status in (:available,:boundary_included),records)
    status=invalid_fit ? :profile_improves_on_fit : complete==length(records) ? :profile_available : :partial
    return (method=:profile,level,intervals=records,attempts,requested=length(requested),
        completed=complete,status,cutoff=quantile(Chisq(1),level),calibrated=false,
        reason="ordinary individual coefficient profile likelihood; exact-zero constant-SD limits are separate; calibration not established")
end

# Inference eligibility is distinct from prediction/simulation provenance.
# Selected coefficient and SD profiles retain every constrained optimization.
function _cjs_selected_profile(fit;level,targets,profile_point=_cjs_selection_profile_point)
    0<level<1 || throw(ArgumentError("confidence level must be strictly between zero and one"))
    prep=fit.prepared
    hasproperty(prep,:selection) && prep.selection!==nothing ||
        throw(ArgumentError("selected CJS profiles require reference-selection conditioning"))
    _cjs_is_individual(prep) && !_cjs_is_shared(prep) ||
        throw(ArgumentError("selected CJS profiles require individual-only integration"))
    diag=fit.diagnostics
    length(fit.beta)==prep.nphi+prep.np+prep.individual.nscale && all(isfinite,fit.beta) && isfinite(fit.loglik) ||
        throw(ArgumentError("selected CJS profile parameter dimensions or values are invalid"))
    names=_cjs_coef_names(fit);nfixed=prep.nphi+prep.np
    raw=targets===nothing ? names[1:nfixed] : collect(targets)
    all(name->name isa AbstractString,raw) || throw(ArgumentError("selected CJS profile targets must be exact labels supplied as strings"))
    requested=String.(raw)
    isempty(requested) && throw(ArgumentError("at least one selected CJS profile target is required"))
    resolved=[_cjs_individual_profile_target(fit,name) for name in requested]
    admitted=hasproperty(diag,:converged) && diag.converged===true &&
        hasproperty(diag,:gradient_norm) && isfinite(diag.gradient_norm) &&
        0<=diag.gradient_norm<=1e-4 && hasproperty(diag,:quadrature_refined) && diag.quadrature_refined===true
    guard_reason=admitted ? nothing :
        (hasproperty(diag,:uncertainty_reason) && diag.uncertainty_reason isa AbstractString &&
         !isempty(diag.uncertainty_reason) ? String(diag.uncertainty_reason) :
         "selected CJS profiles require a converged fit, small gradient and refined quadrature")
    boundary=hasproperty(diag,:boundary) && diag.boundary===true
    if guard_reason!==nothing || boundary
        reason=guard_reason===nothing ? "selected profile boundary fit remains guarded" : guard_reason
        records=NamedTuple[];attempts=NamedTuple[]
        for (name,target) in zip(requested,resolved)
            coordinate=target.alias ? :log_sd : target.scale
            estimate=target.index===nothing ? missing : target.alias ? exp(fit.beta[target.index]) : fit.beta[target.index]
            push!(records,merge(_cjs_record(name,estimate,missing,missing,:guarded_unavailable),
                (coordinate,interval_scale=target.scale)))
            push!(attempts,(target=name,index=target.index,scale=target.scale,coordinate,
                interval_scale=target.scale,status=:guarded_unavailable,reason))
        end
        return (method=:profile,level,intervals=records,attempts,requested=length(requested),
            status=:guarded_unavailable,reason,cutoff=quantile(Chisq(1),level),calibrated=false)
    end
    cutoff=quantile(Chisq(1),level);records=NamedTuple[];attempts=NamedTuple[];invalid_fit=false
    for (name,target) in zip(requested,resolved)
        coordinate=target.alias ? :log_sd : target.scale
        if invalid_fit
            estimate=target.index===nothing ? missing : target.alias ? exp(fit.beta[target.index]) : fit.beta[target.index]
            push!(records,merge(_cjs_record(name,estimate,missing,missing,:not_attempted_invalid_fit),
                (coordinate,interval_scale=target.scale)))
            push!(attempts,merge(_cjs_profile_unattempted(name,target.index,target.scale),
                (coordinate,interval_scale=target.scale)))
            continue
        end
        if target.status!=:ready || diag.boundary
            reason=target.status!=:ready ? target.reason : "selected profile boundary fit remains guarded"
            push!(records,merge(_cjs_record(name,missing,missing,missing,:guarded_unavailable),
                (coordinate,interval_scale=target.scale)))
            push!(attempts,(target=name,index=target.index,scale=target.scale,coordinate,
                interval_scale=target.scale,status=:guarded_unavailable,reason))
            continue
        end
        # The common controller is supplied the selected point routine, whose
        # admission separately checks conditional, numerator and normalizer integrals.
        coefficient_target=merge(target,(scale=target.alias ? :log_sd : target.scale,alias=false))
        selected_points=NamedTuple[]
        traced_point=function(f,index,value;start=f.beta,zero_process=nothing)
            point=try
                profile_point(f,index,value;start,zero_process)
            catch err
                err isa InterruptException && rethrow()
                theta=copy(f.beta);theta[index]=value
                (objective=missing,status=:failed_reoptimization,theta,
                    refinement=NamedTuple[],attempts=NamedTuple[],reason=sprint(showerror,err))
            end
            lr,status=_cjs_selection_profile_lr(point,f)
            tolerance=_cjs_selection_profile_tolerance(point,f)
            phase=value==f.beta[index] ? :center : :endpoint_search
            push!(selected_points,(;value,phase,point,lr,status,tolerance))
            return point
        end
        record,attempt=_cjs_individual_coefficient_profile(fit,coefficient_target;level,name,
            profile_point=traced_point)
        # Preserve the selected public per-side ledger. The center is shared;
        # each side includes only values evaluated in that direction, in call order.
        lower=ismissing(attempt.lower) ? missing : merge(attempt.lower,
            (points=filter(p->p.value<=fit.beta[target.index],selected_points),))
        upper=ismissing(attempt.upper) ? missing : merge(attempt.upper,
            (points=filter(p->p.value>=fit.beta[target.index],selected_points),))
        attempt=merge(attempt,(lower,upper,selected_points=copy(selected_points)))
        if target.alias
            log_record=record;estimate=exp(fit.beta[target.index])
            if record.status==:available
                lower=exp(record.lower);upper=exp(record.upper)
                finite=all(x->isfinite(x)&&x>0,(estimate,lower,upper))
                record=_cjs_record(name,isfinite(estimate)&&estimate>0 ? estimate : missing,
                    finite ? lower : missing,finite ? upper : missing,
                    finite ? :available : :transform_unavailable)
            else
                record=_cjs_record(name,isfinite(estimate)&&estimate>0 ? estimate : missing,
                    missing,missing,record.status)
            end
            attempt=merge(attempt,(scale=:positive_sd,estimate,log_interval=log_record,
                log_estimate=fit.beta[target.index],status=record.status))
        end
        push!(records,merge(record,(coordinate,interval_scale=target.scale)))
        push!(attempts,merge(attempt,(coordinate,interval_scale=target.scale)))
        invalid_fit=record.status==:profile_improves_on_fit
    end
    invalid_fit && _cjs_profile_invalidate_records!(records)
    completed=count(r->r.status==:available,records)
    status=invalid_fit ? :profile_improves_on_fit : completed==length(records) ? :profile_available :
        all(r->r.status==:guarded_unavailable,records) ? :guarded_unavailable : :partial
    return (method=:profile,level,intervals=records,attempts,requested=length(requested),
        completed,status,cutoff,calibrated=false,
        reason="selection-conditioned coefficient profiles; interior constant-SD limits exponentiate valid log-SD limits; exact-zero inference and calibration remain separate")
end

function _cjs_bootstrap(fit; level, targets, nsim, seed, rng, min_usable)
    fit.diagnostics.converged && isfinite(fit.diagnostics.gradient_norm) &&
        fit.diagnostics.gradient_norm <= 1e-4 ||
        throw(ArgumentError("CJS inference requires a converged fit with a small gradient."))
    nsim isa Integer && !(nsim isa Bool) && nsim >= 1 || throw(ArgumentError("bootstrap nsim must be a positive integer"))
    min_usable isa Integer && !(min_usable isa Bool) || throw(ArgumentError("bootstrap min_usable must be an integer"))
    min_usable >= 1 || throw(ArgumentError("min_usable must be a positive integer"))
    indices = _cjs_target_indices(fit, targets)
    drawrng = _cjs_random(rng, seed)
    attempts = NamedTuple[]
    successful = Vector{Vector{Float64}}()
    formulas = (phi=fit.phi_design.formula, p=fit.p_design.formula)
    for replicate in 1:nsim
        try
            simdata = only(simulate(fit; nsim=1, rng=drawrng))
            refit = cjs(simdata; phi=formulas.phi, p=formulas.p,
                        terminal_constraint=fit.prepared.terminal_constraint,
                        start=fit.beta, contrasts=fit.phi_design.contrasts)
            usable = refit.diagnostics.converged && refit.diagnostics.gradient_norm <= 1e-4 &&
                     refit.diagnostics.positive_curvature && !refit.diagnostics.boundary && all(isfinite, refit.beta)
            push!(attempts, (replicate=replicate, status=usable ? :usable : :unusable_fit,
                             converged=refit.diagnostics.converged, usable=usable,
                             reason=refit.diagnostics.uncertainty_reason, coefficients=copy(refit.beta)))
            usable && push!(successful, refit.beta)
        catch err
            err isa InterruptException && rethrow()
            push!(attempts, (replicate=replicate, status=:failed, converged=false, usable=false,
                             reason=sprint(showerror, err), coefficients=missing))
        end
    end
    enough = length(successful) >= min_usable && !isempty(successful)
    loq, hiq = (1-level)/2, 1-(1-level)/2
    records = NamedTuple[]
    for (name, i) in indices
        vals = sort([b[i] for b in successful])
        lower = enough && !isempty(vals) ? _cjs_quantile(vals, loq) : missing
        upper = enough && !isempty(vals) ? _cjs_quantile(vals, hiq) : missing
        status = enough && !isempty(vals) ? :available : :insufficient_usable
        push!(records, _cjs_record(name, fit.beta[i], lower, upper, status;
                                   usable=length(successful), requested=nsim))
    end
    return (method=:bootstrap, level=level, intervals=records, attempts=attempts,
            requested=nsim, usable=length(successful), status=enough ? :available : :insufficient_usable,
            reason="percentile coefficient intervals; calibration not established")
end

function _cjs_quantile(x, p)
    length(x) == 1 && return only(x)
    h = (length(x)-1)*p + 1
    lo = floor(Int, h); hi = ceil(Int, h)
    return x[lo] + (h-lo)*(x[hi]-x[lo])
end

"""CJS uncertainty summaries with explicit availability status.

Native shared-effect `method=:profile` constrains fitted fixed and log-SD
coefficients and reoptimizes nuisance parameters using the joint CJS Laplace
objective. Interior constant-SD targets exponentiate finite log-SD profile
endpoints; SD-formula targets are coefficients on the log-SD scale. Boundary
fits and exact-zero SD limits remain guarded. Returned shared profiles have `calibrated=false`;
numerical availability does not establish interval coverage.
"""
function confint(fit::CJSFit; method=:wald, level=0.95, targets=nothing,
                          nsim=199, seed=nothing, rng=Random.default_rng(), min_usable=nothing)
    0 < level < 1 || throw(ArgumentError("confidence level must be strictly between zero and one"))
    method in (:wald, :profile, :bootstrap) || throw(ArgumentError("method must be :wald, :profile or :bootstrap"))
    if _cjs_is_shared(fit.prepared) && method == :profile
        return _cjs_shared_profile(fit; level, targets)
    end
    if _cjs_is_individual(fit.prepared) && method==:profile
        if hasproperty(fit.prepared,:selection) && fit.prepared.selection!==nothing
            return _cjs_selected_profile(fit;level,targets)
        end
        return _cjs_individual_profile(fit;level,targets)
    end
    if _cjs_is_shared(fit.prepared) || _cjs_is_individual(fit.prepared)
        indices=_cjs_target_indices(fit,targets)
        return (method=method,level=level,
                intervals=[_cjs_record(name,fit.beta[i],missing,missing,:guarded_unavailable) for (name,i) in indices],
                attempts=NamedTuple[],requested=0,status=:guarded_unavailable,
                reason=fit.diagnostics.uncertainty_reason)
    end
    if method == :wald
        return _cjs_wald(fit; level, targets)
    elseif method == :profile
        return _cjs_profile(fit; level, targets)
    end
    nsim isa Integer && !(nsim isa Bool) && nsim >= 1 || throw(ArgumentError("bootstrap nsim must be a positive integer"))
    minimum = min_usable === nothing ? ceil(Int, 0.8nsim) : min_usable
    minimum isa Integer && !(minimum isa Bool) || throw(ArgumentError("min_usable must be an integer"))
    return _cjs_bootstrap(fit; level, targets, nsim, seed, rng, min_usable=minimum)
end
