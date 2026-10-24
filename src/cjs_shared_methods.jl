# Native methods for ordinary shared-effect CJS fits. These methods are kept
# separate from the public dispatch in cjs_methods.jl so the shared route has
# its own admission, alignment, and prediction-integration checks.

using Random: Random
using LinearAlgebra: eigvals, Symmetric, norm

const _CJS_SHARED_INNER_DENSE_CAP = 256

function _cjs_shared_methods_guard(fit)
    fit isa CJSFit || throw(ArgumentError("cjs shared methods require a native CJSFit object"))
    prep = fit.prepared
    _cjs_is_shared(prep) || throw(ArgumentError("cjs shared methods require an ordinary shared-effect fit"))
    sh = prep.shared
    sh.route in (:site_phi, :site_p, :site_pair, :year_vector) ||
        throw(ArgumentError("cjs shared methods require route site_phi, site_p, site_pair or year_vector"))
    active = (phi=sh.phi !== nothing, p=sh.p !== nothing)
    actual_route = if any(m -> m !== nothing && m.varying, (sh.phi, sh.p))
        :year_vector
    elseif active.phi && active.p
        :site_pair
    elseif active.phi
        :site_phi
    elseif active.p
        :site_p
    else
        throw(ArgumentError("cjs shared route has no active process"))
    end
    sh.route == actual_route || throw(ArgumentError("cjs shared route and active process designs disagree"))

    hasproperty(prep, :prediction_contract) ||
        throw(ArgumentError("cjs shared methods require version-one prediction provenance; refit the model"))
    provenance = prep.prediction_contract
    provenance isa NamedTuple && keys(provenance) == (:version, :selection_enabled, :sd_formula_present) &&
        provenance.version isa Integer && !(provenance.version isa Bool) && provenance.version == 1 &&
        provenance.selection_enabled isa Bool && provenance.sd_formula_present isa NamedTuple &&
        keys(provenance.sd_formula_present) == (:phi, :p) &&
        all(x -> x isa Bool, values(provenance.sd_formula_present)) ||
        throw(ArgumentError("cjs shared prediction provenance is malformed; refit the model"))
    selected = hasproperty(prep, :selection) && prep.selection !== nothing
    sd_formula = (phi=active.phi && hasproperty(sh.phi, :sd_formula) && sh.phi.sd_formula !== nothing,
                  p=active.p && hasproperty(sh.p, :sd_formula) && sh.p.sd_formula !== nothing)
    provenance.selection_enabled == selected && provenance.sd_formula_present == sd_formula ||
        throw(ArgumentError("cjs shared prediction provenance disagrees with the prepared model; refit the model"))
    selected && throw(ArgumentError("cjs reference-selection prediction and simulation are unsupported"))
    _cjs_is_individual(prep) && throw(ArgumentError("cjs shared-plus-individual effects are unsupported"))
    hasproperty(prep, :correlation) && prep.correlation !== nothing &&
        throw(ArgumentError("cjs correlated process effects are unsupported"))

    qfixed = prep.nphi + prep.np
    q = qfixed + sh.nscale
    length(fit.beta) == q && isfinite(fit.loglik) && all(isfinite, fit.beta) ||
        throw(ArgumentError("cjs shared fit coefficients, likelihood or parameter dimensions are invalid"))
    sh.nscale == sum(m === nothing ? 0 : hasproperty(m, :scale_range) ? length(m.scale_range) : 1 for m in (sh.phi, sh.p)) ||
        throw(ArgumentError("cjs shared scale dimension disagrees with active processes"))
    diag = fit.diagnostics
    hasproperty(diag, :integration) && diag.integration == :joint_CJS_Laplace ||
        throw(ArgumentError("cjs shared methods require retained native joint CJS Laplace provenance"))
    hasproperty(diag, :converged) && diag.converged === true &&
        hasproperty(diag, :gradient_norm) && diag.gradient_norm isa Real &&
        isfinite(diag.gradient_norm) && 0 <= diag.gradient_norm <= 1e-4 ||
        throw(ArgumentError("cjs shared methods require outer convergence and finite gradient norm at most 1e-4"))
    hasproperty(diag, :gradient) && diag.gradient isa AbstractVector && length(diag.gradient) == q &&
        all(isfinite, diag.gradient) && norm(diag.gradient, Inf) <= 1e-4 ||
        throw(ArgumentError("cjs shared methods require a finite retained outer CJS gradient of the fitted dimension"))
    hasproperty(diag, :inner) && diag.inner !== nothing ||
        throw(ArgumentError("cjs shared methods require retained joint CJS inner curvature evidence"))
    inner = diag.inner
    hasproperty(inner, :mode) && inner.mode isa AbstractVector && length(inner.mode) == sh.dimension &&
        all(isfinite, inner.mode) ||
        throw(ArgumentError("cjs shared inner mode dimension or values do not match the fitted group design"))
    hasproperty(inner, :hessian) && inner.hessian isa AbstractMatrix &&
        size(inner.hessian) == (sh.dimension, sh.dimension) && all(isfinite, inner.hessian) ||
        throw(ArgumentError("cjs shared inner joint Hessian is unavailable or has incompatible dimensions"))
    hasproperty(inner, :gradient_norm) && inner.gradient_norm isa Real &&
        isfinite(inner.gradient_norm) && 0 <= inner.gradient_norm <= 1e-4 &&
        hasproperty(inner, :converged) && inner.converged === true ||
        throw(ArgumentError("cjs shared inner mode is not retained as a stationary converged mode"))
    hasproperty(inner, :route) && inner.route == sh.route ||
        throw(ArgumentError("cjs shared inner Laplace route disagrees with the prepared group design"))
    sh.dimension <= _CJS_SHARED_INNER_DENSE_CAP ||
        throw(ArgumentError("cjs shared methods require retained positive-curvature evidence; joint inner dimension exceeds the dense diagnostic cap of $(_CJS_SHARED_INNER_DENSE_CAP)"))
    λ = try
        eigvals(Symmetric(Matrix{Float64}(inner.hessian)))
    catch err
        err isa InterruptException && rethrow()
        throw(ArgumentError("cjs shared joint inner curvature could not be checked: $(sprint(showerror, err))"))
    end
    λmin, λmax = minimum(λ), maximum(λ)
    isfinite(λmin) && isfinite(λmax) && λmax > 0 && λmin > max(0., λmax * 1e-10) ||
        throw(ArgumentError("cjs shared joint CJS inner Hessian is not positive definite at the required relative threshold"))

    ranges = UnitRange{Int}[]
    for (label, m, des) in ((:phi, sh.phi, fit.phi_design), (:p, sh.p, fit.p_design))
        m === nothing && continue
        haskey(des, :rows) && length(m.index) == length(des.rows) &&
            all(i -> i isa Integer && 0 <= i <= length(m.levels), m.index) ||
            throw(ArgumentError("cjs shared $label group indices do not match the fitted design rows"))
        length(unique(m.levels)) == length(m.levels) && !isempty(m.levels) ||
            throw(ArgumentError("cjs shared $label group labels are empty or duplicated"))
        hasproperty(prep, :data) && hasproperty(prep.data, label == :phi ? :phi_data : :p_data) ||
            throw(ArgumentError("cjs shared $label keyed predictor table is unavailable"))
        table = label == :phi ? prep.data.phi_data : prep.data.p_data
        haskey(table, m.group) || throw(ArgumentError("cjs shared $label grouping column $(m.group) is absent"))
        hasproperty(m, :scale_index) && m.scale_index isa Integer && qfixed < m.scale_index <= q ||
            throw(ArgumentError("cjs shared $label scale index is invalid"))
        _cjs_shared_method_sigmas(fit.beta, m, qfixed, label)
        if hasproperty(m, :range)
            m.range isa UnitRange{<:Integer} && length(m.range) == length(m.levels) &&
                first(m.range) >= 1 && last(m.range) <= sh.dimension ||
                throw(ArgumentError("cjs shared $label latent range dimension is invalid"))
            push!(ranges, m.range)
        end
        values = getproperty(table, m.group)
        all(k -> begin
                row = des.rows[k]
                row isa Integer && 1 <= row <= length(values) || return false
                index = m.index[k]
                index == 0 ? true : (!ismissing(values[row]) && isequal(values[row], m.levels[index]))
            end, eachindex(m.index)) ||
            throw(ArgumentError("cjs shared $label group labels and retained fitted level indices disagree"))
    end
    if length(ranges) == 2
        isempty(intersect(ranges[1], ranges[2])) ||
            throw(ArgumentError("cjs shared process latent ranges overlap"))
    end
    scale_coordinates = vcat([hasproperty(m, :scale_range) ? collect(m.scale_range) : [m.scale_index]
        for m in (sh.phi, sh.p) if m !== nothing]...)
    sort(scale_coordinates) == collect(qfixed+1:q) ||
        throw(ArgumentError("cjs shared SD coefficient ranges overlap or omit fitted coordinates"))
    return (phi=sh.phi === nothing ? 0. : _cjs_shared_method_sigmas(fit.beta, sh.phi, qfixed, :phi),
            p=sh.p === nothing ? 0. : _cjs_shared_method_sigmas(fit.beta, sh.p, qfixed, :p),
            inner_dimension=sh.dimension, inner_minimum_eigenvalue=λmin)
end

function _cjs_shared_group_values(cols, group, n, label)
    haskey(cols, group) || throw(ArgumentError("cjs $label prediction table requires grouping column $group"))
    values = getproperty(cols, group)
    length(values) == n || throw(ArgumentError("cjs $label grouping labels do not align with prediction rows"))
    for x in values
        (ismissing(x) || (x isa AbstractString && isempty(strip(x)))) &&
            throw(ArgumentError("cjs $label grouping labels must be nonmissing and nonempty"))
    end
    return values
end

function _cjs_shared_check_prediction_groups(fit, newdata)
    sh = fit.prepared.shared
    process = Dict{Symbol,Dict{Any,Any}}()
    for (label, mapping, keys) in ((:phi, sh.phi, (:id, :interval_start, :interval_end)),
                                   (:p, sh.p, (:id, :occasion)))
        mapping === nothing && continue
        cols, n = _cjs_table_columns(getproperty(newdata, label), label)
        haskey(cols, :id) || throw(ArgumentError("cjs $label prediction table requires key id"))
        groups = _cjs_shared_group_values(cols, mapping.group, n, label)
        by_id = Dict{Any,Any}()
        for i in 1:n
            id = getproperty(cols, :id)[i]
            if !mapping.varying
                if haskey(by_id, id) && !isequal(by_id[id], groups[i])
                    throw(ArgumentError("cjs $label site prediction rows for one id must use one group label"))
                end
                by_id[id] = groups[i]
            end
        end
        process[label] = by_id
    end
    if sh.phi !== nothing && sh.p !== nothing && !sh.phi.varying && !sh.p.varying
        for id in intersect(Set(keys(process[:phi])), Set(keys(process[:p])))
            isequal(process[:phi][id], process[:p][id]) ||
                throw(ArgumentError("cjs shared paired-process site prediction labels must agree for id $(repr(id))"))
        end
    end
    return nothing
end

function _cjs_shared_prediction_rows(fit, des, table, label, keynames, beta, group; sd_vars=Symbol[])
    cols, n = _cjs_table_columns(table, label)
    all(k -> haskey(cols, k), keynames) || throw(ArgumentError("cjs $label prediction table requires keys $(join(keynames, ", "))"))
    seen = Set{Tuple}()
    for i in 1:n
        key = Tuple(getproperty(cols, k)[i] for k in keynames)
        any(ismissing, key) && throw(ArgumentError("cjs $label prediction keys cannot be missing"))
        key in seen && throw(ArgumentError("cjs $label prediction keys must be unique"))
        push!(seen, key)
    end
    labels = _cjs_shared_group_values(cols, group, n, label)
    vars = StatsModels.termvars(des.rhs)
    offsets = [term.sym for term in des.offset_terms if term isa Term]
    allowed = unique(vcat(collect(keynames), vars, offsets, sd_vars, [group], label == :phi ? [:interval_length] : Symbol[]))
    Set(keys(cols)) == Set(allowed) || throw(ArgumentError("cjs $label prediction table does not match the fitted predictor/group schema"))
    for name in vars
        haskey(cols, name) || throw(ArgumentError("cjs $label prediction predictor $name is absent"))
        any(ismissing, getproperty(cols, name)) && throw(ArgumentError("cjs $label prediction predictor $name cannot be missing"))
    end
    sub = NamedTuple{keys(cols)}(Tuple(collect(getproperty(cols, k)) for k in keys(cols)))
    raw = try
        modelcols(des.applied, sub)
    catch err
        err isa InterruptException && rethrow()
        throw(ArgumentError("cjs $label prediction predictors do not match the fitted schema: $(sprint(showerror, err))"))
    end
    matrix = raw isa Tuple ? hcat(raw...) : raw
    matrix = matrix isa AbstractMatrix ? Matrix{Float64}(matrix) : reshape(Float64.(collect(matrix)), :, 1)
    X = matrix[:, des.columns]
    size(X, 2) == length(beta) && all(isfinite, X) || throw(ArgumentError("cjs $label prediction design is invalid"))
    offset = zeros(Float64, n)
    for term in des.offset_terms
        term isa Term && haskey(cols, term.sym) || throw(ArgumentError("cjs $label prediction offset column is absent"))
        v = getproperty(cols, term.sym)
        any(ismissing, v) && throw(ArgumentError("cjs $label prediction offset cannot be missing"))
        offset .+= Float64.(v)
    end
    all(isfinite, offset) || throw(ArgumentError("cjs $label prediction offsets must be finite"))
    return (cols=cols, n=n, X=X, offset=offset, labels=labels)
end

function _cjs_shared_training_tables(fit)
    phi, p, phi_rows, p_rows = _cjs_training_prediction_tables(fit)
    for (table_name, table, rows, des, mapping) in
        ((:phi, phi, phi_rows, fit.phi_design, fit.prepared.shared.phi),
         (:p, p, p_rows, fit.p_design, fit.prepared.shared.p))
        vals = getproperty(fit.data, table_name == :phi ? :phi_data : :p_data)
        table = _cjs_append_predictors(table, rows, vals, des)
        pairs = Pair{Symbol,Any}[k => getproperty(table, k) for k in keys(table)]
        if mapping !== nothing
            group = mapping.group
            group in keys(table) || push!(pairs, group => [getproperty(vals, group)[row] for (row, _, _) in rows])
        end
        if mapping !== nothing && hasproperty(mapping, :sd_formula) && mapping.sd_formula !== nothing
            sd_vars = _cjs_sd_predictor_names(mapping.sd_design)
            for name in sd_vars
                name in first.(pairs) || push!(pairs, name => [getproperty(vals, name)[row] for (row, _, _) in rows])
            end
        end
        updated = (; pairs...)
        table_name == :phi ? (phi = updated) : (p = updated)
    end
    return phi, p
end

function _cjs_shared_probability_table(fit, des, table, label, keynames, beta, mapping,
                                      σ, response_type, target, rule; training=false)
    sd_vars = hasproperty(mapping, :sd_formula) && mapping.sd_formula !== nothing ? _cjs_sd_predictor_names(mapping.sd_design) : Symbol[]
    rows = _cjs_shared_prediction_rows(fit, des, table, label, keynames, beta, mapping.group; sd_vars)
    η = rows.X * beta + rows.offset
    all(isfinite, η) || throw(ArgumentError("cjs $label prediction link values must be finite"))
    Δ = label == :phi ? Float64.(rows.cols.interval_length) : nothing
    label == :phi && (!all(isfinite, Δ) || !all(>(0), Δ)) &&
        throw(ArgumentError("cjs survival interval lengths must be finite and positive"))
    terminal = falses(rows.n)
    constraint = fit.prepared.terminal_constraint
    if constraint !== nothing && only(keys(constraint)) == label
        terminal = label == :phi ?
            [last(fit.data.occasions) == x for x in rows.cols.interval_end] :
            [last(fit.data.occasions) == x for x in rows.cols.occasion]
        c = only(values(constraint))
        η[terminal] .= log(c / (1 - c))
    end
    sigma_rows = if training && hasproperty(mapping, :sd_X)
        unit_sd = _cjs_shared_sigmas(fit.beta, mapping)
        index = Dict(level => i for (i, level) in enumerate(mapping.levels))
        groups = getproperty(rows.cols, mapping.group)
        [terminal[i] ? 0. : unit_sd[get(index, groups[i], 0)] for i in eachindex(groups)]
    elseif isempty(sd_vars)
        fill(σ isa AbstractVector ? first(σ) : σ, rows.n)
    else
        out = zeros(rows.n)
        active = findall(.!terminal)
        if !isempty(active)
            supplied = NamedTuple{keys(rows.cols)}(Tuple(collect(v)[active] for v in values(rows.cols)))
            out[active] = _cjs_shared_sd_profile(mapping, fit.beta, supplied, label)
        end
        out
    end
    q = response_type ? exp.(_cjs_logistic_log.(η)) : η
    interval = label == :phi && response_type ? q .^ Δ : nothing
    if response_type && target == :integrated && any(>(0), sigma_rows)
        weights = exp.(rule.logweights)
        for i in eachindex(q)
            terminal[i] && continue
            q[i], interval_value = _cjs_shared_integral(η[i], sigma_rows[i], label == :phi ? Δ[i] : 1., rule)
            label == :phi && (interval[i] = interval_value)
        end
    end
    base = Pair{Symbol,Any}[k => getproperty(rows.cols, k) for k in keynames]
    label == :phi && push!(base, :interval_length => Δ)
    if response_type
        push!(base, (label == :phi ? :phi_unit : :p) => q)
        label == :phi && push!(base, :phi_interval => interval)
    else
        push!(base, :eta => η)
    end
    eligible = response_type ? .!terminal : falses(rows.n)
    return (; base...), (probabilities=response_type ? (label == :phi ? vcat(q, interval) : q) : Float64[],
                         eligible=response_type ? (label == :phi ? vcat(eligible, eligible) : eligible) : falses(0))
end

function _cjs_shared_has_unconstrained_rows(fit, table, mapping, label)
    mapping === nothing && return false
    constraint = fit.prepared.terminal_constraint
    constraint === nothing && return true
    only(keys(constraint)) == label || return true
    cols, n = _cjs_table_columns(table, label)
    key = label == :phi ? :interval_end : :occasion
    haskey(cols, key) || throw(ArgumentError("cjs $label prediction table requires key $key"))
    return any(x -> x != last(fit.data.occasions), getproperty(cols, key))
end

function _cjs_shared_integral(eta, sigma, delta, rule)
    weights = exp.(rule.logweights)
    values = exp.(_cjs_logistic_log.(eta .+ sigma .* rule.nodes))
    return sum(weights .* values), sum(weights .* values .^ delta)
end

"""Predict shared CJS process probabilities at supplied profiles.

Shared fits require `target=:zero_effect` or `target=:integrated`. Integrated
predictions average each process row over its fitted independent normal
reference effect; survival interval means integrate `S^Delta` directly.
Training groups and new groups use the same reference law. Parameters are held
fixed, and this is not a history-conditioned or population prediction.
"""
function _cjs_predict_shared(fit, newdata; type=:response, target=nothing)
    scales = _cjs_shared_methods_guard(fit)
    type in (:response, :link, "response", "link") || throw(ArgumentError("cjs prediction type must be :response or :link"))
    response = type in (:response, "response")
    target in (:zero_effect, :integrated) || throw(ArgumentError("cjs shared prediction requires explicit target=:zero_effect or target=:integrated"))
    target == :integrated && !response && throw(ArgumentError("cjs integrated shared prediction is available on the response scale only"))
    training = newdata === nothing
    if training
        phi, p = _cjs_shared_training_tables(fit)
        newdata = (; phi, p, units=fit.data.units)
    else
        newdata isa NamedTuple && all(k -> haskey(newdata, k), (:phi, :p, :units)) ||
            throw(ArgumentError("cjs newdata must be (phi=<keyed table>, p=<keyed table>, units=<fit units>)"))
        newdata.units == fit.data.units || throw(ArgumentError("cjs prediction units must match fitted units $(repr(fit.data.units))"))
    end
    _cjs_shared_check_prediction_groups(fit, newdata)
    sh = fit.prepared.shared
    function tables(rule=target == :integrated ? _cjs_individual_rule(first(_CJS_INDIVIDUAL_ORDERS)) : nothing)
        phi = sh.phi === nothing ? _cjs_prediction_table(fit, fit.phi_design, newdata.phi, :phi,
            (:id, :interval_start, :interval_end), view(fit.beta, 1:fit.prepared.nphi), response) :
            first(_cjs_shared_probability_table(fit, fit.phi_design, newdata.phi, :phi,
                (:id, :interval_start, :interval_end), view(fit.beta, 1:fit.prepared.nphi), sh.phi,
                scales.phi, response, target, rule; training))
        p = sh.p === nothing ? _cjs_prediction_table(fit, fit.p_design, newdata.p, :p,
            (:id, :occasion), view(fit.beta, fit.prepared.nphi+1:fit.prepared.nphi+fit.prepared.np), response) :
            first(_cjs_shared_probability_table(fit, fit.p_design, newdata.p, :p,
                (:id, :occasion), view(fit.beta, fit.prepared.nphi+1:fit.prepared.nphi+fit.prepared.np), sh.p,
                scales.p, response, target, rule; training))
        diagnostics = Pair{Symbol,Any}[]
        for (label, mapping, table) in ((:phi, sh.phi, newdata.phi), (:p, sh.p, newdata.p))
            mapping === nothing && continue
            cols, n = _cjs_table_columns(table, label)
            vals = _cjs_shared_group_values(cols, mapping.group, n, label)
            push!(diagnostics, label => (; variable=mapping.group, labels=unique(string.(vals))))
        end
        return (; phi, p), diagnostics
    end
    result, _ = tables()
    integration = (order=0, refined=true, tolerance=1e-8, maximum_probability_change=0.)
    needs_phi = _cjs_shared_positive_scale(scales.phi) && _cjs_shared_has_unconstrained_rows(fit, newdata.phi, sh.phi, :phi)
    needs_p = _cjs_shared_positive_scale(scales.p) && _cjs_shared_has_unconstrained_rows(fit, newdata.p, sh.p, :p)
    if target == :integrated && (needs_phi || needs_p)
        previous = nothing
        for order in _CJS_INDIVIDUAL_1D_ORDERS
            result, _ = tables(_cjs_individual_rule(order))
            vectors = Any[]
            for (label, mapping, table) in ((:phi, sh.phi, newdata.phi), (:p, sh.p, newdata.p))
                mapping === nothing && continue
                β = label == :phi ? view(fit.beta, 1:fit.prepared.nphi) : view(fit.beta, fit.prepared.nphi+1:fit.prepared.nphi+fit.prepared.np)
                σ = label == :phi ? scales.phi : scales.p
                pred, detail = _cjs_shared_probability_table(fit, label == :phi ? fit.phi_design : fit.p_design,
                    table, label, label == :phi ? (:id, :interval_start, :interval_end) : (:id, :occasion),
                    β, mapping, σ, true, target, _cjs_individual_rule(order))
                append!(vectors, detail.probabilities[detail.eligible])
            end
            change = previous === nothing ? Inf : maximum(abs.(Float64.(vectors) .- previous); init=0.)
            integration = (order=order, refined=change <= 1e-8, tolerance=1e-8,
                           maximum_probability_change=change)
            integration.refined && break
            previous = Float64.(vectors)
        end
        integration.refined || throw(ArgumentError("cjs integrated prediction quadrature is unrefined at order $(last(_CJS_INDIVIDUAL_1D_ORDERS)); returned-probability tolerance 1e-8 was not met"))
    end
    _, group_info = tables()
    return merge(result, (target=target, route=sh.route, groups=(; group_info...),
        mixing=:independent_process_normal_reference, parameters_held_fixed=true,
        integration=integration, qualification=:process_means_only))
end

function _cjs_shared_align_effect(index, u)
    all(j -> j isa Integer && 0 <= j <= length(u), index) ||
        throw(ArgumentError("cjs shared group index is outside the latent vector"))
    return [j == 0 ? 0. : u[j] for j in index]
end

_cjs_shared_draw(rng, σ, n) = σ == 0 ? zeros(n) : randn(rng, n)

function _cjs_shared_draw(rng, σ::AbstractVector, n)
    length(σ) == n && all(x -> isfinite(x) && x >= 0, σ) ||
        throw(ArgumentError("cjs shared simulation needs one finite nonnegative SD per latent unit"))
    u = all(iszero, σ) ? zeros(n) : randn(rng, n)
    u[iszero.(σ)] .= 0
    return u
end

function _cjs_shared_sample_history!(rng, y, observed, first, last, phi_interval, p_occasion)
    length(phi_interval) + 1 == length(y) == length(observed) == length(p_occasion) ||
        throw(ArgumentError("cjs shared simulation probability and history dimensions do not align"))
    1 <= first <= last <= length(y) && y[first] == 1 ||
        throw(ArgumentError("cjs shared simulation requires a retained first capture within the history"))
    alive = true
    for t in first:last-1
        alive = alive && rand(rng) < phi_interval[t]
        next = t + 1
        if alive && observed[next]
            y[next] = rand(rng) < p_occasion[next]
        else
            y[next] = 0
        end
    end
    return y
end

function _cjs_shared_contribution(mapping, σ, rng, nrows)
    mapping === nothing && return zeros(nrows)
    length(mapping.index) == nrows || throw(ArgumentError("cjs shared simulation group incidence does not match process rows"))
    u = _cjs_shared_draw(rng, σ, length(mapping.levels))
    unit_sd = σ isa AbstractVector ? σ : fill(σ, length(mapping.levels))
    return _cjs_shared_align_effect(mapping.index, unit_sd .* u)
end

"""Simulate conditional marked-cohort histories with fresh persistent shared draws.

Each replicate draws one independent latent coordinate per contributing group
and process, shared across all histories and frequency copies bearing that
label. Individual transitions and emissions remain independent conditional on
those draws. First captures, masks, administrative ends and keyed predictor
frames are retained. Returns `Vector{CMRData}` for refitting. The target is
conditional on supplied first captures; no posterior modes, coefficients, SDs,
unmarked groups or population entry process are simulated.
"""
function _cjs_simulate_shared(fit; nsim=1, seed=nothing, rng=Random.default_rng())
    scales = _cjs_shared_methods_guard(fit)
    !(nsim isa Bool) && nsim isa Integer && nsim >= 1 || throw(ArgumentError("nsim must be a positive integer"))
    rng isa Random.AbstractRNG || throw(ArgumentError("rng must be an AbstractRNG"))
    drawrng = _cjs_random(rng, seed)
    d = fit.data
    N, T = size(d.Y)
    d.phi_data === nothing && throw(ArgumentError("cjs shared simulation requires retained survival predictor rows"))
    d.p_data === nothing && throw(ArgumentError("cjs shared simulation requires retained detection predictor rows"))
    phi_index = Dict(row => i for (i, row) in enumerate(fit.phi_design.rows))
    p_index = Dict(row => i for (i, row) in enumerate(fit.p_design.rows))
    out = CMRData[]
    for _ in 1:nsim
        # Fresh effect vectors are generated for every replicate, including a
        # new pair for detection. A zero SD consumes no latent RNG values.
        phi_contrib = _cjs_shared_contribution(fit.prepared.shared.phi, scales.phi, drawrng, length(fit.phi_design.rows))
        p_contrib = _cjs_shared_contribution(fit.prepared.shared.p, scales.p, drawrng, length(fit.p_design.rows))
        ls, lp = _cjs_process_logs(fit.beta, fit.prepared; random_phi=phi_contrib, random_p=p_contrib)
        expanded = [(i, r) for i in 1:N for r in 1:d.frequency[i]]
        ids = [(d.ids[i], r) for (i, r) in expanded]
        Y = zeros(Int, length(expanded), T)
        observed = BitMatrix(undef, length(expanded), T)
        ends = Int[]
        for (j, (i, _)) in enumerate(expanded)
            observed[j, :] .= d.observed[i, :]
            Y[j, 1:d.first[i]] .= d.Y[i, 1:d.first[i]]
            phi_probability = zeros(Float64, T - 1)
            p_probability = zeros(Float64, T)
            for t in d.first[i]:d.end[i]-1
                kphi = get(phi_index, (i-1)*(T-1)+t, 0)
                kphi > 0 || throw(ArgumentError("cjs shared simulation lacks a fitted survival design row"))
                phi_probability[t] = exp(ls[kphi])^d.intervals[t]
                next = t + 1
                if d.observed[i, next]
                    kp = get(p_index, (i-1)*T+next, 0)
                    kp > 0 || throw(ArgumentError("cjs shared simulation lacks a surveyed detection design row"))
                    p_probability[next] = exp(lp[kp])
                end
            end
            _cjs_shared_sample_history!(drawrng, view(Y, j, :), view(observed, j, :),
                d.first[i], d.end[i], phi_probability, p_probability)
            push!(ends, d.end[i])
        end
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
        sim = CMRData(Y; ids, occasions=d.occasions, units=d.units,
            observed, intervals=d.intervals, ends, phi_data, p_data)
        push!(out, sim)
    end
    return out
end

"Evaluate the retained shared log-SD schema on one declared profile per unit."
function _cjs_shared_sd_profile(mapping, theta, table, label)
    cols, n = _cjs_table_columns(table, label)
    groups = _cjs_shared_group_values(cols, mapping.group, n, label)
    units = unique(collect(groups))
    if !hasproperty(mapping, :sd_formula) || mapping.sd_formula === nothing
        return fill(exp(theta[mapping.scale_index]), n)
    end
    hasproperty(mapping, :sd_design) && mapping.sd_design !== nothing ||
        throw(ArgumentError("cjs shared sd_$label retained schema is absent"))
    m = merge(mapping, (levels=units,))
    prep = (data=(phi_data=table, p_data=table),)
    unit_table = _cjs_shared_sd_unit_table(prep, m, mapping.sd_formula, label)
    des = mapping.sd_design
    raw = try
        modelcols(des.applied, unit_table)
    catch err
        err isa InterruptException && rethrow()
        throw(ArgumentError("cjs shared sd_$label prediction does not match fitted factor/contrast schema: $(sprint(showerror, err))"))
    end
    raw = raw isa Tuple ? hcat(raw...) : raw
    X = raw isa AbstractMatrix ? Matrix{Float64}(raw) : reshape(Float64.(collect(raw)), :, 1)
    X = X[:, des.columns]
    size(X) == (length(units), length(mapping.scale_range)) && all(isfinite, X) ||
        throw(ArgumentError("cjs shared sd_$label prediction design is invalid"))
    off = zeros(length(units))
    for term in des.offset_terms
        term isa Term || throw(ArgumentError("cjs shared sd_$label offsets must name numeric columns"))
        value = getproperty(unit_table, term.sym)
        eltype(value) <: Real || throw(ArgumentError("cjs shared sd_$label offsets must be numeric"))
        off .+= Float64.(value)
    end
    sigma = exp.(X * view(theta, mapping.scale_range) + off)
    all(isfinite, off) && all(x -> isfinite(x) && x > 0, sigma) ||
        throw(ArgumentError("cjs shared sd_$label profiles must yield finite positive SDs"))
    index = Dict(unit => i for (i, unit) in enumerate(units))
    return [sigma[index[g]] for g in groups]
end


_cjs_shared_positive_scale(sigma) = sigma isa AbstractVector ? any(>(0), sigma) : sigma > 0

function _cjs_shared_method_sigmas(theta, m, nfixed, label)
    if !hasproperty(m, :sd_X)
        sigma = exp(theta[m.scale_index])
        isfinite(sigma) && sigma >= 0 || throw(ArgumentError("cjs shared $label SD must be finite and nonnegative"))
        return sigma
    end
    all(k -> hasproperty(m, k), (:scale_range, :sd_offset, :sd_names, :sd_formula)) ||
        throw(ArgumentError("cjs shared sd_$label fitted design provenance is absent"))
    r = m.scale_range
    r isa UnitRange{<:Integer} && !isempty(r) && first(r) > nfixed && last(r) <= length(theta) &&
        m.scale_index == first(r) && size(m.sd_X) == (length(m.levels), length(r)) &&
        length(m.sd_offset) == length(m.levels) && length(m.sd_names) == length(r) &&
        all(isfinite, m.sd_X) && all(isfinite, m.sd_offset) ||
        throw(ArgumentError("cjs shared sd_$label coefficient range or unit design is invalid"))
    if m.sd_formula !== nothing
        hasproperty(m, :sd_design) && m.sd_design !== nothing &&
            m.sd_design.names == m.sd_names && m.sd_design.X == m.sd_X &&
            m.sd_design.offset == m.sd_offset && isequal(m.sd_design.formula, m.sd_formula) ||
            throw(ArgumentError("cjs shared sd_$label fitted schema disagrees with its unit design"))
    else
        length(r) == 1 && all(==(1), m.sd_X) && all(iszero, m.sd_offset) ||
            throw(ArgumentError("cjs shared default SD must retain an intercept-only unit design"))
    end
    return _cjs_shared_sigmas(theta, m)
end

# Profiles constrain an existing coefficient coordinate, including log-SD
# design coefficients. The native objective resolves the joint latent mode at
# every evaluation; no fitted mode or covariance approximation is held fixed.
function _cjs_shared_profile_point(fit, index, target; maxiters=1000, start_theta=nothing)
    _cjs_is_shared(fit.prepared) || throw(ArgumentError("shared CJS profile requires a shared-effect fit"))
    index isa Integer && !(index isa Bool) && 1 <= index <= length(fit.beta) ||
        throw(ArgumentError("shared CJS profile coefficient index is invalid"))
    target isa Real && isfinite(target) || throw(ArgumentError("shared CJS profile value must be finite"))
    maxiters isa Integer && !(maxiters isa Bool) && maxiters >= 1 ||
        throw(ArgumentError("shared CJS profile maxiters must be a positive integer"))
    free = [j for j in eachindex(fit.beta) if j != index]
    start = start_theta === nothing ? copy(fit.beta) : Float64.(collect(start_theta))
    length(start) == length(fit.beta) && all(isfinite, start) ||
        throw(ArgumentError("shared CJS profile start_theta must be a finite full coefficient vector"))
    template = start; template[index] = target
    attempts = NamedTuple[]; mode_failures = NamedTuple[]
    expand(x) = begin
        theta = eltype(x).(template)
        theta[free] = x
        theta
    end
    objective = x -> try
        _cjs_shared_laplace(expand(x), fit.prepared)
    catch err
        err isa InterruptException && rethrow()
        if err isa _CJSSharedModeError
            push!(mode_failures, (theta=Float64.(_cjs_primal.(expand(x))),
                reason=err.reason, attempts=err.attempts))
        end
        rethrow()
    end
    start = template[free]; best = Inf
    for method in (Optim.BFGS(), Optim.NelderMead(), Optim.BFGS())
        optimizer = method isa Optim.BFGS ? :BFGS : :NelderMead
        failure_count = length(mode_failures)
        candidate_theta = expand(start)
        try
            result = method isa Optim.BFGS ?
                Optim.optimize(objective, start, method,
                    Optim.Options(iterations=maxiters, g_tol=1e-7, f_reltol=0.0); autodiff=:forward) :
                Optim.optimize(objective, start, method,
                    Optim.Options(iterations=maxiters, f_reltol=1e-12))
            x = Vector{Float64}(Optim.minimizer(result)); theta = expand(x)
            candidate_theta = theta
            inner = _cjs_shared_laplace(theta, fit.prepared; details=true)
            gradient = _cjs_shared_gradient(theta, fit.prepared)[free]
            norm_gradient = isempty(gradient) ? 0. : norm(gradient, Inf)
            available = Optim.converged(result) && all(isfinite, theta) &&
                isfinite(inner.objective) && all(isfinite, gradient) && norm_gradient <= 1e-5 &&
                inner.converged && inner.gradient_norm <= 1e-9 && theta[index] == target
            push!(attempts, (optimizer, start=copy(start), theta=copy(theta),
                objective=Float64(inner.objective), gradient_norm=norm_gradient,
                optimizer_converged=Optim.converged(result), converged=available,
                iterations=Optim.iterations(result), reason=available ?
                    "constrained nuisance optimum and resolved shared mode" : "nuisance gradient or optimizer failed"))
            if available
                return (objective=Float64(inner.objective), status=:available, theta,
                    gradient_norm=norm_gradient, inner, attempts, mode_failures)
            end
            if isfinite(inner.objective) && inner.objective < best
                best = inner.objective; start = x
            end
        catch err
            err isa InterruptException && rethrow()
            if err isa _CJSSharedModeError && length(mode_failures) == failure_count
                push!(mode_failures, (theta=copy(candidate_theta), reason=err.reason, attempts=err.attempts))
            end
            push!(attempts, (optimizer, start=copy(start), theta=expand(start),
                objective=Inf, gradient_norm=Inf, optimizer_converged=false,
                converged=false, iterations=0, reason=sprint(showerror,err)))
        end
    end
    return (objective=missing, status=isempty(mode_failures) ? :failed_reoptimization : :failed_mode,
        theta=missing, gradient_norm=Inf, inner=nothing, attempts, mode_failures)
end

function _cjs_shared_profile_lr(point, fit)
    point.status == :available || return (missing, point.status)
    lr = 2 * (point.objective + fit.loglik)
    tolerance = 1e-8 * max(1., abs(fit.loglik), abs(point.objective))
    isfinite(lr) || return (missing, :failed_reoptimization)
    lr < -2tolerance && return (missing, :profile_improves_on_fit)
    return (max(0., lr), :available)
end

function _cjs_shared_profile_endpoint(fit,index,direction,cutoff,center;
        profile_point=_cjs_shared_profile_point, center_point=nothing)
    direction in (-1,1) || throw(ArgumentError("shared CJS profile direction must be -1 or 1"))
    points = NamedTuple[]
    last_theta = nothing
    result(estimate,status) = (;estimate,status,points=copy(points))
    function evaluate(value,phase; supplied=nothing)
        warm_start_fallback = false
        warm_start_status = nothing
        point = if supplied !== nothing
            supplied
        elseif last_theta === nothing
            profile_point(fit,index,value)
        else
            warm = profile_point(fit,index,value; start_theta=copy(last_theta))
            warm_start_status = warm.status
            if warm.status == :available
                warm
            else
                # Continuation is only an initial value. If it fails the same
                # strict admission checks, retry from the original fitted
                # coefficients so a bad warm start cannot censor the endpoint.
                warm_start_fallback = true
                profile_point(fit,index,value; start_theta=nothing)
            end
        end
        lr,status = _cjs_shared_profile_lr(point,fit)
        status == :available && hasproperty(point,:theta) &&
            (last_theta = copy(point.theta))
        push!(points,(;value,phase,point,lr,status,warm_start_fallback,warm_start_status))
        return lr,status
    end
    lr,status = evaluate(center,:center; supplied=center_point)
    status == :available || return result(missing,status)
    lr < cutoff || return result(missing,:center_outside_cutoff)
    step = .25; inner = center; outer = center+direction*step; crossed = false
    for _ in 1:12
        lr,status = evaluate(outer,:expansion)
        if status == :unrefined_at_cap
            failed_outer = outer
            for _ in 1:16
                mid = (inner+failed_outer)/2
                abs(failed_outer-inner) <= 1e-8 && break
                mid_lr,mid_status = evaluate(mid,:backoff)
                if mid_status == :available
                    if mid_lr >= cutoff
                        outer = mid
                        crossed = true
                        break
                    end
                    inner = mid
                elseif mid_status == :unrefined_at_cap
                    failed_outer = mid
                else
                    return result(missing,mid_status)
                end
            end
            crossed || return result(missing,:unrefined_at_cap)
            break
        elseif status != :available
            return result(missing,status)
        end
        if lr >= cutoff
            crossed = true; break
        end
        inner = outer; step *= 2; outer = center+direction*step
    end
    crossed || return result(missing,:non_crossing)
    lo,hi = min(inner,outer),max(inner,outer)
    for _ in 1:32
        mid = (lo+hi)/2
        lr,status = evaluate(mid,:bisection)
        status == :available || return result(missing,status)
        abs(lr-cutoff) <= 1e-5 && return result(mid,:available)
        if direction < 0
            lr >= cutoff ? (lo=mid) : (hi=mid)
        else
            lr >= cutoff ? (hi=mid) : (lo=mid)
        end
    end
    return result(missing,:endpoint_tolerance_not_met)
end

"""Drop one constant shared SD while retaining the other process in the CJS pair."""
function _cjs_shared_zero_sd_restriction(prep,beta,process)
    process in (:phi,:p) || throw(ArgumentError("shared CJS zero-SD process must be :phi or :p"))
    _cjs_is_shared(prep) || throw(ArgumentError("shared CJS zero-SD restriction requires a shared fit"))
    mapping = getproperty(prep.shared,process)
    mapping === nothing && throw(ArgumentError("shared CJS zero-SD process $process is inactive"))
    hasproperty(mapping,:sd_formula) && mapping.sd_formula !== nothing &&
        throw(ArgumentError("exact-zero shared CJS profile is available only for a constant SD"))
    beta isa AbstractVector && length(beta)==prep.nphi+prep.np+prep.shared.nscale && all(isfinite,beta) ||
        throw(ArgumentError("shared CJS zero-SD coefficients do not match the prepared model"))
    scale_indices = hasproperty(mapping,:scale_range) ? collect(mapping.scale_range) : [mapping.scale_index]
    length(scale_indices)==1 || throw(ArgumentError("exact-zero shared CJS profile requires one SD coefficient"))
    removed = only(scale_indices)
    other_process = process==:phi ? :p : :phi
    other = getproperty(prep.shared,other_process)
    other === nothing && throw(ArgumentError("exact-zero shared CJS profile requires the other process shared effect to remain fitted"))
    nfixed=prep.nphi+prep.np
    removed>nfixed || throw(ArgumentError("shared CJS SD coefficient overlaps a fixed-effect coordinate"))

    other_mapping = if hasproperty(other,:scale_range)
        shifted = map(i -> i>removed ? i-1 : i,other.scale_range)
        merge(other,(scale_index=first(shifted),scale_range=shifted,
            range=1:length(other.levels)))
    else
        merge(other,(scale_index=other.scale_index>removed ? other.scale_index-1 : other.scale_index,
            range=1:length(other.levels)))
    end
    phi = process==:phi ? nothing : other_mapping
    p = process==:p ? nothing : other_mapping
    remaining_label = other_process
    route = other_mapping.varying ? :year_vector : remaining_label==:phi ? :site_phi : :site_p
    names = filter(name -> !startswith(name,"log_sigma_$(process):"),prep.shared.names)
    groups = filter(group -> group.process != process,prep.shared.diagnostics.groups)
    shared = merge(prep.shared,(;phi,p,dimension=length(other_mapping.levels),
        nscale=prep.shared.nscale-1,route,names,
        diagnostics=merge(prep.shared.diagnostics,(;groups))))
    restricted=merge(prep,(;shared))
    start=Float64[beta[j] for j in eachindex(beta) if j!=removed]
    return restricted,start
end

"""Fit and strictly admit the exact-zero constant-SD reduced profile point."""
function _cjs_shared_zero_sd_profile_point(fit,process;maxiters=1000,g_tol=1e-8,
        fit_shared=_cjs_fit_shared)
    prep,start=_cjs_shared_zero_sd_restriction(fit.prepared,fit.beta,process)
    restricted=try
        fit_shared(prep;start,maxiters,g_tol)
    catch err
        err isa InterruptException && rethrow()
        return (objective=missing,status=:failed_reoptimization,reason=sprint(showerror,err),
            theta=missing)
    end
    diagnostics=restricted.diagnostics
    inner=hasproperty(diagnostics,:inner) ? diagnostics.inner : nothing
    status,reason=if !hasproperty(diagnostics,:converged) || !diagnostics.converged
        (:failed_reoptimization,:outer_optimizer)
    elseif !hasproperty(diagnostics,:gradient_norm) || !isfinite(diagnostics.gradient_norm) || diagnostics.gradient_norm>1e-5
        (:failed_reoptimization,:outer_gradient)
    elseif inner===nothing || !inner.converged || !isfinite(inner.gradient_norm) || inner.gradient_norm>1e-9
        (:failed_mode,:inner_mode)
    elseif !isfinite(restricted.loglik)
        (:failed_reoptimization,:nonfinite_objective)
    else
        (:available,:admitted)
    end
    return (objective=status==:available ? -Float64(restricted.loglik) : missing,
        status,reason,theta=restricted.beta,
        diagnostics=(converged=diagnostics.converged,gradient_norm=diagnostics.gradient_norm,
            inner_converged=inner!==nothing && inner.converged,
            inner_gradient_norm=inner===nothing ? missing : inner.gradient_norm))
end

"""Check whether the exact-zero SD endpoint belongs to the profile set."""
function _cjs_shared_profile_zero_sd_endpoint(point,fit,cutoff)
    lr,status=_cjs_shared_profile_lr(point,fit)
    status==:profile_improves_on_fit && return (estimate=missing,status,
        points=[(value=0.0,phase=:exact_zero,point,lr,status)],lr)
    status==:available || return (estimate=missing,status=:zero_boundary_unavailable,
        points=[(value=0.0,phase=:exact_zero,point,lr,status)],lr=missing)
    included=lr<=cutoff
    return (estimate=included ? 0.0 : missing,
        status=included ? :boundary_included : :boundary_excluded,
        points=[(value=0.0,phase=:exact_zero,point,lr,status)],lr)
end

function _cjs_shared_profile_sd_lower_endpoint(fit,process,index,cutoff,center;
        finite_endpoint=_cjs_shared_profile_endpoint,
        zero_profile_point=_cjs_shared_zero_sd_profile_point)
    other_process=process==:phi ? :p : :phi
    other_mapping=getproperty(fit.prepared.shared,other_process)
    if other_mapping===nothing
        return finite_endpoint(fit,index,-1,cutoff,fit.beta[index];center_point=center)
    end
    zero_point=try
        zero_profile_point(fit,process)
    catch err
        err isa InterruptException && rethrow()
        (objective=missing,status=:failed_reoptimization,
            reason=sprint(showerror,err),theta=missing)
    end
    zero_boundary=_cjs_shared_profile_zero_sd_endpoint(zero_point,fit,cutoff)
    if zero_boundary.status in (:boundary_included,:zero_boundary_unavailable,
            :profile_improves_on_fit)
        return merge(zero_boundary,(zero_boundary=zero_boundary,))
    end
    finite=finite_endpoint(fit,index,-1,cutoff,fit.beta[index];center_point=center)
    return merge(finite,(zero_boundary=zero_boundary,))
end

"Transform an available finite log-SD endpoint without asserting an exact-zero limit."
function _cjs_shared_profile_sd_limit(endpoint)
    endpoint.status==:boundary_included && return (0.0,:boundary_included)
    endpoint.status == :available || return (missing,endpoint.status)
    ismissing(endpoint.estimate) && return (missing,:positive_sd_transform_unavailable)
    value = exp(endpoint.estimate)
    isfinite(value) && value > 0 || return (missing,:positive_sd_transform_unavailable)
    return (value,:available)
end

function _cjs_shared_profile_interval_status(lower_status, upper_status, invalid_fit)
    invalid_fit && return :profile_improves_on_fit
    lower_usable = lower_status in (:available,:boundary_included)
    lower_usable && upper_status == :available && return :available
    return lower_usable ? upper_status : lower_status
end

function _cjs_shared_profile(fit; level, targets)
    _cjs_shared_methods_guard(fit)
    names = _cjs_coef_names(fit)
    requested = targets === nothing ? names : String.(collect(targets))
    isempty(requested) && throw(ArgumentError("at least one shared CJS profile target is required"))
    all(name -> name in names || name in ("sd:phi","sd:p"), requested) ||
        throw(ArgumentError("unknown shared CJS profile coefficient; use one of $(join(names, ", "))"))
    records = NamedTuple[]; attempts = NamedTuple[]; cutoff = quantile(Chisq(1),level)
    invalid_fit = false
    for name in requested
        positive_sd = name in ("sd:phi","sd:p")
        index = findfirst(==(name),names)
        if positive_sd
            mapping = getproperty(fit.prepared.shared,name == "sd:phi" ? :phi : :p)
            mapping === nothing && throw(ArgumentError("shared CJS profile target $name is inactive"))
            if hasproperty(mapping,:sd_formula) && mapping.sd_formula !== nothing
                if invalid_fit
                    push!(records,_cjs_record(name,missing,missing,missing,:not_attempted_invalid_fit))
                    push!(attempts,_cjs_profile_unattempted(name,nothing,:positive_sd))
                    continue
                end
                push!(records,_cjs_record(name,missing,missing,missing,:guarded_unavailable))
                push!(attempts,(target=name,status=:guarded_unavailable,
                    reason="a varying SD formula has no scalar constant-SD target"))
                continue
            end
            index = mapping.scale_index
        end
        estimate = positive_sd ? exp(fit.beta[index]) : fit.beta[index]
        if invalid_fit
            push!(records,_cjs_record(name,estimate,missing,missing,:not_attempted_invalid_fit))
            push!(attempts,merge(_cjs_profile_unattempted(name,index,
                positive_sd ? :positive_sd : :coefficient),
                (coordinate=names[index],interval_scale=positive_sd ? :positive_sd : :coefficient)))
            continue
        end
        if fit.diagnostics.boundary || !fit.diagnostics.positive_curvature
            push!(records,_cjs_record(name,estimate,missing,missing,:guarded_unavailable))
            push!(attempts,(target=name,coordinate=names[index],
                interval_scale=positive_sd ? :positive_sd : :coefficient,
                status=:guarded_unavailable,reason="boundary or fitted curvature guard"))
            continue
        end
        center = _cjs_shared_profile_point(fit,index,fit.beta[index])
        lr,status = _cjs_shared_profile_lr(center,fit)
        status == :available && lr > 2e-8*max(1.,abs(fit.loglik)) &&
            (status = :center_inconsistent_with_fit)
        if status != :available
            push!(records,_cjs_record(name,estimate,missing,missing,status))
            push!(attempts,(target=name,coordinate=names[index],
                interval_scale=positive_sd ? :positive_sd : :coefficient,
                center,lower=missing,upper=missing,status))
            invalid_fit = status == :profile_improves_on_fit
            continue
        end
        zero_boundary = nothing
        lower = if positive_sd
            process=name=="sd:phi" ? :phi : :p
            _cjs_shared_profile_sd_lower_endpoint(fit,process,index,cutoff,center)
        else
            _cjs_shared_profile_endpoint(fit,index,-1,cutoff,fit.beta[index];center_point=center)
        end
        upper = lower.status == :profile_improves_on_fit ? missing :
            _cjs_shared_profile_endpoint(fit,index,1,cutoff,fit.beta[index];center_point=center)
        lo,lo_status = positive_sd ? _cjs_shared_profile_sd_limit(lower) : (lower.estimate,lower.status)
        hi,hi_status = ismissing(upper) ? (missing,:not_attempted_invalid_fit) :
            positive_sd ? _cjs_shared_profile_sd_limit(upper) : (upper.estimate,upper.status)
        invalid_fit = lo_status == :profile_improves_on_fit || hi_status == :profile_improves_on_fit
        status = _cjs_shared_profile_interval_status(lo_status,hi_status,invalid_fit)
        push!(records,_cjs_record(name,estimate,invalid_fit ? missing : lo,invalid_fit ? missing : hi,status))
        push!(attempts,(target=name,coordinate=names[index],
            interval_scale=positive_sd ? :positive_sd : :coefficient,center,lower,upper,status))
    end
    invalid_fit && _cjs_profile_invalidate_records!(records)
    completed = count(r -> r.status == :available,records)
    status = invalid_fit ? :profile_improves_on_fit : all(r -> r.status == :available,records) ? :profile_available :
        all(r -> r.status == :guarded_unavailable,records) ? :guarded_unavailable : :partial
    return (method=:profile,level,intervals=records,attempts,requested=length(requested),
        completed,status,cutoff,calibrated=false,
        reason="shared CJS Laplace coefficient profiles with exact-zero checks for paired constant-SD limits; boundary fits guarded and coverage not qualified")
end
