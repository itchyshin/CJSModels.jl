# Per-unit log-SD designs for the shared CJS route. The schema is learned on
# the active shared levels only; repeated process-table rows declare the same
# unit-level values and are checked before one row per unit is constructed.
function _cjs_shared_sd_formula(formula, label)
    formula === nothing && return nothing
    formula isa FormulaTerm && formula.lhs isa Term && formula.lhs.sym == Symbol("sd_", label) ||
        throw(ArgumentError("cjs sd_$label must be a formula with left side sd_$label"))
    _cjs_validate_term(formula.rhs)
    return formula
end

function _cjs_shared_sd_unit_table(prep, mapping, formula, label)
    levels = mapping.levels
    table = label == :phi ? prep.data.phi_data : prep.data.p_data
    cols = try
        Tables.columntable(table)
    catch
        throw(ArgumentError("cjs sd_$label requires a named $label process table"))
    end
    haskey(cols, :id) || throw(ArgumentError("cjs sd_$label process table must contain id keys"))
    haskey(cols, mapping.group) ||
        throw(ArgumentError("cjs sd_$label grouping column $(mapping.group) is absent from the $label table"))
    nrows = length(cols.id)
    all(v -> length(v) == nrows, values(cols)) ||
        throw(ArgumentError("cjs sd_$label process table columns have unequal lengths"))
    group_values = getproperty(cols, mapping.group)
    unit_rows = [findall(v -> isequal(v, level), group_values) for level in levels]
    any(isempty, unit_rows) &&
        throw(ArgumentError("cjs sd_$label active grouping level has no rows in the $label process table"))

    formula === nothing && return (id=collect(levels),)
    terms = unique(vcat(StatsModels.termvars(formula.rhs),
        [t.sym for t in _cjs_offsets(formula.rhs) if t isa Term]))
    any(v -> !haskey(cols, v), terms) &&
        throw(ArgumentError("cjs sd_$label predictors and offsets must be present in the $label table"))
    pairs = Pair{Symbol,Any}[:id => collect(levels)]
    for name in terms
        source = getproperty(cols, name)
        T = Base.nonmissingtype(eltype(source))
        values_by_unit = Vector{T}(undef, length(levels))
        for (j, rows) in enumerate(unit_rows)
            declared = collect(skipmissing(source[rows]))
            isempty(declared) &&
                throw(ArgumentError("cjs sd_$label predictor or offset $name has no nonmissing value for active unit $(repr(levels[j]))"))
            all(x -> isequal(x, first(declared)), declared) ||
                throw(ArgumentError("cjs sd_$label predictor or offset $name must be constant within active unit $(repr(levels[j]))"))
            values_by_unit[j] = first(declared)
        end
        push!(pairs, name => values_by_unit)
    end
    return (; pairs...)
end

function _cjs_shared_sd_design(prep, mapping, formula, label, contrasts)
    unit_table = _cjs_shared_sd_unit_table(prep, mapping, formula, label)
    if formula === nothing
        n = length(mapping.levels)
        return (sd_X=ones(n, 1), sd_offset=zeros(n), sd_names=["(Intercept)"],
            sd_formula=nothing, sd_design=nothing, sd_unit_table=unit_table)
    end
    design = _cjs_design(formula, unit_table, collect(eachindex(mapping.levels)),
        Symbol("sd_", label); contrasts=contrasts)
    ncols = size(design.X, 2)
    ncols > 0 || throw(ArgumentError("cjs sd_$label formula must produce at least one free model column"))
    rank(design.X) == ncols ||
        throw(ArgumentError("cjs sd_$label unit design is rank deficient; simplify the formula or provide more distinct active units"))
    length(design.offset) == length(mapping.levels) ||
        throw(ArgumentError("cjs sd_$label offset design does not match active unit levels"))
    return (sd_X=design.X, sd_offset=design.offset, sd_names=design.names,
        sd_formula=formula, sd_design=design, sd_unit_table=unit_table)
end

function _cjs_prepare_shared_sd(prep, sd_phi, sd_p; contrasts=Dict{Symbol,Any}())
    _cjs_is_shared(prep) || throw(ArgumentError("cjs shared SD preparation requires an already prepared shared route"))
    sh = prep.shared
    formulas = (phi=_cjs_shared_sd_formula(sd_phi, :phi),
                p=_cjs_shared_sd_formula(sd_p, :p))
    mappings = (phi=sh.phi, p=sh.p)
    for label in (:phi, :p)
        mappings[label] === nothing && formulas[label] !== nothing &&
            throw(ArgumentError("cjs sd_$label formula was supplied but that process has no shared effect"))
    end

    nfixed = prep.nphi + prep.np
    cursor = nfixed + 1
    prepared = Dict{Symbol,Any}()
    names = String[]
    for label in (:phi, :p)
        mapping = mappings[label]
        if mapping === nothing
            prepared[label] = nothing
            continue
        end
        design = _cjs_shared_sd_design(prep, mapping, formulas[label], label, contrasts)
        scale_range = cursor:(cursor + size(design.sd_X, 2) - 1)
        cursor = last(scale_range) + 1
        prepared[label] = merge(mapping, design,
            (scale_range=scale_range, scale_index=first(scale_range)))
        if formulas[label] === nothing
            push!(names, "log_sigma_$(label): $(mapping.group)")
        else
            append!(names, ["sd_$(label): $name" for name in design.sd_names])
        end
    end
    return merge(prep, (shared=merge(sh, (phi=prepared[:phi], p=prepared[:p],
        nscale=cursor - nfixed - 1, names=names)),))
end
