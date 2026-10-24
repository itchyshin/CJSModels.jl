# Selection-conditioned profiles retain both integration components.

_cjs_selection_optimize_nuisance(objective,start) = Optim.optimize(objective,start,
    Optim.BFGS(),Optim.Options(iterations=1000,g_tol=1e-7,f_reltol=1e-12);autodiff=:forward)

function _cjs_selection_profile_point(fit,index,target;
        integral_tol=1e-8,objective_tol=1e-8,parameter_tol=1e-5,
        zero_process=nothing,start=fit.beta,optimize_nuisance=_cjs_selection_optimize_nuisance)
    prep=fit.prepared
    hasproperty(prep,:selection) && prep.selection!==nothing ||
        throw(ArgumentError("selected CJS profiles require reference-selection conditioning"))
    1<=index<=length(fit.beta) || throw(ArgumentError("selected CJS profile coefficient index is invalid"))
    isfinite(target) || throw(ArgumentError("selected CJS profile value must be finite"))
    all(x->isfinite(x)&&x>0,(integral_tol,objective_tol,parameter_tol)) ||
        throw(ArgumentError("selected CJS profile tolerances must be finite and positive"))
    length(start)==length(fit.beta) && all(isfinite,start) ||
        throw(ArgumentError("selected CJS profile start must be finite and match the fitted parameter dimension"))
    zero_processes=zero_process===nothing ? Symbol[] :
        zero_process isa Symbol ? [zero_process] : collect(zero_process)
    all(x->x in (:phi,:p),zero_processes) && length(unique(zero_processes))==length(zero_processes) ||
        throw(ArgumentError("zero_process must be :phi, :p, or a unique tuple of them"))
    individual=hasproperty(prep,:individual) ? prep.individual : nothing
    zero_scales=Int[]
    if individual!==nothing
        for process in zero_processes
            mapping=getproperty(individual,process)
            mapping===nothing && continue
            append!(zero_scales,hasproperty(mapping,:scale_range) ?
                collect(mapping.scale_range) : [mapping.scale_index])
        end
    end
    index in zero_scales && throw(ArgumentError("selected profile coordinate is removed by the zero-process constraint"))
    free=[j for j in eachindex(fit.beta) if j!=index && !(j in zero_scales)]
    previous=nothing;refinement=NamedTuple[];attempts=NamedTuple[]
    template=Float64.(start);template[index]=target
    orders=individual===nothing ? _CJS_INDIVIDUAL_PAIRED_ORDERS : _cjs_individual_integration_orders(prep;zero_process)
    for order in orders
        order_index=findfirst(==(order),orders)
        split=individual!==nothing && hasproperty(individual,:integration) &&
            individual.integration===:threshold_split_gauss_legendre
        rule=split ? nothing : _cjs_individual_rule(order);start=copy(template);candidate=copy(start)
        base=(order,start=copy(start),fixed_index=index,fixed_value=target)
        function failed(status,reason;iterations=0,gradient_norm=Inf,
                optimizer_converged=false,comparison_order=missing)
            push!(attempts,merge(base,(status,solution=copy(candidate),iterations,
                gradient_norm,optimizer_converged,comparison_order,reason)))
            return (objective=missing,status,theta=copy(candidate),order,refinement,attempts,reason)
        end
        # The selected preparation supplies numerator and normalizer rows to
        # the same marginal objective; do not replace it with numerator alone.
        objective=_cjs_individual_profile_objective(start,free,prep;order,rule,zero_process)
        result=try
            optimize_nuisance(objective,copy(start[free]))
        catch err
            err isa InterruptException && rethrow()
            return failed(:failed_reoptimization,sprint(showerror,err))
        end
        candidate[free]=Optim.minimizer(result)
        iterations=Optim.iterations(result);optimizer_converged=Optim.converged(result)
        optimizer_converged || return failed(:failed_reoptimization,
            "constrained nuisance optimizer did not converge";iterations,optimizer_converged)
        grad=try
            ForwardDiff.gradient(objective,candidate[free])
        catch err
            err isa InterruptException && rethrow()
            return failed(:failed_reoptimization,sprint(showerror,err);iterations,optimizer_converged)
        end
        gradient_norm=isempty(grad) ? 0. : norm(grad,Inf)
        if !(all(isfinite,grad) && gradient_norm<=1e-5)
            failure=failed(:failed_reoptimization,
                "constrained nuisance gradient exceeded the existing 1e-5 limit";
                iterations,gradient_norm,optimizer_converged)
            if all(isfinite,candidate) && order_index<length(orders)
                # Keep the rejected order in the attempt trace, then warm-start
                # the next rule from its finite candidate. Do not make it the
                # previous refinement solution: only gradient-qualified fits
                # may establish cross-order refinement.
                template=copy(candidate)
                continue
            end
            return failure
        end
        comparison_order=order_index<length(orders) ? orders[order_index+1] : orders[order_index-1]
        parts,comparison=try
            current=_cjs_individual_history_logliks(candidate,prep;order,rule,components=true,zero_process)
            comparison_rule=split ? nothing : _cjs_individual_rule(comparison_order)
            next=_cjs_individual_history_logliks(candidate,prep;order=comparison_order,
                rule=comparison_rule,components=true,zero_process)
            (current,next)
        catch err
            err isa InterruptException && rethrow()
            return failed(:failed_refinement,sprint(showerror,err);iterations,
                gradient_norm,optimizer_converged,comparison_order)
        end
        weighted=sum(prep.data.frequency.*parts.conditional)
        objective_change=abs(sum(prep.data.frequency.*(comparison.conditional.-parts.conditional)))
        parameter_change=previous===nothing ? Inf : norm(candidate-previous,Inf)
        conditional_change=maximum(abs,comparison.conditional.-parts.conditional;init=0.)
        numerator_change=maximum(abs,comparison.numerator.-parts.numerator;init=0.)
        denominator_change=maximum(abs,comparison.denominator.-parts.denominator;init=0.)
        objective_limit=objective_tol*max(1,sum(prep.data.frequency))
        isfinite(weighted) || return failed(:failed_refinement,"nonfinite selected conditional likelihood";
            iterations,gradient_norm,optimizer_converged,comparison_order)
        refined=previous!==nothing && all(isfinite,(objective_change,parameter_change,
            conditional_change,numerator_change,denominator_change)) &&
            objective_change<=objective_limit && parameter_change<=parameter_tol &&
            conditional_change<=integral_tol && numerator_change<=integral_tol && denominator_change<=integral_tol
        push!(refinement,(order,comparison_order,objective_change,parameter_change,
            max_conditional_change=conditional_change,max_numerator_change=numerator_change,
            max_denominator_change=denominator_change,integral_tol,objective_tol=objective_limit,
            parameter_tol,refined))
        push!(attempts,merge(base,(status=refined ? :available : :unrefined,
            solution=copy(candidate),iterations,gradient_norm,optimizer_converged,
            objective=Float64(-weighted),comparison_order,objective_change,parameter_change,
            conditional_change,numerator_change,denominator_change,objective_limit,
            integral_tol,parameter_tol,refined,reason=nothing)))
        refined && return (objective=Float64(-weighted),status=:available,theta=copy(candidate),order,refinement,attempts)
        previous=copy(candidate);template=copy(candidate)
    end
    return (objective=missing,status=:unrefined_at_cap,theta=copy(template),order=last(orders),
        refinement,attempts,reason="selected profile quadrature did not refine at the dimension-specific cap")
end

function _cjs_selection_profile_tolerance(point,fit)
    hasproperty(point,:refinement) && !isempty(point.refinement) &&
        return 2last(point.refinement).objective_tol
    return 200eps(Float64)*max(1,abs(fit.loglik),
        hasproperty(point,:objective) && !ismissing(point.objective) ? abs(point.objective) : 0.)
end

function _cjs_selection_profile_lr(point,fit)
    point.status==:available || return (missing,point.status)
    lr=2*(fit.loglik+point.objective)
    isfinite(lr) || return (missing,:failed_reoptimization)
    lr < -_cjs_selection_profile_tolerance(point,fit) && return (missing,:profile_improves_on_fit)
    return (max(0.,lr),:available)
end

function _cjs_selection_profile_endpoint(fit,index,direction,cutoff,center;
        profile_point=_cjs_selection_profile_point,center_point=nothing)
    direction in (-1,1) || throw(ArgumentError("selected profile direction must be -1 or 1"))
    isfinite(cutoff) && cutoff>0 && isfinite(center) ||
        throw(ArgumentError("selected profile cutoff and center must be finite, with positive cutoff"))
    points=NamedTuple[]
    result(estimate,status)=(;estimate,status,points=copy(points))
    function evaluate(value,phase;supplied=nothing)
        point=try
            supplied===nothing ? profile_point(fit,index,value) : supplied
        catch err
            err isa InterruptException && rethrow()
            (objective=missing,status=:failed_reoptimization,reason=sprint(showerror,err))
        end
        lr,status=_cjs_selection_profile_lr(point,fit)
        tolerance=_cjs_selection_profile_tolerance(point,fit)
        push!(points,(;value,phase,point,lr,status,tolerance))
        return lr,status,tolerance
    end
    lr,status,tolerance=evaluate(center,:center;supplied=center_point)
    status==:available || return result(missing,status)
    lr<cutoff || return result(missing,:center_outside_cutoff)
    lr<=tolerance || return result(missing,:center_inconsistent_with_fit)
    step=.5;inner=center;outer=center+direction*step;crossed=false
    for _ in 1:12
        isfinite(outer) || return result(missing,:endpoint_overflow)
        lr,status,tolerance=evaluate(outer,:expansion)
        status==:available || return result(missing,status)
        if lr>=cutoff;crossed=true;break;end
        inner=outer;step*=2;outer=center+direction*step
    end
    crossed || return result(missing,:non_crossing)
    lo,hi=min(inner,outer),max(inner,outer)
    for _ in 1:45
        mid=_cjs_profile_midpoint(lo,hi)
        lr,status,tolerance=evaluate(mid,:bisection)
        status==:available || return result(missing,status)
        abs(lr-cutoff)<=tolerance && return result(mid,:available)
        if direction<0
            lr>=cutoff ? (lo=mid) : (hi=mid)
        else
            lr>=cutoff ? (hi=mid) : (lo=mid)
        end
    end
    estimate=_cjs_profile_midpoint(lo,hi)
    lr,status,tolerance=evaluate(estimate,:endpoint)
    status==:available || return result(missing,status)
    abs(lr-cutoff)<=tolerance || return result(missing,:endpoint_lr_not_converged)
    return result(estimate,:available)
end
