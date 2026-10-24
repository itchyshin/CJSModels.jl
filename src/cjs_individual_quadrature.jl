# Persistent individual intercepts integrate each complete conditional history.
# This separate, bounded route leaves the verified sparse DRM engine unchanged.
_cjs_is_individual(prep)=hasproperty(prep,:individual) && prep.individual!==nothing
const _CJS_INDIVIDUAL_PAIRED_ORDERS=(25,41,65,97)
const _CJS_INDIVIDUAL_1D_ORDERS=(25,41,65,97,145,217)
const _CJS_INDIVIDUAL_ORDERS=tuple(sort!(unique!(vcat(collect(_CJS_INDIVIDUAL_PAIRED_ORDERS),collect(_CJS_INDIVIDUAL_1D_ORDERS))))...)
const _CJS_INDIVIDUAL_SPLIT_ORDERS=(8,12,16)
const _CJS_INDIVIDUAL_RULES=Dict(K=>let
    nodes,weights=_gauss_hermite(K)
    normalized=weights./sum(weights)
    (order=K,nodes=sqrt(2).*nodes,logweights=log.(normalized))
end for K in _CJS_INDIVIDUAL_ORDERS)

function _cjs_split_gaussian_rule(order::Integer)
    2 <= order <= 32 || throw(ArgumentError("CJS split quadrature order must be between 2 and 32"))
    off=[k/sqrt(4k^2-1) for k in 1:order-1]
    eig=eigen(SymTridiagonal(zeros(order),off))
    return eig.values,2 .* eig.vectors[1,:].^2
end

function _cjs_split_gaussian_edges(cuts)
    raw=collect(cuts)
    isempty(raw) && return [-9.,9.]
    candidates=filter(x->-9 < _cjs_primal(x) < 9,raw)
    sort!(candidates;by=_cjs_primal)
    edges=typeof(first(raw))[]
    push!(edges,zero(first(raw))-9.)
    for cut in candidates
        (length(edges)==1 || abs(_cjs_primal(cut)-_cjs_primal(last(edges)))>1e-12) &&
            push!(edges,cut)
    end
    push!(edges,zero(first(raw))+9.)
    return edges
end

"""Integrate a scalar over two independent standard-normal effects with moving split points."""
function _cjs_split_gaussian_expectation2(integrand,cuts_phi,cuts_p;order::Integer=12)
    nodes,weights=_cjs_split_gaussian_rule(order)
    phi_edges=_cjs_split_gaussian_edges(cuts_phi)
    p_edges=_cjs_split_gaussian_edges(cuts_p)
    total=nothing
    for i in 1:length(phi_edges)-1,j in 1:length(p_edges)-1
        plo,phi=phi_edges[i],phi_edges[i+1]
        qlo,qhi=p_edges[j],p_edges[j+1]
        pscale,qscale=(phi-plo)/2,(qhi-qlo)/2
        pmid,qmid=(phi+plo)/2,(qhi+qlo)/2
        for a in eachindex(nodes),b in eachindex(nodes)
            u=pmid+pscale*nodes[a]
            v=qmid+qscale*nodes[b]
            weight=pscale*qscale*weights[a]*weights[b]*exp(-(u*u+v*v)/2)/(2pi)
            term=weight*integrand(u,v)
            total=total===nothing ? term : total+term
        end
    end
    return total
end

function _cjs_individual_orders(prep;zero_process=nothing)
    re=prep.individual
    zeros=zero_process===nothing ? () : zero_process isa Symbol ? (zero_process,) :
        zero_process isa Tuple ? zero_process : throw(ArgumentError("zero_process must be nothing, a process symbol, or a tuple of process symbols"))
    all(x->x in (:phi,:p),zeros) && length(unique(zeros))==length(zeros) ||
        throw(ArgumentError("zero_process must contain distinct :phi and/or :p symbols"))
    active=Int(re.phi!==nothing && !(:phi in zeros))+Int(re.p!==nothing && !(:p in zeros))
    # Keep the conservative two-dimensional ladder for a zero-dimensional
    # reduced model; exactly one active normal effect uses the longer 1D ladder.
    return active==1 ? _CJS_INDIVIDUAL_1D_ORDERS : _CJS_INDIVIDUAL_PAIRED_ORDERS
end

function _cjs_individual_rule(order::Integer)
    order in _CJS_INDIVIDUAL_ORDERS || throw(ArgumentError("cjs individual quadrature order must be one of 25, 41, 65, 97, 145, 217"))
    return _CJS_INDIVIDUAL_RULES[order]
end

function _cjs_split_random(formula,label)
    formula isa FormulaTerm && formula.lhs isa Term && formula.lhs.sym==label ||
        throw(ArgumentError("cjs $label formula must have the label $label on its left side"))
    terms=formula.rhs isa Tuple ? collect(formula.rhs) : Any[formula.rhs]
    fixed=Any[];group=nothing
    for term in terms
        if term isa FunctionTerm && term.f === (|)
            group===nothing || throw(ArgumentError("cjs permits one random intercept per process; shared-plus-individual effects are unsupported"))
            length(term.args)==2 && term.args[1] isa ConstantTerm && term.args[1].n==1 && term.args[2] isa Term ||
                throw(ArgumentError("cjs random effects require (1 | group); random slopes and correlated effects are unsupported"))
            group=term.args[2].sym
        else
            _cjs_validate_term(term);push!(fixed,term)
        end
    end
    rhs=isempty(fixed) ? ConstantTerm(1) : length(fixed)==1 ? only(fixed) : Tuple(fixed)
    return FormulaTerm(formula.lhs,rhs),group
end

function _cjs_individual_sd_design(formula,table,ids,label; informative=trues(length(ids)), required_mask=nothing, contrasts=Dict{Symbol,Any}())
    formula===nothing && return (X=ones(length(ids),1),offset=zeros(length(ids)),names=["(Intercept)"],formula=nothing,design=nothing)
    formula isa FormulaTerm && formula.lhs isa Term && formula.lhs.sym==Symbol("sd_",label) ||
        throw(ArgumentError("cjs sd_$label must be a formula with left side sd_$label"))
    _cjs_validate_term(formula.rhs)
    cols=Tables.columntable(table);N=length(ids)
    length(informative)==N || throw(ArgumentError("cjs sd_$label information mask does not align with individuals"))
    active=findall(informative)
    isempty(active) && throw(ArgumentError("cjs sd_$label formula has no informative individuals"))
    vars=unique(vcat(StatsModels.termvars(formula.rhs),[t.sym for t in _cjs_offsets(formula.rhs)]))
    any(v->!haskey(cols,v),vars) && throw(ArgumentError("cjs sd_$label predictors must be present in the $label table"))
    Trows=length(cols.id)÷N
    length(cols.id)==N*Trows || throw(ArgumentError("cjs sd_$label table rows do not align with individuals"))
    required_mask=required_mask===nothing ? falses(length(cols.id)) : BitVector(required_mask)
    length(required_mask)==length(cols.id) || throw(ArgumentError("cjs sd_$label required-row mask does not align with the $label table"))
    idcols=Dict{Symbol,Any}(:id=>ids)
    for v in vars
        values=Vector{Base.nonmissingtype(eltype(getproperty(cols,v)))}(undef,N)
        for i in 1:N
            ix=(i-1)*Trows+1:i*Trows
            supplied=collect(getproperty(cols,v)[ix])
            any(ismissing,supplied[required_mask[ix]]) &&
                throw(ArgumentError("cjs sd_$label predictor $v must be complete on required selection prehistory rows for individual $i"))
            vv=collect(skipmissing(supplied))
            if isempty(vv)
                informative[i] && throw(ArgumentError("cjs sd_$label predictor $v is missing within informative individual $i"))
                # This history has likelihood one. The unused design row is
                # internal padding and is not a supplied prediction profile.
                # Filled below from a declared value solely to construct a typed table.
                # Neither this placeholder nor its factor level enters schema fitting.
                continue
            end
            all(x->isequal(x,first(vv)),vv) || throw(ArgumentError("cjs sd_$label predictor $v must be constant within each individual"))
            values[i]=first(vv)
        end
        donor=first(active)
        for i in eachindex(informative)
            if !informative[i] && isempty(collect(skipmissing(getproperty(cols,v)[(i-1)*Trows+1:i*Trows])))
                values[i]=values[donor]
            end
        end
        idcols[v]=values
    end
    tab=(;id=ids,(v=>idcols[v] for v in vars)...)
    des=_cjs_design(formula,tab,active,Symbol("sd_",label);contrasts=contrasts)
    X=zeros(N,size(des.X,2)); off=zeros(N)
    X[active,:]=des.X; off[active]=des.offset
    # Unused rows have likelihood one. Preserve a neutral intercept row without
    # adding invented levels or predictor values to the fitted schema.
    intercept=findfirst(==("(Intercept)"),des.names)
    intercept===nothing || (X[.!informative,intercept].=1.)
    return (X=X,offset=off,names=des.names,formula=formula,design=des)
end

_cjs_offsets(t::FunctionTerm)=nameof(t.f)==:offset ? Any[t.args...] : reduce(vcat,(_cjs_offsets(x) for x in t.args);init=Any[])
_cjs_offsets(t::Tuple)=reduce(vcat,(_cjs_offsets(x) for x in t);init=Any[])
_cjs_offsets(t)=Any[]

function _cjs_prepare_sd(prep,sd_phi,sd_p;selection=nothing,contrasts=Dict{Symbol,Any}())
    re=prep.individual;d=prep.data;nfixed=prep.nphi+prep.np;cursor=nfixed+1
    out=Any[]
    for (label,m,formula,table) in ((:phi,re.phi,sd_phi,d.phi_data),(:p,re.p,sd_p,d.p_data))
        m===nothing && (push!(out,nothing);continue)
        informative=BitVector(d.first .< d.end)
        required_mask = if selection===nothing
            nothing
        else
            nrows=length(Tables.columntable(table).id)
            mask=falses(nrows)
            rows=getproperty(selection,Symbol(label,"_rows"))
            constraint=prep.terminal_constraint
            if constraint!==nothing && only(keys(constraint))==label
                T=size(d.Y,2);block=label==:phi ? T-1 : T
                fixed_rows=Set(row for row in 1:nrows if mod1(row,block)==block)
                # A terminal-constrained probability is fixed by the
                # constraint and does not depend on the individual effect.
                rows=filter(row->!(row in fixed_rows),rows)
            end
            mask[rows].=true
            # A process-specific SD profile is needed only when that process
            # contributes either a post-capture likelihood row or a selection
            # normalizer row for this individual. Final-first histories can
            # have no survival contribution while still informing detection.
            Trows=nrows÷length(d.ids)
            for i in eachindex(informative)
                informative[i] |= any(@view mask[(i-1)*Trows+1:i*Trows])
            end
            mask
        end
        des=_cjs_individual_sd_design(formula,table,d.ids,label;
            informative,required_mask,contrasts=contrasts)
        r=cursor:cursor+size(des.X,2)-1;cursor=last(r)+1
        push!(out,merge(m,(scale_index=first(r),scale_range=r,sd_X=des.X,sd_offset=des.offset,sd_names=des.names,sd_formula=des.formula,sd_design=des.design)))
    end
    sm,pm=out
    return merge(prep,(individual=merge(re,(phi=sm,p=pm,nscale=cursor-nfixed-1,
        names=vcat(sm===nothing ? String[] : ["sd_phi: "*n for n in sm.sd_names],
                   pm===nothing ? String[] : ["sd_p: "*n for n in pm.sd_names]))),))
end

# Only informative post-capture rows need grouping values; pre-capture rows,
# masked detections, and administrative suffixes do not select an effect.
function _cjs_individual_group(data,group,label)
    group===nothing && return false
    table=label==:phi ? data.phi_data : data.p_data
    haskey(table,group) || throw(ArgumentError("cjs $label grouping column $group is absent"))
    group==:id && return true
    vals=getproperty(table,group);_,T=size(data.Y);seen=Set{Any}()
    for i in eachindex(data.ids)
        rows=label==:phi ? [(i-1)*(T-1)+t for t=data.first[i]:data.end[i]-1 if any(view(data.observed,i,t+1:data.end[i]))] :
                          [(i-1)*T+t for t=data.first[i]+1:data.end[i] if data.observed[i,t]]
        isempty(rows) && continue
        values=vals[rows]
        any(ismissing,values) && throw(ArgumentError("cjs $label group cannot be missing on contributing rows"))
        length(unique(values))==1 || return false
        v=first(values);v in seen && return false;push!(seen,v)
    end
    return !isempty(seen)
end

function _cjs_prepare_random(data,phi,p;terminal_constraint=nothing,contrasts=Dict{Symbol,Any}())
    fphi,gphi=_cjs_split_random(phi,:phi);fp,gp=_cjs_split_random(p,:p)
    individual_phi=_cjs_individual_group(data,gphi,:phi)
    individual_p=_cjs_individual_group(data,gp,:p)
    (individual_phi || individual_p) || return _cjs_prepare_shared(data,phi,p;terminal_constraint,contrasts)
    (gphi!==nothing && !individual_phi || gp!==nothing && !individual_p) &&
        throw(ArgumentError("cjs shared-plus-individual effects are unsupported"))
    prep=_cjs_prepare(data,fphi,fp;terminal_constraint,contrasts)
    nfixed=prep.nphi+prep.np
    sm=individual_phi ? (group=gphi,scale_index=nfixed+1) : nothing
    pm=individual_p ? (group=gp,scale_index=nfixed+(individual_phi ? 2 : 1)) : nothing
    for (label,m) in ((:phi,sm),(:p,pm))
        m===nothing && continue
        des=label==:phi ? prep.phi_design : prep.p_design
        terminal=label==:phi ? prep.phi_terminal : prep.p_terminal
        constrained=prep.terminal_constraint!==nothing && only(keys(prep.terminal_constraint))==label
        any(k->!(constrained && terminal[k]) && (label==:phi ?
            any(view(data.observed,cld(des.rows[k],size(data.Y,2)-1),mod1(des.rows[k],size(data.Y,2)-1)+1:data.end[cld(des.rows[k],size(data.Y,2)-1)])) :
            data.observed[cld(des.rows[k],size(data.Y,2)),mod1(des.rows[k],size(data.Y,2))]),eachindex(des.rows)) ||
            throw(ArgumentError("cjs $label individual effect has no informative rows after masks and terminal constraints"))
    end
    names=["log_sigma_$(label): $(m.group)" for (label,m) in ((:phi,sm),(:p,pm)) if m!==nothing]
    route=sm===nothing ? :individual_p : pm===nothing ? :individual_phi : :individual_pair
    individual=(phi=sm,p=pm,nscale=length(names),route=route,names=names,order=25)
    return merge(prep,(phi_design=merge(prep.phi_design,(formula=phi,)),
                       p_design=merge(prep.p_design,(formula=p,)),individual=individual))
end

# log(expm1(x)/x) has a removable singularity at zero. This Taylor
# branch has error below Float64 rounding for |x|<=1e-4 and avoids reciprocal
# subnormal quantities in both first and second automatic derivatives.
function _cjs_individual_logexprel(x)
    if abs(_cjs_primal(x))<=1e-4
        return log1p(x*(1/2+x*(1/6+x*(1/24+x*(1/120+x*(1/720+x/5040))))))
    end
    return log(expm1(x)/x)
end

function _cjs_individual_log1msurvival(l,lc,delta)
    delta==1 && return lc
    if _cjs_primal(l)>=-1e-4
        # 1-s^delta=(1-s)*delta*exprel(delta*log(s))/exprel(log(s)).
        return lc+log(delta)+_cjs_individual_logexprel(delta*l)-_cjs_individual_logexprel(l)
    end
    return _cjs_log1mexp(delta*l)
end

function _cjs_individual_selection_logweight(i,a,b,beta,sp,sq,prep,zn,pn)
    hasproperty(prep,:selection) || return zero(eltype(beta))
    sel=prep.selection;d=prep.data;T=size(d.Y,2)
    logw=zero(eltype(beta))
    for j in eachindex(sel.phi_by_id[i])
        eta=dot(view(sel.phi_X_by_id[i],j,:),view(beta,1:prep.nphi))+sel.phi_offset_by_id[i][j]
        re=prep.individual.phi;re===nothing || (eta+=sp*zn[a])
        row=sel.phi_by_id[i][j];t=mod1(row,T-1)
        if prep.terminal_constraint!==nothing && only(keys(prep.terminal_constraint))==:phi && t==T-1
            logw+=d.intervals[t]*log(only(values(prep.terminal_constraint)))
        else
            logw+=d.intervals[t]*_cjs_logistic_log(eta)
        end
    end
    for j in eachindex(sel.p_by_id[i])
        eta=dot(view(sel.p_X_by_id[i],j,:),view(beta,prep.nphi+1:prep.nphi+prep.np))+sel.p_offset_by_id[i][j]
        re=prep.individual.p;re===nothing || (eta+=sq*pn[b])
        row=sel.p_by_id[i][j];t=mod1(row,T)
        if prep.terminal_constraint!==nothing && only(keys(prep.terminal_constraint))==:p && t==T
            logw+=t==d.first[i] ? log(only(values(prep.terminal_constraint))) : log1p(-only(values(prep.terminal_constraint)))
        else
            logw+=t==d.first[i] ? _cjs_logistic_log(eta) : _cjs_logistic_log(-eta)
        end
    end
    return logw
end

"Internal normalized Gaussian log integral for every complete first-capture-conditional history."
function _cjs_individual_history_logliks(theta,prep;order=prep.individual.order,
        rule=nothing,components=false,zero_process=nothing)
    re=prep.individual;nfixed=prep.nphi+prep.np
    length(theta)==nfixed+re.nscale || throw(ArgumentError("CJS individual-effect parameter dimension mismatch"))
    zero_processes=zero_process===nothing ? () : zero_process isa Symbol ? (zero_process,) :
        zero_process isa Tuple ? zero_process : throw(ArgumentError("zero_process must be nothing, a process symbol, or a tuple of process symbols"))
    all(x->x in (:phi,:p),zero_processes) && length(unique(zero_processes))==length(zero_processes) ||
        throw(ArgumentError("zero_process must contain distinct :phi and/or :p symbols"))
    if hasproperty(re,:integration) && re.integration===:threshold_split_gauss_legendre
        return _cjs_individual_split_history_logliks(theta,prep;order,components,zero_process)
    end
    rule===nothing && (rule=_cjs_individual_rule(order))
    beta=view(theta,1:nfixed);d=prep.data;N,T=size(d.Y)
    eta_phi=prep.phi_design.X*view(beta,1:prep.nphi)+prep.phi_design.offset
    eta_p=prep.p_design.X*view(beta,prep.nphi+1:nfixed)+prep.p_design.offset
    zero_phi=re.phi===nothing || (:phi in zero_processes)
    zero_p=re.p===nothing || (:p in zero_processes)
    zn=zero_phi ? (0.,) : rule.nodes;zw=zero_phi ? (0.,) : rule.logweights
    pn=zero_p ? (0.,) : rule.nodes;pw=zero_p ? (0.,) : rule.logweights
    history_loglik=zeros(eltype(theta),N);log_numerator=zeros(eltype(theta),N);log_denominator=zeros(eltype(theta),N);pos=1;c=prep.terminal_constraint
    logzero=oftype(zero(eltype(theta)),-Inf)
    for i=1:N
        k=d.end[i]-d.first[i]
        # Zero-contribution histories still need a valid selection event. Keep
        # its actual normalizer in diagnostics/refinement, and reject a zero
        # conditioning mass before simplifying numerator / denominator to one.
        if k==0 || !any(view(d.observed,i,d.first[i]+1:d.end[i]))
            if hasproperty(prep,:selection)
                sp=zero_phi ? zero(eltype(theta)) : _cjs_sd_value(re.phi,theta,i)
                sq=zero_p ? zero(eltype(theta)) : _cjs_sd_value(re.p,theta,i)
                denominator=logzero
                for a in eachindex(zn),b in eachindex(pn)
                    lognode=zw[a]+pw[b]
                    logw=_cjs_individual_selection_logweight(i,a,b,beta,sp,sq,prep,zn,pn)
                    denominator=_cjs_logadd(denominator,lognode+logw)
                end
                isfinite(_cjs_primal(denominator)) || throw(ArgumentError("cjs selection event has zero or non-finite conditioning probability for history $(d.ids[i])"))
                history_loglik[i]=zero(eltype(theta));log_numerator[i]=denominator;log_denominator[i]=denominator
            else
                history_loglik[i]=zero(eltype(theta));log_numerator[i]=zero(eltype(theta));log_denominator[i]=zero(eltype(theta))
            end
            pos+=k;continue
        end
        sp=zero_phi ? zero(eltype(theta)) : _cjs_sd_value(re.phi,theta,i)
        sq=zero_p ? zero(eltype(theta)) : _cjs_sd_value(re.p,theta,i)
        all(x->isfinite(_cjs_primal(x)),(sp,sq)) || throw(ArgumentError("cjs individual SD must be finite"))
        ls=zeros(eltype(theta),T-1,length(zn));lcs=fill(logzero,T-1,length(zn))
        lp=zeros(eltype(theta),T,length(pn));lcp=fill(logzero,T,length(pn))
        for a in eachindex(zn),j=0:k-1
            t=d.first[i]+j;x=eta_phi[pos+j]+sp*zn[a]
            l=_cjs_logistic_log(x);lc=_cjs_logistic_log(-x)
            if c!==nothing && only(keys(c))==:phi && prep.phi_terminal[pos+j]
                l=zero(x)+log(c.phi);lc=zero(x)+log1p(-c.phi)
            end
            ls[t,a]=l*d.intervals[t]
            lcs[t,a]=_cjs_individual_log1msurvival(l,lc,d.intervals[t])
        end
        for b in eachindex(pn),j=0:k-1
            t=d.first[i]+j+1;x=eta_p[pos+j]+sq*pn[b]
            lp[t,b]=_cjs_logistic_log(x);lcp[t,b]=_cjs_logistic_log(-x)
            if c!==nothing && only(keys(c))==:p && prep.p_terminal[pos+j]
                lp[t,b]=zero(x)+log(c.p);lcp[t,b]=zero(x)+log1p(-c.p)
            end
        end
        marginal=logzero;denominator=logzero
        for a in eachindex(zn),b in eachindex(pn)
            ll=_cjs_history_loglik_logs(view(d.Y,i,:),view(d.observed,i,:),d.first[i],d.end[i],
                view(ls,:,a),view(lp,:,b),view(lcs,:,a),view(lcp,:,b))
            lognode=zw[a]+pw[b]
            logw=_cjs_individual_selection_logweight(i,a,b,beta,sp,sq,prep,zn,pn)
            marginal=_cjs_logadd(marginal,lognode+logw+ll)
            hasproperty(prep,:selection) && (denominator=_cjs_logadd(denominator,lognode+logw))
        end
        log_numerator[i]=marginal;log_denominator[i]=hasproperty(prep,:selection) ? denominator : zero(eltype(theta))
        history_loglik[i]=marginal-log_denominator[i];pos+=k
    end
    return components ? (conditional=history_loglik,numerator=log_numerator,denominator=log_denominator) : history_loglik
end

function _cjs_individual_split_selection_logweight(i,u,v,beta,sp,sq,prep)
    hasproperty(prep,:selection) || return zero(eltype(beta))
    sel=prep.selection;d=prep.data;T=size(d.Y,2)
    value=zero(eltype(beta))
    for j in eachindex(sel.phi_by_id[i])
        eta=dot(view(sel.phi_X_by_id[i],j,:),view(beta,1:prep.nphi))+
            sel.phi_offset_by_id[i][j]
        row=sel.phi_by_id[i][j];t=mod1(row,T-1)
        if prep.terminal_constraint!==nothing && only(keys(prep.terminal_constraint))==:phi && t==T-1
            value+=d.intervals[t]*log(only(values(prep.terminal_constraint)))
        else
            value+=d.intervals[t]*_cjs_logistic_log(eta+sp*u)
        end
    end
    for j in eachindex(sel.p_by_id[i])
        eta=dot(view(sel.p_X_by_id[i],j,:),view(beta,prep.nphi+1:prep.nphi+prep.np))+
            sel.p_offset_by_id[i][j]
        row=sel.p_by_id[i][j];t=mod1(row,T)
        if prep.terminal_constraint!==nothing && only(keys(prep.terminal_constraint))==:p && t==T
            probability=only(values(prep.terminal_constraint))
            value+=t==d.first[i] ? log(probability) : log1p(-probability)
        else
            value+=t==d.first[i] ? _cjs_logistic_log(eta+sq*v) :
                _cjs_logistic_log(-eta-sq*v)
        end
    end
    return value
end

function _cjs_individual_split_cuts(prep,theta,i,process,sigma)
    d=prep.data;T=size(d.Y,2);beta=view(theta,1:prep.nphi+prep.np)
    cuts=[zero(theta[1])+x for x in (-6.,-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.,6.)]
    function add_transition!(eta)
        _cjs_primal(sigma)>0 || return
        for level in (-8.,-4.,0.,4.,8.)
            cut=(level-eta)/sigma
            isfinite(_cjs_primal(cut)) && push!(cuts,cut)
        end
    end
    pos=1+sum((d.end[j]-d.first[j] for j in 1:i-1);init=0)
    if process==:phi
        mapping=prep.individual.phi
        mapping===nothing && return cuts
        for j in 0:d.end[i]-d.first[i]-1
            row=pos+j;t=d.first[i]+j
            constrained=prep.terminal_constraint!==nothing &&
                only(keys(prep.terminal_constraint))==:phi && prep.phi_terminal[row]
            constrained && continue
            eta=dot(view(prep.phi_design.X,row,:),view(beta,1:prep.nphi))+
                prep.phi_design.offset[row]
            add_transition!(eta)
        end
        if hasproperty(prep,:selection)
            for j in eachindex(prep.selection.phi_by_id[i])
                row=prep.selection.phi_by_id[i][j];t=mod1(row,T-1)
                prep.terminal_constraint!==nothing &&
                    only(keys(prep.terminal_constraint))==:phi && t==T-1 && continue
                eta=dot(view(prep.selection.phi_X_by_id[i],j,:),view(beta,1:prep.nphi))+
                    prep.selection.phi_offset_by_id[i][j]
                add_transition!(eta)
            end
        end
    else
        mapping=prep.individual.p
        mapping===nothing && return cuts
        for j in 0:d.end[i]-d.first[i]-1
            row=pos+j;t=d.first[i]+j+1
            d.observed[i,t] || continue
            constrained=prep.terminal_constraint!==nothing &&
                only(keys(prep.terminal_constraint))==:p && prep.p_terminal[row]
            constrained && continue
            eta=dot(view(prep.p_design.X,row,:),view(beta,prep.nphi+1:prep.nphi+prep.np))+
                prep.p_design.offset[row]
            add_transition!(eta)
        end
        if hasproperty(prep,:selection)
            for j in eachindex(prep.selection.p_by_id[i])
                row=prep.selection.p_by_id[i][j];t=mod1(row,T)
                prep.terminal_constraint!==nothing &&
                    only(keys(prep.terminal_constraint))==:p && t==T && continue
                eta=dot(view(prep.selection.p_X_by_id[i],j,:),view(beta,prep.nphi+1:prep.nphi+prep.np))+
                    prep.selection.p_offset_by_id[i][j]
                add_transition!(eta)
            end
        end
    end
    return cuts
end

"""Split-rule numerator and selection denominator for one persistent-effect history."""
function _cjs_individual_split_components(theta,prep,i;order::Integer=12,zero_process=nothing)
    re=prep.individual;d=prep.data;T=size(d.Y,2)
    1<=i<=length(d.ids) || throw(BoundsError(d.ids,i))
    length(theta)==prep.nphi+prep.np+re.nscale ||
        throw(ArgumentError("CJS individual-effect parameter dimension mismatch"))
    zero_processes=zero_process===nothing ? () : zero_process isa Symbol ? (zero_process,) :
        zero_process isa Tuple ? zero_process : throw(ArgumentError("zero_process must be nothing, a process symbol, or a tuple of process symbols"))
    all(x->x in (:phi,:p),zero_processes) && length(unique(zero_processes))==length(zero_processes) ||
        throw(ArgumentError("zero_process must contain distinct :phi and/or :p symbols"))
    beta=view(theta,1:prep.nphi+prep.np)
    phi=re.phi===nothing || (:phi in zero_processes) ? zero(eltype(theta)) : _cjs_sd_value(re.phi,theta,i)
    p=re.p===nothing || (:p in zero_processes) ? zero(eltype(theta)) : _cjs_sd_value(re.p,theta,i)
    all(x->isfinite(_cjs_primal(x)),(phi,p)) ||
        throw(ArgumentError("cjs individual SD must be finite"))
    phi_cuts=_cjs_individual_split_cuts(prep,theta,i,:phi,phi)
    p_cuts=_cjs_individual_split_cuts(prep,theta,i,:p,p)
    pos=1+sum((d.end[j]-d.first[j] for j in 1:i-1);init=0)
    zero_log=zero(eltype(theta))-Inf
    # The tensor rule calls this closure many thousands of times per history.
    # Reuse typed work buffers instead of allocating four full vectors at every
    # node; each call consumes the likelihood before the next node mutates them.
    ls=fill(zero_log,T-1);lcs=fill(zero_log,T-1)
    lp=fill(zero_log,T);lcp=fill(zero_log,T)
    y=view(d.Y,i,:);observed=view(d.observed,i,:)
    function log_components(u,v)
        fill!(ls,zero_log);fill!(lcs,zero_log)
        fill!(lp,zero_log);fill!(lcp,zero_log)
        for j in 0:d.end[i]-d.first[i]-1
            row=pos+j;t=d.first[i]+j
            eta=dot(view(prep.phi_design.X,row,:),view(beta,1:prep.nphi))+
                prep.phi_design.offset[row]
            log_survival=_cjs_logistic_log(eta+phi*u)
            log_failure=_cjs_logistic_log(-eta-phi*u)
            if prep.terminal_constraint!==nothing &&
                    only(keys(prep.terminal_constraint))==:phi && prep.phi_terminal[row]
                probability=only(values(prep.terminal_constraint))
                log_survival=zero(log_survival)+log(probability)
                log_failure=zero(log_failure)+log1p(-probability)
            end
            delta=d.intervals[t]
            ls[t]=log_survival*delta
            lcs[t]=_cjs_individual_log1msurvival(log_survival,log_failure,delta)
        end
        for j in 0:d.end[i]-d.first[i]-1
            row=pos+j;t=d.first[i]+j+1
            d.observed[i,t] || continue
            eta=dot(view(prep.p_design.X,row,:),view(beta,prep.nphi+1:prep.nphi+prep.np))+
                prep.p_design.offset[row]
            lp[t]=_cjs_logistic_log(eta+p*v)
            lcp[t]=_cjs_logistic_log(-eta-p*v)
            if prep.terminal_constraint!==nothing &&
                    only(keys(prep.terminal_constraint))==:p && prep.p_terminal[row]
                probability=only(values(prep.terminal_constraint))
                lp[t]=zero(lp[t])+log(probability)
                lcp[t]=zero(lcp[t])+log1p(-probability)
            end
        end
        log_history=_cjs_history_loglik_logs(y,observed,d.first[i],d.end[i],ls,lp,lcs,lcp)
        log_weight=_cjs_individual_split_selection_logweight(i,u,v,beta,phi,p,prep)
        return log_history,log_weight
    end
    log_numerator=_cjs_split_gaussian_expectation2((u,v)->begin
        log_history,log_weight=log_components(u,v)
        exp(log_history+log_weight)
    end,phi_cuts,p_cuts;order)
    isfinite(_cjs_primal(log_numerator)) && log_numerator>0 ||
        throw(ArgumentError("cjs split numerator is zero or non-finite for history $(d.ids[i])"))
    numerator=log(log_numerator)
    denominator=if hasproperty(prep,:selection)
        value=_cjs_split_gaussian_expectation2((u,v)->
            exp(last(log_components(u,v))),phi_cuts,p_cuts;order)
        isfinite(_cjs_primal(value)) && value>0 ||
            throw(ArgumentError("cjs split conditioning probability is zero or non-finite for history $(d.ids[i])"))
        log(value)
    else
        zero(numerator)
    end
    return (numerator=numerator,denominator=denominator,conditional=numerator-denominator)
end

function _cjs_individual_split_history_logliks(theta,prep;order::Integer=12,components=false,zero_process=nothing)
    re=prep.individual;d=prep.data
    length(theta)==prep.nphi+prep.np+re.nscale ||
        throw(ArgumentError("CJS individual-effect parameter dimension mismatch"))
    parts=[_cjs_individual_split_components(theta,prep,i;order,zero_process) for i in eachindex(d.ids)]
    conditional=[part.conditional for part in parts]
    components ? (conditional,numerator=[part.numerator for part in parts],
        denominator=[part.denominator for part in parts]) : conditional
end

function _cjs_individual_split_objective(theta,prep;order::Integer=12)
    history_loglik=_cjs_individual_split_history_logliks(theta,prep;order)
    return -sum(prep.data.frequency[i]*history_loglik[i] for i in eachindex(history_loglik))
end

function _cjs_individual_split_refinement(theta,previous,prep,order;
        integral_tol=1e-8,parameter_tol=1e-5)
    idx=findfirst(==(order),_CJS_INDIVIDUAL_SPLIT_ORDERS)
    idx===nothing && throw(ArgumentError("unsupported selected CJS split-quadrature order"))
    comparison_order=_CJS_INDIVIDUAL_SPLIT_ORDERS[idx<length(_CJS_INDIVIDUAL_SPLIT_ORDERS) ? idx+1 : idx-1]
    current=_cjs_individual_split_history_logliks(theta,prep;order,components=true)
    comparison=_cjs_individual_split_history_logliks(theta,prep;order=comparison_order,components=true)
    history_changes=comparison.conditional.-current.conditional
    numerator_changes=comparison.numerator.-current.numerator
    denominator_changes=comparison.denominator.-current.denominator
    value=-sum(prep.data.frequency[i]*current.conditional[i] for i in eachindex(current.conditional))
    comparison_value=-sum(prep.data.frequency[i]*comparison.conditional[i] for i in eachindex(comparison.conditional))
    objective_change=abs(comparison_value-value)
    parameter_change=previous===nothing ? Inf : norm(theta-previous,Inf)
    refined=previous!==nothing && _cjs_individual_refinement_accepts(
        vcat(history_changes,numerator_changes,denominator_changes),parameter_change;
        integral_tol,parameter_tol)
    return (order,comparison_order,objective=value,comparison_objective=comparison_value,
        objective_change,max_history_log_integral_change=isempty(history_changes) ? 0. : maximum(abs,history_changes),
        max_numerator_log_integral_change=isempty(numerator_changes) ? 0. : maximum(abs,numerator_changes),
        max_denominator_log_integral_change=isempty(denominator_changes) ? 0. : maximum(abs,denominator_changes),
        history_log_integral_abs_tol=integral_tol,parameter_change,parameter_limit=parameter_tol,
        refined,status=refined ? :refined : idx==length(_CJS_INDIVIDUAL_SPLIT_ORDERS) ? :unrefined_at_cap : :needs_refinement)
end

function _cjs_individual_objective(theta,prep;order=prep.individual.order,
        rule=nothing,zero_process=nothing)
    if hasproperty(prep.individual,:integration) &&
            prep.individual.integration===:threshold_split_gauss_legendre
        return _cjs_individual_split_objective(theta,prep;order)
    end
    rule===nothing && (rule=_cjs_individual_rule(order))
    history_loglik=_cjs_individual_history_logliks(theta,prep;order,rule,zero_process)
    return -sum(prep.data.frequency[i]*history_loglik[i] for i in eachindex(history_loglik))
end

_cjs_uses_split_integration(prep) = hasproperty(prep,:selection) && prep.selection!==nothing &&
    ((prep.individual.phi!==nothing && prep.individual.phi.sd_formula!==nothing) ||
     (prep.individual.p!==nothing && prep.individual.p.sd_formula!==nothing))

function _cjs_individual_integration_orders(prep;zero_process=nothing)
    split = hasproperty(prep.individual,:integration) ?
        prep.individual.integration===:threshold_split_gauss_legendre : _cjs_uses_split_integration(prep)
    return split ? _CJS_INDIVIDUAL_SPLIT_ORDERS : _cjs_individual_orders(prep;zero_process)
end

_cjs_sd_coefficients(m,theta)=hasproperty(m,:scale_range) ? theta[m.scale_range] : [theta[m.scale_index]]
_cjs_sd_value(m,theta,i)=hasproperty(m,:sd_X) ? exp(dot(view(m.sd_X,i,:),view(theta,m.scale_range))+m.sd_offset[i]) : exp(theta[m.scale_index])

function _cjs_individual_refinement_accepts(history_changes,parameter_change;
        integral_tol=1e-8,parameter_tol=1e-5)
    return all(isfinite,history_changes) && isfinite(parameter_change) &&
        all(x->abs(x)<=integral_tol,history_changes) && parameter_change<=parameter_tol
end

# Compare rules only between optimizer runs. At the cap, the last lower rule
# supplies the comparison; the fit is retained even when the check fails.
function _cjs_individual_refinement(theta,previous,prep,order;integral_tol=1e-8,parameter_tol=1e-5)
    orders=_cjs_individual_orders(prep)
    idx=findfirst(==(order),orders)
    idx===nothing && throw(ArgumentError("unsupported individual quadrature order"))
    next_order=idx<length(orders) ? orders[idx+1] : nothing
    comparison_order=next_order===nothing ? orders[idx-1] : next_order
    current_parts=_cjs_individual_history_logliks(theta,prep;order,components=true)
    comparison_parts=_cjs_individual_history_logliks(theta,prep;order=comparison_order,components=true)
    history_changes=comparison_parts.conditional.-current_parts.conditional
    numerator_changes=comparison_parts.numerator.-current_parts.numerator
    denominator_changes=comparison_parts.denominator.-current_parts.denominator
    value=-sum(prep.data.frequency[i]*current_parts.conditional[i] for i in eachindex(current_parts.conditional))
    comparison=-sum(prep.data.frequency[i]*comparison_parts.conditional[i] for i in eachindex(comparison_parts.conditional))
    objective_change=abs(comparison-value)
    parameter_change=previous===nothing ? Inf : norm(theta-previous,Inf)
    history_change=isempty(history_changes) ? 0. : maximum(abs,history_changes)
    numerator_change=isempty(numerator_changes) ? 0. : maximum(abs,numerator_changes)
    denominator_change=isempty(denominator_changes) ? 0. : maximum(abs,denominator_changes)
    refined=previous!==nothing && _cjs_individual_refinement_accepts(vcat(history_changes,numerator_changes,denominator_changes),parameter_change;
        integral_tol,parameter_tol)
    return (order=order,next_order=next_order,comparison_order=comparison_order,
        objective=value,comparison_objective=comparison,objective_change=objective_change,
        max_history_log_integral_change=history_change,max_numerator_log_integral_change=numerator_change,
        max_denominator_log_integral_change=denominator_change,history_log_integral_abs_tol=integral_tol,
        parameter_change=parameter_change,parameter_limit=parameter_tol,
        refined=refined,status=refined ? :refined : next_order===nothing ? :unrefined_at_cap : :needs_refinement)
end

function _cjs_fit_individual(prep;start=nothing,maxiters=1000,g_tol=1e-8,
                             integral_tol=1e-8,parameter_tol=1e-5)
    maxiters>0 && isfinite(g_tol) && g_tol>0 || throw(ArgumentError("cjs optimizer iterations and gradient tolerance must be positive"))
    isfinite(integral_tol) && integral_tol>0 && isfinite(parameter_tol) && parameter_tol>0 ||
        throw(ArgumentError("cjs quadrature refinement tolerances must be finite and positive"))
    re=prep.individual;q=prep.nphi+prep.np+re.nscale
    theta=start===nothing ? vcat(zeros(prep.nphi+prep.np),fill(log(.5),re.nscale)) : Float64.(collect(start))
    length(theta)==q && all(isfinite,theta) || throw(ArgumentError("cjs start must be finite with one value per fixed coefficient and active individual log SD"))
    attempts=NamedTuple[];refinement=NamedTuple[];previous=nothing;converged=false
    split_integration=_cjs_uses_split_integration(prep)
    integration=split_integration ? :threshold_split_gauss_legendre : :normalized_Gauss_Hermite
    optimizer=:BFGS;order=25
    orders=split_integration ? _CJS_INDIVIDUAL_SPLIT_ORDERS : _cjs_individual_orders(prep)
    for K in orders
        order=K
        rule=split_integration ? nothing : _cjs_individual_rule(K)
        activeprep=split_integration ? merge(prep,(individual=merge(re,
            (order=K,integration=:threshold_split_gauss_legendre)),)) : prep
        fun=x->_cjs_individual_objective(x,activeprep;order=K,rule)
        value=fun(theta);converged=false
        for method in (Optim.BFGS(),Optim.NewtonTrustRegion())
            optimizer=method isa Optim.BFGS ? :BFGS : :NewtonTrustRegion
            try
                result=Optim.optimize(fun,theta,method,
                    Optim.Options(iterations=maxiters,g_tol=g_tol,f_reltol=1e-12);autodiff=:forward)
                candidate=Vector{Float64}(Optim.minimizer(result));v=fun(candidate)
                g=ForwardDiff.gradient(fun,candidate);gnorm=norm(g,Inf)
                success=Optim.converged(result) && isfinite(v) && all(isfinite,g) && gnorm<=max(1e-5,g_tol)
                push!(attempts,(order=K,optimizer=optimizer,start=copy(theta),solution=copy(candidate),
                    objective=v,converged=success,iterations=Optim.iterations(result),gradient_norm=gnorm,
                    reason=split_integration ? "moving-threshold split Gaussian marginal ML optimization" :
                        "fixed-node individual marginal ML optimization"))
                if isfinite(v) && v<=value+1e-12*max(1.,abs(value))
                    theta=candidate;value=v;converged=success
                end
                converged && break
            catch err
                err isa InterruptException && rethrow()
                push!(attempts,(order=K,optimizer=optimizer,start=copy(theta),solution=copy(theta),
                    objective=Inf,converged=false,iterations=0,gradient_norm=Inf,reason=sprint(showerror,err)))
            end
        end
        check=split_integration ? _cjs_individual_split_refinement(theta,previous,prep,K;
            integral_tol,parameter_tol) : _cjs_individual_refinement(theta,previous,prep,K;
            integral_tol,parameter_tol)
        push!(refinement,check)
        check.refined && break
        previous=copy(theta)
    end
    finalprep=merge(prep,(individual=merge(re,(order=order,
        integration=split_integration ? :threshold_split_gauss_legendre : :normalized_Gauss_Hermite)),))
    fun=x->_cjs_individual_objective(x,finalprep)
    value=fun(theta);grad=ForwardDiff.gradient(fun,theta);H=ForwardDiff.hessian(fun,theta)
    eigen=all(isfinite,H) ? eigvals(Symmetric(H)) : fill(NaN,q)
    positive=all(isfinite,eigen) && minimum(eigen)>max(0.,maximum(eigen)*1e-10)
    gnorm=all(isfinite,grad) ? norm(grad,Inf) : Inf
    scales=[begin
        logs=hasproperty(m,:sd_X) ? Vector{Float64}(m.sd_X*theta[m.scale_range]+m.sd_offset) : fill(Float64(theta[m.scale_index]),length(prep.data.ids))
        (process=label,group=m.group,log_sd_coefficients=copy(_cjs_sd_coefficients(m,theta)),
         log_sigma_min=minimum(logs),log_sigma_max=maximum(logs),sigma_min=minimum(exp.(logs)),sigma_max=maximum(exp.(logs)))
    end for (label,m) in ((:phi,re.phi),(:p,re.p)) if m!==nothing]
    ls,lp=_cjs_process_logs(theta,finalprep);d=prep.data;_,T=size(d.Y)
    delta=[d.intervals[mod1(row,T-1)] for row in prep.phi_design.rows]
    variance_boundary=any(x->x.sigma_min<=1e-6,scales)
    boundary=variance_boundary || any(x->x<=1e-8||x>=1-1e-8,vcat(exp.(ls),exp.(ls.*delta),exp.(lp)))
    refined=last(refinement).refined
    reason="Wald and bootstrap intervals are unavailable; profile intervals require the implemented route to pass eligibility and endpoint checks"
    diagnostics=(objective=Float64(value),optimizer=optimizer,method=:ML,converged=converged,
        gradient=grad,gradient_norm=gnorm,hessian=H,curvature=eigen,positive_curvature=positive,
        boundary=boundary,variance_boundary=variance_boundary,attempts=attempts,
        integration,integration_order=order,quadrature_order=order,quadrature_refined=refined,
        integration_status=refined ? :refined : :unrefined_at_cap,refinement=refinement,
        refinement_criterion=(history_log_integral_abs_tol=integral_tol,parameter_abs_tol=parameter_tol),
        individual_scales=scales,route=re.route,terminal_constraint=prep.terminal_constraint,
        enrolled=sum(d.frequency),informative=sum(d.frequency[i] for i in eachindex(d.first) if any(view(d.observed,i,d.first[i]+1:d.end[i]))),
        individual_rows=length(d.ids),contributing_occasions=sum(d.frequency.*(d.end.-d.first)),
        surveyed_occasions=sum(d.frequency[i]*count(view(d.observed,i,d.first[i]+1:d.end[i])) for i in eachindex(d.first)),
        recaptured_histories=sum(d.frequency[i] for i in eachindex(d.ids) if sum(view(d.Y,i,d.first[i]+1:d.end[i]))>0),
        uncertainty_available=false,uncertainty_reason=reason,qualification=:numerical_route_only,
        profile_shape=:unavailable,variance_recovery=:unqualified)
    co=(phi=theta[1:prep.nphi],p=theta[prep.nphi+1:prep.nphi+prep.np],
        sd_phi=re.phi===nothing ? nothing : _cjs_sd_coefficients(re.phi,theta),
        sd_p=re.p===nothing ? nothing : _cjs_sd_coefficients(re.p,theta),
        log_sigma_phi=re.phi===nothing ? nothing : length(_cjs_sd_coefficients(re.phi,theta))==1 ? only(_cjs_sd_coefficients(re.phi,theta)) : nothing,
        log_sigma_p=re.p===nothing ? nothing : length(_cjs_sd_coefficients(re.p,theta))==1 ? only(_cjs_sd_coefficients(re.p,theta)) : nothing)
    return CJSFit(d,theta,co,prep.phi_design,prep.p_design,-Float64(value),fill(NaN,q,q),diagnostics,finalprep)
end
