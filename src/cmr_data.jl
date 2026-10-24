"""
    CMRData(Y; ids, occasions, units, observed=nothing, intervals=nothing,
            ends=nothing, frequency=nothing, phi_data=nothing, p_data=nothing)

Validated first-capture-conditional capture histories. Rows are individuals;
columns are occasions. `ends` is administrative follow-up, never last sighting.
`observed=false` omits an emission while retaining survival transitions.
`frequency` contains positive integer replication counts, not arbitrary weights.

Predictor tables are keyed by `id, interval_start, interval_end` (survival) and
`id, occasion` (detection). Survival requires every post-capture interval; detection requires only surveyed
post-capture occasions. Earlier and unsurveyed detection rows are optional.
Input row order is ignored.
`intervals` uses the declared `units`; otherwise numeric occasion differences
are used. Missing encounters require an explicit false observation mask.
"""
struct CMRData{I,O,S,P}
    Y::Matrix{Int}
    observed::BitMatrix
    ids::Vector{I}
    occasions::Vector{O}
    intervals::Vector{Float64}
    first::Vector{Int}
    var"end"::Vector{Int}
    frequency::Vector{Int}
    units::String
    phi_data::S
    p_data::P
end

function _cmr_keyed_table(table, defaults, keynames, required)
    table === nothing && return defaults
    cols=Tables.columntable(table)
    all(k->haskey(cols,k),keynames) || throw(ArgumentError("CMR predictor table requires keys $(join(keynames, ", "))"))
    n=length(first(values(cols)))
    all(v->length(v)==n,values(cols)) || throw(ArgumentError("CMR predictor columns have unequal lengths"))
    index=Dict{Tuple,Int}()
    for j in 1:n
        key=Tuple(getproperty(cols,k)[j] for k in keynames)
        any(ismissing,key) && throw(ArgumentError("CMR predictor keys cannot be missing"))
        haskey(index,key) && throw(ArgumentError("CMR predictor keys must be unique"))
        index[key]=j
    end
    expected=[Tuple(getproperty(defaults,k)[j] for k in keynames) for j in eachindex(defaults.id)]
    all(k->k in expected,keys(index)) || throw(ArgumentError("CMR predictor table has unknown keys"))
    all(j->haskey(index,expected[j]),required) || throw(ArgumentError("CMR predictor table lacks required post-capture keys through administrative end"))
    order=[get(index,k,0) for k in expected]
    return NamedTuple{keys(cols)}(Tuple(k in keynames ? getproperty(defaults,k) : [j==0 ? missing : getproperty(cols,k)[j] for j in order] for k in keys(cols)))
end

function CMRData(Y::AbstractMatrix; ids, occasions, units,
                 observed=nothing, intervals=nothing, ends=nothing,
                 frequency=nothing, phi_data=nothing, p_data=nothing)
    N,T=size(Y)
    N>0 && T>=1 || throw(ArgumentError("CMRData requires individuals and at least one occasion"))
    ids=collect(ids); occasions=collect(occasions)
    length(ids)==N && !any(ismissing,ids) && length(unique(ids))==N || throw(ArgumentError("CMR IDs must be explicit, unique, nonmissing and match history rows"))
    length(occasions)==T && !any(ismissing,occasions) || throw(ArgumentError("CMR occasions must match history columns and be nonmissing"))
    all(occasions[t]<occasions[t+1] for t=1:T-1) || throw(ArgumentError("CMR occasions must be strictly ordered"))
    units isa AbstractString && !isempty(strip(units)) || throw(ArgumentError("CMR interval units must be declared"))
    delta=intervals===nothing ? try Float64.(diff(occasions)) catch; throw(ArgumentError("Supply numeric intervals in declared units for these occasions")); end : Float64.(collect(intervals))
    length(delta)==T-1 && all(x->isfinite(x)&&x>0,delta) || throw(ArgumentError("CMR intervals must be finite, positive and have length T-1"))
    if observed===nothing
        any(ismissing,Y) && throw(ArgumentError("Missing encounters require an explicit observed=false mask"))
        mask=trues(N,T)
    else
        size(observed)==size(Y) && all(x->x isa Bool,observed) || throw(ArgumentError("CMR observed must be a Boolean N by T matrix"))
        mask=BitMatrix(observed)
    end
    yy=zeros(Int,N,T)
    for j in eachindex(Y)
        if ismissing(Y[j])
            !mask[j] || throw(ArgumentError("Missing encounter is marked observed"))
        else
            Y[j] in (0,1) || throw(ArgumentError("CMR encounters must be zero or one"))
            yy[j]=Int(Y[j])
            yy[j]==1 && !mask[j] && throw(ArgumentError("A capture cannot be marked unobserved"))
        end
    end
    firsts=[findfirst(==(1),view(yy,i,:)) for i=1:N]
    any(isnothing,firsts) && throw(ArgumentError("CJS excludes all-zero histories: first observed capture is required"))
    ends=ends===nothing ? fill(T,N) : collect(ends)
    length(ends)==N && all(x->x isa Integer,ends) || throw(ArgumentError("CMR end must contain one integer occasion index per history"))
    ends=Int.(ends); ff=Int.(firsts)
    all(ff[i]<=ends[i]<=T for i=1:N) || throw(ArgumentError("Administrative end must be at or after first capture and no later than T"))
    all(!any(==(1),view(yy,i,ends[i]+1:T)) for i=1:N) || throw(ArgumentError("Capture occurs after administrative end"))
    freq=frequency===nothing ? ones(Int,N) : collect(frequency)
    length(freq)==N && all(x->x isa Real && isfinite(x) && x>=1 && isinteger(x),freq) || throw(ArgumentError("CMR frequency must contain positive integer replication counts"))
    freq=Int.(freq)
    pd=(id=repeat(ids,inner=T),occasion=repeat(occasions,N))
    sd=(id=repeat(ids,inner=T-1),interval_start=repeat(occasions[1:T-1],N),interval_end=repeat(occasions[2:T],N))
    sr=[(i-1)*(T-1)+t for i=1:N for t=ff[i]:ends[i]-1]
    pr=[(i-1)*T+t for i=1:N for t=ff[i]+1:ends[i] if mask[i,t]]
    return CMRData(yy,mask,ids,occasions,delta,ff,ends,freq,String(units),
        _cmr_keyed_table(phi_data,sd,(:id,:interval_start,:interval_end),sr),
        _cmr_keyed_table(p_data,pd,(:id,:occasion),pr))
end
