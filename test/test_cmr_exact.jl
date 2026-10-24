using Test, CJSModels, StatsModels, LinearAlgebra
@testset "Exact CJS production kernel" begin
    @test isdefined(CJSModels, :_cjs_history_loglik)
    n=0
    for line in Iterators.drop(eachline(joinpath(@__DIR__,"../docs/dev-log/cmr/fixtures/exact.tsv")),1)
        h,f,ph,pp,prob=split(line,'\t')
        y=[c=='1' ? 1 : 0 for c in h]; obs=[c!='?' for c in h]
        phi=parse.(Float64,split(ph,',')); p=parse.(Float64,split(pp,','))
        actual=CJSModels._cjs_history_loglik(y,obs,parse(Int,f),length(y),phi,p)
        @test exp(actual) ≈ parse(Float64,prob) atol=1e-10 rtol=1e-10
        n+=1
    end
    @test n == 719
    # Administrative suffix differs from last sighting; no first-capture emission.
    ll=CJSModels._cjs_history_loglik([1,0,0],trues(3),1,3,[.8,.6],[.1,.3,.4])
    @test exp(ll) ≈ .6256
    @test ll != log(.1*.6256)
    @test ll != CJSModels._cjs_history_loglik([1,0,0],trues(3),1,1,[.8,.6],[.1,.3,.4])
    @test exp(ll) != (1-.8*.3)*(1-.6*.4)
    @test CJSModels._cjs_history_loglik([1,1],trues(2),1,2,[0.],[.2,.5]) == -Inf
    @test CJSModels._cjs_history_loglik([0,0,1],trues(3),3,3,[0.,0.],[0.,0.,0.]) == 0.
    d=CMRData([1 0 0;0 0 1];ids=[1,2],occasions=1:3,units="years",frequency=[3,2])
    @test CJSModels._cjs_loglik(d,[.8 .6;.8 .6],[.1 .3 .4;.1 .3 .4]) ≈ 3ll
end
@testset "Fixed effects RHS-only designs and fit diagnostics" begin
    y=[1 0 0 0;1 0 1 0;1 1 0 0;1 1 1 0;1 0 0 1;1 0 1 1;1 1 0 1;1 1 1 1]
    d=CMRData(y;ids=1:8,occasions=1:4,units="years",frequency=[12,4,8,4,4,4,4,4])
    f=cjs(d;phi=@formula(phi~1),p=@formula(p~1))
    @test f isa CJSFit
    @test isfinite(f.loglik)
    @test norm(f.diagnostics.gradient,Inf) < 1e-5
    @test f.diagnostics.enrolled == 44
    @test f.diagnostics.informative == 44
    @test f.phi_design.names == ["(Intercept)"]
    @test !haskey(f.data.phi_data,:phi)
    @test size(f.vcov) == (2,2)
    @test_throws ArgumentError cjs(d;sd_phi=@formula(sd_phi~1))
    @test_throws ArgumentError cjs(d;selection=(eligibility=1,))
    @test CJSModels._cjs_is_individual(CJSModels._cjs_prepare(d,@formula(phi~1+(1|id)),@formula(p~1)))
end

@testset "CMR labelled RHS, offsets and terminal constraints" begin
    y=[1 0 0 0;1 0 1 0;1 1 0 0;1 1 1 0;1 0 0 1;1 0 1 1;1 1 0 1;1 1 1 1]
    ids=collect(1:8)
    sd=(id=repeat(ids,inner=3),interval_start=repeat(1:3,8),interval_end=repeat(2:4,8),year=repeat(["a","b","c"],8),x=repeat([-.5,0.,.5],8),off=fill(.2,24))
    pd=(id=repeat(ids,inner=4),occasion=repeat(1:4,8),year=repeat(["a","b","c","d"],8),x=repeat([-1.,-.5,0.,.5],8))
    d=CMRData(y;ids=ids,occasions=1:4,units="years",phi_data=sd,p_data=pd)
    des=CJSModels._cjs_prepare(d,@formula(phi~x+offset(off)),@formula(p~1))
    @test des.phi_design.names == ["(Intercept)","x"]
    @test des.phi_design.offset == fill(.2,24)
    @test_throws ArgumentError CJSModels._cjs_prepare(d,@formula(phi~year),@formula(p~year))
    free=CJSModels._cjs_prepare(d,@formula(phi~year),@formula(p~1))
    @test free.nphi == 3
    constrained=CJSModels._cjs_prepare(d,@formula(phi~year),@formula(p~year);terminal_constraint=(p=.5,))
    @test constrained.np == 2
    @test rank(constrained.p_design.X[.!constrained.p_terminal,:])==constrained.np
    @test_throws ArgumentError CJSModels._cjs_prepare(d,@formula(phi~1),@formula(p~1);terminal_constraint=(p=1.,))
    @test_throws ArgumentError CJSModels._cjs_prepare(d,@formula(phi~1),@formula(p~1);terminal_constraint=(p=.5,phi=.8))
    # Probability constraint fixes unit-time survival before interval power.
    di=CMRData([1 1 1];ids=[1],occasions=[0.,1.,3.],units="years")
    pi=CJSModels._cjs_prepare(di,@formula(phi~1),@formula(p~1);terminal_constraint=(phi=.8,))
    @test exp(-CJSModels._cjs_objective([0.,0.],pi)) ≈ .5*.5*.8^2*.5
end

@testset "CMR fully fixed offset model has no free coefficients" begin
    d=CMRData([1 1 0];ids=[1],occasions=1:3,units="years",phi_data=(id=[1,1],interval_start=[1,2],interval_end=[2,3],off=[0.,0.]),p_data=(id=[1,1,1],occasion=[1,2,3],off=[0.,0.,0.]))
    f=cjs(d;phi=@formula(phi~0+offset(off)),p=@formula(p~0+offset(off)))
    @test isempty(f.beta)
    @test f.loglik ≈ log(.1875)
    @test f.diagnostics.converged
end

@testset "Detection design uses surveyed contributing rows only" begin
    y=[1 0 1 0]; obs=reshape([true,false,true,true],1,:)
    d=CMRData(y;ids=[1],occasions=1:4,units="years",observed=obs,p_data=(id=[1,1],occasion=[3,4],x=[.4,.8]))
    dummy=CMRData(y;ids=[1],occasions=1:4,units="years",observed=obs,p_data=(id=[1,1,1,1],occasion=1:4,x=[100.,999.,.4,.8]))
    a=CJSModels._cjs_prepare(d,@formula(phi~1),@formula(p~x))
    b=CJSModels._cjs_prepare(dummy,@formula(phi~1),@formula(p~x))
    beta=[.2,-.3,.4]
    @test a.p_design.X==b.p_design.X
    @test a.p_design.X[1,:]==[0.,0.]
    @test a.p_design.rows==[2,3,4]
    @test a.p_design.training_rows==[3,4]
    @test CJSModels._cjs_objective(beta,a) ≈ CJSModels._cjs_objective(beta,b) atol=1e-12
    categorical=CMRData(y;ids=[1],occasions=1:4,units="years",observed=obs,p_data=(id=fill(1,4),occasion=1:4,category=["pre","unsurveyed","a","b"]))
    pc=CJSModels._cjs_prepare(categorical,@formula(phi~1),@formula(p~category))
    @test length(pc.p_design.names)==2
    @test !any(n->occursin("unsurveyed",n)||occursin("pre",n),pc.p_design.names)
    allmasked=CMRData([1 0 0];ids=[1],occasions=1:3,units="years",observed=reshape([true,false,false],1,:))
    @test_throws ArgumentError cjs(allmasked)
end

@testset "CJS survival design rank uses only survey-informative intervals" begin
    y=reshape([1,0,0,0],1,:)
    observed=reshape([true,true,false,false],1,:)
    # The predictor varies only after the final survey, so it contributes no
    # information even though its full-history design matrix has rank two.
    d=CMRData(y;ids=[1],occasions=1:4,units="years",observed,
        phi_data=(id=[1,1,1],interval_start=1:3,interval_end=2:4,x=[0.,0.,1.]))
    @test_throws ArgumentError CJSModels._cjs_prepare(d,@formula(phi~1+x),@formula(p~1))
    # Selection still needs the pre-capture rows even though the post-capture
    # schema/rank check excludes intervals with no later surveyed occasion.
    selected=CMRData(reshape([0,1,0,0],1,:);ids=[1],occasions=1:4,units="years",
        observed=trues(1,4),phi_data=(id=fill(1,3),interval_start=1:3,interval_end=2:4,x=[0.,0.,1.]))
    prep=CJSModels._cjs_prepare(selected,@formula(phi~1+x),@formula(p~1))
    @test size(prep.phi_design.X,1)==2
    sel=CJSModels._cjs_selection_prepare(prep,(eligibility=[1],assumption=:alive_available))
    @test sel.selection.phi_rows==[1]
end
