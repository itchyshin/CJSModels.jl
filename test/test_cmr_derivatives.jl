using Test, CJSModels, ForwardDiff, StatsModels
@testset "CJS interior derivatives at non-optimal parameters" begin
    d=CMRData([1 0 0 1;1 1 0 0;0 1 1 0];ids=1:3,occasions=[0.,.5,2.,4.],units="years",frequency=[3,1,2])
    prep=CJSModels._cjs_prepare(d,@formula(phi~1),@formula(p~1))
    fun=b->CJSModels._cjs_objective(b,prep)
    b=[.3,-.7]; ad=ForwardDiff.gradient(fun,b)
    @test maximum(abs.(ad)) > .01
    fd(h)=[(fun(b+[j==k ? h : 0. for j=1:2])-fun(b-[j==k ? h : 0. for j=1:2]))/(2h) for k=1:2]
    @test maximum(abs.(fd(1e-4)-fd(2e-5))) < 1e-6
    @test maximum(abs.(ad-fd(2e-5))) < 1e-6
    @test all(isfinite,ForwardDiff.hessian(fun,b))
    final=CMRData([0 0 1];ids=[1],occasions=1:3,units="years")
    z=CJSModels._cjs_prepare(final,@formula(phi~1),@formula(p~1))
    @test CJSModels._cjs_objective(b,z)==0.
    @test ForwardDiff.gradient(x->CJSModels._cjs_objective(x,z),b)==[0.,0.]
end

@testset "CJS extreme finite links retain complements" begin
    d=CMRData([1 0 0];ids=[1],occasions=1:3,units="years")
    prep=CJSModels._cjs_prepare(d,@formula(phi~1),@formula(p~1))
    @test isfinite(CJSModels._cjs_objective([1000.,1000.],prep))
    @test CJSModels._cjs_objective([1000.,1000.],prep) ≈ 1000. atol=1e-10
    @test all(isfinite,ForwardDiff.gradient(b->CJSModels._cjs_objective(b,prep),[1000.,1000.]))
end

@testset "CJS two rare alternatives at extreme links" begin
    d=CMRData([1 0];ids=[1],occasions=1:2,units="years",phi_data=(id=[1],interval_start=[1],interval_end=[2],off=[1000.]),p_data=(id=[1],occasion=[2],off=[1000.]))
    prep=CJSModels._cjs_prepare(d,@formula(phi~0+offset(off)),@formula(p~0+offset(off)))
    @test CJSModels._cjs_objective(Float64[],prep) ≈ 1000-log(2) atol=1e-10
end
