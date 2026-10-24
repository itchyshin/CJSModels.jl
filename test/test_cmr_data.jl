using Test, CJSModels
@testset "CMR data API exists" begin
    @test isdefined(CJSModels, :CMRData)
end
@testset "CMR data validation and keyed alignment" begin
    mk(y; kw...) = CMRData(y; ids=["a", "b"], occasions=[0., 1., 3.], units="years", kw...)
    d=mk([1 0 0; 0 0 1])
    @test d.first == [1,3]
    @test d.end == [3,3]
    @test d.intervals == [1.,2.]
    @test d.frequency == [1,1]
    @test_throws ArgumentError mk([0 0 0; 0 0 1])
    @test_throws ArgumentError mk([1 2 0; 0 0 1])
    @test_throws ArgumentError mk([1 0 0; 0 0 1]; observed=Bool[0 1 1;1 1 1])
    @test_throws ArgumentError mk([1 0 0; 0 0 1]; ends=[0,3])
    @test_throws ArgumentError mk([1 0 0; 0 0 1]; frequency=[1.,1.5])
    @test_throws ArgumentError CMRData([1 0;1 0]; ids=["a","a"], occasions=[1,2], units="years")
    @test_throws ArgumentError CMRData([1 0]; ids=["a"], occasions=[2,1], units="years")
    m=CMRData(reshape([1,missing,0],1,:); ids=[1], occasions=1:3, units="years", observed=reshape([true,false,true],1,:))
    @test m.Y == [1 0 0]
    @test_throws ArgumentError CMRData(reshape([1,missing,0],1,:); ids=[1], occasions=1:3, units="years")
    pd=(id=["b","a","b","a","b","a"], occasion=[3.,3.,1.,1.,0.,0.], x=[6.,3.,5.,2.,4.,1.])
    d=mk([1 0 0;0 0 1];p_data=pd)
    @test d.p_data.x == [1.,2.,3.,4.,5.,6.]
    @test_throws ArgumentError mk([1 0 0;0 0 1];p_data=(id=["a"],occasion=[0.],x=[1.]))
end

@testset "CMR requires post-capture predictors only" begin
    d=CMRData([0 1 0];ids=[1],occasions=1:3,units="years",p_data=(id=[1],occasion=[3],x=[.5]),phi_data=(id=[1],interval_start=[2],interval_end=[3],x=[.2]))
    @test d.p_data.x[3] == .5
    @test ismissing(d.p_data.x[1])
end

@testset "Single-occasion CMR retains final-first" begin
    d=CMRData(reshape([1,1],2,1);ids=[1,2],occasions=[1],units="years")
    @test d.first == d.end == [1,1]
    @test isempty(d.intervals)
    @test_throws ArgumentError cjs(d)
end

@testset "Unsurveyed detection predictor keys are optional" begin
    y=[1 0 1 0]; obs=reshape([true,false,true,true],1,:)
    partial=(id=[1,1],occasion=[3,4],x=[.4,.8])
    d=CMRData(y;ids=[1],occasions=1:4,units="years",observed=obs,p_data=partial)
    @test ismissing(d.p_data.x[2])
    full=CMRData(y;ids=[1],occasions=1:4,units="years",observed=obs,p_data=(id=[1,1,1,1],occasion=[1,2,3,4],x=[missing,missing,.4,.8]))
    @test full.Y==d.Y
end
