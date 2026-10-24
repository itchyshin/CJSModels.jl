using Test, CJSModels, StatsModels, ForwardDiff, LinearAlgebra

function _cmr_matrix_bridge_fixture()
    Y=[1 0 0 1; 0 1 1 0; 0 0 0 1]
    observed=trues(3,4); observed[1,2]=false
    first=[1,2,4]; ends=[4,3,4]; intervals=[1.,.5,1.5]; frequency=[2,3,4]
    X_phi=zeros(4,0); X_p=zeros(4,0)
    logit(x)=log(x/(1-x))
    offset_phi=logit.([.8,.6,.7,.9]); offset_p=logit.([.3,.4,.5,.6])
    return (;Y,observed,first,ends,intervals,frequency,X_phi,offset_phi,X_p,offset_p)
end

@testset "CJS matrix bridge objective includes constrained matrix derivatives" begin
    d=_cmr_matrix_bridge_fixture()
    X_phi=hcat(ones(4),[-.7,.4,1.2,.8]); X_p=ones(4,1)
    prep=CJSModels._cjs_matrix_bridge_prepare(d.Y,d.observed,d.first,d.ends,d.intervals,d.frequency,
        X_phi,d.offset_phi,X_p,d.offset_p,"phi",.65)
    beta=[.2,-.3,.1]
    objective=x->CJSModels._cjs_objective(x,prep)
    logistic(x)=1/(1+exp(-x))
    survival=logistic.(X_phi*beta[1:2]+d.offset_phi)
    detection=logistic.(X_p*beta[3:3]+d.offset_p)
    expected=2log(survival[1]*sqrt(survival[2])*(1-detection[2])*.65^1.5*detection[3])+
             3log(sqrt(survival[4])*detection[4])
    @test objective(beta)≈-expected atol=1e-10
    automatic=ForwardDiff.gradient(objective,beta)
    difference(h)=[(objective(beta+[j==k ? h : 0. for j=1:3])-
                    objective(beta-[j==k ? h : 0. for j=1:3]))/(2h) for k=1:3]
    coarse=difference(2e-4); fine=difference(1e-4)
    @test norm(automatic,Inf)>.01
    @test maximum(abs.(coarse-fine))<1e-6
    @test maximum(abs.(automatic-fine))<1e-6
    H=ForwardDiff.hessian(objective,beta)
    @test all(isfinite,H) && isapprox(H,H';atol=1e-10)
    @info "CJS matrix bridge AD vs step-halved FD" max_ad_fd=maximum(abs.(automatic-fine)) max_fd_halving=maximum(abs.(coarse-fine))
end

@testset "CJS matrix bridge optimizer matches native fixed-effect fitting" begin
    Y=[1 0 0;1 0 1;1 1 0;1 1 1]; frequency=[37,15,25,23]
    data=CMRData(Y;ids=1:4,occasions=1:3,units="years",frequency)
    native=cjs(data;maxiters=100)
    bridge=CJSModels.cjs_matrix_bridge(Y,trues(4,3),ones(Int,4),fill(3,4),ones(2),frequency,
        ones(8,1),zeros(8),ones(8,1),zeros(8),"",.5,100,1e-8)
    @test native.diagnostics.converged && bridge.converged
    @test bridge.beta≈native.beta atol=1e-8
    @test bridge.loglik≈native.loglik atol=1e-8
    @test bridge.gradient≈native.diagnostics.gradient atol=1e-6
    @test bridge.hessian≈native.diagnostics.hessian atol=1e-6
end

function _cmr_matrix_bridge_call(d; first=d.first, X_phi=d.X_phi, offset_phi=d.offset_phi,
                                 X_p=d.X_p, offset_p=d.offset_p,
                                 terminal_name="", terminal_value=0.5,
                                 maxiters=100, g_tol=1e-8)
    CJSModels.cjs_matrix_bridge(d.Y,d.observed,first,d.ends,d.intervals,d.frequency,
        X_phi,offset_phi,X_p,offset_p,terminal_name,terminal_value,maxiters,g_tol)
end

@testset "CJS matrix bridge rejects inconsistent capture indices before fitting" begin
    d=_cmr_matrix_bridge_fixture()
    @test_throws ArgumentError _cmr_matrix_bridge_call(d;first=[1,1,4])
    @test_throws ArgumentError _cmr_matrix_bridge_call(d;first=[1,2])
    @test_throws ArgumentError _cmr_matrix_bridge_call(d;first=[1,2,NaN])
end

@testset "CJS matrix bridge validates supplied designs and terminal identification" begin
    d=_cmr_matrix_bridge_fixture()
    for name in ("survival",:phi,missing,2)
        @test_throws ArgumentError _cmr_matrix_bridge_call(d;terminal_name=name)
    end
    for value in (0.,1.,-0.1,NaN,Inf)
        @test_throws ArgumentError _cmr_matrix_bridge_call(d;terminal_name="phi",terminal_value=value)
    end
    @test_throws ArgumentError _cmr_matrix_bridge_call(d;offset_phi=[0.,0.,NaN,0.])
    @test_throws ArgumentError _cmr_matrix_bridge_call(d;offset_p=[0.,0.,0.,Inf])
    @test_throws ArgumentError _cmr_matrix_bridge_call(d;X_phi=zeros(3,0))
    @test_throws ArgumentError _cmr_matrix_bridge_call(d;X_p=zeros(3,0))
    @test_throws ArgumentError _cmr_matrix_bridge_call(d;offset_phi=zeros(3))
    @test_throws ArgumentError _cmr_matrix_bridge_call(d;offset_p=zeros(3))
    @test_throws ArgumentError _cmr_matrix_bridge_call(d;X_phi=zeros(4,1),maxiters=2)
    # A detection column confined to the unsurveyed slot cannot be estimated.
    @test_throws ArgumentError _cmr_matrix_bridge_call(d;X_p=reshape([1.,0.,0.,0.],4,1),maxiters=2)
    final_direction=[0.,0.,1.,0.]
    @test_throws ArgumentError _cmr_matrix_bridge_call(d;
        X_phi=hcat(ones(4),final_direction),X_p=hcat(ones(4),final_direction),maxiters=2)
    # Caller coefficient order is preserved: flat constrained columns must be
    # removed in R, rather than silently retained or re-pivoted in Julia.
    @test_throws ArgumentError _cmr_matrix_bridge_call(d;
        X_phi=reshape(final_direction,4,1),terminal_name="phi",terminal_value=.7,maxiters=2)
    for maxiters in (0,-1,1.5,Inf,true)
        @test_throws ArgumentError _cmr_matrix_bridge_call(d;maxiters=maxiters)
    end
    for tolerance in (0.,-1.,NaN,Inf,true)
        @test_throws ArgumentError _cmr_matrix_bridge_call(d;g_tol=tolerance)
    end
end

@testset "CJS matrix bridge zero columns retain exact row and interval semantics" begin
    d=_cmr_matrix_bridge_fixture()
    out=_cmr_matrix_bridge_call(d)
    expected=2log(.8*sqrt(.6)*.6*.7^1.5*.5)+3log(sqrt(.9)*.6)
    @test out.loglik≈expected atol=1e-10
    @test out.objective≈-expected atol=1e-10
    @test out.beta==Float64[] && out.gradient==Float64[]
    @test size(out.hessian)==(0,0) && out.converged && out.iterations==0
    @test hasproperty(out,:row_layout)
    if hasproperty(out,:row_layout)
        @test out.row_layout.phi_rows==[1,2,3,5]
        @test out.row_layout.p_rows==[2,3,4,7]
        @test out.row_layout.p_surveyed==Bool[0,1,1,1]
    end
    fixed_phi=_cmr_matrix_bridge_call(d;terminal_name="phi",terminal_value=.65)
    fixed_p=_cmr_matrix_bridge_call(d;terminal_name="p",terminal_value=.25)
    @test fixed_phi.loglik≈2log(.8*sqrt(.6)*.6*.65^1.5*.5)+3log(sqrt(.9)*.6) atol=1e-10
    @test fixed_p.loglik≈2log(.8*sqrt(.6)*.6*.7^1.5*.25)+3log(sqrt(.9)*.6) atol=1e-10
    # Detection offset at the masked occasion has no likelihood contribution.
    @test _cmr_matrix_bridge_call(d;offset_p=[1000.;d.offset_p[2:end]]).loglik≈out.loglik atol=1e-10
    @test _cmr_matrix_bridge_call(d;terminal_name=nothing).loglik≈out.loglik atol=1e-10
    for (X_phi,X_p) in ((zeros(4,0),ones(4,1)),(ones(4,1),zeros(4,0)))
        prep=CJSModels._cjs_matrix_bridge_prepare(d.Y,d.observed,d.first,d.ends,d.intervals,d.frequency,
            X_phi,d.offset_phi,X_p,d.offset_p,"",.5)
        @test prep.nphi+prep.np==1
        @test CJSModels._cjs_objective([0.],prep)≈-expected atol=1e-10
    end
    final=CJSModels.cjs_matrix_bridge(reshape([0,0,1],1,3),trues(1,3),[3],[3],[1.,1.],[5],
        zeros(0,0),Float64[],zeros(0,0),Float64[],"",.5,1,1e-8)
    @test final.objective==0 && final.loglik==0 && final.beta==Float64[]
    @test isempty(final.row_layout.phi_rows) && isempty(final.row_layout.p_rows)
end

@testset "CJS matrix bridge survival rank ignores post-survey suffix" begin
    Y=reshape([1,0,0,0],1,:); observed=reshape([true,true,false,false],1,:)
    # Matrix inputs retain all positional rows, but suffix-only columns cannot
    # pass the rank gate as if the unsurveyed tail informed survival.
    @test_throws ArgumentError CJSModels.cjs_matrix_bridge(Y,observed,[1],[4],[1.,1.,1.],[1],
        hcat(ones(3),[0.,0.,1.]),zeros(3),ones(3,1),zeros(3),"",.5,5,1e-8)
end
