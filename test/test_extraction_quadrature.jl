using Test, CJSModels, LinearAlgebra
@testset "Standalone Gaussian quadrature dependency" begin
    for k in (25, 41)
        x,w=CJSModels._gauss_hermite(k)
        u=sqrt(2).*x; weights=w./sum(w)
        @test sum(weights) ≈ 1.0 atol=1e-14
        @test sum(weights.*u) ≈ 0.0 atol=1e-14
        @test sum(weights.*u.^2) ≈ 1.0 atol=1e-13
        @test sum(weights.*u.^4) ≈ 3.0 atol=1e-12
    end
end
