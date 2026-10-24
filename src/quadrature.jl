function _gauss_hermite(K::Int)
    β = [sqrt(k / 2) for k in 1:(K-1)]
    E = eigen(SymTridiagonal(zeros(K), β))
    return E.values, sqrt(π) .* (E.vectors[1, :]) .^ 2
end
