using Distributions, SpecialFunctions, Random, Roots, LinearAlgebra
# import SpecialFunctions: lbeta
"""
TSmile
To be used alone as passive learners with random policy
or as Testimate parameter of SmallBackups
"""
struct TSmile
    ns::Int
    na::Int
    m::Float64
    stochasticity::Float64
    Ps1a0s0::Array{Dict{Tuple{Int, Int}, Float64}, 1}
    alphas::Array{Array{Float64,1}, 2}
end
function TSmile(;ns = 10, na = 4, m = .1, stochasticity = .01)
    Ps1a0s0 = [Dict{Tuple{Int, Int}, Float64}() for _ in 1:ns]
    alphas = Array{Array{Float64,1}}(undef, na, ns)
    [alphas[a0, s0] = stochasticity .* ones(ns) for a0 in 1:na for s0 in 1:ns]
    TSmile(ns, na, m, stochasticity, Ps1a0s0, alphas)
end
export TSmile
function updatet!(learnerT::TSmile, s0, a0, s1)
    betas = learnerT.stochasticity .* ones(learnerT.ns)
    betas[s1] += 1.
    #@show learnerT.m
    Scc = KL(learnerT.alphas[a0, s0], betas)
    Bmax = KL(betas, learnerT.alphas[a0, s0])
    B = learnerT.m * Scc/(1. + learnerT.m * Scc) * Bmax
    γ0 = find_γ0(betas, learnerT.alphas[a0, s0], B)
    @. learnerT.alphas[a0, s0] = (1. - γ0) .* learnerT.alphas[a0, s0] + γ0 .* betas

    computePs1a0s0!(learnerT, s0, a0)
end
export updatet!
function KL(α1, α2)
    max(lbeta(α2) - lbeta(α1) + dot(α1 .- α2, digamma.(α1) .- digamma(sum(α1))), 0.)
end
function lbeta(α::Array{Float64,1})
    α0 = 0.
    lmnB = 0.
    for i in 1:length(α)
        αi = α[i]
        α0 += αi
        lmnB += lgamma(αi)
    end
    lmnB -= lgamma(α0)
    lmnB
end
function computePs1a0s0!(learnerT::TSmile, s0, a0)
    for s in 1:learnerT.ns
        learnerT.Ps1a0s0[s][(a0, s0)] = learnerT.alphas[a0, s0][s] / sum(learnerT.alphas[a0, s0])
    end
end
function find_γ0(betas, alphas, B)
    f = γ -> KL(γ .* betas .+ (1. - γ) .* alphas, alphas) - B
    # γ0 = find_zero(f, (0., 1.))
    if abs(f(0.)) < 5*eps()
        γ0 = 0.
        # println("HERE WE ARE! 0!")
    elseif abs(f(1.)) < 5*eps()
        γ0 = 1.
        # println("HERE WE ARE! 1!")
    else
        γ0 = find_zero(f, (0., 1.))
    end
    γ0
end
