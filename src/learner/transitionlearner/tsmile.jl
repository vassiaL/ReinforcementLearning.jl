using Distributions, SpecialFunctions, Random, Roots
"""
TEstimateSmile
To be used alone as passive learners with random policy
or as Testimate parameter of SmallBackups
"""
struct TEstimateSmile
    ns::Int
    na::Int
    m::Float64
    stochasticity::Float64
    Ps1a0s0::Array{Dict{Tuple{Int, Int}, Float64}, 1}
    alphas::Array{Array{Float64,1}, 2}
    seed::Any
    rng::MersenneTwister
end
function TEstimateSmile(;ns = 10, na = 4, m = .1, stochasticity = .01, seed = 3)
    Ps1a0s0 = [Dict{Tuple{Int, Int}, Float64}() for _ in 1:ns]
    alphas = Array{Array{Float64,1}}(undef, na, ns)
    [alphas[a0, s0] = stochasticity .* ones(ns) for a0 in 1:na for s0 in 1:ns]
    rng = MersenneTwister(seed)
    TEstimateSmile(ns, na, m, stochasticity, Ps1a0s0, alphas, m, seed, rng)
end
export TEstimateSmile
function updatet!(learnerT::TEstimateSmile, s0, a0, s1)

    betas = learnerT.stochasticity .* ones(learnerT.ns)
    betas[s1] += 1

    Scc = KL(learnerT.alphas[a0, s0], b)
    Bmax = KL(b, learnerT.alphas[a0, s0])
    B = learnerT.m * Scc/(1. + learnerT.m * Scc) * Bmax
    f = γ -> KL(γ .* betas .+ (1 - γ) .* learnerT.alphas[a0, s0], learnerT.alphas[a0, s0]) - B
    γ0 = find_zero(f, (0, 1))
    @. learnerT.alphas[a0, s0] = (1 - γ0) * learnerT.alphas[a0, s0] + γ0 * betas
    #
    # if !haskey(learnerT.Ps1a0s0[s1], (a0, s0))
    #     for s in 1:learnerT.ns
    #         learnerT.Ps1a0s0[s][(a0, s0)] = 1. /learnerT.ns
    #     end
    # end
    computePs1a0s0!(learnerT, s0, a0)
end
export updatet!
function KL(α1, α2)
    lbeta(α2) - lbeta(α1) + dot(α1 .- α2, digamma.(α1) .- digamma(sum(α1)))
end
function lbeta(α::Vector)
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
function computePs1a0s0!(learnerT::TEstimateSmile, s0, a0)
    for s in 1:learnerT.ns
        learnerT.Ps1a0s0[s][(a0, s0)] = learnerT.alphas[a0, s0][s] / sum(learnerT.alphas[a0, s0])
        expectedvaluethetas[s]
    end
end
