using Distributions, SpecialFunctions, Roots, LinearAlgebra
# import SpecialFunctions: lbeta
"""
VarSMiLe (TSmileextended)
To be used alone as passive learners with random policy
or as Testimate parameter of SmallBackups
"""
struct TVarSmile <: TPs1a0s0
    ns::Int
    na::Int
    m::Float64
    stochasticity::Float64
    pcprime::Float64
    Ps1a0s0::Array{Dict{Tuple{Int, Int}, Float64}, 1}
    alphas::Array{Array{Float64,1}, 2}
    terminalstates::Array{Int,1}
end
function TVarSmile(;ns = 10, na = 4, m = .1, stochasticity = .01)
    pcprime = m/(1. + m)
    Ps1a0s0 = [Dict{Tuple{Int, Int}, Float64}() for _ in 1:ns]
    [Ps1a0s0[sprime][(a, s)] = 1. /ns for sprime in 1:ns for a in 1:na for s in 1:ns]
    alphas = Array{Array{Float64,1}}(undef, na, ns)
    [alphas[a0, s0] = stochasticity .* ones(ns) for a0 in 1:na for s0 in 1:ns]
    TVarSmile(ns, na, m, stochasticity, pcprime, Ps1a0s0, alphas, Int[])
end
export TVarSmile
function updatet!(learnerT::TVarSmile, s0, a0, s1, done)
    alpha0 = learnerT.stochasticity .* ones(learnerT.ns)
    Sgm = calcSgm(learnerT.alphas[a0, s0], alpha0, s1)
    # @show Sgm
    γ0 = learnerT.m * Sgm/(1. + learnerT.m * Sgm)
    betas = zeros(learnerT.ns)
    betas[s1] += 1.
    @. learnerT.alphas[a0, s0] = (1. - γ0) .* learnerT.alphas[a0, s0] +
                                        γ0 .* alpha0 .+ betas
    computePs1a0s0!(learnerT, s0, a0)
    computeterminalPs1a0s0!(learnerT, s1, done)
    leakothers!(learnerT, s0, a0)
end
export updatet!
function calcSgm(αn, α0, s1)
    p0 = α0[s1] / sum(α0)
    pn = αn[s1] / sum(αn)
    p0 / pn
end
function computePs1a0s0!(learnerT::Union{TSmile, TVarSmile}, s0, a0)
    for s in 1:learnerT.ns
        learnerT.Ps1a0s0[s][(a0, s0)] = learnerT.alphas[a0, s0][s] / sum(learnerT.alphas[a0, s0])
    end
end
function leakothers!(learnerT::TVarSmile, s0, a0)
    pairs = getactionstatepairs!(learnerT, s0, a0)
    for sa in pairs # sa[1] = action, sa[2] = state
        # @show sa
        # @show learnerT.alphas[sa[1], sa[2]]
        if !in(sa[2], learnerT.terminalstates) # Dont leak outgoing transitions of terminalstates
            if !all(learnerT.alphas[sa[1], sa[2]] .== learnerT.stochasticity)
            #if !all(@. all(learnerT.alphas[sa[1], sa[2], :] == [learnerT.stochasticity .* ones(learnerT.ns)]))
                learnerT.alphas[sa[1], sa[2]] = learnerT.pcprime * learnerT.stochasticity .* ones(learnerT.ns) + # Do not move the "+" to the line below!!!
                                    (1. - learnerT.pcprime) .* learnerT.alphas[sa[1], sa[2]]
                computePs1a0s0!(learnerT, sa[2], sa[1])
                # @show sa, learnerT.alphas[sa[1], sa[2]]
            end
        end
    end
end
