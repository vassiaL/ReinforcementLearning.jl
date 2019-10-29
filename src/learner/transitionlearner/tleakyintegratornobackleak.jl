"""
TLeakyIntegrator without leaky transitions in the background (non-visited ones)
To be used alone as passive learners with random policy
or as Testimate parameter of SmallBackups
"""
struct TLeakyIntegratorNoBackLeak <: TNs1a0s0
    ns::Int
    na::Int
    etaleak::Float64
    lowerbound::Float64
    Nsa::Array{Float64, 2}
    Ns1a0s0::Array{Dict{Tuple{Int, Int}, Float64}, 1}
    terminalstates::Array{Int,1}
end
function TLeakyIntegratorNoBackLeak(; ns = 10, na = 4, etaleak = .9,
                                        lowerbound = eps())
    Nsa = zeros(na, ns) .+ ns*eps()
    Ns1a0s0 = [Dict{Tuple{Int, Int}, Float64}() for _ in 1:ns]
    [Ns1a0s0[sprime][(a, s)] = eps() for sprime in 1:ns for a in 1:na for s in 1:ns]
    TLeakyIntegratorNoBackLeak(ns, na, etaleak, lowerbound, Nsa, Ns1a0s0, Int[])
end
export TLeakyIntegratorNoBackLeak
""" X[t] = etaleak * X[t-1] + etaleak * I[.] """
function updatet!(learnerT::Union{TLeakyIntegratorNoBackLeak, TLeakyIntegratorJump}, s0, a0, s1, done)
    leaka0s0!(learnerT, s0, a0, s1, done)
    computeterminalNs1a0s0!(learnerT, s1, done)
end
