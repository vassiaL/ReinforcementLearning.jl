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
function leaka0s0!(learnerT::Union{TLeakyIntegratorNoBackLeak,
                    TLeakyIntegratorJump},
                    s0, a0, s1, done)
    learnerT.Nsa[a0, s0] *= learnerT.etaleak # Discount transition
    learnerT.Nsa[a0, s0] += learnerT.etaleak # Increase observed transition
    learnerT.Nsa[a0, s0] += (1. - learnerT.etaleak) * learnerT.ns * learnerT.lowerbound # Bound it
    # @show a0, s0, s1, done learnerT.Nsa[a0, s0]
    nextstates = [s for s in 1:learnerT.ns if haskey(learnerT.Ns1a0s0[s],(a0,s0))]
    for sprime in nextstates
        learnerT.Ns1a0s0[sprime][(a0, s0)] *= learnerT.etaleak # Discount all outgoing transitions
        learnerT.Ns1a0s0[sprime][(a0, s0)] += (1. - learnerT.etaleak) * learnerT.lowerbound # Bound it
    end
    if haskey(learnerT.Ns1a0s0[s1], (a0, s0))
        learnerT.Ns1a0s0[s1][(a0, s0)] += learnerT.etaleak # Increase observed
    else
        # println("i m here")
        learnerT.Ns1a0s0[s1][(a0, s0)] = learnerT.etaleak
        learnerT.Ns1a0s0[s1][(a0, s0)] += (1. - learnerT.etaleak) * learnerT.lowerbound # Bound it
    end
    # @show [learnerT.Ns1a0s0[s][a0, s0] for s in nextstates]
end
function computeterminalNs1a0s0!(learnerT::Union{TLeakyIntegratorNoBackLeak,
                                TLeakyIntegratorJump},
                                s1, done)
    if done
        if !in(s1, learnerT.terminalstates)
            push!(learnerT.terminalstates, s1)
            for a in 1:learnerT.na
                for s in 1:learnerT.ns
                    if s == s1
                        learnerT.Nsa[a, s1] = 1.
                        learnerT.Ns1a0s0[s][(a, s1)] = 1.
                    else
                        learnerT.Ns1a0s0[s][(a, s1)] = 0.
                    end
                end
                # @show a, s1
                # @show [learnerT.Ns1a0s0[snext][(a, s1)]/learnerT.Nsa[a, s1] for snext in 1:learnerT.ns]
            end
        end
    end
end
