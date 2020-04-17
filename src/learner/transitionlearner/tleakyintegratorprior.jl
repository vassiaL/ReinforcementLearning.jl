"""
TLeakyIntegrator with prior information for the calculation of the Ps1a0s0
To be used alone as passive learners with random policy
or as Testimate parameter of SmallBackups
"""
struct TLeakyIntegratorPrior <: TPs1a0s0
    ns::Int
    na::Int
    etaleak::Float64
    etaleakbckground::Float64
    stochasticity::Float64
    Nsa::Array{Float64, 2}
    Ns1a0s0::Array{Dict{Tuple{Int, Int}, Float64}, 1}
    Ps1a0s0::Array{Dict{Tuple{Int, Int}, Float64}, 1}
    terminalstates::Array{Int,1}
end
function TLeakyIntegratorPrior(; ns = 10, na = 4, etaleak = .9,
                            etaleakbckground = etaleak,
                            stochasticity = eps())
    Nsa = zeros(na, ns) #.+ ns*eps()
    Ns1a0s0 = [Dict{Tuple{Int, Int}, Float64}() for _ in 1:ns]
    [Ns1a0s0[sprime][(a, s)] = 0. for sprime in 1:ns for a in 1:na for s in 1:ns]
    Ps1a0s0 = [Dict{Tuple{Int, Int}, Float64}() for _ in 1:ns]
    [Ps1a0s0[sprime][(a, s)] = 1. /ns for sprime in 1:ns for a in 1:na for s in 1:ns]
    TLeakyIntegratorPrior(ns, na, etaleak, etaleakbckground, stochasticity,
                            Nsa, Ns1a0s0, Ps1a0s0, Int[])
end
export TLeakyIntegratorPrior
""" X[t] = etaleak * X[t-1] + etaleak * I[.] """
function updatet!(learnerT::TLeakyIntegratorPrior, s0, a0, s1, done)
    leaka0s0!(learnerT, s0, a0, s1, done)
    computeterminalNs1a0s0!(learnerT, s1, done)
    computePs1a0s0!(learnerT, s0, a0)
    leakothers!(learnerT, s0, a0)
end
function leaka0s0!(learnerT::TLeakyIntegratorPrior,
                    s0, a0, s1, done)
    learnerT.Nsa[a0, s0] *= learnerT.etaleak # Discount transition
    learnerT.Nsa[a0, s0] += learnerT.etaleak # Increase observed transition
    # @show a0, s0, learnerT.Nsa[a0, s0]
    nextstates = [s for s in 1:learnerT.ns if haskey(learnerT.Ns1a0s0[s],(a0,s0))]
    for sprime in nextstates
        learnerT.Ns1a0s0[sprime][(a0, s0)] *= learnerT.etaleak # Discount all outgoing transitions
    end
    if haskey(learnerT.Ns1a0s0[s1], (a0, s0))
        learnerT.Ns1a0s0[s1][(a0, s0)] += learnerT.etaleak # Increase observed
    else
        learnerT.Ns1a0s0[s1][(a0, s0)] = learnerT.etaleak
    end
    # @show a0, s0, s1, learnerT.Ns1a0s0[s1][(a0, s0)]
end
function leakothers!(learnerT::TLeakyIntegratorPrior, s0, a0)
    pairs = getactionstatepairs!(learnerT, s0, a0)
    for sa in pairs # sa[1] = action, sa[2] = state
        # @show sa
        if !in(sa[2], learnerT.terminalstates) # Dont leak outgoing transitions of terminal states
            if learnerT.Nsa[sa[1], sa[2]] != 0. # Doesn't make sense to leak if it is already 0
                learnerT.Nsa[sa[1], sa[2]] *= learnerT.etaleakbckground
                # @show sa, learnerT.Nsa[sa[1], sa[2]]
                nextstates = [s for s in 1:learnerT.ns if haskey(learnerT.Ns1a0s0[s], sa)]
                for sprime in nextstates
                    learnerT.Ns1a0s0[sprime][sa] *= learnerT.etaleakbckground # Discount all outgoing transitions
                    # @show sprime, learnerT.Ns1a0s0[sprime][sa]
                end
                computePs1a0s0!(learnerT, sa[2], sa[1])
            end
        end
    end
end
function computeterminalNs1a0s0!(learnerT::TLeakyIntegratorPrior,
                                s1, done)
    if done
        if !in(s1, learnerT.terminalstates)
            push!(learnerT.terminalstates, s1)
            for a in 1:learnerT.na
                for s in 1:learnerT.ns
                    if s == s1
                        learnerT.Nsa[a, s1] = 1.
                        learnerT.Ns1a0s0[s][(a, s1)] = 1.
                        learnerT.Ps1a0s0[s][(a, s1)] = 1.
                    else
                        learnerT.Ns1a0s0[s][(a, s1)] = 0.
                        learnerT.Ps1a0s0[s][(a, s1)] = 0.
                    end
                    # @show a, s1, s, learnerT.Ns1a0s0[s][(a, s1)]
                end
                # @show a, s1
                # @show [learnerT.Ns1a0s0[snext][a, s1] for snext in 1:learnerT.ns]
                # @show [learnerT.Ps1a0s0[snext][a, s1] for snext in 1:learnerT.ns]
            end
        end
    end
end
function computePs1a0s0!(learnerT::TLeakyIntegratorPrior, s0, a0)
    nextstates = [s for s in 1:learnerT.ns if haskey(learnerT.Ns1a0s0[s],(a0, s0))]
    denominator = sum(learnerT.stochasticity .+ [learnerT.Ns1a0s0[s][a0, s0] for s in nextstates])
    # @show [learnerT.Ns1a0s0[s][a0, s0] for s in nextstates]
    # @show denominator
   for s in 1:learnerT.ns
       # learnerT.Ps1a0s0[s][(a0, s0)] = learnerT.Ns1a0s0[s][(a0, s0)] / learnerT.Nsa[a0, s0]
       learnerT.Ps1a0s0[s][(a0, s0)] = (learnerT.stochasticity + learnerT.Ns1a0s0[s][(a0, s0)])
       learnerT.Ps1a0s0[s][(a0, s0)] /= denominator
       # @show s, learnerT.Ps1a0s0[s][(a0, s0)]
   end
end
export updatet!
