"""
TLeakyIntegrator
To be used alone as passive learners with random policy
or as Testimate parameter of SmallBackups
"""
struct TLeakyIntegrator <: TNs1a0s0
    ns::Int
    na::Int
    etaleak::Float64
    etaleakbckground::Float64
    lowerbound::Float64
    Nsa::Array{Float64, 2}
    Ns1a0s0::Array{Dict{Tuple{Int, Int}, Float64}, 1}
    terminalstates::Array{Int,1}
end
function TLeakyIntegrator(; ns = 10, na = 4, etaleak = .9,
                            etaleakbckground = etaleak, lowerbound = eps())
    Nsa = zeros(na, ns) .+ ns*eps()
    Ns1a0s0 = [Dict{Tuple{Int, Int}, Float64}() for _ in 1:ns]
    [Ns1a0s0[sprime][(a, s)] = eps() for sprime in 1:ns for a in 1:na for s in 1:ns]
    TLeakyIntegrator(ns, na, etaleak, etaleakbckground, lowerbound, Nsa, Ns1a0s0, Int[])
end
export TLeakyIntegrator
""" X[t] = etaleak * X[t-1] + etaleak * I[.] """
function updatet!(learnerT::TLeakyIntegrator, s0, a0, s1, done)
    leaka0s0!(learnerT, s0, a0, s1, done)
    computeterminalNs1a0s0!(learnerT, s1, done)
    leakothers!(learnerT, s0, a0)
end
function leaka0s0!(learnerT::Union{TLeakyIntegrator,
                    TLeakyIntegratorNoBackLeak,
                    TLeakyIntegratorJump},
                    s0, a0, s1, done)
    learnerT.Nsa[a0, s0] *= learnerT.etaleak # Discount transition
    learnerT.Nsa[a0, s0] += learnerT.etaleak # Increase observed transition
    learnerT.Nsa[a0, s0] += (1. - learnerT.etaleak) * learnerT.ns * learnerT.lowerbound # Bound it
    # @show a0, s0, learnerT.Nsa[a0, s0]
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
end
function leakothers!(learnerT::TLeakyIntegrator, s0, a0)
    pairs = getactionstatepairs!(learnerT, s0, a0)
    for sa in pairs # sa[1] = action, sa[2] = state
        # @show sa
        if !in(sa[2], learnerT.terminalstates) # Dont leak outgoing transitions of terminal states
            if learnerT.Nsa[sa[1], sa[2]] != learnerT.ns * learnerT.lowerbound
                learnerT.Nsa[sa[1], sa[2]] *= learnerT.etaleakbckground
                learnerT.Nsa[sa[1], sa[2]] += (1. - learnerT.etaleakbckground) * learnerT.ns * learnerT.lowerbound # Bound it
                # @show sa, learnerT.Nsa[sa[1], sa[2]]
                nextstates = [s for s in 1:learnerT.ns if haskey(learnerT.Ns1a0s0[s], sa)]
                for sprime in nextstates
                    learnerT.Ns1a0s0[sprime][sa] *= learnerT.etaleakbckground # Discount all outgoing transitions
                    learnerT.Ns1a0s0[sprime][sa] += (1. - learnerT.etaleakbckground) * learnerT.lowerbound # Bound it
                    # @show sprime, learnerT.Ns1a0s0[sprime][sa]
                end
            end
        end
    end
end
function computeterminalNs1a0s0!(learnerT::Union{TLeakyIntegrator,
                                TLeakyIntegratorNoBackLeak,
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
                    #@show a, s1, s, learnerT.Ns1a0s0[s][(a, s1)]
                end
            end
        end
    end
end
## ----- Delete me: init with 0s
# function updatet!(learnerT::TLeakyIntegrator, s0, a0, s1, done)
#     # ------ VERSION 2: Leave rest s0, a0 untouched
#     learnerT.Nsa[a0, s0] *= learnerT.etaleak # Discount transition
#     learnerT.Nsa[a0, s0] += learnerT.etaleak # Increase observed transition
#     if !done
#         for s in 1:learnerT.ns # Initialize all transitions to 0
#             if haskey(learnerT.Ns1a0s0[s], (a0, s0))
#                 learnerT.Ns1a0s0[s][(a0, s0)] *= learnerT.etaleak # Discount all outgoing transitions
#             else
#                 learnerT.Ns1a0s0[s][(a0, s0)] = 0.
#             end
#         end
#         learnerT.Ns1a0s0[s1][(a0, s0)] += learnerT.etaleak # Increase observed
#         # computePs1a0s0!(learnerT::TLeakyIntegrator, s0, a0)
#     end
# end
# # function computePs1a0s0!(learnerT::TLeakyIntegrator, s0, a0)
# #    for s in 1:learnerT.ns
# #        learnerT.Ps1a0s0[s][(a0, s0)] = learnerT.Ns1a0s0[s][(a0, s0)] / learnerT.Nsa[a0, s0]
# #        # @show learnerT.Ps1a0s0[s][(a0, s0)]
# #    end
# # end
export updatet!
