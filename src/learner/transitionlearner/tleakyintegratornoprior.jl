"""
TLeakyIntegrator
To be used alone as passive learners with random policy
or as Testimate parameter of SmallBackups
    - etaleak for current (s,a) pair
    - etaleakbckground for all not currently visited (s,a) pairs in the background
If the background counts go beyond the point where 1/Nsa = Inf,
stop leaking, keep them in this same low value.
    - No info about the (prior) stochasticity of the environment in the calculation
    of the Ps1a0s0.
"""
struct TLeakyIntegrator <: TPs1a0s0 #TNs1a0s0
    ns::Int
    na::Int
    etaleak::Float64
    etaleakbckground::Float64
    lowerbound::Float64
    Nsa::Array{Float64, 2}
    Ns1a0s0::Array{Dict{Tuple{Int, Int}, Float64}, 1}
    Ps1a0s0::Array{Dict{Tuple{Int, Int}, Float64}, 1}
    terminalstates::Array{Int,1}
end
function TLeakyIntegrator(; ns = 10, na = 4, etaleak = .9,
                            etaleakbckground = etaleak, lowerbound = eps())
    # Nsa = zeros(na, ns) .+ ns*eps()
    # [Ns1a0s0[sprime][(a, s)] = eps() for sprime in 1:ns for a in 1:na for s in 1:ns]
    Nsa = zeros(na, ns) #.+ ns*eps()
    Ns1a0s0 = [Dict{Tuple{Int, Int}, Float64}() for _ in 1:ns]
    [Ns1a0s0[sprime][(a, s)] = 0. for sprime in 1:ns for a in 1:na for s in 1:ns]
    Ps1a0s0 = [Dict{Tuple{Int, Int}, Float64}() for _ in 1:ns]
    [Ps1a0s0[sprime][(a, s)] = 1. /ns for sprime in 1:ns for a in 1:na for s in 1:ns]
    TLeakyIntegrator(ns, na, etaleak, etaleakbckground, lowerbound, Nsa, Ns1a0s0, Ps1a0s0, Int[])
end
export TLeakyIntegrator
function leakothers!(learnerT::TLeakyIntegrator, s0, a0)
    pairs = getactionstatepairs!(learnerT, s0, a0)
    for sa in pairs # sa[1] = action, sa[2] = state
        # @show sa
        if !in(sa[2], learnerT.terminalstates) # Dont leak outgoing transitions of terminal states
            if learnerT.Nsa[sa[1], sa[2]] != 0. # Doesn't make sense to leak if it is already 0
                if !isinf(1. / learnerT.Nsa[sa[1], sa[2]]) # If it becomes too low, stop leaking to avoid numerical errors
                    learnerT.Nsa[sa[1], sa[2]] *= learnerT.etaleakbckground
                    # @show sa, learnerT.Nsa[sa[1], sa[2]]
                    nextstates = [s for s in 1:learnerT.ns if haskey(learnerT.Ns1a0s0[s], sa)]
                    for sprime in nextstates
                        learnerT.Ns1a0s0[sprime][sa] *= learnerT.etaleakbckground # Discount all outgoing transitions
                        # @show sprime, learnerT.Ns1a0s0[sprime][sa]
                    end
                    computePs1a0s0!(learnerT, sa[2], sa[1])
                    # @show [learnerT.Ns1a0s0[s][sa] for s in nextstates]
                    # @show [learnerT.Ps1a0s0[s][sa] for s in nextstates]
                # else
                #     println("I reached infinity!")
                #     # @show learnerT.Nsa[sa[1], sa[2]]
                #     # @show 1. / learnerT.Nsa[sa[1], sa[2]]
                    # @show sa
                #     @show [learnerT.Ns1a0s0[s][sa] for s in  1:learnerT.ns]
                #     @show [learnerT.Ps1a0s0[s][sa] for s in  1:learnerT.ns]
                end
            end
        end
    end
end
function computePs1a0s0!(learnerT::TLeakyIntegrator, s0, a0)
    if learnerT.Nsa[a0, s0] != 0. # Don't do anything to Ps1a0s0 if state has not been visited: keep them at 1/ns
        nextstates = [s for s in 1:learnerT.ns if haskey(learnerT.Ns1a0s0[s],(a0, s0))]
        denominator = sum([learnerT.Ns1a0s0[s][a0, s0] for s in nextstates])
        # @show [learnerT.Ns1a0s0[s][a0, s0] for s in nextstates]
        # @show denominator
        for s in 1:learnerT.ns
            learnerT.Ps1a0s0[s][(a0, s0)] = learnerT.Ns1a0s0[s][(a0, s0)] / denominator
            # @show s, learnerT.Ps1a0s0[s][(a0, s0)]
        end
    end
    # @show [learnerT.Ns1a0s0[s][(a0, s0)] for s in 1:learnerT.ns]
    # @show [learnerT.Ps1a0s0[s][(a0, s0)] for s in 1:learnerT.ns]
end
export updatet!

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
