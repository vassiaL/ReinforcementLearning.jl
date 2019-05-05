"""
TLeakyIntegrator
To be used alone as passive learners with random policy
or as Testimate parameter of SmallBackups
"""
struct TLeakyIntegrator
    ns::Int
    na::Int
    etaleak::Float64
    Nsa::Array{Float64, 2}
    Ns1a0s0::Array{Dict{Tuple{Int, Int}, Float64}, 1}
end
function TLeakyIntegrator(; ns = 10, na = 4, etaleak = .9)
    Nsa = zeros(na, ns)
    Ns1a0s0 = [Dict{Tuple{Int, Int}, Float64}() for _ in 1:ns]
    # for s in 1:ns # Initialize all transitions to 0
    #     for a in 1:na
    #         for sprime in 1:ns
    #             Ns1a0s0[sprime][(a, s)] = 0.
    #         end
    #     end
    # end
    TLeakyIntegrator(ns, na, etaleak, Nsa, Ns1a0s0)
end
export TLeakyIntegrator
""" X[t] = etaleak * X[t-1] + etaleak * I[.] """
function updatet!(learnerT::TLeakyIntegrator, s0, a0, s1, done)
    # ------ VERSION 2: Leave rest s0, a0 untouched
    learnerT.Nsa[a0, s0] *= learnerT.etaleak # Discount transition
    learnerT.Nsa[a0, s0] += learnerT.etaleak # Increase observed transition

    if !done
        nextstates = [s for s in 1:learnerT.ns if haskey(learnerT.Ns1a0s0[s],(a0,s0))]
        for sprime in nextstates
            learnerT.Ns1a0s0[sprime][(a0, s0)] *= learnerT.etaleak # Discount all outgoing transitions
        end
        if haskey(learnerT.Ns1a0s0[s1], (a0, s0))
            learnerT.Ns1a0s0[s1][(a0, s0)] += learnerT.etaleak # Increase observed transition
        else
            learnerT.Ns1a0s0[s1][(a0, s0)] = learnerT.etaleak # Increase observed transition
        end
    end
end
export updatet!
# Original:
# for sprime in 1:learnerT.ns
#     learnerT.Ns1a0s0[sprime][(a0, s0)] *= learnerT.etaleak # Discount all outgoing transitions
# end
# learnerT.Ns1a0s0[s1][(a0, s0)] += learnerT.etaleak # Increase observed transition

# If we initialize with 0s:
# if !done
#     learnerT.Ns1a0s0[s1][(a0, s0)] *= learnerT.etaleak # Discount all outgoing transitions
#     learnerT.Ns1a0s0[s1][(a0, s0)] += learnerT.etaleak # Increase observed transition
#
#     for sprime in 1:learnerT.ns
#         learnerT.Ns1a0s0[sprime][(a0, s0)] *= learnerT.etaleak # Discount all outgoing transitions
#     end
# else
#     learnerT.Ns1a0s0[s1][(a0, s0)] = learnerT.etaleak # Increase observed transition
# end
