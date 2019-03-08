"""
TEstimateLeakyIntegrator
To be used alone as passive learners with random policy
or as Testimate parameter of SmallBackups
"""
struct TEstimateLeakyIntegrator
    ns::Int
    na::Int
    etaleak::Float64
    Nsa::Array{Float64, 2}
    Ns1a0s0::Array{Dict{Tuple{Int, Int}, Float64}, 1}
end
function TEstimateLeakyIntegrator(; ns = 10, na = 4, etaleak = .9)
    Nsa = zeros(na, ns)
    Ns1a0s0 = [Dict{Tuple{Int, Int}, Float64}() for _ in 1:ns]
    for s in 1:ns # Initialize all transitions to 0
        for a in 1:na
            for sprime in 1:ns
                Ns1a0s0[sprime][(a, s)] = 0.
            end
        end
    end
    TEstimateLeakyIntegrator(ns, na, etaleak, Nsa, Ns1a0s0)
end
export TEstimateLeakyIntegrator
""" X[t] = etaleak * X[t-1] + etaleak * I[.] """
function updatet!(learnerT::TEstimateLeakyIntegrator, s0, a0, s1)
    for s in 1:learnerT.ns
        for a in 1:learnerT.na
            learnerT.Nsa[a, s] *= learnerT.etaleak # Discount everything
            for sprime in 1:learnerT.ns
                learnerT.Ns1a0s0[sprime][(a, s)] *= learnerT.etaleak # Discount everything
            end
        end
    end
    learnerT.Nsa[a0, s0] += learnerT.etaleak # Update observed
    learnerT.Ns1a0s0[s1][(a0, s0)] += learnerT.etaleak # And increase the one that happened
end
export updatet!
