"""
TEstimateIntegrator
To be used alone as passive learners with random policy
or as Testimate parameter of SmallBackups
"""
struct TEstimateIntegrator
    ns::Int
    na::Int
    Nsa::Array{Int, 2}
    Ns1a0s0::Array{Dict{Tuple{Int, Int}, Int}, 1}
end
function TEstimateIntegrator(; ns = 10, na = 4)
    Nsa = zeros(Int, na, ns)
    Ns1a0s0 = [Dict{Tuple{Int, Int}, Int}() for _ in 1:ns]
    TEstimateIntegrator(ns, na, Nsa, Ns1a0s0)
end
export TEstimateIntegrator
function updatet!(learnerT::TEstimateIntegrator, s0, a0, s1)
    learnerT.Nsa[a0, s0] += 1
    if haskey(learnerT.Ns1a0s0[s1], (a0, s0))
        learnerT.Ns1a0s0[s1][(a0, s0)] += 1
    else
        learnerT.Ns1a0s0[s1][(a0, s0)] = 1
    end
end
