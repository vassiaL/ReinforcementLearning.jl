"""
TIntegrator
To be used alone as passive learners with random policy
or as Testimate parameter of SmallBackups
"""
struct TIntegrator
    ns::Int
    na::Int
    Nsa::Array{Int, 2}
    Ns1a0s0::Array{Dict{Tuple{Int, Int}, Int}, 1}
end
function TIntegrator(; ns = 10, na = 4)
    Nsa = zeros(Int, na, ns)
    Ns1a0s0 = [Dict{Tuple{Int, Int}, Int}() for _ in 1:ns]
    TIntegrator(ns, na, Nsa, Ns1a0s0)
end
export TIntegrator
function updatet!(learnerT::TIntegrator, s0, a0, s1, done)
    learnerT.Nsa[a0, s0] += 1
    # ---- If updating transitions to the goal doesn't matter (eg R(s,a) ):
    if !done
        if haskey(learnerT.Ns1a0s0[s1], (a0, s0))
            learnerT.Ns1a0s0[s1][(a0, s0)] += 1
        else
            learnerT.Ns1a0s0[s1][(a0, s0)] = 1
        end
    end
    # ---- If it matters...
    # learnerT.Ns1a0s0[s1][(a0, s0)] += 1
end
