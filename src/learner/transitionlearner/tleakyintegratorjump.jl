"""
TLeakyIntegratorJump
To be used alone as passive learners with random policy
or as Testimate parameter of SmallBackups
No background leak!!!
"""
struct TLeakyIntegratorJump <: TNs1a0s0
    ns::Int
    na::Int
    etaleak::Float64
    lowerbound::Float64
    Nsa::Array{Float64, 2}
    Ns1a0s0::Array{Dict{Tuple{Int, Int}, Float64}, 1}
    terminalstates::Array{Int,1}
end
function TLeakyIntegratorJump(; ns = 10, na = 4, etaleak = .9, lowerbound = eps())
    Nsa = zeros(na, ns) .+ ns*eps()
    Ns1a0s0 = [Dict{Tuple{Int, Int}, Float64}() for _ in 1:ns]
    [Ns1a0s0[sprime][(a, s)] = eps() for sprime in 1:ns for a in 1:na for s in 1:ns]
    TLeakyIntegratorJump(ns, na, etaleak, lowerbound, Nsa, Ns1a0s0, Int[])
end
export TLeakyIntegratorJump
# function computePs1a0s0!(learnerT::TLeakyIntegratorJump, s0, a0)
#    for s in 1:learnerT.ns
#        learnerT.Ps1a0s0[s][(a0, s0)] = learnerT.Ns1a0s0[s][(a0, s0)] / learnerT.Nsa[a0, s0]
#        # @show learnerT.Ps1a0s0[s][(a0, s0)]
#    end
# end
