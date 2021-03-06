struct ExplorationBonusDummy end
ExplorationBonusDummy()
updatebonus!(::ExplorationBonusDummy, learnerT, s0, a0, s1, done) = nothing
# updateQaugmented!(::ExplorationBonusDummy, learner) = nothing
export ExplorationBonusDummy

mutable struct ExplorationBonusLeaky
    ns::Int
    na::Int
    etaleakbonus::Float64
    smallestvalue::Float64
    #α0::Array{Float64, 2}
    Nsa::Array{Float64, 2}
    backupstep::Int
    backupcounter::Int
    β::Float64
    rewardbonus::Array{Float64, 2}
    Qaugmented::Array{Float64, 2}
end
function ExplorationBonusLeaky(; ns = 10, na = 4, etaleakbonus = .9,
                                    smallestvalue = eps(),
                                    backupstep = 30, backupcounter = 0,
                                    β = 0.1)
    #α0 = smallestvalue .* ones(na, ns)
    Nsa = zeros(na, ns)
    # rewardbonus = β ./ sqrt.(α0)
    # Qaugmented = β ./ sqrt.(α0)
    rewardbonus = zeros(na, ns)
    Qaugmented = zeros(na, ns)
    ExplorationBonusLeaky(ns, na, etaleakbonus, smallestvalue, Nsa,
                                backupstep, backupcounter, β,
                                rewardbonus, Qaugmented)
end
export ExplorationBonusLeaky
# function updatebonus!(bonus::ExplorationBonusLeaky, learnerT::TLeakyIntegrator,
#                         s0, a0, s1, done)
#     bonus.backupcounter += 1
#     @show (s0, a0, s1, done)
#     # leakbonus!(bonus)
#     computeRewardBonus!(bonus, learnerT)
# end
function updatebonus!(bonus::ExplorationBonusLeaky, learnerT, s0, a0, s1, done)
    bonus.backupcounter += 1
    @show bonus.backupcounter
    # leakbonus!(bonus)
    getNsa!(bonus, learnerT)
    for s in 1:learnerT.ns
        @show bonus.Nsa[:, s]
    end
    computeRewardBonus!(bonus)
    for s in 1:learnerT.ns
        @show bonus.rewardbonus[:, s]
    end
end
export updatebonus!
function getNsa!(bonus::ExplorationBonusLeaky, learnerT::TLeakyIntegrator)
    bonus.Nsa = deepcopy(learnerT.Nsa)
end
function getNsa!(bonus::ExplorationBonusLeaky, learnerT::TParticleFilter)
    for a in 1:learnerT.na
        # @show a
        for s in 1:learnerT.ns
            # @show s
            # @show learnerT.counts[a, s, :]
            if all(@. all(learnerT.counts[a, s, :] == [zeros(learnerT.ns)]))
                bonus.Nsa[a, s] = bonus.smallestvalue
                # println("they are zeros")
            else
                countsweighted = getcountsweighted(learnerT, a, s)
                # bonus.α0[a, s] = bonus.smallestvalue + sum(sum(countsweighted, dims = 1))
                bonus.Nsa[a, s] = sum(sum(countsweighted, dims = 1))
            end
        end
    end
end
function getNsa!(bonus::ExplorationBonusLeaky, learnerT::TVarSmile)
    for a in 1:learnerT.na
        for s in 1:learnerT.ns
            bonus.Nsa[a, s] = sum(learnerT.alphas[a, s] -
                                learnerT.stochasticity .* ones(learnerT.na, learnerT.ns))
        end
    end
end
function getcountsweighted(learnerT::TParticleFilter, a, s)
    countsweighted = zeros(learnerT.nparticles, learnerT.ns)
    for i in 1:learnerT.nparticles
        countsweighted[i,:] = learnerT.weights[a, s, i] .* learnerT.counts[a, s, i]
    end
    countsweighted
end
function computeRewardBonus!(bonus::ExplorationBonusLeaky)
    bonus.rewardbonus = bonus.β ./ sqrt.(bonus.Nsa)
end
# function computeRewardBonus!(bonus::ExplorationBonusLeaky, learnerT::TLeakyIntegrator)
#     bonus.rewardbonus = bonus.β ./ sqrt.(bonus.Nsa)
# end
# function computeRewardBonus!(bonus::ExplorationBonusLeaky, learnerT)
#     bonus.rewardbonus = bonus.β ./ sqrt.(bonus.α0)
# end
