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
    α0::Array{Float64, 2}
    backupstep::Int
    backupcounter::Int
    beta::Float64
    rewardbonus::Array{Float64, 2}
    Qaugmented::Array{Float64, 2}
end
function ExplorationBonusLeaky(; ns = 10, na = 4, etaleakbonus = .9,
                                    smallestvalue = eps(),
                                    backupstep = 30, backupcounter = 0,
                                    beta = 2.5)
    α0 = smallestvalue .* ones(na, ns)
    rewardbonus = beta ./ sqrt.(α0)
    Qaugmented = beta ./ sqrt.(α0)
    ExplorationBonusLeaky(ns, na, etaleakbonus, smallestvalue,
                                α0, backupstep, backupcounter, beta,
                                rewardbonus, Qaugmented)
end
export ExplorationBonusLeaky
function updatebonus!(bonus::ExplorationBonusLeaky, learnerT::TLeakyIntegrator,
                        s0, a0, s1, done)
    bonus.backupcounter += 1
    @show (s0, a0, s1, done)
    leakbonus!(bonus)
    bonus.α0[a0, s0] = copy(learnerT.α0[a0, s0]) # Get a0, s0 counts
    bonus.rewardbonus = computeRewardBonus!(bonus)
end
function updatebonus!(bonus::ExplorationBonusLeaky, learnerT::TParticleFilter,
                        s0, a0, s1, done)
    bonus.backupcounter += 1
    @show (s0, a0, s1, done)
    leakbonus!(bonus)
    if !done
        countsweighted = getcountsweighted(learnerT, a0, s0) # Get a0, s0 counts
        @show countsweighted
        bonus.α0[a0, s0] = bonus.smallestvalue + sum(sum(countsweighted, dims = 1))
    else
        bonus.α0[a0, s0] += 1.
    end
    bonus.rewardbonus = computeRewardBonus!(bonus)
end
function updatebonus!(bonus::ExplorationBonusLeaky, learnerT::TVarSmile,
                                s0, a0, s1, done)
    bonus.backupcounter += 1
    @show (s0, a0, s1, done)
    leakbonus!(bonus)
    if !done # Get a0, s0 counts
        bonus.α0[a0, s0] = sum(learnerT.alphas[a0, s0])
    else
        bonus.α0[a0, s0] += 1.
    end
    bonus.rewardbonus = computeRewardBonus!(bonus)
end
export updatebonus!
function getcountsweighted(learnerT::TParticleFilter, a0, s0)
    countsweighted = zeros(learnerT.nparticles, learnerT.ns)
    for i in 1:learnerT.nparticles
        countsweighted[i,:] = learnerT.weights[a0, s0, i] .*
                                learnerT.counts[a0, s0, i]
    end
    countsweighted
end
function leakbonus!(bonus::ExplorationBonusLeaky)
    @. bonus.α0 .= bonus.etaleakbonus .* bonus.smallestvalue +
                (1 - bonus.etaleakbonus) .* bonus.α0 # Diffuse everything
end
function computeRewardBonus!(bonus::ExplorationBonusLeaky)
    bonus.rewardbonus = bonus.beta ./ sqrt.(bonus.α0)
end
# function updateQaugmented!(bonus, learner)
#     bonus.backupcounter += 1
#     if bonus.backupcounter == bonus.backupstep
#         backupbonus!(bonus, learner)
#         bonus.Qaugmented = copy(learner.Q)
#         bonus.backupcounter = 0
#     else
#         @. bonus.Qaugmented = bonus.rewardbonus + learner.Q
#     end
# end
# export updateQaugmented!
# function backupbonus!(bonus, learner)
#     # for all states and actions : # Update reward
#     for a in 1:bonus.na
#         for s in 1:bonus.ns
#             learner.Restimate.R[a, s] += bonus.rewardbonus
#             for sprime in 1:bonus.ns
#                 # Update Q values
#                 updateq!(learner, a, s, sprime, nothing, false)
#                 learner.V[s] = maximumbelowInf(learner.Q[:, s])
#                 # processqueue
#                 p = abs(learner.V[s] - learner.U[s])
#                 if p > learner.minpriority; addtoqueue!(learner.queue, s, p); end
#                 processqueue!(learner)
#             end
#         end
#     end
# end
# export backupbonus!
