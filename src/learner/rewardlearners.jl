"""
REstimateIntegrator, REstimateDummy, REstimateLeakyIntegrator
To be used as Restimate parameter of SmallBackups
"""

struct REstimateIntegrator
    ns::Int
    na::Int
    Nsa::Array{Int, 2}
    R::Array{Float64, 2}
end
function REstimateIntegrator(; ns = 10, na = 4)
    Nsa = zeros(Int, na, ns)
    R = zeros(na, ns)
    REstimateIntegrator(ns, na, Nsa, R)
end
export REstimateIntegrator
function updater!(learnerR::REstimateIntegrator, s0, a0, r)
    learnerR.Nsa[a0, s0] += 1
    learnerR.R[a0, s0] -= learnerR.R[a0, s0] / learnerR.Nsa[a0, s0]
    learnerR.R[a0, s0] += r / learnerR.Nsa[a0, s0]
end

struct REstimateDummy end
REstimateDummy()
updater!(::REstimateDummy, s0, a0, r) = nothing
export REstimateDummy

mutable struct REstimateLeakyIntegrator
    ns::Int
    na::Int
    etaleak::Float64
    Nsa::Array{Float64, 2}
    Rsum::Array{Float64, 2}
    R::Array{Float64, 2}
end
function REstimateLeakyIntegrator(; ns = 10, na = 4, etaleak = 0.9)
    Nsa = zeros(na, ns)
    Rsum = zeros(na, ns)
    R = zeros(na, ns)
    REstimateLeakyIntegrator(ns, na, etaleak, Nsa, Rsum, R)
end
export REstimateLeakyIntegrator
""" X[t] = etaleak * X[t-1] + etaleak * I[.] """
function updater!(learnerR::REstimateLeakyIntegrator, s0, a0, r)
    for s in 1:learnerR.ns
        for a in 1:learnerR.na
            learnerR.Nsa[a, s] *= learnerR.etaleak # Discount everything
        end
    end
    learnerR.Nsa[a0, s0] += learnerR.etaleak # Update observed

    for s in 1:learnerR.ns
        for a in 1:learnerR.na
            learnerR.Rsum[a, s] *= learnerR.etaleak * learnerR.Rsum[a, s] # Discount everything
        end
    end
    learnerR.Rsum[a0, s0] += learnerR.etaleak * r # Update observed

    learnerR.R = learnerR.Rsum ./ learnerR.Nsa
end

struct REstimateParticleFilter
    ns::Int
    na::Int
    nparticles::Int # Per state and action
    Rsumparticles::Array{Float64, 3}
    R::Array{Float64, 2}
end
function REstimateParticleFilter(; ns = 10, na = 4, nparticles = 6)
    Rsumparticles = zeros(na, ns, nparticles)
    R = zeros(na, ns)
    REstimateParticleFilter(ns, na, nparticles, Rsumparticles, R)
end
export REstimateParticleFilter

function updater!(learnerR::REstimateParticleFilter, s0, a0, r, particlesswitch, weights, counts)
    updateRsum!(learnerR, s0, a0, r, particlesswitch)
    computeR!(learnerR, s0, a0, weights, counts)
end

function updateRsum!(learnerR::REstimateParticleFilter, s0, a0, r, particlesswitch)
    for i in 1:learnerR.nparticles
        if particlesswitch[a0, s0, i] # if new hidden state
            learnerR.Rsumparticles[a0, s0, i] =  0.
        end
        learnerR.Rsumparticles[a0, s0, i] += r # Last hidden state. +1 for s'
    end
end
export updateRsum!

function computeR!(learnerR::REstimateParticleFilter, s0, a0, weights, counts)
    # lastweights = [weights[a0, s0, i][end] for i in 1:learnerR.nparticles]
    # lastcounts = [sum(counts[a0, s0, i][end]) for i in 1:learnerR.nparticles]
    # lastRsum = [learnerR.Rsumparticles[a0, s0, i][end] for i in 1:learnerR.nparticles]
    learnerR.R[a0, s0] = sum(weights[a0, s0, i] .* (learnerR.Rsumparticles[a0, s0, i] ./ sum(counts[a0, s0, i])))
end
export computeR!


function defaultpolicy(learner::Union{REstimateDummy, REstimateIntegrator, REstimateLeakyIntegrator}, actionspace,
                       buffer)
    RandomPolicy(actionspace)
end

function update!(learner::Union{REstimateDummy, REstimateIntegrator, REstimateLeakyIntegrator}, buffer)
    a0 = buffer.actions[1]
    a1 = buffer.actions[2]
    s0 = buffer.states[1]
    s1 = buffer.states[2]
    r = buffer.rewards[1]
    updater!(learner, s0, a0, r)
end
export updater!
export update!
