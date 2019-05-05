"""
RIntegrator, REstimateDummy, RLeakyIntegrator
To be used as Restimate parameter of SmallBackups
"""

struct RIntegrator
    ns::Int
    na::Int
    Nsa::Array{Int, 2}
    Rsum::Array{Float64, 2}
    R::Array{Float64, 2}
end
function RIntegrator(; ns = 10, na = 4)
    Nsa = zeros(Int, na, ns)
    Rsum = zeros(na, ns)
    R = zeros(na, ns)
    RIntegrator(ns, na, Nsa, Rsum, R)
end
export RIntegrator
function updater!(learnerR::RIntegrator, s0, a0, r)
    learnerR.Nsa[a0, s0] += 1
    learnerR.Rsum[a0, s0] += r
    learnerR.R[a0, s0] = learnerR.Rsum[a0, s0] / learnerR.Nsa[a0, s0]
    # learnerR.R[a0, s0] -= learnerR.R[a0, s0] / learnerR.Nsa[a0, s0]
    # learnerR.R[a0, s0] += r / learnerR.Nsa[a0, s0]
end

struct REstimateDummy end
REstimateDummy()
updater!(::REstimateDummy, s0, a0, r) = nothing
export REstimateDummy

mutable struct RLeakyIntegrator
    ns::Int
    na::Int
    etaleak::Float64
    Nsa::Array{Float64, 2}
    Rsum::Array{Float64, 2}
    R::Array{Float64, 2}
end
function RLeakyIntegrator(; ns = 10, na = 4, etaleak = 0.9)
    Nsa = zeros(na, ns)
    Rsum = zeros(na, ns)
    R = zeros(na, ns)
    RLeakyIntegrator(ns, na, etaleak, Nsa, Rsum, R)
end
export RLeakyIntegrator
""" X[t] = etaleak * X[t-1] + etaleak * I[.] """
function updater!(learnerR::RLeakyIntegrator, s0, a0, r)
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

struct RParticleFilter
    ns::Int
    na::Int
    nparticles::Int # Per state and action
    Rsumparticles::Array{Float64, 3}
    R::Array{Float64, 2}
end
function RParticleFilter(; ns = 10, na = 4, nparticles = 6)
    Rsumparticles = zeros(na, ns, nparticles)
    R = zeros(na, ns)
    RParticleFilter(ns, na, nparticles, Rsumparticles, R)
end
export RParticleFilter

function updater!(learnerR::RParticleFilter, s0, a0, r, particlesswitch, weights, counts)
    updateRsum!(learnerR, s0, a0, r, particlesswitch)
    computeR!(learnerR, s0, a0, weights, counts)
end

function updateRsum!(learnerR::RParticleFilter, s0, a0, r, particlesswitch)
    for i in 1:learnerR.nparticles
        if particlesswitch[a0, s0, i] # if new hidden state
            learnerR.Rsumparticles[a0, s0, i] =  0.
        end
        learnerR.Rsumparticles[a0, s0, i] += r # Last hidden state. +1 for s'
    end
end
export updateRsum!

function computeR!(learnerR::RParticleFilter, s0, a0, weights, counts)
    # lastweights = [weights[a0, s0, i][end] for i in 1:learnerR.nparticles]
    # lastcounts = [sum(counts[a0, s0, i][end]) for i in 1:learnerR.nparticles]
    # lastRsum = [learnerR.Rsumparticles[a0, s0, i][end] for i in 1:learnerR.nparticles]
    learnerR.R[a0, s0] = sum(weights[a0, s0, i] .* (learnerR.Rsumparticles[a0, s0, i] ./ sum(counts[a0, s0, i])))
end
export computeR!


function defaultpolicy(learner::Union{REstimateDummy, RIntegrator, RLeakyIntegrator}, actionspace,
                       buffer)
    RandomPolicy(actionspace)
end

function update!(learner::Union{REstimateDummy, RIntegrator, RLeakyIntegrator}, buffer)
    a0 = buffer.actions[1]
    a1 = buffer.actions[2]
    s0 = buffer.states[1]
    s1 = buffer.states[2]
    r = buffer.rewards[1]
    updater!(learner, s0, a0, r)
end
export updater!
export update!
