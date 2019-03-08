using Distributions, SpecialFunctions, Random
"""
TEstimateParticleFilter
To be used alone as passive learners with random policy
or as Testimate parameter of SmallBackups
"""
struct TEstimateParticleFilter
    ns::Int
    na::Int
    nparticles::Int # Per state and action
    Neffthrs::Float64
    stayprobability::Float64
    stochasticity::Float64
    particlesswitch::Array{Bool, 3} # wannabe ns x na x nparticles.
    weights::Array{Float64, 3} # wannabe ns x na x nparticles.
    Ps1a0s0::Array{Dict{Tuple{Int, Int}, Float64}, 1}
    alphas::Array{Array{Float64,1}, 3}
    seed::Any
    rng::MersenneTwister
end
function TEstimateParticleFilter(;ns = 10, na = 4, nparticles = 6, stayprobability = .999, stochasticity = .01, seed = 3)
    Neffthrs=nparticles/2.
    particlesswitch = Array{Bool, 3}(undef, na, ns, nparticles)
    weights = Array{Float64, 3}(undef, na, ns, nparticles)
    Ps1a0s0 = [Dict{Tuple{Int, Int}, Float64}() for _ in 1:ns]
    alphas = Array{Array{Float64,1}}(undef, na, ns, nparticles)
    rng = MersenneTwister(seed)
    TEstimateParticleFilter(ns, na, nparticles, Neffthrs, stayprobability, stochasticity, particlesswitch, weights, Ps1a0s0, alphas, seed, rng)
end
export TEstimateParticleFilter
function updatet!(learnerT::TEstimateParticleFilter, s0, a0, s1)
    if haskey(learnerT.Ps1a0s0[s1], (a0, s0))
        stayterms, switchterm = computeupdateterms(learnerT, s0, a0, s1)
        getweights!(learnerT, s0, a0, stayterms, switchterm)
        sampleparticles!(learnerT, s0, a0, stayterms, switchterm)
        Neff = 1. /sum((@view learnerT.weights[a0, s0, :]) .^2) # Evaluate Neff = approx nr of particles that have a weight which meaningfully contributes to the probability distribution.
        if Neff <= learnerT.Neffthrs # Resample!
            resample!(learnerT, s0, a0)
        end
        updatealphas!(learnerT, s0, a0, s1)
    else # First visit
        for i in 1:learnerT.nparticles
            learnerT.particlesswitch[a0, s0, i] = false
            learnerT.weights[a0, s0, i] = 1. /learnerT.nparticles
            learnerT.alphas[a0, s0, i] = zeros(learnerT.ns)
            learnerT.alphas[a0, s0, i][s1] += 1
        end
        for s in 1:learnerT.ns
            learnerT.Ps1a0s0[s][(a0, s0)] = 1. /learnerT.ns
        end
    end
    computePs1a0s0!(learnerT, s0, a0)
end
export updatet!
function computeupdateterms(learnerT::TEstimateParticleFilter, s0, a0, s1)
    stayterms = zeros(learnerT.nparticles)
    for i in 1:learnerT.nparticles # particlealphas_htminus1_ytminus1 = deepcopy(learnerT.alphas[a0, s0, i])
        stayterms[i] = (learnerT.stochasticity + learnerT.alphas[a0, s0, i][s1]) / sum(learnerT.stochasticity .+ learnerT.alphas[a0, s0, i])
    end
    switchterm = 1. / learnerT.ns
    stayterms, switchterm
end
export computeupdateterms!
function getweights!(learnerT::TEstimateParticleFilter, s0, a0, stayterms, switchterm)
    for i in 1:learnerT.nparticles #firstratio = B(s + a(h_t-1)') / B(s + a(h_t-1)). secondratio = B(s + a(h_t=h_t-1 + 1)) / B(s)
        particleweightupdate = learnerT.stayprobability * stayterms[i] + (1. - learnerT.stayprobability) * switchterm
        learnerT.weights[a0, s0, i] *= particleweightupdate
    end
    learnerT.weights[a0, s0, :] /= sum(learnerT.weights[a0, s0, :]) # Normalize
end
export getweights!
function sampleparticles!(learnerT::TEstimateParticleFilter, s0, a0, stayterms, switchterm)
    for i in 1:learnerT.nparticles
        particlestayprobability = computeproposaldistribution(learnerT, stayterms[i], switchterm)
        r = rand(learnerT.rng) # Draw and possibly update
        if r > particlestayprobability
            learnerT.particlesswitch[a0, s0, i] = true
        end
    end
end
export sampleparticles!
function computeproposaldistribution(learnerT::TEstimateParticleFilter, istayterm, switchterm)
    particlestayprobability = 1. /(1. + (((1. - learnerT.stayprobability) * switchterm) / (learnerT.stayprobability * istayterm)))
end
export computeproposaldistribution
function updatealphas!(learnerT::TEstimateParticleFilter, s0, a0, s1)
    for i in 1:learnerT.nparticles
        if learnerT.particlesswitch[a0, s0, i] # if new hidden state
            learnerT.alphas[a0, s0, i] =  zeros(learnerT.ns)
            learnerT.particlesswitch[a0, s0, i] = false
        end
        learnerT.alphas[a0, s0, i][s1] += 1 # Last hidden state. +1 for s'
    end
end
export updatealphas!
function resample!(learnerT::TEstimateParticleFilter, s0, a0)
    d = Categorical(learnerT.weights[a0, s0, :])
    tempcopyparticlesswitch = copy(learnerT.particlesswitch[a0, s0, :])
    tempcopyalphas = deepcopy(learnerT.alphas[a0, s0, :])
    for i in 1:learnerT.nparticles
        sampledindex = rand(learnerT.rng, d)
        learnerT.particlesswitch[a0, s0, i] = tempcopyparticlesswitch[sampledindex]
        learnerT.weights[a0, s0, i] = 1. /learnerT.nparticles
        learnerT.alphas[a0, s0, i] = deepcopy(tempcopyalphas[sampledindex])
    end
end
export resample!
function computePs1a0s0!(learnerT::TEstimateParticleFilter, s0, a0)
    thetasweighted = zeros(learnerT.nparticles, learnerT.ns)
    for i in 1:learnerT.nparticles
        thetas = (learnerT.stochasticity .+ learnerT.alphas[a0, s0, i])/sum(learnerT.stochasticity .+ learnerT.alphas[a0, s0, i])
        thetasweighted[i,:] = learnerT.weights[a0, s0, i] * thetas
    end
    expectedvaluethetas = sum(thetasweighted, dims = 1)
    for s in 1:learnerT.ns
        learnerT.Ps1a0s0[s][(a0, s0)] = expectedvaluethetas[s]
    end
end
function defaultpolicy(learner::Union{TEstimateIntegrator, TEstimateLeakyIntegrator, TEstimateParticleFilter}, actionspace, buffer)
    RandomPolicy(actionspace)
end
function update!(learner::Union{TEstimateIntegrator, TEstimateLeakyIntegrator, TEstimateParticleFilter}, buffer)
    a0 = buffer.actions[1]
    a1 = buffer.actions[2]
    s0 = buffer.states[1]
    s1 = buffer.states[2]
    r = buffer.rewards[1]
    updatet!(learner, s0, a0, s1)
end
export update!
