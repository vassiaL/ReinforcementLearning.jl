using Distributions, SpecialFunctions, Random

"""
TEstimateIntegrator, TEstimateLeakyIntegrator, TEstimateParticleFilter
To be used alone as passive learners with random policy
or as Testimate parameter of SmallBackups
"""

struct TEstimateIntegrator
    ns::Int64
    na::Int64
    Nsa::Array{Int64, 2}
    Ns1a0s0::Array{Dict{Tuple{Int64, Int64}, Int64}, 1}
end
function TEstimateIntegrator(; ns = 10, na = 4)
    Nsa = zeros(Int64, na, ns)
    Ns1a0s0 = [Dict{Tuple{Int64, Int64}, Int64}() for _ in 1:ns]
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

struct TEstimateLeakyIntegrator
    ns::Int64
    na::Int64
    etaleak::Float64
    Nsa::Array{Float64, 2}
    Ns1a0s0::Array{Dict{Tuple{Int64, Int64}, Float64}, 1}
end
function TEstimateLeakyIntegrator(; ns = 10, na = 4, etaleak = .9)
    Nsa = zeros(na, ns)
    Ns1a0s0 = [Dict{Tuple{Int64, Int64}, Float64}() for _ in 1:ns]
    # Initialize all transitions to 0
    for s in 1:ns
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
        end
    end
    learnerT.Nsa[a0, s0] += learnerT.etaleak # Update observed

    for s in 1:learnerT.ns
        for a in 1:learnerT.na
            for sprime in 1:learnerT.ns
                learnerT.Ns1a0s0[sprime][(a, s)] *= learnerT.etaleak # Discount everything
            end
        end
    end
    learnerT.Ns1a0s0[s1][(a0, s0)] += learnerT.etaleak # And increase the one that happened
end

struct TEstimateParticleFilter
    ns::Int
    na::Int
    nparticles::Int # Per state and action
    Neffthrs::Float64
    stayprobability::Float64
    stochasticity::Float64
    particles::Array{Array{Int64,1}, 3} # wannabe ns x na x nparticles. Each element is an array that increases in time
    weights::Array{Array{Float64,1}, 3} # wannabe ns x na x nparticles. Each element is an array that increases in time
    Ps1a0s0::Array{Dict{Tuple{Int64, Int64}, Float64}, 1}
    alphas::Array{Array{Array{Float64,1}}, 3}
    seed::Any
    rng::MersenneTwister
end
function TEstimateParticleFilter(;ns = 10, na = 4, nparticles = 6, stayprobability = .999, stochasticity = .01, seed = 3)
    Neffthrs=nparticles/2.
    particles = Array{Array{Int64,1}}(undef, na, ns, nparticles)
    weights = Array{Array{Float64,1}}(undef, na, ns, nparticles)
    Ps1a0s0 = [Dict{Tuple{Int64, Int64}, Float64}() for _ in 1:ns]
    alphas = Array{Array{Array{Float64,1}}}(undef, na, ns, nparticles)
    rng = MersenneTwister(seed)
    TEstimateParticleFilter(ns, na, nparticles, Neffthrs, stayprobability, stochasticity, particles, weights, Ps1a0s0, alphas, seed, rng)
end
export TEstimateParticleFilter
function updatet!(learnerT::TEstimateParticleFilter, s0, a0, s1)
    if haskey(learnerT.Ps1a0s0[s1], (a0, s0))
        stayterms, switchterm = computeupdateterms(learnerT, s0, a0, s1)
        getweights!(learnerT, s0, a0, stayterms, switchterm)
        sampleparticles!(learnerT, s0, a0, stayterms, switchterm)

        lastweights = [learnerT.weights[a0, s0, i][end] for i in 1:learnerT.nparticles]
        Neff = 1. /sum(lastweights.^2) # Evaluate Neff = approx No of particles that have a weight which meaningfully contributes to the probability distribution.
        if Neff <= learnerT.Neffthrs # Resample!
            # println("I'm resampling!")
            resample!(learnerT, s0, a0)
        end
        updatealphas!(learnerT, s0, a0, s1)
    else # First visit
        for i in 1:learnerT.nparticles
            learnerT.particles[a0, s0, i] = [1]
            learnerT.weights[a0, s0, i] = [1. /learnerT.nparticles]
            learnerT.alphas[a0, s0, i] = [zeros(learnerT.ns)]
            learnerT.alphas[a0, s0, i][1][s1] += 1
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
    for i in 1:learnerT.nparticles
        particlealphas_htminus1_ytminus1 = deepcopy(learnerT.alphas[a0, s0, i][end])
        stayterms[i] = (learnerT.stochasticity + particlealphas_htminus1_ytminus1[s1]) / sum(learnerT.stochasticity .+ particlealphas_htminus1_ytminus1)
    end
    switchterm = 1. / learnerT.ns
    stayterms, switchterm
end

function getweights!(learnerT::TEstimateParticleFilter, s0, a0, stayterms, switchterm)
    sumofweights = 0.
    for i in 1:learnerT.nparticles
        #firstratio = B(s + a(h_t-1)') / B(s + a(h_t-1)). secondratio = B(s + a(h_t=h_t-1 + 1)) / B(s)
        particleweightupdate = learnerT.stayprobability * stayterms[i] + (1. - learnerT.stayprobability) * switchterm
        push!(learnerT.weights[a0, s0, i], particleweightupdate * learnerT.weights[a0, s0, i][end])
        sumofweights += learnerT.weights[a0, s0, i][end]
    end
    for i in 1:learnerT.nparticles # Normalize weights w_i[t] = w̃_i[t] / sum_j (w̃_j[t])
        learnerT.weights[a0, s0, i][end] /= sumofweights
    end
end
export getweights!

function sampleparticles!(learnerT::TEstimateParticleFilter, s0, a0, stayterms, switchterm) #beta_s_htminus1_yt, beta_s_htminus1_ytminus1, beta_s_htswitched_yt, beta_s)
    for i in 1:learnerT.nparticles
        push!(learnerT.particles[a0, s0, i], learnerT.particles[a0, s0, i][end]) # Same h as previous time
        particlestayprobability = computeproposaldistribution(learnerT, stayterms[i], switchterm)#beta_s_htminus1_yt[i], beta_s_htminus1_ytminus1[i], beta_s_htswitched_yt, beta_s) # Sample from proposal
        r=rand(learnerT.rng) # Draw and possibly update:
        if r > particlestayprobability
            learnerT.particles[a0, s0, i][end] += 1
        end
    end
end
export sampleparticles!

function computeproposaldistribution(learnerT::TEstimateParticleFilter, istayterm, switchterm)#ibeta_s_htminus1_yt, ibeta_s_htminus1_ytminus1, beta_s_htswitched_yt, beta_s)
    particlestayprobability = 1. /(1. + (((1. - learnerT.stayprobability) * switchterm) / (learnerT.stayprobability * istayterm)))
end
export computeproposaldistribution
function updatealphas!(learnerT::TEstimateParticleFilter, s0, a0, s1)
    for i in 1:learnerT.nparticles
        if size(learnerT.alphas[a0, s0, i], 1) != learnerT.particles[a0, s0, i][end] # If hidden state changed
            push!(learnerT.alphas[a0, s0, i], zeros(learnerT.ns))
        end
        learnerT.alphas[a0, s0, i][end][s1] += 1 # Last hidden state. +1 for s'
    end
end
export updatealphas!
function resample!(learnerT::TEstimateParticleFilter, s0, a0)
    lastweights = [learnerT.weights[a0, s0, i][end] for i in 1:learnerT.nparticles]
    d = Categorical(lastweights)
    tempcopyparticles = deepcopy(learnerT.particles)
    tempcopyalphas = deepcopy(learnerT.alphas)
    for i in 1:learnerT.nparticles
        sampledindex = rand(learnerT.rng, d)
        learnerT.particles[a0, s0, i] = deepcopy(tempcopyparticles[a0, s0, sampledindex])
        learnerT.weights[a0, s0, i][end] = 1. /learnerT.nparticles
        learnerT.alphas[a0, s0, i] = deepcopy(tempcopyalphas[a0, s0, sampledindex])
    end
end
export resample!
function computePs1a0s0!(learnerT::TEstimateParticleFilter, s0, a0)
    # lastht = [learnerT.particles[a0, s0, i][end] for i in 1:learnerT.nparticles]
    lastweights = [learnerT.weights[a0, s0, i][end] for i in 1:learnerT.nparticles]
    lastalphas = [learnerT.alphas[a0, s0, i][end] for i in 1:learnerT.nparticles]

    thetasweighted = zeros(learnerT.nparticles, learnerT.ns)
    for i in 1:learnerT.nparticles
        thetas = (learnerT.stochasticity .+ lastalphas[i])/sum(learnerT.stochasticity .+ lastalphas[i])
        thetasweighted[i,:] = lastweights[i] * thetas
    end
    expectedvaluethetas = sum(thetasweighted, dims = 1)
    for s in 1:learnerT.ns
        learnerT.Ps1a0s0[s][(a0, s0)] = deepcopy(expectedvaluethetas[s])
    end
end

function defaultpolicy(learner::Union{TEstimateIntegrator, TEstimateLeakyIntegrator, TEstimateParticleFilter}, actionspace,
                       buffer)
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
