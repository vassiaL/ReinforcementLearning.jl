using Distributions, SpecialFunctions, Random
"""
TParticleFilter
To be used alone as passive learners with random policy
or as Testimate parameter of SmallBackups
"""
struct TParticleFilter
    ns::Int
    na::Int
    nparticles::Int # Per state and action
    Neffthrs::Float64 # Neff = approx nr of particles that have a weight which
                    # meaningfully contributes to the probability distribution.
    changeprobability::Float64
    stochasticity::Float64
    particlesswitch::Array{Bool, 3} # wannabe ns x na x nparticles.
    weights::Array{Float64, 3} # wannabe ns x na x nparticles.
    Ps1a0s0::Array{Dict{Tuple{Int, Int}, Float64}, 1}
    counts::Array{Array{Float64,1}, 3}
    seed::Any
    rng::MersenneTwister
end
function TParticleFilter(;ns = 10, na = 4, nparticles = 6, changeprobability = .01,
                        stochasticity = .01, seed = 3)
    Neffthrs = nparticles/2. #Neffthrs = nparticles/10.
    particlesswitch = Array{Bool, 3}(undef, na, ns, nparticles)
    weights = Array{Float64, 3}(undef, na, ns, nparticles)
    Ps1a0s0 = [Dict{Tuple{Int, Int}, Float64}() for _ in 1:ns]
    counts = Array{Array{Float64,1}}(undef, na, ns, nparticles)
    rng = MersenneTwister(seed)
    [weights[a0, s0, i] = 1. /nparticles for a0 in 1:na for s0 in 1:ns for i in 1:nparticles]
    [counts[a0, s0, i] = zeros(ns) for a0 in 1:na for s0 in 1:ns for i in 1:nparticles]
    TParticleFilter(ns, na, nparticles, Neffthrs, changeprobability, stochasticity,
                    particlesswitch, weights, Ps1a0s0, counts, seed, rng)
end
export TParticleFilter
function updatet!(learnerT::TParticleFilter, s0, a0, s1, done)
    # if !done
    learnerT.particlesswitch[a0, s0, :] .= false
    stayterms = computestayterms(learnerT, s0, a0, s1)
    getweights!(learnerT, s0, a0, stayterms, 1. / learnerT.ns)
    sampleparticles!(learnerT, s0, a0, stayterms, 1. / learnerT.ns)
    Neff = 1. /sum(learnerT.weights[a0, s0, :] .^2)
    if Neff <= learnerT.Neffthrs; resample!(learnerT, s0, a0); end
    updatecounts!(learnerT, s0, a0, s1)
    computePs1a0s0!(learnerT, s0, a0)
    leakothers!(learnerT, a0, s0)
end
export updatet!
function computestayterms(learnerT::TParticleFilter, s0, a0, s1)
    stayterms = zeros(learnerT.nparticles)
    for i in 1:learnerT.nparticles
        stayterms[i] = (learnerT.stochasticity + learnerT.counts[a0, s0, i][s1])
        stayterms[i] /= sum(learnerT.stochasticity .+ learnerT.counts[a0, s0, i])
    end #switchterm = 1. / learnerT.ns
    stayterms#, switchterm
end
export computestayterms!
function getweights!(learnerT::TParticleFilter, s0, a0, stayterms, switchterm)
    for i in 1:learnerT.nparticles #firstratio = B(s + a(h_t-1)') / B(s + a(h_t-1)). secondratio = B(s + a(h_t=h_t-1 + 1)) / B(s)
        particleweightupdate = (1. - learnerT.changeprobability) * stayterms[i]
        particleweightupdate += learnerT.changeprobability * switchterm
        learnerT.weights[a0, s0, i] *= particleweightupdate
    end
    learnerT.weights[a0, s0, :] ./= sum(learnerT.weights[a0, s0, :]) # Normalize
end
export getweights!
function sampleparticles!(learnerT::TParticleFilter, s0, a0, stayterms, switchterm)
    for i in 1:learnerT.nparticles
        particlestayprobability = computeproposaldistribution(learnerT, stayterms[i], switchterm)
        r = rand(learnerT.rng) # Draw and possibly update
        if r > particlestayprobability
            learnerT.particlesswitch[a0, s0, i] = true
        end
    end
end
export sampleparticles!
function computeproposaldistribution(learnerT::TParticleFilter, istayterm, switchterm)
    particlestayprobability = 1. /(1. + ((learnerT.changeprobability * switchterm) /
                                ((1. - learnerT.changeprobability) * istayterm)))
end
export computeproposaldistribution
function updatecounts!(learnerT::TParticleFilter, s0, a0, s1)
    for i in 1:learnerT.nparticles
        if learnerT.particlesswitch[a0, s0, i] # if new hidden state
            learnerT.counts[a0, s0, i] =  zeros(learnerT.ns)
        end
        learnerT.counts[a0, s0, i][s1] += 1 # Last hidden state. +1 for s'
    end
end
export updatecounts!
function resample!(learnerT::TParticleFilter, s0, a0)
    tempcopyparticleweights = deepcopy(learnerT.weights[a0, s0, :])
    d = Categorical(tempcopyparticleweights)
    tempcopyparticlesswitch = copy(learnerT.particlesswitch[a0, s0, :])
    tempcopycounts = deepcopy(learnerT.counts[a0, s0, :])
    for i in 1:learnerT.nparticles
        sampledindex = rand(learnerT.rng, d)
        learnerT.particlesswitch[a0, s0, i] = copy(tempcopyparticlesswitch[sampledindex])
        learnerT.weights[a0, s0, i] = 1. /learnerT.nparticles
        learnerT.counts[a0, s0, i] = deepcopy(tempcopycounts[sampledindex])
    end
end
export resample!
function computePs1a0s0!(learnerT::TParticleFilter, s0, a0)
    thetasweighted = zeros(learnerT.nparticles, learnerT.ns)
    for i in 1:learnerT.nparticles
        # thetas = (learnerT.stochasticity .+ learnerT.counts[a0, s0, i])/sum(learnerT.stochasticity .+ learnerT.counts[a0, s0, i])
        thetas = (learnerT.stochasticity .+ learnerT.counts[a0, s0, i])
        thetas /= sum(learnerT.stochasticity .+ learnerT.counts[a0, s0, i])
        thetasweighted[i,:] = learnerT.weights[a0, s0, i] .* thetas
    end
    expectedvaluethetas = sum(thetasweighted, dims = 1)
    for s in 1:learnerT.ns
        learnerT.Ps1a0s0[s][(a0, s0)] = copy(expectedvaluethetas[s])
        #@show learnerT.Ps1a0s0[s][(a0, s0)]
    end
end
function leakothers!(learnerT::TParticleFilter, a0, s0)
    pairs = getstateactionpairs!(learnerT, a0, s0)
    for sa in pairs
        @show sa
        if !all(@. all(learnerT.counts[sa[1], sa[2], :] == [zeros(learnerT.ns)]))
            for i in 1:learnerT.nparticles
                @show learnerT.counts[sa[1], sa[2], i]
                r = rand(learnerT.rng) # Draw and possibly update
                if r < learnerT.changeprobability
                    println("Yes!")
                    learnerT.counts[sa[1], sa[2], i] = zeros(learnerT.ns)
                end
                @show learnerT.counts[sa[1], sa[2], i]
            end
        end
    end
end
function getstateactionpairs!(learnerT, a0, s0)
    pairs = collect(Iterators.product(1:learnerT.na, 1:learnerT.ns))[:]
    deleteat!(pairs, findall([x == (a0,s0) for x in pairs])[1])
    pairs
end
function defaultpolicy(learner::Union{TIntegrator, TLeakyIntegrator, TParticleFilter, TSmile, TVarSmile},
                        actionspace, buffer)
    RandomPolicy(actionspace)
end
function update!(learner::Union{TIntegrator, TLeakyIntegrator, TParticleFilter, TSmile, TVarSmile},
                buffer)
    a0 = buffer.actions[1]
    a1 = buffer.actions[2]
    s0 = buffer.states[1]
    s1 = buffer.states[2]
    r = buffer.rewards[1]
    done = buffer.done[1]
    updatet!(learner, s0, a0, s1, done)
end
export update!
