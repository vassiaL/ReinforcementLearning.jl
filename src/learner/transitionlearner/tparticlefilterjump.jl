using Distributions, SpecialFunctions, Random
"""
TParticleFilter for single Multinomial-Dirichlet task
"""
struct TParticleFilterJump <: TPs1a0s0
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
    terminalstates::Array{Int,1}
end
function TParticleFilterJump(;ns = 10, na = 4, nparticles = 6, changeprobability = .01,
                        stochasticity = .01, seed = 3)
    Neffthrs = nparticles/2. #Neffthrs = nparticles/10.
    particlesswitch = Array{Bool, 3}(undef, na, ns, nparticles)
    weights = Array{Float64, 3}(undef, na, ns, nparticles)
    Ps1a0s0 = [Dict{Tuple{Int, Int}, Float64}() for _ in 1:ns]
    [Ps1a0s0[sprime][(a, s)] = 1. /ns for sprime in 1:ns for a in 1:na for s in 1:ns]
    counts = Array{Array{Float64,1}}(undef, na, ns, nparticles)
    rng = MersenneTwister(seed)
    [weights[a0, s0, i] = 1. /nparticles for a0 in 1:na for s0 in 1:ns for i in 1:nparticles]
    [counts[a0, s0, i] = zeros(ns) for a0 in 1:na for s0 in 1:ns for i in 1:nparticles]
    TParticleFilterJump(ns, na, nparticles, Neffthrs, changeprobability, stochasticity,
                    particlesswitch, weights, Ps1a0s0, counts, seed, rng, Int[])
end
export TParticleFilterJump
function updatet!(learnerT::TParticleFilterJump, s0, a0, s1, done)
    learnerT.particlesswitch[a0, s0, :] .= false
    stayterms = computestayterms(learnerT, s0, a0, s1)
    getweights!(learnerT, s0, a0, stayterms, 1. / learnerT.ns)
    sampleparticles!(learnerT, s0, a0, stayterms, 1. / learnerT.ns)
    Neff = 1. /sum(learnerT.weights[a0, s0, :] .^2)
    if Neff <= learnerT.Neffthrs; resample!(learnerT, s0, a0); end
    updatecounts!(learnerT, s0, a0, s1)
    computePs1a0s0!(learnerT, s0, a0)
    computeterminalPs1a0s0!(learnerT, s1, done)
    # leakothers!(learnerT, s0, a0)
end
export updatet!
function updatecounts!(learnerT::TParticleFilterJump, s0, a0, s1)
    for i in 1:learnerT.nparticles
        if !learnerT.particlesswitch[a0, s0, i] # if it is not a surprise trial: Integrate
            learnerT.counts[a0, s0, i][s1] += 1 # +1 for s'
        end
    end
end
export updatecounts!
