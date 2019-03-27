"""
    mutable struct SmallBackups{TREstimate,TTEstimate} <: AbstractReinforcementLearner
        ns::Int = 10
        na::Int = 4
        γ::Float64 = .9
        initvalue::Float64 = Inf64
        maxcount::UInt = 3
        minpriority::Float64 = 1e-8
        M::Int = 1
        counter::Int = 0
        Q::Array{Float64, 2} = zeros(na, ns) .+ initvalue
        V::Array{Float64, 1} = zeros(ns) .+ (initvalue == Inf64 ? 0. : initvalue)
        U::Array{Float64, 1} = zeros(ns) .+ (initvalue == Inf64 ? 0. : initvalue)
        Restimate::TREstimate
        Testimate::TTEstimate
        queue::PriorityQueue = PriorityQueue(Base.Order.Reverse, zip(Int[], Float64[]))

See [Harm Van Seijen, Rich Sutton ; Proceedings of the 30th International Conference on Machine Learning, PMLR 28(3):361-369, 2013.](http://proceedings.mlr.press/v28/vanseijen13.html)

`maxcount` defines the maximal number of backups per action, `minpriority` is
the smallest priority still added to the queue.
"""

mutable struct SmallBackups{TREstimate,TTEstimate}
    ns::Int
    na::Int
    γ::Float64
    initvalue::Float64
    maxcount::Int #UInt64
    minpriority::Float64
    M::Int
    counter::Int
    Q::Array{Float64, 2}
    V::Array{Float64, 1}
    U::Array{Float64, 1}
    Restimate::TREstimate
    Testimate::TTEstimate
    queue::PriorityQueue
end
function SmallBackups(; ns = 10, na = 4, γ = .9, initvalue = Inf64, maxcount = 3, minpriority = 1e-8, M = 1, counter = 0,
    Q = zeros(na, ns) .+ initvalue, V = zeros(ns) .+ (initvalue == Inf64 ? 0. : initvalue),
    U = zeros(ns) .+ (initvalue == Inf64 ? 0. : initvalue), Restimatetype = RIntegrator,
    Testimatetype = TIntegrator,
    queue = PriorityQueue(Base.Order.Reverse, zip(Int[], Float64[])),
    nparticles = 6, stayprobability = .999, stochasticity = .01, etaleak = .9, seedparticlefilter = 3)

    if Testimatetype == TIntegrator
        Testimate = TIntegrator(ns = ns, na = na)
    elseif Testimatetype == TParticleFilter
        Testimate = TParticleFilter(ns = ns, na = na, nparticles = nparticles, stayprobability = stayprobability, stochasticity = stochasticity, seed = seedparticlefilter)
    elseif Testimatetype == TIntegrator
        Testimate = TIntegrator(ns = ns, na = na, etaleak = etaleak)
    end

    if Restimatetype == RIntegrator
        if Testimatetype == TIntegrator
            Restimate = REstimateDummy()
        else
            Restimate = RIntegrator(ns = ns, na = na)
        end
    elseif Restimatetype == RLeakyIntegrator
        Restimate = RLeakyIntegrator(ns = ns, na = na, etaleak = etaleak)
    elseif Restimatetype == RParticleFilter
        Restimate = RParticleFilter(ns = ns, na = na, nparticles = nparticles)
    end

    SmallBackups(ns, na, γ, initvalue, maxcount, minpriority, M, counter, Q, V, U, Restimate, Testimate, queue)
end
export SmallBackups
function defaultpolicy(learner::Union{SmallBackups, MonteCarlo}, actionspace,
                       buffer)
    EpsilonGreedyPolicy(.1, actionspace, s -> getvalue(learner.Q, s))
end
function addtoqueue!(q, s, p)
    if haskey(q, s)
        if q[s] > p; q[s] = p; end
    else
        enqueue!(q, s, p)
    end
end
function processqueue!(learner::SmallBackups)
    # Main function to process Queue for any SmallBackups learner. Calls processqueueupdateq!
    while length(learner.queue) > 0 && learner.counter < learner.maxcount
        learner.counter += 1
        s1 = dequeue!(learner.queue)
        ΔV = learner.V[s1] - learner.U[s1]
        learner.U[s1] = learner.V[s1]
        processqueueupdateq!(learner, s1, ΔV)
    end
    learner.counter = 0
end
function processqueueupdateq!(learner::Union{SmallBackups{REstimateDummy, TIntegrator}, SmallBackups{RLeakyIntegrator, TIntegrator}, SmallBackups{RIntegrator, TIntegrator}}, s1, ΔV)
    # Queue update Q for RDummy+TIntegrator or RLeaky+TLeaky or RIntegrator+TLeaky
    if length(learner.Testimate.Ns1a0s0[s1]) > 0
        for ((a0, s0), n) in learner.Testimate.Ns1a0s0[s1]
            if learner.Testimate.Nsa[a0, s0] >= learner.M
                learner.Q[a0, s0] += learner.γ * ΔV * n/learner.Testimate.Nsa[a0, s0]
                learner.V[s0] = maximumbelowInf(learner.Q[:, s0])
                p = abs(learner.V[s0] - learner.U[s0])
                if p > learner.minpriority; addtoqueue!(learner.queue, s0, p); end
            end
        end
    end
end
function processqueueupdateq!(learner::SmallBackups{<:Union{RIntegrator, RLeakyIntegrator, RParticleFilter}, TParticleFilter}, s1, ΔV)
    # Queue update Q for RIntegrator+TParticle or RLeaky+TParticle
    if length(learner.Testimate.Ps1a0s0[s1]) > 0
        for ((a0, s0), n) in learner.Testimate.Ps1a0s0[s1]
            learner.Q[a0, s0] += learner.γ * ΔV * n
            learner.V[s0] = maximumbelowInf(learner.Q[:, s0])
            p = abs(learner.V[s0] - learner.U[s0])
            if p > learner.minpriority; addtoqueue!(learner.queue, s0, p); end
        end
    end
end
function updateq!(learner::Union{SmallBackups{REstimateDummy, TIntegrator}, SmallBackups{RLeakyIntegrator, TIntegrator}}, a0, s0, s1, r, done)
    # Either RDummy+Tintegrator or RLeaky+TLeaky
    if done
        if learner.Q[a0, s0] == Inf; learner.Q[a0, s0] = 0; end
        if learner.Testimate.Nsa[a0, s0] >= learner.M
            learner.Q[a0, s0] = (learner.Q[a0, s0] * (learner.Testimate.Nsa[a0, s0] - 1) + r) /
                               learner.Testimate.Nsa[a0, s0]
        end
    else
        if learner.Q[a0, s0] == Inf64; learner.Q[a0, s0] = 0.; end
        nextv = learner.γ * learner.U[s1]
        if learner.Testimate.Nsa[a0, s0] >= learner.M
            learner.Q[a0, s0] = (learner.Q[a0, s0] * (learner.Testimate.Nsa[a0, s0] - 1) +
                                r + nextv) / learner.Testimate.Nsa[a0, s0]
        end
    end
end
""" Full backup """
function updateq!(learner::SmallBackups{<:Union{RIntegrator, RLeakyIntegrator, RParticleFilter}, TParticleFilter}, a0, s0, s1, r, done)
    # Full backup: TParticle
    if done
        if learner.Q[a0, s0] == Inf; learner.Q[a0, s0] = 0; end
        learner.Q[a0, s0] = learner.Restimate.R[a0, s0]
    else
        if learner.Q[a0, s0] == Inf64; learner.Q[a0, s0] = 0.; end
        nextstates = [s for s in 1:learner.ns if haskey(learner.Testimate.Ps1a0s0[s],(a0,s0))]
        nextvs = [learner.U[s] for s in nextstates]
        nextps = [learner.Testimate.Ps1a0s0[s][a0, s0] for s in nextstates]
        learner.Q[a0, s0] = learner.Restimate.R[a0, s0] + learner.γ * sum(nextps .* nextvs)
    end
end
# function update!(learner::SmallBackups{TT, TR}, buffer) where {TT, TR}
function update!(learner::SmallBackups, buffer)
    a0 = buffer.actions[1]
    a1 = buffer.actions[2]
    s0 = buffer.states[1]
    s1 = buffer.states[2]
    r = buffer.rewards[1]
    updatet!(learner.Testimate, s0, a0, s1)
    if typeof(learner.Restimate) == RParticleFilter
        updater!(learner.Restimate, s0, a0, r, learner.Testimate.particlesswitch, learner.Testimate.weights, learner.Testimate.counts)
    else
        updater!(learner.Restimate, s0, a0, r)
    end
    updateq!(learner, a0, s0, s1, r, buffer.done[1])
    learner.V[s0] = maximumbelowInf(learner.Q[:, s0])
    p = abs(learner.V[s0] - learner.U[s0])
    if p > learner.minpriority; addtoqueue!(learner.queue, s0, p); end
    processqueue!(learner)
end
export update!
