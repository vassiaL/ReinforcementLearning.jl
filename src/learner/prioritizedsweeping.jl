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
    initvalueR::Float64
    initvalue::Float64
    maxcount::Int #UInt64
    minpriority::Float64
    M::Float64
    counter::Int
    Q::Array{Float64, 2}
    V::Array{Float64, 1}
    U::Array{Float64, 1}
    Restimate::TREstimate
    Testimate::TTEstimate
    queue::PriorityQueue
end
function SmallBackups(; ns = 10, na = 4, γ = .9, initvalueR = 1.,
    initvalue = initvalueR / (1. - γ),#initvalue = Inf64, #initvalue = 1. / (1. - γ), # initvalue = 0.
    maxcount = 3,
    minpriority = 1e-8, M = 1., counter = 0,
    Q = zeros(na, ns) .+ initvalue, V = zeros(ns) .+ (initvalue == Inf64 ? 0. : initvalue),
    U = zeros(ns) .+ (initvalue == Inf64 ? 0. : initvalue),
    Restimatetype = RIntegratorStateActionReward,
    Testimatetype = TIntegrator,
    queue = PriorityQueue(Base.Order.Reverse, zip(Int[], Float64[])),
    nparticles = 6, changeprobability = .01, stochasticity = .01, etaleak = .9,
    seedlearner = 3, msmile = 0.1)

    if Testimatetype == TIntegrator
        Testimate = TIntegrator(ns = ns, na = na)
    elseif Testimatetype == TParticleFilter
        Testimate = TParticleFilter(ns = ns, na = na, nparticles = nparticles,
                                    changeprobability = changeprobability,
                                    stochasticity = stochasticity,
                                    seed = seedlearner)
    elseif Testimatetype == TLeakyIntegrator
        Testimate = TLeakyIntegrator(ns = ns, na = na, etaleak = etaleak)
        M = M > etaleak ? etaleak : M
    elseif Testimatetype == TSmile
        Testimate = TSmile(ns = ns, na = na, m = msmile,
                            stochasticity = stochasticity)
    elseif Testimatetype == TVarSmile
        Testimate = TVarSmile(ns = ns, na = na, m = msmile,
                            stochasticity = stochasticity)
    end

    if Restimatetype == RIntegratorStateActionReward
        if Testimatetype == TIntegrator
            Restimate = REstimateDummy()
        else
            Restimate = RIntegratorStateActionReward(ns = ns, na = na,
                                                    initvalueR = initvalueR)
        end
    elseif Restimatetype == RIntegratorNextStateReward
        Restimate = RIntegratorNextStateReward(ns = ns, initvalueR = initvalueR)
    elseif Restimatetype == RLeakyIntegratorStateActionReward
        Restimate = RLeakyIntegrator(ns = ns, na = na, initvalueR = initvalueR,
                                    etaleak = etaleak)
    # elseif Restimatetype == RParticleFilter
    #     Restimate = RParticleFilter(ns = ns, na = na, nparticles = nparticles)
    end

    SmallBackups(ns, na, γ, initvalueR, initvalue, maxcount, minpriority, M, counter,
                Q, V, U, Restimate, Testimate, queue)
end
export SmallBackups
function defaultpolicy(learner::Union{SmallBackups, MonteCarlo}, actionspace,
                       buffer)
    EpsilonGreedyPolicy(.1, actionspace, s -> getvalue(learner.Q, s))
end
export defaultpolicy
function addtoqueue!(q, s, p)
    if haskey(q, s)
        if q[s] > p; q[s] = p; end
    else
        enqueue!(q, s, p)
    end
end
function processqueue!(learner::SmallBackups)
    @show learner.queue
    while length(learner.queue) > 0 && learner.counter < learner.maxcount
        learner.counter += 1
        s1 = dequeue!(learner.queue)
        @show s1
        ΔV = learner.V[s1] - learner.U[s1]
        @show learner.V[s1], learner.U[s1], ΔV
        learner.U[s1] = learner.V[s1]
        processqueueupdateq!(learner, s1, ΔV)
    end
    learner.counter = 0
end
 """ For Testimate that uses counts (Ns1a0s0) """
function processqueueupdateq!(learner::SmallBackups{TR, <:Union{TIntegrator, TLeakyIntegrator}} where TR,
                            s1, ΔV)
    if length(learner.Testimate.Ns1a0s0[s1]) > 0
        for ((a0, s0), n) in learner.Testimate.Ns1a0s0[s1]
        @show ((a0, s0), n)
            if n > 0. && learner.Testimate.Nsa[a0, s0] >= learner.M
                learner.Q[a0, s0] += learner.γ * ΔV * n/learner.Testimate.Nsa[a0, s0]
                learner.V[s0] = maximumbelowInf(learner.Q[:, s0])
                @show learner.V[s0]
                p = abs(learner.V[s0] - learner.U[s0])
                if p > learner.minpriority; addtoqueue!(learner.queue, s0, p); end
            end
        end
    end
end
""" For Testimate that uses probabilites (Ps1a0s0) """
function processqueueupdateq!(learner::SmallBackups{TR, <:Union{TParticleFilter, TSmile, TVarSmile}} where TR,
                            s1, ΔV)
    if length(learner.Testimate.Ps1a0s0[s1]) > 0
        for ((a0, s0), n) in learner.Testimate.Ps1a0s0[s1]
            @show ((a0, s0), n)
            if learner.Testimate.Ps1a0s0[s1][a0, s0] > 0.
                learner.Q[a0, s0] += learner.γ * ΔV * n
                learner.V[s0] = maximumbelowInf(learner.Q[:, s0])
                @show learner.V[s0]
                p = abs(learner.V[s0] - learner.U[s0])
                if p > learner.minpriority; addtoqueue!(learner.queue, s0, p); end
            end
        end
    end
end
""" Original small backups version : RDummy+TIntegrator or RLeaky+TLeaky"""
function updateq!(learner::Union{SmallBackups{REstimateDummy, TIntegrator}, SmallBackups{RLeakyIntegratorStateActionReward, TLeakyIntegrator}},
                a0, s0, s1, r, done)
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
""" Full backup  -- R[s,a]"""
function updateq!(learner::Union{SmallBackups{RIntegratorStateActionReward, TT} where TT,
                            SmallBackups{RLeakyIntegratorStateActionReward, <:Union{TParticleFilter, TSmile, TVarSmile}}},
                a0, s0, s1, r, done)
    if learner.Q[a0, s0] == Inf64; learner.Q[a0, s0] = 0.; end
    nextvs, nextps, nextstates = getnextstates(learner, a0, s0)
    learner.Q[a0, s0] = learner.Restimate.R[a0, s0] + learner.γ * sum(nextps .* nextvs)
end
""" Full backup  -- R[s']"""
function updateq!(learner::SmallBackups{RIntegratorNextStateReward, TT} where TT,
                 a0, s0, s1, r, done)
    if learner.Q[a0, s0] == Inf64; learner.Q[a0, s0] = 0.; end
    nextvs, nextps, nextstates = getnextstates(learner, a0, s0)
    Rbar = [learner.Restimate.R[s] for s in nextstates]
    @show Rbar
    learner.Q[a0, s0] = sum(nextps.* (Rbar + learner.γ * nextvs))
end
""" Get next states and probabilities -- Ps1a0s0 """
function getnextstates(learner::SmallBackups{TR, <:Union{TParticleFilter, TSmile, TVarSmile}} where TR, a0, s0)
    nextstates = [s for s in 1:learner.ns if haskey(learner.Testimate.Ps1a0s0[s],(a0,s0))]
    nextvs = [learner.U[s] for s in nextstates]
    @show nextvs
    nextps = [learner.Testimate.Ps1a0s0[s][a0, s0] for s in nextstates]
    @show nextps
    nextvs, nextps, nextstates
end
""" Get next states and probabilities -- Ns1a0s0 """
function getnextstates(learner::SmallBackups{TR, TLeakyIntegrator} where TR, a0, s0)
    nextstates = [s for s in 1:learner.ns if haskey(learner.Testimate.Ns1a0s0[s],(a0,s0))]
    # @show nextstates
    nextvs = [learner.U[s] for s in nextstates]
    # @show nextvs
    nextps = [learner.Testimate.Ns1a0s0[s][a0, s0]/learner.Testimate.Nsa[a0, s0] for s in nextstates]
    # @show nextps
    nextvs, nextps, nextstates
end
# function update!(learner::SmallBackups{TT, TR}, buffer) where {TT, TR}
function update!(learner::SmallBackups, buffer)
    a0 = buffer.actions[1]
    a1 = buffer.actions[2]
    s0 = buffer.states[1]
    s1 = buffer.states[2]
    r = buffer.rewards[1]
    done = buffer.done[1]
    sprime = done ? buffer.terminalstates[1] : s1

    println("--------------")
    @show a0, s0, s1, a1, r, done
    @show a0, s0, sprime, a1, r, done

    for s in 1:learner.ns
        @show learner.Testimate.counts[a0, s0, :]
        @show learner.Testimate.Ps1a0s0[s][(a0, s0)]
    end

    updatet!(learner.Testimate, s0, a0, sprime, done)
    updater!(learner.Restimate, s0, a0, sprime, r)
    @show learner.Restimate.R
    @show learner.Q
    updateq!(learner, a0, s0, sprime, r, done)
    @show learner.Q
    learner.V[s0] = maximumbelowInf(learner.Q[:, s0])
    @show learner.V[s0]
    p = abs(learner.V[s0] - learner.U[s0])
    if p > learner.minpriority; addtoqueue!(learner.queue, s0, p); end
    processqueue!(learner)
    for s in 1:size(learner.Q,2)
           @show learner.Q[:, s]
    end
end
export update!
