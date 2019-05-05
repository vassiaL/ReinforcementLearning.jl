using Test, Random, Parameters
import nonstationaryrl: MDP, ConstantNumberSteps, ENV_RNG, SmallBackups,
TLeakyIntegrator, TParticleFilter, TSmile
# import ReinforcementLearning, ReinforcementLearningEnvironmentDiscrete
import ReinforcementLearning: PriorityQueue, RLSetup, EpsilonGreedyPolicy, learn!,
        getvalue, maximumbelowInf, addtoqueue!, dequeue!
import ReinforcementLearningEnvironmentDiscrete: DiscreteMaze
import nonstationaryrl.defaultpolicy
import nonstationaryrl.update!


""" Original implementation of Johanni """
@with_kw mutable struct SmallBackupsOriginal
    ns::Int64 = 10
    na::Int64 = 4
    γ::Float64 = .9
    initvalue::Float64 = Inf64
    maxcount::UInt64 = 3
    minpriority::Float64 = 1e-8
    M::Int64 = 1
    counter::Int64 = 0
    Q::Array{Float64, 2} = zeros(na, ns) .+ initvalue
    V::Array{Float64, 1} = zeros(ns) .+ (initvalue == Inf64 ? 0. : initvalue)
    U::Array{Float64, 1} = zeros(ns) .+ (initvalue == Inf64 ? 0. : initvalue)
    Nsa::Array{Int64, 2} = zeros(Int64, na, ns)
    Ns1a0s0::Array{Dict{Tuple{Int64, Int64}, Int64}, 1} = [Dict{Tuple{Int64, Int64}, Int64}() for _ in 1:ns]
    queue::PriorityQueue = PriorityQueue(Base.Order.Reverse, zip(Int64[], Float64[]))
end
export SmallBackupsOriginal
# defaultpolicy(learner::SmallBackupsOriginal, actionspace, buffer) = defaultpolicy(learner::SmallBackups, actionspace, buffer)
function defaultpolicy(learner::SmallBackupsOriginal, actionspace,
                       buffer)
    EpsilonGreedyPolicy(.1, actionspace, s -> getvalue(learner.Q, s))
end
export defaultpolicy
function processqueue!(learner::SmallBackupsOriginal)
    while length(learner.queue) > 0 && learner.counter < learner.maxcount
        learner.counter += 1
        s1 = dequeue!(learner.queue)
        ΔV = learner.V[s1] - learner.U[s1]
        learner.U[s1] = learner.V[s1]
        if length(learner.Ns1a0s0[s1]) > 0
            for ((a0, s0), n) in learner.Ns1a0s0[s1]
                if learner.Nsa[a0, s0] >= learner.M
                    learner.Q[a0, s0] += learner.γ * ΔV * n/learner.Nsa[a0, s0]
                    learner.V[s0] = maximumbelowInf(learner.Q[:, s0])
                    p = abs(learner.V[s0] - learner.U[s0])
                    if p > learner.minpriority; addtoqueue!(learner.queue, s0, p); end
                end
            end
        end
    end
    learner.counter = 0
end

function update!(learner::SmallBackupsOriginal, buffer)
    a0 = buffer.actions[1]
    a1 = buffer.actions[2]
    s0 = buffer.states[1]
    s1 = buffer.states[2]
    r = buffer.rewards[1]

    if buffer.done[1]
        learner.Nsa[a0, s0] += 1
        if learner.Q[a0, s0] == Inf; learner.Q[a0, s0] = 0; end
        if learner.Nsa[a0, s0] >= learner.M
            learner.Q[a0, s0] = (learner.Q[a0, s0] * (learner.Nsa[a0, s0] - 1) + r) /
                               learner.Nsa[a0, s0]
        end
        @show learner.Q[a0, s0]
    else
        learner.Nsa[a0, s0] += 1
        if haskey(learner.Ns1a0s0[s1], (a0, s0))
            learner.Ns1a0s0[s1][(a0, s0)] += 1
        else
            learner.Ns1a0s0[s1][(a0, s0)] = 1
        end
        if learner.Q[a0, s0] == Inf64; learner.Q[a0, s0] = 0.; end
        nextv = learner.γ * learner.U[s1]
        if learner.Nsa[a0, s0] >= learner.M
            learner.Q[a0, s0] = (learner.Q[a0, s0] * (learner.Nsa[a0, s0] - 1) +
                                r + nextv) / learner.Nsa[a0, s0]
        end
    end
    learner.V[s0] = maximumbelowInf(learner.Q[:, s0])
    p = abs(learner.V[s0] - learner.U[s0])
    if p > learner.minpriority; addtoqueue!(learner.queue, s0, p); end
    processqueue!(learner)
end



""" ------------------ """
function testSmallBackups()
    nsteps = 10^3

    println("Original")
    Random.seed!(ENV_RNG, 123)
    #mdp = MDP(ns = 4, na = 2, init = "random")
    env = DiscreteMaze(nx = 6, ny = 6)
    Random.seed!(123)
    # x = RLSetup(SmallBackupsOriginal(ns = mdp.observationspace.n, na = mdp.actionspace.n),
    #             mdp, ConstantNumberSteps(10^4));
    x = RLSetup(SmallBackupsOriginal(ns = env.mdp.observationspace.n, na = env.mdp.actionspace.n),
                 env, ConstantNumberSteps(nsteps));
    learn!(x)

    println("TIntegrator")
    Random.seed!(ENV_RNG, 123)
    env = DiscreteMaze(nx = 6, ny = 6)
    Random.seed!(123)
    y = RLSetup(SmallBackups(ns = env.mdp.observationspace.n, na = env.mdp.actionspace.n),
                            env, ConstantNumberSteps(nsteps));
    learn!(y)

    println("TLeaky")
    Random.seed!(ENV_RNG, 123)
    #mdp = MDP(ns = 4, na = 2, init = "random")
    env = DiscreteMaze(nx = 6, ny = 6)
    Random.seed!(123)
    # z = RLSetup(SmallBackups(Testimatetype = TLeakyIntegrator, etaleak = 1.,
    #             ns = mdp.observationspace.n, na = mdp.actionspace.n),
    #             mdp, ConstantNumberSteps(nsteps));
    z = RLSetup(SmallBackups(Testimatetype = TLeakyIntegrator, etaleak = 1.,
                            ns = env.mdp.observationspace.n, na = env.mdp.actionspace.n),
                            env, ConstantNumberSteps(nsteps));
    learn!(z)

    # Q values of original should be same as for TIntegrator
    @test all(y.learner.Q[:, 1:end-1] .== x.learner.Q[:, 1:end-1])

    # Q values of original should be approx same as for TLeakyIntegrator
    # with very low tolerance
    @test all(isapprox(z.learner.Q[:, 1:end-1], x.learner.Q[:, 1:end-1], atol = .00002))

    println("TParticleFilter")
    Random.seed!(ENV_RNG, 123)
    env = DiscreteMaze(nx = 6, ny = 6)
    Random.seed!(123)
    k = RLSetup(SmallBackups(Testimatetype = TParticleFilter,
                            ns = env.mdp.observationspace.n, na = env.mdp.actionspace.n),
                            env, ConstantNumberSteps(nsteps));
    learn!(k)

    # Q values of original should be approx same as for TParticleFilter
    @test all(isapprox(k.learner.Q[:, 1:end-1], y.learner.Q[:, 1:end-1], atol = .1))

    nsteps = 10^3
    println("TSmile")
    Random.seed!(ENV_RNG, 123)
    #mdp = MDP(ns = 4, na = 2, init = "random")
    env = DiscreteMaze(nx = 6, ny = 6)
    Random.seed!(123)
    m = RLSetup(SmallBackups(Testimatetype = TSmile,
                            ns = env.mdp.observationspace.n, na = env.mdp.actionspace.n),
                            env, ConstantNumberSteps(nsteps));
    learn!(m)

    # Q values of original should be approx same as for TSmile
    @test all(isapprox(m.learner.Q[:, 1:end-1], y.learner.Q[:, 1:end-1], atol = .1))
end
testSmallBackups()
