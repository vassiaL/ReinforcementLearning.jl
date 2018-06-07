"""
    mutable struct DQN{Tnet,TnetT,ToptT,Topt}
        γ::Float64 = .99
        net::TnetT
        targetnet::Tnet = Flux.mapleaves(Flux.Tracker.data, deepcopy(net))
        policynet::Tnet = Flux.mapleaves(Flux.Tracker.data, net)
        updatetargetevery::Int64 = 500
        t::Int64 = 0
        updateevery::Int64 = 1
        opttype::ToptT = Flux.ADAM
        opt::Topt = opttype(Flux.params(net))
        startlearningat::Int64 = 10^3
        minibatchsize::Int64 = 32
        doubledqn::Bool = true
        nmarkov::Int64 = 1
        replaysize::Int64 = 10^4
"""
@with_kw mutable struct DQN{Tnet,TnetT,ToptT,Topt}
    γ::Float64 = .99
    net::TnetT
    policynet::Tnet = Flux.mapleaves(Flux.Tracker.data, net)
    targetnet::Tnet = deepcopy(policynet)
    updatetargetevery::Int64 = 500
    t::Int64 = 0
    updateevery::Int64 = 1
    opttype::ToptT = Flux.ADAM
    opt::Topt = opttype(Flux.params(net))
    startlearningat::Int64 = 10^3
    minibatchsize::Int64 = 32
    doubledqn::Bool = true
    nmarkov::Int64 = 1
    replaysize::Int64 = 10^4
    loss::Function = Flux.mse
end
export DQN
DQN(net; kargs...) = DQN(; net = Flux.gpu(net), kargs...)
function defaultbuffer(learner::Union{DQN, DeepActorCritic}, env, preprocessor)
    state = preprocessstate(preprocessor, getstate(env)[1])
    ArrayStateBuffer(capacity = typeof(learner) <: DQN ? learner.replaysize :
                                                         learner.nsteps + learner.nmarkov, 
                     arraytype = typeof(state).name.wrapper,
                     datatype = typeof(state[1]),
                     elemshape = size(state))
end
function defaultpolicy(learner::Union{DQN, DeepActorCritic}, buffer)
    if learner.nmarkov == 1
        typeof(learner) <: DQN ? EpsilonGreedyPolicy(.1) : SoftmaxPolicy()
    else
        a = buffer.states.data
        data = getindex(a, map(x -> 1:x, size(a)[1:end-1])..., 1:learner.nmarkov)
        NMarkovPolicy(typeof(learner) <: DQN ? EpsilonGreedyPolicy(.1) : 
                                               SoftmaxPolicy(),
                      ArrayCircularBuffer(data, learner.nmarkov, 0, 0, false))
    end
end

@with_kw struct NMarkovPolicy{Tpol, Tbuf}
    policy::Tpol = EpsilonGreedyPolicy(.1)
    buffer::Tbuf
end
@inline setepsilon(policy::NMarkovPolicy, val) = policy.policy.ϵ = val
@inline incrementepsilon(policy::NMarkovPolicy, val) = policy.policy.ϵ += val

huberloss(yhat, y::Flux.TrackedArray) = -2*dot(clamp.(yhat - y.data, -1, 1), y)/length(y)
export huberloss

@inline function selectaction(learner::Union{DQN, DeepActorCritic}, policy, state)
    if learner.nmarkov == 1
        selectaction(policy, learner.policynet(state))
    else
        push!(policy.buffer, state)
        selectaction(policy.policy, 
                     learner.policynet(nmarkovgetindex(policy.buffer, 
                                                endof(policy.buffer),
                                                learner.nmarkov)))
    end
end
function selecta(q, a)
    na, t = size(q)
    q[na * collect(0:t-1) .+ a]
end
import StatsBase
function update!(learner::DQN, b)
    learner.t += 1
    if learner.t % learner.updatetargetevery == 0
        learner.targetnet = deepcopy(learner.policynet)
    end
    (learner.t < learner.startlearningat || 
     learner.t % learner.updateevery != 0) && return
    indices = StatsBase.sample(1:length(b.rewards), learner.minibatchsize, 
                               replace = false)
    qa = learner.net(nmarkovgetindex(b.states, indices, learner.nmarkov))
    qat = learner.targetnet(nmarkovgetindex(b.states, indices + 1, learner.nmarkov))
    q = selecta(qa, b.actions[indices])
    rs = Float64[]
    for (k, i) in enumerate(indices)
        r, γeff = discountedrewards(b.rewards[i], b.done[i], learner.γ)
        if γeff > 0
            if learner.doubledqn
                r += γeff * qat[indmax(qa.data[:,k]), k]
            else
                r += γeff * maximum(qat[:, k])
            end
        end
        push!(rs, r)
    end
    Flux.back!(learner.loss(Flux.gpu(rs), q))
    learner.opt()
end
