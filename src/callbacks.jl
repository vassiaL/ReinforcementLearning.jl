"""
    mutable struct ReduceEpsilonPerEpisode
        ϵ0::Float64
        counter::Int64

Reduces ϵ of an [`EpsilonGreedyPolicy`](@ref) after each episode.

In episode n, ϵ = ϵ0/n
"""
mutable struct ReduceEpsilonPerEpisode
    ϵ0::Float64
    counter::Int64
end
"""
    ReduceEpsilonPerEpisode()

Initialize callback.
"""
ReduceEpsilonPerEpisode() = ReduceEpsilonPerEpisode(0., 1)
function callback!(c::ReduceEpsilonPerEpisode, rlsetup, sraw, a, r, done)
    if done
        if c.counter == 1
            c.ϵ0 = rlsetup.policy.ϵ
        end
        c.counter += 1
        rlsetup.policy.ϵ = c.ϵ0 / c.counter
    end
end
export ReduceEpsilonPerEpisode

"""
    mutable struct ReduceEpsilonPerT
        ϵ0::Float64
        T::Int64
        n::Int64
        counter::Int64

Reduces ϵ of an [`EpsilonGreedyPolicy`](@ref) after every `T` steps.

After n * T steps, ϵ = ϵ0/n
"""
mutable struct ReduceEpsilonPerT
    ϵ0::Float64
    T::Int64
    n::Int64
    counter::Int64
end
"""
    ReduceEpsilonPerT()

Initialize callback.
"""
ReduceEpsilonPerT(T) = ReduceEpsilonPerT(0., T, 1, 1)
function callback!(c::ReduceEpsilonPerT, rlsetup, sraw, a, r, done)
    c.counter += 1
    if c.counter == c.T
        c.counter == 1
        if c.n == 1
            c.ϵ0 = rlsetup.policy.ϵ
        end
        c.n += 1
        rlsetup.policy.ϵ = c.ϵ0 / c.n
    end
end
export ReduceEpsilonPerT

"""
    mutable struct LinearDecreaseEpsilon
        start::Int64
        stop::Int64
        initval::Float64
        finalval::Float64
        t::Int64
        step::Float64

Linearly decrease ϵ of an [`EpsilonGreedyPolicy`](@ref) from `initval` until
step `start` to `finalval` at step `stop`.

Stepsize `step` = (finalval - initval)/(stop - start).
"""
mutable struct LinearDecreaseEpsilon
    start::Int64
    stop::Int64
    initval::Float64
    finalval::Float64
    t::Int64
    step::Float64
end
export LinearDecreaseEpsilon
"""
    LinearDecreaseEpsilon(start, stop, initval, finalval)
"""
function LinearDecreaseEpsilon(start, stop, initval, finalval)
    step = (finalval - initval)/(stop - start)
    LinearDecreaseEpsilon(start, stop, initval, finalval, 0, step)
end
@inline setepsilon(policy, val) = policy.ϵ = val
@inline incrementepsilon(policy, val) = policy.ϵ += val
@inline setepsilon(policy::NMarkovPolicy, val) = policy.policy.ϵ = val
@inline incrementepsilon(policy::NMarkovPolicy, val) = policy.policy.ϵ += val
function callback!(c::LinearDecreaseEpsilon, rlsetup, sraw, a, r, done)
    c.t += 1
    if c.t == 1
        setepsilon(rlsetup.policy, c.initval)
    elseif c.t >= c.start && c.t < c.stop
        incrementepsilon(rlsetup.policy, c.step)
    elseif c.t == c.stop
        setepsilon(rlsetup.policy, c.finalval)
    end
    nothing
end

"""
    mutable struct Progress
        steps::Int64
        laststopcountervalue::Int64

Show `steps` times progress information during learning.
"""
mutable struct Progress
    steps::Int64
    laststopcountervalue::Int64
end
"""
    Progress(steps = 10) = Progress(steps, 0)
"""
Progress(steps = 10) = Progress(steps, 0)
export Progress
progressunit(stop::ConstantNumberSteps) = "steps"
progressunit(stop::ConstantNumberEpisodes) = "episodes"
function callback!(c::Progress, rlsetup, sraw, a, r, done)
    stop = rlsetup.stoppingcriterion
    if stop.counter != c.laststopcountervalue && stop.counter % div(stop.N, c.steps) == 0
        c.laststopcountervalue = stop.counter
        lastvaluestring = join([getlastvaluestring(c) for c in rlsetup.callbacks])
        if lastvaluestring != ""
            lastvaluestring = "latest " * lastvaluestring
        end
        @info("$(now())\t $(lpad(stop.counter, 9))/$(stop.N) $(progressunit(stop))\t $lastvaluestring.")
    end
end

getlastvaluestring(c) = ""


mutable struct Episode
    N::Int64
    t::Int64
end
Episode(N) = Episode(N, 0)
function step!(c::Episode, done)
    if done
        c.t += 1
        c.t % c.N == 0
    else
        false
    end
end
mutable struct Step
    N::Int64
    t::Int64
end
Step(N) = Step(N, 0)
function step!(c::Step, done)
    c.t += 1
    c.t % c.N == 0
end

@with_kw mutable struct EvaluateGreedy{T,Tc,Tu}
    ingreedy::Bool = false
    callback::Tc
    rlsetupcallbacks::Array{Any, 1} = []
    rlsetuppolicyparameter::Float64 = 1
    rlsetupstoppingcriterion::Any = 1
    stoppingcriterion::T
    every::Tu = Episode(10)
    values::Array{Any, 1} = []
end
"""
    EvaluateGreedy(callback, stoppincriterion; every = Episode(10))

Evaluate an rlsetup greedily by leaving the normal learning loop and evaluating
the agent with `callback` until `stoppingcriterion` is met, at which point
normal learning is resumed. This is done `every` Nth Episode (where N = 10 by
default) or every Nth Step (e.g. `every = Step(10)`).

Example:

    eg = EvaluateGreedy(EvaluationPerEpisode(TotalReward(), returnmean = true),
                        ConstantNumberEpisodes(10), every = Episode(100))
    rlsetup = RLSetup(learner, environment, stoppingcriterion, callbacks = [eg])
    learn!(rlsetup)
    getvalue(eg)

Leaves the learning loop every 100th episode to estimate the average total reward
per episode, by running a greedy policy for 10 episodes.
"""
function EvaluateGreedy(callback, stoppingcriterion; every = Episode(10))
    EvaluateGreedy(callback = callback, stoppingcriterion = stoppingcriterion,
                   every = every)
end


function callback!(c::EvaluateGreedy, rlsetup, sraw, a, r, done)
    if c.ingreedy
        callback!(c.callback, rlsetup, sraw, a, r, done)
        if isbreak!(c.stoppingcriterion, sraw, a, r, done)
            push!(c.values, getvalue(c.callback))
            reset!(c.callback)
            c.ingreedy = false
            rlsetup.islearning = true
            rlsetup.fillbuffer = true
            rlsetup.callbacks = c.rlsetupcallbacks
            ungreedify!(rlsetup.policy, c.rlsetuppolicyparameter)
            rlsetup.stoppingcriterion = c.rlsetupstoppingcriterion
        end
    end
    if !c.ingreedy && step!(c.every, done)
        c.ingreedy = true
        rlsetup.islearning = false
        rlsetup.fillbuffer = false
        c.rlsetupcallbacks = rlsetup.callbacks
        c.rlsetupstoppingcriterion = deepcopy(rlsetup.stoppingcriterion)
        rlsetup.stoppingcriterion = typeof(rlsetup.stoppingcriterion).name.wrapper(typemax(Int))
        rlsetup.callbacks = [c]
        c.rlsetuppolicyparameter = greedify!(rlsetup.policy)
    end
end
getvalue(c::EvaluateGreedy) = c.values

export EvaluateGreedy, Step, Episode
function greedify!(p::EpsilonGreedyPolicy)
    tmp = p.ϵ
    p.ϵ = 0
    tmp
end
ungreedify!(p::EpsilonGreedyPolicy, ϵ) = p.ϵ = ϵ
function greedify!(p::SoftmaxPolicy)
    tmp = p.β
    p.β = Inf
    tmp
end
ungreedify!(p::SoftmaxPolicy, β) = p.β = β

import FileIO:save
"""
    @with_kw struct SaveLearner{T}
        every::T = Step(10^3)
        filename::String = tempname()

Save learner every Nth Step (or Nth Episode) to filename_i.jld2, where i is the
step (or episode) at which the learner is saved.
"""
@with_kw struct SaveLearner{T}
    every::T = Step(10^3)
    filename::String = tempname()
end
export SaveLearner
function callback!(c::SaveLearner, rlsetup, sraw, a, r, done)
    if step!(c.every, done)
        save(c.filename * "_$(c.every.t).jld2",
             Dict("learner" => rlsetup.learner))
    end
end

"""
    struct RecordAll
        rewards::Array{Float64, 1}
        actions::Array{Int64, 1}
        states::Array{Int64, 1}
        done::Array{Bool, 1}

Records everything.
"""
struct RecordAll
    rewards::Array{Float64, 1}
    actions::Array{Int64, 1}
    states::Array{Any, 1}
    done::Array{Bool, 1}
end
"""
    RecordAll()

Initializes with empty arrays.
"""
RecordAll() = RecordAll(Float64[], Int64[], [], Bool[])
function callback!(p::RecordAll, rlsetup, sraw, a, r, done)
    push!(p.rewards, r)
    push!(p.actions, deepcopy(a))
    push!(p.states, deepcopy(sraw))
    push!(p.done, done)
end
function reset!(p::RecordAll)
    empty!(p.rewards); empty!(p.actions); empty!(p.states); empty!(p.done)
end
getvalue(p::RecordAll) = p
export RecordAll

"""
    struct AllRewards
        rewards::Array{Float64, 1}

Records all rewards.
"""
struct AllRewards
    rewards::Array{Float64, 1}
end
"""
    AllRewards()

Initializes with empty array.
"""
AllRewards() = AllRewards(Float64[])
function callback!(p::AllRewards, rlsetup, sraw, a, r, done)
    push!(p.rewards, r)
end
function reset!(p::AllRewards)
    empty!(p.rewards)
end
getvalue(p::AllRewards) = p.rewards
export AllRewards


"""
    mutable struct Visualize
        plot
        wait::Float64
"""
mutable struct Visualize
    wait::Float64
end
"""
    Visualize(; wait = .15)

A callback to be used in an `RLSetup` to visualize an environment during
running or learning.
"""
Visualize(; wait = .15) = Visualize(wait)
export Visualize
function callback!(c::Visualize, rlsetup, s, a, r, done)
    plotenv(rlsetup.environment)
    sleep(c.wait)
end
plotenv(env) = warn("Visualization not implemented for environments of type $(typeof(env)).")


"""
Callback for Transition estimation (prioritizedsweeping or transitionlearners)
"""
struct RecordTransitionEstimation
    Nsa_history::Array{Any, 1} # It is Any, so that it can be used both for TEstimateIntegrator (Int) and TEstimateLeakyIntegrator (Float64)
    Ns1a0s0_history::Array{Array{Float64,1}, 1}
    Ps1a0s0_history::Array{Array{Float64,1}, 1}
end
RecordTransitionEstimation() = RecordTransitionEstimation([], Array{Array{Float64,1}}(undef, 1), Array{Array{Float64,1}}(undef, 1))
function callback!(p::RecordTransitionEstimation, rlsetup, sraw, a, r, done)

    if in(:Testimate, fieldnames(typeof(rlsetup.learner)))
        learnervariable = learner.Testimate
    else
        learnervariable = learner
    end

    a0 = rlsetup.buffer.actions[1]
    s0 = rlsetup.buffer.states[1]
    if in(:Ns1a0s0, fieldnames(typeof(rlsetup.learnervariable))) # TEstimateIntegrator TEstimateLeakyIntegrator
        Nsprimea0s0 = zeros(rlsetup.learner.ns)
        [ Nsprimea0s0[s] = rlsetup.learnervariable.Ns1a0s0[s][a0, s0] for s in 1:rlsetup.learner.ns if haskey(rlsetup.learnervariable.Ns1a0s0[s], (a0, s0)) ]
        if !isassigned(p.Ns1a0s0_history) # If very first step
            p.Ns1a0s0_history[1] = deepcopy(Nsprimea0s0)
        else
            push!(p.Ns1a0s0_history, deepcopy(Nsprimea0s0))
        end
        push!(p.Nsa_history, deepcopy(rlsetup.learnervariable.Nsa[a0, s0]))

    elseif in(:Ps1a0s0, fieldnames(typeof(rlsetup.learnervariable))) # TEstimateParticleFilter
        Psprimea0s0 = [rlsetup.learnervariable.Ps1a0s0[s][a0, s0] for s in 1:rlsetup.learner.ns]
        if !isassigned(p.Ps1a0s0_history) # If very first step
            p.Ps1a0s0_history[1] = deepcopy(Psprimea0s0)
        else
            push!(p.Ps1a0s0_history, deepcopy(Psprimea0s0))
        end
    end
end
function reset!(p::RecordTransitionEstimation)
    empty!(p.Nsa_history); empty!(p.Ns1a0s0_history); empty!(p.Ps1a0s0_history)
end
getvalue(p::RecordTransitionEstimation) = p
export RecordTransitionEstimation


"""
Callback for Reward estimation (prioritizedsweeping or rewardslearners)
"""
struct RecordRewardEstimation
    R_history::Array{Float64, 1}
end
RecordRewardEstimation() = RecordRewardEstimation(Float64[])
function callback!(p::RecordRewardEstimation, rlsetup, sraw, a, r, done)
    if in(:Restimate, fieldnames(typeof(rlsetup.learner)))
        learnervariable = learner.Restimate
    else
        learnervariable = learner
    end
    if in(:R, fieldnames(typeof(rlsetup.learnervariable))) # REstimateIntegrator and REstimateLeakyIntegrator
        push!(p.R_history, deepcopy(rlsetup.learnervariable.R[rlsetup.buffer.actions[1], rlsetup.buffer.states[1]]))
    end
end
function reset!(p::RecordRewardEstimation)
    empty!(p.R_history)
end
getvalue(p::RecordRewardEstimation) = p
export RecordRewardsEstimation

struct RecordEnvironmentTransitions
    trans_probs_history::Array{Array{Float64,1}, 1}
    switchflag::Array{Bool, 1}
end
RecordEnvironmentTransitions() = RecordEnvironmentTransitions(Array{Array{Float64,1}}(undef, 1), Bool[])
function callback!(p::RecordEnvironmentTransitions, rlsetup, sraw, a, r, done)

    if in(:trans_probs, fieldnames(typeof(rlsetup.environment))) # MDP
        envvariable = rlsetup.environment.trans_probs
    elseif in(:mdp, fieldnames(typeof(rlsetup.environment))) # ChangeMDP
        push!(p.switchflag, deepcopy(rlsetup.environment.switchflag))
        envvariable = rlsetup.environment.mdp.trans_probs
    elseif in(:discretemaze, fieldnames(typeof(rlsetup.environment))) # ChangeDiscreteMaze
        envvariable = rlsetup.environment.discretemaze.mdp.trans_probs
    end
    a0 = rlsetup.buffer.actions[1]
    s0 = rlsetup.buffer.states[1]
    if !isassigned(p.trans_probs_history) # If very first step
        p.trans_probs_history[1] = deepcopy(envvariable[a0, s0])
    else
        push!(p.trans_probs_history, deepcopy(envvariable[a0, s0]))
    end
end
function reset!(p::RecordEnvironmentTransitions)
    empty!(p.trans_probs_history); empty!(p.switchflag)
end
getvalue(p::RecordEnvironmentTransitions) = p
export RecordEnvironmentTransitions
