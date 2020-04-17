"""
    @with_kw struct MDPLearner
        mdp::MDP = MDP()
        γ::Float64 = .9
        policy::Array{Int64, 1} = ones(Int64, mdp.observationspace.n)
        values::Array{Float64, 1} = zeros(mdp.observationspace.n)

Used to solve `mdp` with discount factor `γ`.
"""
@with_kw struct MDPLearner{T}
    mdp::T
    γ::Float64 = .9
    policy::Array{Int64, 1} = ones(Int64, mdp.observationspace.n)
    values::Array{Float64, 1} = zeros(mdp.observationspace.n)
end
export MDPLearner

function defaultpolicy(learner::MDPLearner, actionspace, buffer)
    EpsilonGreedyPolicy(.1, 1:actionspace.n, s -> learner.policy[s])
end

# solve MDP

function get_optimal_policy_given_values!(mdplearner::MDPLearner)
    for state in findall(x -> x == 0, mdplearner.mdp.isterminal)
        mdplearner.policy[state], vmax = argmaxvalue(mdplearner, state)
    end
    return mdplearner.policy
end

function argmaxvalue(mdplearner, state)
    amax = 0; vmax = -Inf64
    for a in 1:mdplearner.mdp.actionspace.n
        v = mdplearner.mdp.reward[a, state] + mdplearner.γ *
                dot(mdplearner.mdp.trans_probs[a, state], mdplearner.values)
        if vmax < v
            vmax = v
            amax = a
        end
    end
    amax, vmax
end

function geteffectivetandr(mdplearner)
    trans_probs = []
    reward_immediate = zeros(mdplearner.mdp.observationspace.n)
    for state = 1:mdplearner.mdp.observationspace.n
        if size(mdplearner.mdp.trans_probs, 2) < state ||
            mdplearner.mdp.isterminal[state] == 1
            push!(trans_probs, SparseVector(mdplearner.mdp.observationspace.n, Int64[], Float64[]))
        else
            push!(trans_probs, mdplearner.mdp.trans_probs[mdplearner.policy[state], state])
        end
        # --- It used to be like this
        # reward[state] = mdplearner.mdp.reward[mdplearner.policy[state], state]
        # --- To work it shoul now be like this:
        reward[state] = mdplearner.mdp.reward.value[mdplearner.policy[state], state]
        # --- The following should now work for DeterministicNextStateReward and DeterministicStateActionReward
        # --- if one can import the function reward() from ReinforcementLearningEnvironmentDiscrete
        # reward_immediate[state] = reward(mdplearner.mdp.reward, state, mdplearner.policy[state], state)
        # reward(r::DeterministicNextStateReward, s, a, s′)
    end
    hcat(trans_probs...), reward_immediate
end

function get_values_given_policy!(mdplearner::MDPLearner)
    trans_probs, reward = geteffectivetandr(mdplearner)
    mdplearner.values[:] = get_value(reward, trans_probs, mdplearner.γ)
end

"""
    policy_iteration!(mdplearner::MDPLearner)

Solve MDP with policy iteration using [`MDPLearner`](@ref).
"""
function policy_iteration!(mdplearner::MDPLearner)
    oldpolicy = zeros(mdplearner.mdp.observationspace.n)
    while sum(abs.(oldpolicy - mdplearner.policy)) > 0
        oldpolicy[:] = mdplearner.policy[:]
        get_values_given_policy!(mdplearner)
        get_optimal_policy_given_values!(mdplearner)
    end
end
export policy_iteration!

function value_iteration!(mdplearner::MDPLearner; eps = 1.e-8)
    diff = 1.
    values = zeros(mdplearner.mdp.observationspace.n)
    while diff > eps
        for state in findall(x -> x == 0, mdplearner.mdp.isterminal)
            amax, values[state] = argmaxvalue(mdplearner, state)
        end
        diff = norm(values - mdplearner.values)
        mdplearner.values[:] = values[:]
    end
end
export value_iteration!

# utilities

function get_Q_values(mdplearner::MDPLearner)
    [mdplearner.mdp.reward[action, state] + mdplearner.γ * (transpose(mdplearner.values) * mdplearner.mdp.trans_probs[:, action, state])[1,1] for action = 1:mdplearner.mdp.actionspace.n, state = 1:mdplearner.mdp.observationspace.n]
end

function get_value(reward, trans_probs, γ)
    return (sparse(Matrix(1.0I, length(reward), length(reward))) -
            γ * transpose(trans_probs)) \ reward
end

# update!(::MDPLearner) = Nothing
update!(::MDPLearner) = nothing
update!(::MDPLearner, buffer) = nothing
