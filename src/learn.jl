# using nonstationaryrl
# using PyPlot
@inline function step!(rlsetup, a)
    @unpack learner, policy, buffer, preprocessor, environment, fillbuffer = rlsetup
    s0, r0, done0 = interact!(environment, a)
    #@show s0, r0, done0
    s, r, done = preprocess(preprocessor, s0, r0, done0)
    # @show s, r, done
    if fillbuffer; pushreturn!(buffer, r, done) end
    # @show buffer.states
    if done
        if fillbuffer; pushterminalstates!(buffer, s) end
        s0, = reset!(environment)
        # @show s0
        s = preprocessstate(preprocessor, s0)
        # @show s
    end
    if fillbuffer; pushstate!(buffer, s) end
    a = policy(s)
    # @show a
    if fillbuffer pushaction!(buffer, a) end
    # @show buffer.states
    # @show buffer.actions

    # println("--------------")
    # @show s, a, r, done
    s0, a, r, done
end
@inline function firststateaction!(rlsetup)
    @unpack learner, policy, buffer, preprocessor, environment, fillbuffer = rlsetup
    if isempty(buffer.actions)
        sraw, done = getstate(environment)
        if done; sraw, = reset!(environment); end
        s = preprocessstate(preprocessor, sraw)
        if fillbuffer; pushstate!(buffer, s) end
        a = policy(s)
        if fillbuffer; pushaction!(buffer, a) end
        a
    else
        buffer.actions[end]
    end
end

"""
    learn!(rlsetup)

Runs an [`rlsetup`](@ref RLSetup) with learning.
"""
function learn!(rlsetup)
    @unpack learner, buffer = rlsetup
    a = firststateaction!(rlsetup) #TODO: callbacks don't see first state action
    # @show a
    # t = 0
    # tplot = 0
    # start = time()
    # plotenv(rlsetup.environment)
    # fig = figure(); ax = gca()
    # v = nonstationaryrl.plotvalues(rlsetup, ax, sizeofmaze=(13,13))
    while true
        # t+=1
        # tplot+=1
        # @show t
        sraw, a, r, done = step!(rlsetup, a)
        if rlsetup.islearning; update!(learner, buffer); end
        for callback in rlsetup.callbacks
            callback!(callback, rlsetup, sraw, a, r, done)
        end
        # plotenv(rlsetup.environment)
        # if t == 100
        #      elapsedtime = time() - start
        #      @show elapsedtime
        #      t = 0
        #      start = time()
        # end
        # if tplot == 100
        #     v = nonstationaryrl.plotvalues(rlsetup, ax, sizeofmaze=(13,13))
        #     sleep(0.01)
        #      tplot = 0
        # end
        if isbreak!(rlsetup.stoppingcriterion, sraw, a, r, done); break; end
    end
end

"""
    run!(rlsetup)

Runs an [`rlsetup`](@ref RLSetup) without learning.
"""
function run!(rlsetup; fillbuffer = false)
    @unpack islearning = rlsetup
    rlsetup.islearning = false
    tmp = rlsetup.fillbuffer
    rlsetup.fillbuffer = fillbuffer
    learn!(rlsetup)
    rlsetup.islearning = islearning
    rlsetup.fillbuffer = tmp
end

export learn!, run!
