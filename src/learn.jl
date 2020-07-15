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
        # @show r
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
    # rsum = 0
    # start = time()
    # plotenv(rlsetup.environment)
    # sizeofmaze = (10,10)
    # sizeofmaze = (5, 12)
    # fig = figure(); ax = gca()
    # v = nonstationaryrl.plotvalues(rlsetup, ax, sizeofmaze=(10,10))
    while true
        # t+=1
        # tplot+=1
        # @show t
        sraw, a, r, done = step!(rlsetup, a)
        # @show sraw, a, r, done
        # rsum+=r
        # plotenv(rlsetup.environment)
        if rlsetup.islearning; update!(learner, buffer); end
        for callback in rlsetup.callbacks
            callback!(callback, rlsetup, sraw, a, r, done)
        end
        # if t == 100
        #      elapsedtime = time() - start
        #      @show elapsedtime
        #      t = 0
        #      start = time()
        # end
        # if tplot == 500 #50
        #     # Vvalues = round.(transpose(reshape(rlsetup.learner.V,(10,10))), digits=2)
        #     Vvalues = round.(reshape(rlsetup.learner.V, sizeofmaze), digits=2)
        #
        #     @show Vvalues
            # @show rlsetup.learner.Q[:, 27]
            # @show rlsetup.learner.Testimate.Ps1a0s0[28][1, 27]
            # @show rlsetup.learner.Testimate.Ps1a0s0[28][2, 27]
            # @show rlsetup.learner.Testimate.Ps1a0s0[28][3, 27]
            # @show rlsetup.learner.Testimate.Ps1a0s0[28][4, 27]
            # # ----
            # @show rlsetup.learner.Testimate.Ps1a0s0[74][2, 27]
            # @show rlsetup.learner.Testimate.Ps1a0s0[79][2, 27]
            # @show rlsetup.learner.Testimate.Ps1a0s0[43][2, 27]
            # @show rlsetup.learner.Q[:, 29]
            # @show rlsetup.learner.Testimate.Ps1a0s0[28][1, 29]
            # @show rlsetup.learner.Testimate.Ps1a0s0[28][2, 29]
            # @show rlsetup.learner.Testimate.Ps1a0s0[28][3, 29]
            # @show rlsetup.learner.Testimate.Ps1a0s0[28][4, 29]
            # @show rlsetup.learner.Q[:, 18]
            # @show rlsetup.learner.Testimate.Ps1a0s0[28][1, 18]
            # @show rlsetup.learner.Testimate.Ps1a0s0[28][2, 18]
            # @show rlsetup.learner.Testimate.Ps1a0s0[28][3, 18]
            # @show rlsetup.learner.Testimate.Ps1a0s0[28][4, 18]
            # @show rlsetup.learner.Q[:, 13]
            # @show rlsetup.learner.Testimate.Ps1a0s0[14][1, 13]
            # @show rlsetup.learner.Testimate.Ps1a0s0[14][2, 13]
            # @show rlsetup.learner.Testimate.Ps1a0s0[14][3, 13]
            # @show rlsetup.learner.Testimate.Ps1a0s0[14][4, 13]
            # @show rlsetup.learner.Q[:, 78]
            # @show rlsetup.learner.Testimate.Ps1a0s0[79][1, 78]
            # @show rlsetup.learner.Testimate.Ps1a0s0[79][2, 78]
            # @show rlsetup.learner.Testimate.Ps1a0s0[79][3, 78]
            # @show rlsetup.learner.Testimate.Ps1a0s0[79][4, 78]
            # @show rlsetup.learner.Q[:, 74]
            # @show rlsetup.learner.Testimate.Ps1a0s0[75][1, 74]
            # @show rlsetup.learner.Testimate.Ps1a0s0[75][2, 74]
            # @show rlsetup.learner.Testimate.Ps1a0s0[75][3, 74]
            # @show rlsetup.learner.Testimate.Ps1a0s0[75][4, 74]
            # @show rlsetup.learner.Q[:, 69]
            # @show rlsetup.learner.Testimate.Ps1a0s0[79][1, 69]
            # @show rlsetup.learner.Testimate.Ps1a0s0[79][2, 69]
            # @show rlsetup.learner.Testimate.Ps1a0s0[79][3, 69]
            # @show rlsetup.learner.Testimate.Ps1a0s0[79][4, 69]
            #---
            # @show rlsetup.learner.Q[:, 19]
            # @show rlsetup.learner.Testimate.Ps1a0s0[20][1, 19]
            # @show rlsetup.learner.Testimate.Ps1a0s0[20][2, 19]
            # @show rlsetup.learner.Testimate.Ps1a0s0[20][3, 19]
            # @show rlsetup.learner.Testimate.Ps1a0s0[20][4, 19]
            # @show rlsetup.learner.Q[:, 25]
            # @show rlsetup.learner.Testimate.Ps1a0s0[20][1, 25]
            # @show rlsetup.learner.Testimate.Ps1a0s0[20][2, 25]
            # @show rlsetup.learner.Testimate.Ps1a0s0[20][3, 25]
            # @show rlsetup.learner.Testimate.Ps1a0s0[20][4, 25]
            # @show rlsetup.learner.Q[:, 15]
            # @show rlsetup.learner.Testimate.Ps1a0s0[20][1, 15]
            # @show rlsetup.learner.Testimate.Ps1a0s0[20][2, 15]
            # @show rlsetup.learner.Testimate.Ps1a0s0[20][3, 15]
            # @show rlsetup.learner.Testimate.Ps1a0s0[20][4, 15]
            # @show [rlsetup.learner.Testimate.Ps1a0s0[s][2, 27] for s in  1:rlsetup.learner.Testimate.ns]
            # @show rlsetup.learner.Q[:, 88]
              # v = nonstationaryrl.plotvalues(rlsetup, ax, sizeofmaze=sizeofmaze)
            # sleep(0.01)
         #    @show t
         #    # @show sraw, a, r, done
         #     @show rsum
         #    tplot = 0
         # end
        # if t>=5*10^4
        #     @show t
        #     @show sraw, a, r, done
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
