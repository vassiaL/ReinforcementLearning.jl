using ReinforcementLearning
import ReinforcementLearning: getactionprobabilities, selectaction, update!
using Base.Test

@testset "ϵ-greedy policies" begin include("epsilongreedypolicies.jl") end
@testset "traces" begin include("traces.jl") end
@testset "tdlearning" begin include("tdlearning.jl") end
@testset "mdp solver" begin include("learnmdp.jl") end
@testset "learn" begin include("learn.jl") end
@testset "linear function approximation" begin include("linfuncapprox.jl") end
@testset "preprocessor" begin include("preprocessors.jl") end
@testset "buffers" begin include("buffer.jl") end
@testset "monte carlo" begin include("montecarlo.jl") end
@testset "small backups" begin include("smallbackups.jl") end
@testset "losses" begin include("losses.jl") end
