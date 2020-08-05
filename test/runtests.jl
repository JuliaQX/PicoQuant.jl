using PicoQuant
using Test
using TestSetExtensions

include("test_utils.jl")

@testset "All the tests" begin
    @includetests ARGS
end
