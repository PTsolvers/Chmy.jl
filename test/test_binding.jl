using Test
using Chmy

@testset "binding" begin
    @testset "constructor deduplicates keys" begin
        @scalars a b

        binding = Binding(a => 1, b => 2, a => 3)

        @test length(binding) == 2
        @test keys(binding) === (b, a)
        @test binding[a] == 3
        @test binding[b] == 2
    end
end
