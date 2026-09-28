using Test
using Chmy

@testset "differences" begin
    @scalars f
    i = 𝑖

    @testset "constructors" begin
        @test (@inferred Difference()) === CentralDifference{3}()
        @test (@inferred Difference{5}()) === CentralDifference{5}()
        @test (@inferred StaggeredDifference()) === StaggeredCentralDifference{2}()
        @test (@inferred StaggeredDifference{4}()) === StaggeredCentralDifference{4}()
        @test (@inferred CentralDifference()) === Difference{3, 0}()
        @test (@inferred StaggeredCentralDifference()) === StaggeredDifference{2, 0}()
        @test (@inferred ForwardDifference()) === Difference{2, 1 // 2}()
        @test (@inferred BackwardDifference()) === Difference{2, -1 // 2}()
        @test (@inferred StaggeredForwardDifference()) === StaggeredDifference{2, 1 // 1}()
        @test (@inferred StaggeredBackwardDifference()) === StaggeredDifference{2, -1 // 1}()
        @test (@inferred ForwardDifference{4}()) === Difference{4, 3 // 2}()
        @test (@inferred BackwardDifference{4}()) === Difference{4, -3 // 2}()
        @test (@inferred StaggeredForwardDifference{3}()) === StaggeredDifference{3, 3 // 2}()
        @test (@inferred StaggeredBackwardDifference{3}()) === StaggeredDifference{3, -3 // 2}()
        @test CentralDifference{5}() isa AbstractDifference
        @test StaggeredForwardDifference{3}() isa AbstractDifference
    end

    @testset "nodes and weights" begin
        @test (@inferred stencil_nodes(Difference{3, -1}())) === (-2, -1, 0)
        @test (@inferred stencil_nodes(Difference{3, 0}())) === (-1, 0, 1)
        @test (@inferred stencil_nodes(Difference{3, 1}())) === (0, 1, 2)
        @test (@inferred stencil_weights(Difference{3, -1}())) === (1 // 2, -2, 3 // 2)
        @test (@inferred stencil_weights(Difference{3, 0}())) === (-1 // 2, 0, 1 // 2)
        @test (@inferred stencil_weights(Difference{3, 1}())) === (-3 // 2, 2, -1 // 2)
        @test (@inferred stencil_nodes(CentralDifference{5}())) === (-2, -1, 0, 1, 2)
        @test (@inferred stencil_weights(CentralDifference{5}())) === (1 // 12, -2 // 3, 0, 2 // 3, -1 // 12)
        @test (@inferred stencil_nodes(StaggeredCentralDifference())) === (-1 // 2, 1 // 2)
        @test (@inferred stencil_weights(StaggeredCentralDifference())) === (-1, 1)
        @test (@inferred stencil_nodes(StaggeredCentralDifference{4}())) === (-3 // 2, -1 // 2, 1 // 2, 3 // 2)
        @test (@inferred stencil_weights(StaggeredCentralDifference{4}())) === (1 // 24, -9 // 8, 9 // 8, -1 // 24)
        @test (@inferred stencil_nodes(StaggeredDifference{3, 1 // 2}())) === (-1 // 2, 1 // 2, 3 // 2)
        @test (@inferred stencil_weights(StaggeredDifference{3, 1 // 2}())) === (-1, 1, 0)
        @test (@inferred stencil_nodes(StaggeredDifference{3, -1 // 2}())) === (-3 // 2, -1 // 2, 1 // 2)
        @test (@inferred stencil_weights(StaggeredDifference{3, -1 // 2}())) === (0, -1, 1)
        @test (@inferred stencil_nodes(Difference{4, 1 // 2}())) === (-1, 0, 1, 2)
        @test (@inferred stencil_weights(Difference{4, 1 // 2}())) === (-1 // 3, -1 // 2, 1, -1 // 6)
    end

    @testset "standard stencils" begin
        @test (@inferred stencil_rule(CentralDifference(), f, i)) ==ₛ (1 // 2) * (f[i + 1] - f[i - 1])
        @test (@inferred stencil_rule(CentralDifference{5}(), f, i)) ==ₛ (2 // 3) * (f[i + 1] - f[i - 1]) - (1 // 12) * (f[i + 2] - f[i - 2])
        @test (@inferred stencil_rule(CentralDifference{7}(), f, i)) ==ₛ (3 // 4) * (f[i + 1] - f[i - 1]) - (3 // 20) * (f[i + 2] - f[i - 2]) + (1 // 60) * (f[i + 3] - f[i - 3])
        @test (@inferred stencil_rule(ForwardDifference(), f, i)) ==ₛ f[i + 1] - f[i]
        @test (@inferred stencil_rule(BackwardDifference(), f, i)) ==ₛ f[i] - f[i - 1]
        forward = @inferred stencil_rule(ForwardDifference{3}(), f, i)
        @test forward ==ₛ 2 * f[i + 1] - (3 // 2) * f[i] - (1 // 2) * f[i + 2]
        leading_product = first(arguments(first(arguments(forward))))
        @test value(first(arguments(leading_product))) isa Int
        @test value(first(arguments(last(arguments(forward))))) isa Rational{Int}
        @test (@inferred stencil_rule(BackwardDifference{3}(), f, i)) ==ₛ (1 // 2) * f[i - 2] + (3 // 2) * f[i] - 2 * f[i - 1]
        @test (@inferred stencil_rule(Difference{3, 2}(), f, i)) ==ₛ 4 * f[i + 2] - (5 // 2) * f[i + 1] - (3 // 2) * f[i + 3]
        @test (@inferred stencil_rule(Difference{4, 1 // 2}(), f, i)) ==ₛ f[i + 1] - (1 // 3) * f[i - 1] - (1 // 2) * f[i] - (1 // 6) * f[i + 2]
        for loc in (𝓅, 𝓈)
            @test (@inferred stencil_rule(CentralDifference{5}(), f, loc, i)) ==ₛ (2 // 3) * (f[loc][i + 1] - f[loc][i - 1]) - (1 // 12) * (f[loc][i + 2] - f[loc][i - 2])
            @test (@inferred stencil_rule(ForwardDifference(), f, loc, i)) ==ₛ f[loc][i + 1] - f[loc][i]
        end
    end

    @testset "staggered stencils" begin
        @test (@inferred stencil_rule(StaggeredCentralDifference(), f, 𝓅, i)) ==ₛ f[𝓈][i] - f[𝓈][i - 1]
        @test (@inferred stencil_rule(StaggeredCentralDifference(), f, 𝓈, i)) ==ₛ f[𝓅][i + 1] - f[𝓅][i]
        @test (@inferred stencil_rule(StaggeredCentralDifference{4}(), f, 𝓅, i)) ==ₛ (9 // 8) * (f[𝓈][i] - f[𝓈][i - 1]) - (1 // 24) * (f[𝓈][i + 1] - f[𝓈][i - 2])
        @test (@inferred stencil_rule(StaggeredCentralDifference{4}(), f, 𝓈, i)) ==ₛ (9 // 8) * (f[𝓅][i + 1] - f[𝓅][i]) - (1 // 24) * (f[𝓅][i + 2] - f[𝓅][i - 1])
        @test (@inferred stencil_rule(StaggeredForwardDifference(), f, 𝓅, i)) ==ₛ f[𝓈][i + 1] - f[𝓈][i]
        @test (@inferred stencil_rule(StaggeredForwardDifference(), f, 𝓈, i)) ==ₛ f[𝓅][i + 2] - f[𝓅][i + 1]
        @test (@inferred stencil_rule(StaggeredBackwardDifference(), f, 𝓅, i)) ==ₛ f[𝓈][i - 1] - f[𝓈][i - 2]
        @test (@inferred stencil_rule(StaggeredBackwardDifference(), f, 𝓈, i)) ==ₛ f[𝓅][i] - f[𝓅][i - 1]
        @test (@inferred stencil_rule(StaggeredForwardDifference{3}(), f, 𝓅, i)) ==ₛ 3 * f[𝓈][i + 1] - 2 * f[𝓈][i] - f[𝓈][i + 2]
        @test (@inferred stencil_rule(StaggeredBackwardDifference{3}(), f, 𝓈, i)) ==ₛ f[𝓅][i - 2] + 2 * f[𝓅][i] - 3 * f[𝓅][i - 1]
        @test (@inferred stencil_rule(StaggeredDifference{3, 1 // 2}(), f, 𝓅, i)) ==ₛ f[𝓈][i] - f[𝓈][i - 1]
        @test (@inferred stencil_rule(StaggeredDifference{3, 1 // 2}(), f, 𝓈, i)) ==ₛ f[𝓅][i + 1] - f[𝓅][i]
        @test (@inferred stencil_rule(StaggeredDifference{3, -1 // 2}(), f, 𝓅, i)) ==ₛ f[𝓈][i] - f[𝓈][i - 1]
        @test (@inferred stencil_rule(StaggeredDifference{3, -1 // 2}(), f, 𝓈, i)) ==ₛ f[𝓅][i + 1] - f[𝓅][i]
    end

    @testset "invalid parameters" begin
        for op in (Difference, StaggeredDifference)
            @test_throws ArgumentError op{1}()
            @test_throws ArgumentError op{-2}()
            @test_throws ArgumentError op{2.0}()
            @test_throws ArgumentError op{2, 0.0}()
            @test_throws ArgumentError op{2, 1 // 3}()
        end
        @test_throws ArgumentError CentralDifference{2}()
        @test_throws ArgumentError Difference{4}()
        @test_throws ArgumentError StaggeredCentralDifference{3}()
        @test_throws ArgumentError Difference{2, 1}()
        @test_throws ArgumentError StaggeredDifference{2, 1 // 2}()
        @test_throws ArgumentError ForwardDifference{1}()
        @test_throws ArgumentError BackwardDifference{1}()
        @test_throws ArgumentError StaggeredForwardDifference{1}()
        @test_throws ArgumentError StaggeredBackwardDifference{1}()
    end

end
