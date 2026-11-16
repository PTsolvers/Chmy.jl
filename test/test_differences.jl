using Test
using Chmy

@testset "differences" begin
    @scalars f

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

    # stencils at the grid point of the kernel
    i = Coord(:, 0)

    @testset "standard stencils" begin
        @test (@inferred lower_rule(CentralDifference(), f, i)) ==ₛ (1 // 2) * (f[(:, 1)] - f[(:, -1)])
        @test (@inferred lower_rule(CentralDifference{5}(), f, i)) ==ₛ (2 // 3) * (f[(:, 1)] - f[(:, -1)]) - (1 // 12) * (f[(:, 2)] - f[(:, -2)])
        @test (@inferred lower_rule(CentralDifference{7}(), f, i)) ==ₛ (3 // 4) * (f[(:, 1)] - f[(:, -1)]) - (3 // 20) * (f[(:, 2)] - f[(:, -2)]) + (1 // 60) * (f[(:, 3)] - f[(:, -3)])
        @test (@inferred lower_rule(ForwardDifference(), f, i)) ==ₛ f[(:, 1)] - f[(:, 0)]
        @test (@inferred lower_rule(BackwardDifference(), f, i)) ==ₛ f[(:, 0)] - f[(:, -1)]
        forward = @inferred lower_rule(ForwardDifference{3}(), f, i)
        @test forward ==ₛ 2 * f[(:, 1)] - (3 // 2) * f[(:, 0)] - (1 // 2) * f[(:, 2)]
        # integer weights are stored as `Int`
        @test map(t -> value(first(arguments(t))), arguments(forward)) === (-1 // 2, 2, -3 // 2)
        @test (@inferred lower_rule(BackwardDifference{3}(), f, i)) ==ₛ (1 // 2) * f[(:, -2)] + (3 // 2) * f[(:, 0)] - 2 * f[(:, -1)]
        @test (@inferred lower_rule(Difference{3, 2}(), f, i)) ==ₛ 4 * f[(:, 2)] - (5 // 2) * f[(:, 1)] - (3 // 2) * f[(:, 3)]
        @test (@inferred lower_rule(Difference{4, 1 // 2}(), f, i)) ==ₛ f[(:, 1)] - (1 // 3) * f[(:, -1)] - (1 // 2) * f[(:, 0)] - (1 // 6) * f[(:, 2)]
    end

    @testset "staggered stencils" begin
        @test (@inferred lower_rule(StaggeredCentralDifference(), f, i)) ==ₛ f[(:, 1 // 2)] - f[(:, -1 // 2)]
        @test (@inferred lower_rule(StaggeredCentralDifference{4}(), f, i)) ==ₛ (9 // 8) * (f[(:, 1 // 2)] - f[(:, -1 // 2)]) - (1 // 24) * (f[(:, 3 // 2)] - f[(:, -3 // 2)])
        @test (@inferred lower_rule(StaggeredForwardDifference(), f, i)) ==ₛ f[(:, 3 // 2)] - f[(:, 1 // 2)]
        @test (@inferred lower_rule(StaggeredBackwardDifference(), f, i)) ==ₛ f[(:, -1 // 2)] - f[(:, -3 // 2)]
        @test (@inferred lower_rule(StaggeredForwardDifference{3}(), f, i)) ==ₛ 3 * f[(:, 3 // 2)] - 2 * f[(:, 1 // 2)] - f[(:, 5 // 2)]
        @test (@inferred lower_rule(StaggeredBackwardDifference{3}(), f, i)) ==ₛ f[(:, -5 // 2)] + 2 * f[(:, -1 // 2)] - 3 * f[(:, -3 // 2)]
        @test (@inferred lower_rule(StaggeredDifference{3, 1 // 2}(), f, i)) ==ₛ f[(:, 1 // 2)] - f[(:, -1 // 2)]
        @test (@inferred lower_rule(StaggeredDifference{3, -1 // 2}(), f, i)) ==ₛ f[(:, 1 // 2)] - f[(:, -1 // 2)]
        # in the frame of the wall, and at traces, which are constant along the axis
        @test (@inferred lower_rule(StaggeredCentralDifference(), f, Coord(↓, 1 // 2))) ==ₛ f[(↓, 1)] - f[(↓, 0)]
        @test (@inferred lower_rule(StaggeredCentralDifference(), f, Trace(↑))) ==ₛ Literal(0)
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
