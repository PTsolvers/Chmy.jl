using Test
using Chmy

@testset "finite difference" begin
    @scalars f

    @testset "constructors" begin
        @test (@inferred CentralDifference()) == Interpolate(1, [-1, 0, 1])
        @test (@inferred StaggeredCentralDifference()) == Interpolate(1, (-1 // 2, 1 // 2))
        @test ForwardDifference() == Interpolate(1, 0:1)
        @test BackwardDifference() == Interpolate(1, -1:0)
        @test StaggeredForwardDifference() == Interpolate(1, (1 // 2, 3 // 2))
        @test StaggeredBackwardDifference() == Interpolate(1, (-3 // 2, -1 // 2))
        @test CentralDifference(5).weights == [1 // 12, -2 // 3, 0, 2 // 3, -1 // 12]
        @test StaggeredCentralDifference(4).weights == [1 // 24, -9 // 8, 9 // 8, -1 // 24]
        @test ForwardDifference(3).weights == [-3 // 2, 2, -1 // 2]
        @test BackwardDifference(3).weights == [1 // 2, -2, 3 // 2]
    end

    # stencils at the grid point of the kernel
    i = Coord(:, 0)

    @testset "stencils" begin
        # central differences are antisymmetric pairs
        @test (@inferred lower_rule(CentralDifference(), f, i)) ==ₛ (1 // 2) * (f[(:, 1)] - f[(:, -1)])
        @test (@inferred lower_rule(CentralDifference(5), f, i)) ==ₛ (2 // 3) * (f[(:, 1)] - f[(:, -1)]) - (1 // 12) * (f[(:, 2)] - f[(:, -2)])
        @test (@inferred lower_rule(StaggeredCentralDifference(), f, i)) ==ₛ f[(:, 1 // 2)] - f[(:, -1 // 2)]
        @test (@inferred lower_rule(StaggeredCentralDifference(4), f, i)) ==ₛ (9 // 8) * (f[(:, 1 // 2)] - f[(:, -1 // 2)]) - (1 // 24) * (f[(:, 3 // 2)] - f[(:, -3 // 2)])
        # one-sided differences are plain sums
        @test (@inferred lower_rule(ForwardDifference(), f, i)) ==ₛ f[(:, 1)] - f[(:, 0)]
        @test (@inferred lower_rule(BackwardDifference(), f, i)) ==ₛ f[(:, 0)] - f[(:, -1)]
        @test (@inferred lower_rule(ForwardDifference(3), f, i)) ==ₛ 2 * f[(:, 1)] - (3 // 2) * f[(:, 0)] - (1 // 2) * f[(:, 2)]
        @test (@inferred lower_rule(StaggeredForwardDifference(), f, i)) ==ₛ f[(:, 3 // 2)] - f[(:, 1 // 2)]
        @test (@inferred lower_rule(StaggeredBackwardDifference(3), f, i)) ==ₛ f[(:, -5 // 2)] + 2 * f[(:, -1 // 2)] - 3 * f[(:, -3 // 2)]
        # in the frame of the wall, and at traces, which are constant along the axis
        @test (@inferred lower_rule(StaggeredCentralDifference(), f, Coord(↓, 1 // 2))) ==ₛ f[(↓, 1)] - f[(↓, 0)]
        @test (@inferred lower_rule(StaggeredCentralDifference(), f, Trace(↑))) ==ₛ Literal(0)
    end

    @testset "invalid parameters" begin
        @test_throws ArgumentError CentralDifference(2)
        @test_throws ArgumentError StaggeredCentralDifference(3)
        @test_throws ArgumentError ForwardDifference(1)
        @test_throws ArgumentError StaggeredBackwardDifference(1)
    end
end
