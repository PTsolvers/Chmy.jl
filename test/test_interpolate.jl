using Test
using Chmy

@testset "interpolate" begin
    @scalars f u g a b
    @uniform @scalars α

    @testset "fornberg" begin
        # columns are the weights of the derivatives at zero, from the 0th to the `m`-th
        @test (@inferred Chmy.fornberg([-1 // 1, 0, 1], 2)) == [0 -1 // 2 1; 1 0 -2; 0 1 // 2 1]
        @test Chmy.fornberg((-2 // 1):2, 2)[:, 2:3] == [1 // 12 -1 // 12; -2 // 3 4 // 3; 0 -5 // 2; 2 // 3 4 // 3; -1 // 12 -1 // 12]
        @test Chmy.fornberg([-1 // 1, 0, 1, 2], 1)[:, 2] == [-1 // 3, -1 // 2, 1, -1 // 6]
        @test Chmy.fornberg([-1 // 2, 1 // 2], 0) == reshape([1 // 2, 1 // 2], 2, 1)
        # derivatives above the degree of the polynomial vanish
        @test Chmy.fornberg([0 // 1, 1], 3)[:, 3:4] == zeros(2, 2)
    end

    @testset "constructors" begin
        @test (@inferred Interpolate((-1 // 2, 1 // 2))) == Interpolate(0, [-1 // 2, 1 // 2])
        @test (@inferred Interpolate(2, -1:1)).weights == [1, -2, 1]
        @test Interpolate(1, (-1, 0, 1, 2)).weights == [-1 // 3, -1 // 2, 1, -1 // 6]
        # the values at the position of the Lagrange basis polynomials, of the evaluation point first
        @test (@inferred Extrapolate(1 // 2, (1, 2))).weights[:, 1] == [3 // 8, 3 // 4, -1 // 8]
    end

    # stencils at the grid point of the kernel
    i = Coord(:, 0)

    @testset "interpolation" begin
        # stencils on mirrored nodes are paired by the parity of the order
        @test (@inferred lower_rule(Interpolate((-1 // 2, 1 // 2)), f, i)) ==ₛ (1 // 2) * (f[(:, 1 // 2)] + f[(:, -1 // 2)])
        @test (@inferred lower_rule(Interpolate(2, -1:1), f, i)) ==ₛ (f[(:, 1)] + f[(:, -1)]) - 2 * f[(:, 0)]
        # other stencils are plain sums
        @test (@inferred lower_rule(Interpolate((1 // 2, 3 // 2)), f, i)) ==ₛ (3 // 2) * f[(:, 1 // 2)] - (1 // 2) * f[(:, 3 // 2)]
        # nodes with zero weights are not sampled
        @test (@inferred lower_rule(Interpolate((-1, 0, 1, 2)), f, i)) ==ₛ f[(:, 0)]
        @test (@inferred lower_rule(Interpolate(1, (-1 // 2, 1 // 2, 3 // 2)), f, i)) ==ₛ f[(:, 1 // 2)] - f[(:, -1 // 2)]
        # at traces, which are constant along the axis
        @test (@inferred lower_rule(Interpolate((-1 // 2, 1 // 2)), f, Trace(↑))) ==ₛ f[↑]
    end

    @testset "extrapolation" begin
        # the ghost next to the lower wall, half a spacing away
        x = Coord(↓, -1 // 2)
        e = Extrapolate(1 // 2, (1,))
        # the value on the wall and the derivative along the axis
        @test (@inferred lower_rule(e, (u, g, Literal(1)), x)) ==ₛ 2 * g[(↓, 0)] - u[(↓, 1 // 2)]
        @test (@inferred lower_rule(e, (u, g, Literal(0), Literal(1)), x)) ==ₛ u[(↓, 1 // 2)] - g[(↓, 0)]
        @test (@inferred lower_rule(Extrapolate(1 // 2, (1, 2)), (u, g, Literal(1)), x)) ==ₛ
            (8 // 3) * g[(↓, 0)] - 2 * u[(↓, 1 // 2)] + (1 // 3) * u[(↓, 3 // 2)]
        # symbolic coefficients of a Robin condition, sampled on the wall
        r0, r1 = a[(↓, 0)] / 2 - b[(↓, 0)], a[(↓, 0)] / 2 + b[(↓, 0)]
        @test e(u, g, a, b)[x] ==ₛ g[(↓, 0)] / r0 - r1 / r0 * u[(↓, 1 // 2)]
        @test e(u, g, α, 1)[x] ==ₛ g[(↓, 0)] / (α / 2 - 1) - (α / 2 + 1) / (α / 2 - 1) * u[(↓, 1 // 2)]
    end

    @testset "lifting" begin
        x = (Coord(:, 0), Coord(↓, -1 // 2))
        @test Interpolate((-1 // 2, 1 // 2))[2](f)[x...] ==ₛ (1 // 2) * (f[(:, 0), (↓, 0)] + f[(:, 0), (↓, -1)])
        @test Extrapolate(1 // 2, (1,))[2](u, g, 1)[x...] ==ₛ 2 * g[(:, 0), (↓, 0)] - u[(:, 0), (↓, 1 // 2)]
    end

    @testset "invalid parameters" begin
        @test_throws ArgumentError Interpolate((0, 0))
        @test_throws ArgumentError Interpolate((0, 1 // 3))
        @test_throws ArgumentError Interpolate(2, (0, 1))
        @test_throws ArgumentError Interpolate(-1, (0, 1))
        @test_throws ArgumentError Interpolate((0, 1))(u, g)
        @test_throws ArgumentError Extrapolate(1 // 2, (0, 1))
        @test_throws ArgumentError Extrapolate(1 // 3, (1,))
        e = Extrapolate(1 // 2, (1,))
        @test_throws ArgumentError e(u, g)
        @test_throws ArgumentError e(u, g, a, b, a)
        # the value on a node does not depend on the extrapolated value
        @test_throws ArgumentError Extrapolate(1, (1,))(u, g, 1)[(:, 0)]
    end
end
