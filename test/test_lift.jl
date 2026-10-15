using Test
using Chmy

struct WeightedStencil <: Operator end
Chmy.stencil_rule(::WeightedStencil, args::Tuple{DTerm, DTerm}, ::Val{1}) = (1 // 3) * args[1][0] + args[2][1]

struct ShiftedStencil <: Operator end
Chmy.stencil_rule(::ShiftedStencil, args::Tuple{DTerm}, ::Val{2}) = args[1][1, 0] - args[1][0, -1]

@testset "lifting" begin
    @scalars f g
    d, s = CentralDifference(), StaggeredCentralDifference()

    @test (@inferred d[2]) == Lifted(d, 2)
    @test Lifted(d, 2) == Lifted(d, (2,))
    @test (@inferred stencil_rule(d[2], (f,), Val(2))) ==ₛ (1 // 2) * (f[0, 1] - f[0, -1])
    @test (@inferred stencil_rule(s[1], (f,), Val(2))) ==ₛ f[1 // 2, 0] - f[-1 // 2, 0]
    @test (@inferred stencil_rule(Lifted(WeightedStencil(), 2), (f, g), Val(2))) ==ₛ (1 // 3) * f[0, 0] + g[0, 1]

    @test (@inferred stencil_rule(s[-1], (f,), Val(2))) ==ₛ f[-1 // 2, 0] - f[1 // 2, 0]
    @test (@inferred stencil_rule(Lifted(WeightedStencil(), -2), (f, g), Val(2))) ==ₛ (1 // 3) * f[0, 0] + g[0, -1]

    @test_throws ArgumentError Lifted(d, 0)
    @test_throws ArgumentError stencil_rule(d[3], (f,), Val(2))
end

@testset "lifting into multiple axes" begin
    @scalars f
    op = ShiftedStencil()

    @test (@inferred stencil_rule(Lifted(op, (1, 2)), (f,), Val(2))) ==ₛ stencil_rule(op, (f,), Val(2))
    @test (@inferred stencil_rule(Lifted(op, (2, 1)), (f,), Val(2))) ==ₛ f[0, 1] - f[-1, 0]
    @test (@inferred stencil_rule(Lifted(op, (3, 1)), (f,), Val(3))) ==ₛ f[0, 0, 1] - f[-1, 0, 0]
    @test (@inferred stencil_rule(Lifted(op, (1, -2)), (f,), Val(2))) ==ₛ f[1, 0] - f[0, 1]
    @test (@inferred stencil_rule(Lifted(op, (-2, 1)), (f,), Val(2))) ==ₛ f[0, -1] - f[-1, 0]

    @test_throws ArgumentError Lifted(op, ())
    @test_throws ArgumentError Lifted(op, (1, 1))
    @test_throws ArgumentError Lifted(op, (2, 0))
    @test_throws ArgumentError Lifted(op, (1, -1))
    @test_throws ArgumentError stencil_rule(Lifted(op, (1, 3)), (f,), Val(2))
    @test_throws ArgumentError stencil_rule(Lifted(op, (1, -3)), (f,), Val(2))
end
