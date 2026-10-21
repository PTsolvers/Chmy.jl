using Test
using Chmy

struct WeightedStencil <: Operator end
Chmy.lower_rule(::WeightedStencil, args::Tuple{DTerm, DTerm}) = (1 // 3) * args[1][0] + args[2][1]

struct ShiftedStencil <: Operator end
Chmy.lower_rule(::ShiftedStencil, args::Tuple{DTerm}, offs::NTuple{2}, block) = args[1][1, 0] - args[1][0, -1]

@testset "lifting" begin
    @scalars f g
    d, s = CentralDifference(), StaggeredCentralDifference()
    offs, far = (0 // 1, 0 // 1), (Far(), Far())

    @test (@inferred d[2]) == Lifted(d, 2)
    @test Lifted(d, 2) == Lifted(d, (2,))
    @test (@inferred lower_rule(d[2], (f,), (), offs, far)) ==ₛ (1 // 2) * (f[0, 1] - f[0, -1])
    @test (@inferred lower_rule(s[1], (f,), (), offs, far)) ==ₛ f[1 // 2, 0] - f[-1 // 2, 0]
    @test (@inferred lower_rule(Lifted(WeightedStencil(), 2), (f, g), (), offs, far)) ==ₛ (1 // 3) * f[0, 0] + g[0, 1]

    @test (@inferred lower_rule(s[-1], (f,), (), offs, far)) ==ₛ f[-1 // 2, 0] - f[1 // 2, 0]
    @test (@inferred lower_rule(Lifted(WeightedStencil(), -2), (f, g), (), offs, far)) ==ₛ (1 // 3) * f[0, 0] + g[0, -1]

    @test_throws ArgumentError Lifted(d, 0)
    @test_throws ArgumentError lower_rule(d[3], (f,), (), offs, far)
end

@testset "lifting into multiple axes" begin
    @scalars f
    op = ShiftedStencil()
    offs2, far2 = (0 // 1, 0 // 1), (Far(), Far())
    offs3, far3 = (0 // 1, 0 // 1, 0 // 1), (Far(), Far(), Far())

    @test (@inferred lower_rule(Lifted(op, (1, 2)), (f,), (), offs2, far2)) ==ₛ lower_rule(op, (f,), (), offs2, far2)
    @test (@inferred lower_rule(Lifted(op, (2, 1)), (f,), (), offs2, far2)) ==ₛ f[0, 1] - f[-1, 0]
    @test (@inferred lower_rule(Lifted(op, (3, 1)), (f,), (), offs3, far3)) ==ₛ f[0, 0, 1] - f[-1, 0, 0]
    @test (@inferred lower_rule(Lifted(op, (1, -2)), (f,), (), offs2, far2)) ==ₛ f[1, 0] - f[0, 1]
    @test (@inferred lower_rule(Lifted(op, (-2, 1)), (f,), (), offs2, far2)) ==ₛ f[0, -1] - f[-1, 0]

    @test_throws ArgumentError Lifted(op, ())
    @test_throws ArgumentError Lifted(op, (1, 1))
    @test_throws ArgumentError Lifted(op, (2, 0))
    @test_throws ArgumentError Lifted(op, (1, -1))
    @test_throws ArgumentError lower_rule(Lifted(op, (1, 3)), (f,), (), offs2, far2)
    @test_throws ArgumentError lower_rule(Lifted(op, (1, -3)), (f,), (), offs2, far2)
end

@testset "lifting tensors" begin
    # tensor components are not mapped to the lifted axes, so only scalar calls of scalars can be lifted
    @scalars f
    @vectors V
    d = CentralDifference()
    offs, far = (0 // 1, 0 // 1), (Far(), Far())
    @test_throws ArgumentError lower_rule(Lifted(Gradient(d), (2, 1)), (f,), (1,), offs, far)
    @test_throws ArgumentError lower_rule(Lifted(Divergence(d), (2, 1)), (V,), (), offs, far)
end

@testset "lifting Near axes" begin
    @scalars f
    op = ShiftedStencil()
    offs = (1 // 1, -1 // 1, 0 // 1)
    block = (Near(Lower(), 1 // 2), Near(Upper(), -1 // 2), Far())

    @test (@inferred lower_rule(Lifted(op, (1, 2)), (f,), (), offs, block)) ==ₛ f[1, 0, 0] - f[0, -1, 0]
end
