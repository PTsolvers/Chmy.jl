using Test
using Chmy
using Chmy: LoweringContext

struct WeightedStencil <: Operator end
Chmy.lower_rule(::WeightedStencil, args::Tuple{DTerm, DTerm}) = (1 // 3) * args[1][0] + args[2][1]

struct ShiftedStencil <: Operator end
Chmy.lower_rule(::ShiftedStencil, args::Tuple{DTerm}, coord::NTuple{2}, ctx) = args[1][1, 0] - args[1][0, -1]

# a stencil reading the context it is lowered in
struct ContextStencil <: Operator end
Chmy.lower_rule(::ContextStencil, args::Tuple{DTerm}, coord::NTuple{1}, ctx) = ctx.origin[1] * Literal(Chmy.outward(ctx.face[1])) * args[1][0]

@testset "lifting" begin
    @scalars f g
    d, s = CentralDifference(), StaggeredCentralDifference()
    offs = (0 // 1, 0 // 1)
    ctx = LoweringContext((:, :), offs)

    @test (@inferred d[2]) == Lifted(d, 2)
    @test Lifted(d, 2) == Lifted(d, (2,))
    @test (@inferred lower_rule(d[2], (f,), (), offs, ctx)) ==ₛ (1 // 2) * (f[0, 1] - f[0, -1])
    @test (@inferred lower_rule(s[1], (f,), (), offs, ctx)) ==ₛ f[1 // 2, 0] - f[-1 // 2, 0]
    @test (@inferred lower_rule(Lifted(WeightedStencil(), 2), (f, g), (), offs, ctx)) ==ₛ (1 // 3) * f[0, 0] + g[0, 1]

    @test (@inferred lower_rule(s[-1], (f,), (), offs, ctx)) ==ₛ f[-1 // 2, 0] - f[1 // 2, 0]
    @test (@inferred lower_rule(Lifted(WeightedStencil(), -2), (f, g), (), offs, ctx)) ==ₛ (1 // 3) * f[0, 0] + g[0, -1]

    @test_throws ArgumentError Lifted(d, 0)
    @test_throws ArgumentError lower_rule(d[3], (f,), (), offs, ctx)
end

@testset "lifting into multiple axes" begin
    @scalars f
    op = ShiftedStencil()
    offs2, offs3 = (0 // 1, 0 // 1), (0 // 1, 0 // 1, 0 // 1)
    ctx2, ctx3 = LoweringContext((:, :), offs2), LoweringContext((:, :, :), offs3)

    @test (@inferred lower_rule(Lifted(op, (1, 2)), (f,), (), offs2, ctx2)) ==ₛ lower_rule(op, (f,), (), offs2, ctx2)
    @test (@inferred lower_rule(Lifted(op, (2, 1)), (f,), (), offs2, ctx2)) ==ₛ f[0, 1] - f[-1, 0]
    @test (@inferred lower_rule(Lifted(op, (3, 1)), (f,), (), offs3, ctx3)) ==ₛ f[0, 0, 1] - f[-1, 0, 0]
    @test (@inferred lower_rule(Lifted(op, (1, -2)), (f,), (), offs2, ctx2)) ==ₛ f[1, 0] - f[0, 1]
    @test (@inferred lower_rule(Lifted(op, (-2, 1)), (f,), (), offs2, ctx2)) ==ₛ f[0, -1] - f[-1, 0]

    @test_throws ArgumentError Lifted(op, ())
    @test_throws ArgumentError Lifted(op, (1, 1))
    @test_throws ArgumentError Lifted(op, (2, 0))
    @test_throws ArgumentError Lifted(op, (1, -1))
    @test_throws ArgumentError lower_rule(Lifted(op, (1, 3)), (f,), (), offs2, ctx2)
    @test_throws ArgumentError lower_rule(Lifted(op, (1, -3)), (f,), (), offs2, ctx2)
end

@testset "lifting tensors" begin
    # tensor components are not mapped to the lifted axes, so only scalar calls of scalars can be lifted
    @scalars f
    @vectors V
    d = CentralDifference()
    offs = (0 // 1, 0 // 1)
    ctx = LoweringContext((:, :), offs)
    @test_throws ArgumentError lower_rule(Lifted(Gradient(d), (2, 1)), (f,), (1,), offs, ctx)
    @test_throws ArgumentError lower_rule(Lifted(Divergence(d), (2, 1)), (V,), (), offs, ctx)
end

@testset "lifting the context" begin
    @scalars f
    offs = (1 // 1, -1 // 1, 0 // 1)
    ctx = LoweringContext((↓, ↑, :), (1 // 2, -1 // 2, 1 // 2))
    @test (@inferred lower_rule(Lifted(ShiftedStencil(), (1, 2)), (f,), (), offs, ctx)) ==ₛ f[1, 0, 0] - f[0, -1, 0]
    # the outward distance is kept on mirrored sides, which flip, and the lattice coordinate changes sign
    @test lower_rule(Lifted(ContextStencil(), 1), (f,), (), offs, ctx) ==ₛ (1 // 2) * Literal(-1) * f[0, 0, 0]
    @test lower_rule(Lifted(ContextStencil(), -1), (f,), (), offs, ctx) ==ₛ (1 // 2) * Literal(1) * f[0, 0, 0]
    @test lower_rule(Lifted(ContextStencil(), -2), (f,), (), offs, ctx) ==ₛ (-1 // 2) * Literal(-1) * f[0, 0, 0]
end
