using Test
using Chmy
using Chmy: LowerContext

struct WeightedStencil <: Operator end
Chmy.stencil_rule(::WeightedStencil, args::Tuple{DTerm, DTerm}, ::LowerContext{1}) = (1 // 3) * args[1][0] + args[2][1]

struct ShiftedStencil <: Operator end
Chmy.stencil_rule(::ShiftedStencil, args::Tuple{DTerm}, ::LowerContext{2}) = args[1][1, 0] - args[1][0, -1]

# uses its argument both at the evaluation point and through a sample
struct PointwiseStencil <: Operator end
Chmy.stencil_rule(::PointwiseStencil, args::Tuple{DTerm}, ::LowerContext{1}) = args[1] + args[1][1]

@testset "lifting" begin
    @scalars f g
    d, s = CentralDifference(), StaggeredCentralDifference()

    ctx = LowerContext(
        (𝓈, 𝓈),
        (0 // 1, 0 // 1),
        (Far(), Far())
    )

    @test (@inferred d[2]) == Lifted(d, 2)
    @test Lifted(d, 2) == Lifted(d, (2,))
    @test (@inferred stencil_rule(d[2], (f,), ctx)) ==ₛ (1 // 2) * (f[0, 1] - f[0, -1])
    @test (@inferred stencil_rule(s[1], (f,), ctx)) ==ₛ f[1 // 2, 0] - f[-1 // 2, 0]
    @test (@inferred stencil_rule(Lifted(WeightedStencil(), 2), (f, g), ctx)) ==ₛ (1 // 3) * f[0, 0] + g[0, 1]

    @test (@inferred stencil_rule(s[-1], (f,), ctx)) ==ₛ f[-1 // 2, 0] - f[1 // 2, 0]
    @test (@inferred stencil_rule(Lifted(WeightedStencil(), -2), (f, g), ctx)) ==ₛ (1 // 3) * f[0, 0] + g[0, -1]

    @test_throws ArgumentError Lifted(d, 0)
    @test_throws ArgumentError stencil_rule(d[3], (f,), ctx)
end

@testset "lifting into multiple axes" begin
    @scalars f
    op = ShiftedStencil()

    ctx2 = LowerContext(
        (𝓈, 𝓈),
        (0 // 1, 0 // 1),
        (Far(), Far())
    )

    ctx3 = LowerContext(
        (𝓈, 𝓈, 𝓈),
        (0 // 1, 0 // 1, 0 // 1),
        (Far(), Far(), Far())
    )

    @test (@inferred stencil_rule(Lifted(op, (1, 2)), (f,), ctx2)) ==ₛ stencil_rule(op, (f,), ctx2)
    @test (@inferred stencil_rule(Lifted(op, (2, 1)), (f,), ctx2)) ==ₛ f[0, 1] - f[-1, 0]
    @test (@inferred stencil_rule(Lifted(op, (3, 1)), (f,), ctx3)) ==ₛ f[0, 0, 1] - f[-1, 0, 0]
    @test (@inferred stencil_rule(Lifted(op, (1, -2)), (f,), ctx2)) ==ₛ f[1, 0] - f[0, 1]
    @test (@inferred stencil_rule(Lifted(op, (-2, 1)), (f,), ctx2)) ==ₛ f[0, -1] - f[-1, 0]

    @test_throws ArgumentError Lifted(op, ())
    @test_throws ArgumentError Lifted(op, (1, 1))
    @test_throws ArgumentError Lifted(op, (2, 0))
    @test_throws ArgumentError Lifted(op, (1, -1))
    @test_throws ArgumentError stencil_rule(Lifted(op, (1, 3)), (f,), ctx2)
    @test_throws ArgumentError stencil_rule(Lifted(op, (1, -3)), (f,), ctx2)
end

@testset "lifting Near axes" begin
    @scalars f
    op = ShiftedStencil()

    ctx = LowerContext(
        (𝓈, 𝓈, 𝓈),
        (1 // 1, -1 // 1, 0 // 1),
        (Near(Lower(), 1 // 2), Near(Upper(), -1 // 2), Far())
    )

    @test (@inferred stencil_rule(Lifted(op, (1, 2)), (f,), ctx)) ==ₛ f[1, 0, 0] - f[0, -1, 0]
end

@testset "lifting keeps the arguments" begin
    @scalars u
    # the argument holds a global sample: only the sample created by the rule is mirrored
    @test lower(Lifted(PointwiseStencil(), -1)(u[1]), 𝓅) ==ₛ u[𝓅][1] + u[𝓅][0]
    @test lower(Lifted(PointwiseStencil(), 1)(u[1]), 𝓅) ==ₛ u[𝓅][1] + u[𝓅][2]
    # in 2D the argument's samples have another length, and are kept as well
    @test lower(Lifted(PointwiseStencil(), -2)(u[1, 1]), 𝓅, 𝓅) ==ₛ u[𝓅, 𝓅][1, 1] + u[𝓅, 𝓅][1, 0]
end
