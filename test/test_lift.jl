using Test
using Chmy

struct WeightedStencil <: Operator end
Chmy.lower_rule(::WeightedStencil, args::Tuple{DTerm, DTerm}) = (1 // 3) * args[1][0] + args[2][1]

struct ShiftedStencil <: Operator end
Chmy.lower_rule(::ShiftedStencil, args::Tuple{DTerm}, face, x::NTuple{2}) = args[1][1, 0] - args[1][0, -1]

# a stencil reading the face and the position it is lowered at
struct PositionStencil <: Operator end
Chmy.lower_rule(::PositionStencil, args::Tuple{DTerm}, face, x::NTuple{1}) = x[1] * Literal(Chmy.outward(face[1])) * args[1][0]

# a stencil reading the trace on the lower wall
struct TraceStencil <: Operator end
Chmy.lower_rule(::TraceStencil, args::Tuple{DTerm}) = args[1][↓]

@testset "lifting" begin
    @scalars f g
    d, s = CentralDifference(), StaggeredCentralDifference()
    face, x = (:, :), (0 // 1, 0 // 1)

    @test (@inferred d[2]) == Lifted(d, 2)
    @test Lifted(d, 2) == Lifted(d, (2,))
    @test (@inferred lower_rule(d[2], (f,), face, x)) ==ₛ (1 // 2) * (f[0, 1] - f[0, -1])
    @test (@inferred lower_rule(s[1], (f,), face, x)) ==ₛ f[1 // 2, 0] - f[-1 // 2, 0]
    @test (@inferred lower_rule(Lifted(WeightedStencil(), 2), (f, g), face, x)) ==ₛ (1 // 3) * f[0, 0] + g[0, 1]

    @test (@inferred lower_rule(s[-1], (f,), face, x)) ==ₛ f[-1 // 2, 0] - f[1 // 2, 0]
    @test (@inferred lower_rule(Lifted(WeightedStencil(), -2), (f, g), face, x)) ==ₛ (1 // 3) * f[0, 0] + g[0, -1]

    @test_throws ArgumentError Lifted(d, 0)
    @test_throws ArgumentError lower_rule(d[3], (f,), face, x)
end

@testset "lifting into multiple axes" begin
    @scalars f
    op = ShiftedStencil()
    face2, x2 = (:, :), (0 // 1, 0 // 1)
    face3, x3 = (:, :, :), (0 // 1, 0 // 1, 0 // 1)

    @test (@inferred lower_rule(Lifted(op, (1, 2)), (f,), face2, x2)) ==ₛ lower_rule(op, (f,), face2, x2)
    @test (@inferred lower_rule(Lifted(op, (2, 1)), (f,), face2, x2)) ==ₛ f[0, 1] - f[-1, 0]
    @test (@inferred lower_rule(Lifted(op, (3, 1)), (f,), face3, x3)) ==ₛ f[0, 0, 1] - f[-1, 0, 0]
    @test (@inferred lower_rule(Lifted(op, (1, -2)), (f,), face2, x2)) ==ₛ f[1, 0] - f[0, 1]
    @test (@inferred lower_rule(Lifted(op, (-2, 1)), (f,), face2, x2)) ==ₛ f[0, -1] - f[-1, 0]

    @test_throws ArgumentError Lifted(op, ())
    @test_throws ArgumentError Lifted(op, (1, 1))
    @test_throws ArgumentError Lifted(op, (2, 0))
    @test_throws ArgumentError Lifted(op, (1, -1))
    @test_throws ArgumentError lower_rule(Lifted(op, (1, 3)), (f,), face2, x2)
    @test_throws ArgumentError lower_rule(Lifted(op, (1, -3)), (f,), face2, x2)
end

@testset "lifting tensors" begin
    # tensor components are not mapped to the lifted axes, so only scalar calls of scalars can be lifted
    @scalars f
    @vectors V
    d = CentralDifference()
    face, x = (:, :), (0, 0)
    @test_throws ArgumentError lower(Lifted(Gradient(d), (2, 1))(f)[1], face, x)
    @test_throws ArgumentError lower(Lifted(Divergence(d), (2, 1))(V), face, x)
end

@testset "lifting the face and the position" begin
    @scalars f
    face, x = (↓, ↑, :), (1 // 2, -1 // 2, 1 // 2)
    @test (@inferred lower_rule(Lifted(ShiftedStencil(), (1, 2)), (f,), face, x)) ==ₛ f[1, 0, 0] - f[0, -1, 0]
    # mirrored axes flip the side and the sign of the position, keeping the outward distance from the wall
    @test lower_rule(Lifted(PositionStencil(), 1), (f,), face, x) ==ₛ (1 // 2) * Literal(-1) * f[0, 0, 0]
    @test lower_rule(Lifted(PositionStencil(), -1), (f,), face, x) ==ₛ (-1 // 2) * Literal(1) * f[0, 0, 0]
    @test lower_rule(Lifted(PositionStencil(), -2), (f,), face, x) ==ₛ (1 // 2) * Literal(-1) * f[0, 0, 0]
    # traces in the samples are mirrored as well
    @test (@inferred lower_rule(Lifted(TraceStencil(), -2), (f,), face, x)) ==ₛ f[0, ↑, 0]
end
