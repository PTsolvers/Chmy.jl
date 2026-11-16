using Test
using Chmy

struct WeightedStencil <: Operator end
Chmy.lower_rule(::WeightedStencil, (f, g)::Tuple{DTerm, DTerm}, i::Position) = (1 // 3) * f[i] + g[i + 1]

struct ShiftedStencil <: Operator end
Chmy.lower_rule(::ShiftedStencil, f::DTerm, (i, j)::NTuple{2, Position}) = f[i + 1, j] - f[i, j - 1]

# a stencil reading the outward distance of the position from the wall
struct DistanceStencil <: Operator end
Chmy.lower_rule(::DistanceStencil, f::DTerm, i::Position) = Chmy.distance(i) * f[i]

# a stencil reading the trace on the lower wall
struct TraceStencil <: Operator end
Chmy.lower_rule(::TraceStencil, f::DTerm, i::Position) = f[↓]

@testset "lifting" begin
    @scalars f g
    d, s = CentralDifference(), StaggeredCentralDifference()
    x = (Coord(:, 0), Coord(:, 0))

    @test (@inferred d[2]) == Lifted(d, 2)
    @test Lifted(d, 2) == Lifted(d, (2,))
    @test (@inferred lower_rule(d[2], (f,), x)) ==ₛ (1 // 2) * (f[(:, 0), (:, 1)] - f[(:, 0), (:, -1)])
    @test (@inferred lower_rule(s[1], (f,), x)) ==ₛ f[(:, 1 // 2), (:, 0)] - f[(:, -1 // 2), (:, 0)]
    @test (@inferred lower_rule(Lifted(WeightedStencil(), 2), (f, g), x)) ==ₛ (1 // 3) * f[(:, 0), (:, 0)] + g[(:, 0), (:, 1)]

    @test_throws ArgumentError Lifted(d, 0)
    @test_throws ArgumentError Lifted(d, -1)
    @test_throws ArgumentError lower_rule(d[3], (f,), x)
end

@testset "lifting into multiple axes" begin
    @scalars f g
    op = ShiftedStencil()
    x2 = (Coord(:, 0), Coord(:, 0))
    x3 = (Coord(:, 0), Coord(:, 0), Coord(:, 0))

    @test (@inferred lower_rule(Lifted(op, (1, 2)), (f,), x2)) ==ₛ lower_rule(op, (f,), x2)
    @test (@inferred lower_rule(Lifted(op, (2, 1)), (f,), x2)) ==ₛ f[(:, 0), (:, 1)] - f[(:, -1), (:, 0)]
    @test (@inferred lower_rule(Lifted(op, (3, 1)), (f,), x3)) ==ₛ f[(:, 0), (:, 0), (:, 1)] - f[(:, -1), (:, 0), (:, 0)]
    # the samples inside the arguments are not mapped to the lifted axes
    w = g[(↓, 0), (:, 1)]
    @test lower(Lifted(op, (2, 1))(w * f)[x2...]) ==ₛ w * f[(:, 0), (:, 1)] - w * f[(:, -1), (:, 0)]

    @test_throws ArgumentError Lifted(op, ())
    @test_throws ArgumentError Lifted(op, (1, 1))
    @test_throws ArgumentError Lifted(op, (2, 0))
    @test_throws ArgumentError Lifted(op, (1, -2))
    @test_throws ArgumentError lower_rule(Lifted(op, (1, 3)), (f,), x2)
end

@testset "lifting tensors" begin
    # tensor components are not mapped to the lifted axes, so only scalar calls of scalars can be lifted
    @scalars f
    @vectors V
    d = CentralDifference()
    x = ((:, 0), (:, 0))
    @test_throws ArgumentError lower(Lifted(Gradient(d), (2, 1))(f)[1][x...])
    @test_throws ArgumentError lower(Lifted(Divergence(d), (2, 1))(V)[x...])
end

@testset "lifting at positions" begin
    @scalars f
    x = (Coord(↓, 1 // 2), Coord(↑, -1 // 2), Coord(:, 1 // 2))
    @test (@inferred lower_rule(Lifted(ShiftedStencil(), (1, 2)), (f,), x)) ==ₛ
        f[(↓, 3 // 2), (↑, -1 // 2), (:, 1 // 2)] - f[(↓, 1 // 2), (↑, -3 // 2), (:, 1 // 2)]
    # the rules read the frames of the positions along the lifted axes
    @test lower_rule(Lifted(DistanceStencil(), 1), (f,), x) ==ₛ (-1 // 2) * f[x...]
    @test lower_rule(Lifted(DistanceStencil(), 2), (f,), x) ==ₛ (-1 // 2) * f[x...]
    # traces in the rules are taken along the lifted axis
    @test (@inferred lower_rule(Lifted(TraceStencil(), 2), (f,), x)) ==ₛ f[(↓, 1 // 2), ↓, (:, 1 // 2)]
end
