using Test
using Chmy

# a pointwise operator defined only by its rule, which keeps the call and samples the argument here
struct Pointwise <: Operator end
Chmy.lower_rule(op::Pointwise, args::Tuple{DTerm}, face, x) = op(args[1])

# a scalar operator depending on the number of dimensions
struct Scaled <: Operator end
Chmy.lower_rule(::Scaled, args::Tuple{DTerm}, face, x) = args[1] / length(x)

# an operator whose rule returns a tensor in component form instead of a scalar
struct Unindexed <: Operator end
Chmy.lower_rule(::Unindexed, args::Tuple{DTerm}, face, x) = pack([args[1], args[1]])

# an operator whose rule reads its argument at the evaluation point without sampling it
struct Unsampled <: Operator end
Chmy.lower_rule(::Unsampled, args::Tuple{DTerm}, face, x) = 2 * args[1]

@testset "lowering" begin
    @scalars f g
    @uniform @scalars a
    @vectors u v
    @tensors 2 A B
    d, s = CentralDifference(), StaggeredCentralDifference()
    divg, grad = Divergence(s), Gradient(s)

    @testset "samples" begin
        # fields are sampled at the position, uniform terms are kept
        @test (@inferred lower(sin(f) + a * v[1], (:, :), (0, 0))) ==ₛ sin(f[0, 0]) + a * v[1][0, 0]
        @test (@inferred lower(a, (:,), (0,))) ==ₛ a
        # nested samples shift the position
        @test (@inferred lower(sin(f[1 // 2, 0]), (:, :), (0, 0))) ==ₛ sin(f[1 // 2, 0])
        @test (@inferred lower(sin(f)[-1 // 2], (:,), (0,))) ==ₛ sin(f[-1 // 2])
        @test (@inferred lower(f[1 // 2][1], (:,), (1 // 2,))) ==ₛ f[2]
        # uniform calls are expanded as well
        @test (@inferred lower(d[1](a), (:,), (0,))) ==ₛ Literal(0)
    end

    @testset "laws" begin
        e = divg(grad(f)) * f[1 // 2, 0]
        x = (1 // 2, -1 // 2)
        # the position is an outer sample
        @test lower(e, (↓, :), x) ==ₛ lower(e[x...], (↓, :), (0, 0))
        # samples compose with the position
        @test lower(e[1, ↓], (↓, :), x) ==ₛ lower(e, (↓, :), (3 // 2, ↓))
        # lowered expressions are valid input
        @test lower(lower(e, (↓, :), x), (↓, :), (0, 0)) ==ₛ lower(e, (↓, :), x)
    end

    @testset "global positions" begin
        # a global position absorbs offsets, and the innermost one wins
        @test (@inferred lower(f[1 // 2, 0][↓, 0], (:, :), (0, 0))) ==ₛ f[↓, 0]
        @test (@inferred lower(f[↓, 0][1 // 2, 0], (:, :), (0, 0))) ==ₛ f[↓, 0]
        @test (@inferred lower(f[↓, 0][↑, 0], (:, :), (0, 0))) ==ₛ f[↓, 0]
        @test (@inferred lower(f[:, 0][↓, 1], (:, :), (0, 1 // 2))) ==ₛ f[:, 3 // 2]
        # traces are constant along their axis, so their normal derivative vanishes
        @test lower(s[1](f[↓, 0]), (:, :), (0, 1 // 2)) ==ₛ Literal(0)
        @test lower(s[2](f[↓, 0]), (:, :), (0, 1 // 2)) ==ₛ f[↓, 1] - f[↓, 0]
        # lowering at a global position evaluates the expression on the traces
        @test lower(sin(f) * g[1 // 2, 0], (:, :), (0, :)) ==ₛ sin(f[0, :]) * g[1 // 2, :]
    end

    @testset "laplacian" begin
        Δ = divg(grad(f))
        interior = f[-1 // 2] - 2 * f[1 // 2] + f[3 // 2]
        @test lower(Δ, (:,), (1 // 2,)) ==ₛ interior
        # the same stencil next to the wall, relative to the wall point
        @test lower(Δ, (↓,), (1 // 2,)) ==ₛ interior
        @test lower(Δ, (↓,), (3 // 2,)) ==ₛ f[1 // 2] - 2 * f[3 // 2] + f[5 // 2]
        @test lower(divg(grad(f)), (:, :), (0, 0)) ==ₛ
            f[-1, 0] + f[0, -1] - 4 * f[0, 0] + f[1, 0] + f[0, 1]
    end

    @testset "rules" begin
        @test (@inferred lower(d[1](d[2](f)), (:, :), (0, 0))) ==ₛ
            (1 // 2) * ((1 // 2) * (f[1, 1] - f[1, -1]) - (1 // 2) * (f[-1, 1] - f[-1, -1]))
        @test (@inferred lower(s[1](s[2](f)), (:, :), (0, 1 // 2))) ==ₛ
            (f[1 // 2, 1] - f[1 // 2, 0]) - (f[-1 // 2, 1] - f[-1 // 2, 0])
        # calls returned by rules are kept, and the operators inside their arguments are expanded
        @test lower(Pointwise()(s[1](f)), (:, :), (0, 0)) ==ₛ Pointwise()(f[1 // 2, 0] - f[-1 // 2, 0])
        @test lower(Pointwise()(Pointwise()(f)), (:,), (0,)) ==ₛ Pointwise()(Pointwise()(f[0]))
        # rules can depend on the number of dimensions
        @test (@inferred lower(Scaled()(f), (:, :), (0, 0))) ==ₛ f[0, 0] / 2
        # unsampled terms in the result of a rule are read at the evaluation point
        @test (@inferred lower(Unsampled()(s[1](f)), (:, :), (0, 0))) ==ₛ 2 * (f[1 // 2, 0] - f[-1 // 2, 0])
        @test (@inferred lower(Unsampled()(f), (:, :), (↓, 1 // 2))) ==ₛ 2 * f[↓, 1 // 2]
    end

    @testset "tensor expressions" begin
        @test (@inferred lower(divg(u), (:, :), (1 // 2, 1 // 2))) ==ₛ
            (u[1][1, 1 // 2] - u[1][0, 1 // 2]) + (u[2][1 // 2, 1] - u[2][1 // 2, 0])
        @test (@inferred lower((A ⋅ B)[1, 2], (:, :), (0, 0))) ==ₛ
            A[1, 1][0, 0] * B[1, 2][0, 0] + A[1, 2][0, 0] * B[2, 2][0, 0]
        @test (@inferred lower(f .+ g, (:, :), (0, 0))) ==ₛ f[0, 0] + g[0, 0]
        # components exceeding the number of dimensions
        @test_throws BoundsError lower(u[3], (:, :), (0, 0))
        @test_throws BoundsError lower(divg(A)[3], (:, :), (0, 0))
    end

    @testset "tensors in component form" begin
        gravity = pack([0, -1])
        # a call with all tensor arguments in component form is evaluated right away
        @test ispack(f * gravity)
        @test lower((f * gravity)[2], (:, :), (0, 0)) ==ₛ -f[0, 0]
        # mixed with coordinate-free tensors, the call is evaluated during lowering
        @test lower((u + gravity)[2], (:, :), (0, 0)) ==ₛ u[2][0, 0] - 1
        @test_throws DimensionMismatch lower((u + gravity)[2], (:, :, :), (0, 0, 0))
    end

    @testset "sampling with At" begin
        @vectors W
        # tensors are sampled through `At`, component by component
        @test (@inferred lower(At(↓, 0)(W)[2], (:, :), (0, 1 // 2))) ==ₛ W[2][↓, 1 // 2]
        @test (@inferred lower(At(1 // 2, 0)(W)[1], (:, :), (0, 0))) ==ₛ W[1][1 // 2, 0]
        # expressions are sampled as a whole, uniform terms are kept
        @test lower(At(↓, 0)(sin(f) + a), (:, :), (1, 1)) ==ₛ sin(f[↓, 1]) + a
        @test_throws ArgumentError lower(At(↓)(f), (:, :), (0, 0))
    end

    @testset "extrusion" begin
        # surfaces at the bottom and the top, interpolated with a normalised vertical coordinate
        @scalars B S c
        Z = c[:, 0] * S[0, ↑] + (1 - c[:, 0]) * B[0, ↓]
        @test (@inferred lower(Z, (:, :), (0, 0))) ==ₛ Z
        @test lower(Z, (:, :), (1 // 2, 1)) ==ₛ c[:, 1] * S[1 // 2, ↑] + (1 - c[:, 1]) * B[1 // 2, ↓]
    end

    @testset "coordinates" begin
        # point `1` lies on the lower wall, at the coordinate 0
        @test (@inferred lower(ξ¹ * f, (:,), (1 // 2,))) ==ₛ (𝑖 - 1 // 2) * f[1 // 2]
        @test (@inferred lower(ξ² * f, (:, :), (1 // 2, 0))) ==ₛ (𝑗 - 1) * f[1 // 2, 0]
        @test (@inferred lower(s[1](ξ¹ * f), (:,), (0,))) ==ₛ (𝑖 - 1 // 2) * f[1 // 2] - (𝑖 - 3 // 2) * f[-1 // 2]
        # the trace on the lower wall is at the coordinate 0
        @test (@inferred lower(ξ¹ * f, (:, :), (↓, 0))) ==ₛ Literal(0)
        @test_throws ArgumentError lower(ξ¹ * f, (:, :), (↑, 0))
        @test_throws ArgumentError lower(ξ¹ * f, (:, :), (:, 0))
    end

    @testset "errors" begin
        @test_throws ArgumentError lower(f, (), ())
        @test_throws ArgumentError lower(f, (:, :), (0,))
        @test_throws ArgumentError lower(f, (:,), (1 // 3,))
        @test_throws ArgumentError lower(v, (:, :), (0, 0))
        @test_throws ArgumentError lower(𝑖 * f, (:,), (0,))
        @test_throws ArgumentError lower(f[0], (:, :), (0, 0))
        @test_throws ArgumentError lower(d[3](f), (:, :), (0, 0))
        # 1D operators must be lifted to more dimensions
        @test_throws ArgumentError lower(d(f), (:, :), (0, 0))
        # a rule must return a scalar of samples
        @test_throws ArgumentError lower(Unindexed()(f), (:, :), (0, 0))
        @test_throws ArgumentError lower(ξ³ * f, (:,), (0,))
    end
end
