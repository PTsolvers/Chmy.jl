using Test
using Chmy

# a pointwise operator defined only by its rule, which keeps the call and samples the argument here
struct Pointwise <: Operator end
Chmy.lower_rule(op::Pointwise, args::Tuple{DTerm}, coord, ctx) = op(args[1][map(zero, coord)...])

# a scalar operator depending on the number of dimensions
struct Scaled <: Operator end
Chmy.lower_rule(::Scaled, args::Tuple{DTerm}, coord, ctx) = (args[1] / length(coord))[map(zero, coord)...]

# an operator whose rule returns a tensor in component form instead of a scalar
struct Unindexed <: Operator end
Chmy.lower_rule(::Unindexed, args::Tuple{DTerm}, coord, ctx) = pack([args[1], args[1]])

@testset "lowering" begin
    @scalars f g @uniform(a)
    @vectors u v
    @tensors 2 A B
    d, s = CentralDifference(), StaggeredCentralDifference()
    divg, grad = Divergence(s), Gradient(s)

    @testset "samples" begin
        # fields are located at the evaluation point, uniform terms are kept
        @test (@inferred lower(sin(f) + a * v[1], (:, :), (0, 0))) ==ₛ sin(f[𝓅, 𝓅][0, 0]) + a * v[1][𝓅, 𝓅][0, 0]
        @test (@inferred lower(a, (:,), (0,))) ==ₛ a
        # the offsets of nested samples accumulate, half-integer coordinates are segments
        @test (@inferred lower(sin(f[1 // 2, 0]), (:, :), (0, 0))) ==ₛ sin(f[𝓈, 𝓅][0, 0])
        @test (@inferred lower(sin(f)[-1 // 2], (:,), (0,))) ==ₛ sin(f[𝓈][-1])
        @test (@inferred lower(f[1 // 2][1], (:,), (0,))) ==ₛ f[𝓈][1]
        # uniform calls are expanded as well
        @test (@inferred lower(d[1](a), (:,), (0,))) ==ₛ (1 // 2) * (a - a)
    end

    @testset "origin" begin
        laplacian(l) = f[l, l][-1, 0] + f[l, l][0, -1] - 4 * f[l, l][0, 0] + f[l, l][1, 0] + f[l, l][0, 1]
        @test simplify(lower(divg(grad(f)), (:, :), (0, 0))) ==ₛ simplify(laplacian(𝓅))
        @test simplify(lower(divg(grad(f)), (:, :), (1 // 2, 1 // 2))) ==ₛ simplify(laplacian(𝓈))
        @test (@inferred lower(f, (:, :), (1 // 2, 0))) ==ₛ f[𝓈, 𝓅][0, 0]
        @test (@inferred lower(f[1 // 2, -1], (:, :), (0, 0))) ==ₛ f[𝓈, 𝓅][0, -1]
        @test (@inferred lower(f[3 // 2], (:,), (1 // 2,))) ==ₛ f[𝓅][2]
        # along the sides of a face, the origin is the outward distance from the wall, and the indices are relative
        # to the wall point
        @test (@inferred lower(f, (↓,), (-1 // 2,))) ==ₛ f[𝓈][0]
        @test (@inferred lower(f, (↑,), (-1 // 2,))) ==ₛ f[𝓈][-1]
        @test (@inferred lower(f, (↓,), (0,))) ==ₛ f[𝓅][0]
        @test (@inferred lower(f, (↑,), (1 // 2,))) ==ₛ f[𝓈][0]
        @test (@inferred lower(s[1](f), (↓, :), (0, 1 // 2))) ==ₛ f[𝓈, 𝓈][0, 0] - f[𝓈, 𝓈][-1, 0]
        @test (@inferred lower(s[1](f), (↑, ↓), (0, -1 // 2))) ==ₛ f[𝓈, 𝓈][0, 0] - f[𝓈, 𝓈][-1, 0]
    end

    @testset "rules" begin
        @test (@inferred lower(d[1](d[2](f)), (:, :), (0, 0))) ==ₛ
            (1 // 2) * ((1 // 2) * (f[𝓅, 𝓅][1, 1] - f[𝓅, 𝓅][1, -1]) - (1 // 2) * (f[𝓅, 𝓅][-1, 1] - f[𝓅, 𝓅][-1, -1]))
        @test (@inferred lower(s[1](s[2](f)), (:, :), (0, 1 // 2))) ==ₛ
            (f[𝓈, 𝓅][0, 1] - f[𝓈, 𝓅][0, 0]) - (f[𝓈, 𝓅][-1, 1] - f[𝓈, 𝓅][-1, 0])
        # calls returned by rules are kept, and the operators inside their arguments are expanded
        @test lower(Pointwise()(s[1](f)), (:, :), (0, 0)) ==ₛ Pointwise()(f[𝓈, 𝓅][0, 0] - f[𝓈, 𝓅][-1, 0])
        @test lower(Pointwise()(Pointwise()(f)), (:,), (0,)) ==ₛ Pointwise()(Pointwise()(f[𝓅][0]))
        # rules can depend on the number of dimensions
        @test (@inferred lower(Scaled()(f), (:, :), (0, 0))) ==ₛ f[𝓅, 𝓅][0, 0] / 2
    end

    @testset "tensor expressions" begin
        @test (@inferred lower(divg(u), (:, :), (0, 0))) ==ₛ
            (u[1][𝓈, 𝓅][0, 0] - u[1][𝓈, 𝓅][-1, 0]) + (u[2][𝓅, 𝓈][0, 0] - u[2][𝓅, 𝓈][0, -1])
        @test (@inferred lower((A ⋅ B)[1, 2], (:, :), (0, 0))) ==ₛ
            A[1, 1][𝓅, 𝓅][0, 0] * B[1, 2][𝓅, 𝓅][0, 0] + A[1, 2][𝓅, 𝓅][0, 0] * B[2, 2][𝓅, 𝓅][0, 0]
        @test (@inferred lower(f .+ g, (:, :), (0, 0))) ==ₛ f[𝓅, 𝓅][0, 0] + g[𝓅, 𝓅][0, 0]
        # components exceeding the number of dimensions
        @test_throws BoundsError lower(u[3], (:, :), (0, 0))
        @test_throws BoundsError lower(divg(A)[3], (:, :), (0, 0))
    end

    @testset "tensors in component form" begin
        gravity = pack([0, -1])
        # a call with all tensor arguments in component form is evaluated right away
        @test ispack(f * gravity)
        @test simplify(lower((f * gravity)[2], (:, :), (0, 0))) ==ₛ -f[𝓅, 𝓅][0, 0]
        # mixed with coordinate-free tensors, the call is evaluated during lowering
        @test simplify(lower((u + gravity)[2], (:, :), (0, 0))) ==ₛ u[2][𝓅, 𝓅][0, 0] - 1
        @test_throws DimensionMismatch lower((u + gravity)[2], (:, :, :), (0, 0, 0))
    end

    @testset "coordinates" begin
        # point `1` lies on the lower wall, at the coordinate 0
        @test (@inferred lower(ξ¹ * f, (:,), (1 // 2,))) ==ₛ (𝑖 - 1 // 2) * f[𝓈][0]
        @test (@inferred lower(ξ² * f, (:, :), (1 // 2, 0))) ==ₛ (𝑗 - 1) * f[𝓈, 𝓅][0, 0]
        @test (@inferred lower(s[1](ξ¹ * f), (:,), (0,))) ==ₛ (𝑖 - 1 // 2) * f[𝓈][0] - (𝑖 - 3 // 2) * f[𝓈][-1]
        # the grid point of the kernel is the wall point, which is point `1` at the lower wall
        @test (@inferred lower(ξ¹ * f, (↓,), (-1 // 2,))) ==ₛ (𝑖 - 1 // 2) * f[𝓈][0]
        @test (@inferred lower(ξ¹ * f, (↑,), (-1 // 2,))) ==ₛ (𝑖 - 3 // 2) * f[𝓈][-1]
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
        # locations are produced by lowering
        @test_throws ArgumentError lower(f[𝓅, 𝓅][1 // 2, -1], (:, :), (0, 0))
        @test_throws ArgumentError lower(sin(f)[𝓈], (:,), (0,))
        # a rule must return a scalar
        @test_throws ArgumentError lower(Unindexed()(f), (:, :), (0, 0))
        @test_throws ArgumentError lower(ξ³ * f, (:,), (0,))
    end
end
