using Test
using Chmy

# a pointwise operator defined only by its rule, which keeps the call of the sampled argument
struct Pointwise <: Operator end
Chmy.lower_rule(op::Pointwise, f::DTerm, inds::Tuple) = op(f[inds...])

# an operator of one argument in one dimension
struct Shifted <: Operator end
Chmy.lower_rule(::Shifted, f::DTerm, i::Position) = f[i + 1]

# a scalar operator depending only on the number of dimensions
struct Scaled <: Operator end
Chmy.evaluate(::Scaled, args::Tuple{DTerm}, dims) = args[1] / dims

# an operator whose rule returns a tensor in component form instead of a scalar
struct Unindexed <: Operator end
Chmy.lower_rule(::Unindexed, f::DTerm, inds::Tuple) = pack([f, f])

# a tensor-valued operator without a component form
struct Unexpanded <: Operator end
Chmy.tensorrank(::Unexpanded, args) = 1
Chmy.checkranks(::Unexpanded, ::Tuple{Vararg{DTerm}}) = nothing

@testset "lowering" begin
    @scalars f g
    @uniform @scalars a
    @vectors u v
    @tensors 2 A B
    d, s = CentralDifference(), StaggeredCentralDifference()
    divg, grad = Divergence(s), Gradient(s)
    # the grid point of the kernel in 2D
    o = ((:, 0), (:, 0))

    @testset "samples" begin
        # fields are sampled at the positions, uniform terms are kept
        @test (@inferred (sin(f) + a * v[1])[o...]) ==ₛ sin(f[o...]) + a * v[1][o...]
        @test (@inferred a[(:, 0)]) ==ₛ a
        # uniform calls are expanded as well
        @test (@inferred d[1](a)[(:, 0)]) ==ₛ Literal(0)
        # samples are lowered on construction
        @test (@inferred d[1](g)[(:, 0)]) ==ₛ (1 // 2) * (g[(:, 1)] - g[(:, -1)])
        # samples cannot be sampled again, also inside expressions
        @test_throws ArgumentError sin(f[(:, 1 // 2), (↓, 0)])[o...]
        @test_throws ArgumentError (f[(:, 0)] + g)[(:, 0)]
    end

    @testset "laws" begin
        e = divg(grad(f)) * Project(↓, 2)(f)
        x = ((↓, 1 // 2), (:, -1 // 2))
        # a projection replaces the position along its axis
        @test Project(↑, 1)(e)[x...] ==ₛ e[↑, (:, -1 // 2)]
    end

    @testset "traces" begin
        # projections replace the position along their axis, and the innermost one wins
        @test (@inferred Project(↓, 1)(f)[(:, 1 // 2), (:, 0)]) ==ₛ f[↓, (:, 0)]
        @test (@inferred Project(↑, 1)(Project(↓, 1)(f))[o...]) ==ₛ f[↓, (:, 0)]
        @test (@inferred Project(↓, 2)(Project(:, 1)(f))[(:, 0), (:, 1 // 2)]) ==ₛ f[:, ↓]
        # traces are constant along their axis, so their normal derivative vanishes, and not along the other axes
        @test s[1](Project(↓, 1)(f))[(:, 0), (:, 1 // 2)] ==ₛ Literal(0)
        @test s[2](Project(↓, 1)(f))[(:, 0), (:, 1 // 2)] ==ₛ f[↓, (:, 1)] - f[↓, (:, 0)]
        @test s[1](f)[↓, (:, 1 // 2)] ==ₛ Literal(0)
        @test s[2](f)[↓, (:, 1 // 2)] ==ₛ f[↓, (:, 1)] - f[↓, (:, 0)]
        # lowering at a trace evaluates the expression on the traces
        @test (sin(f) * s[1](g))[(:, 0), :] ==ₛ sin(f[(:, 0), :]) * (g[(:, 1 // 2), :] - g[(:, -1 // 2), :])
    end

    @testset "laplacian" begin
        Δ = divg(grad(f))
        @test Δ[(:, 1 // 2)] ==ₛ f[(:, -1 // 2)] - 2 * f[(:, 1 // 2)] + f[(:, 3 // 2)]
        # the same stencil next to the walls, in the frame of the wall
        @test Δ[(↓, 1 // 2)] ==ₛ f[(↓, -1 // 2)] - 2 * f[(↓, 1 // 2)] + f[(↓, 3 // 2)]
        @test Δ[(↑, -1 // 2)] ==ₛ f[(↑, -3 // 2)] - 2 * f[(↑, -1 // 2)] + f[(↑, 1 // 2)]
        @test Δ[o...] ==ₛ f[(:, -1), (:, 0)] + f[(:, 0), (:, -1)] - 4 * f[o...] + f[(:, 1), (:, 0)] + f[(:, 0), (:, 1)]
    end

    @testset "rules" begin
        @test (@inferred d[1](d[2](f))[o...]) ==ₛ
            (1 // 2) * ((1 // 2) * (f[(:, 1), (:, 1)] - f[(:, 1), (:, -1)]) - (1 // 2) * (f[(:, -1), (:, 1)] - f[(:, -1), (:, -1)]))
        @test (@inferred s[1](s[2](f))[(:, 0), (:, 1 // 2)]) ==ₛ
            (f[(:, 1 // 2), (:, 1)] - f[(:, 1 // 2), (:, 0)]) - (f[(:, -1 // 2), (:, 1)] - f[(:, -1 // 2), (:, 0)])
        # rules of one argument at one position
        @test (@inferred Shifted()(f)[(↓, 0)]) ==ₛ f[(↓, 1)]
        # rules sample their arguments, which expands the operators inside them
        @test Pointwise()(s[1](f))[o...] ==ₛ Pointwise()(f[(:, 1 // 2), (:, 0)] - f[(:, -1 // 2), (:, 0)])
        @test Pointwise()(Pointwise()(f))[(:, 0)] ==ₛ Pointwise()(Pointwise()(f[(:, 0)]))
        # rules can depend on the number of dimensions
        @test (@inferred Scaled()(f)[o...]) ==ₛ f[o...] / 2
    end

    @testset "tensor expressions" begin
        @test (@inferred divg(u)[(:, 1 // 2), (:, 1 // 2)]) ==ₛ
            (u[1][(:, 1), (:, 1 // 2)] - u[1][(:, 0), (:, 1 // 2)]) + (u[2][(:, 1 // 2), (:, 1)] - u[2][(:, 1 // 2), (:, 0)])
        @test (@inferred (A ⋅ B)[1, 2][o...]) ==ₛ A[1, 1][o...] * B[1, 2][o...] + A[1, 2][o...] * B[2, 2][o...]
        @test (@inferred (f .+ g)[o...]) ==ₛ f[o...] + g[o...]
        # components exceeding the number of dimensions
        @test_throws BoundsError u[3][o...]
        @test_throws BoundsError divg(A)[3][o...]
    end

    @testset "tensors in component form" begin
        gravity = pack([0, -1])
        # a call with all tensor arguments in component form is evaluated right away
        @test ispack(f * gravity)
        @test (f * gravity)[2][o...] ==ₛ -f[o...]
        # mixed with coordinate-free tensors, the call is evaluated component by component
        @test (u + gravity)[2] ==ₛ u[2] - 1
        @test ((u + gravity) ⋅ u)[o...] ==ₛ u[1][o...]^2 + u[2][o...] * (u[2][o...] - 1)
        @test_throws DimensionMismatch ((u + gravity) ⋅ u)[(:, 0), (:, 0), (:, 0)]
    end

    @testset "projection" begin
        @vectors W
        # tensors are projected component by component
        @test (@inferred Project(↓, 1)(W)[2][(:, 0), (:, 1 // 2)]) ==ₛ W[2][↓, (:, 1 // 2)]
        # expressions are projected as a whole, uniform terms are kept
        @test Project(↓, 1)(sin(f) + a)[(:, 1), (:, 1)] ==ₛ sin(f[↓, (:, 1)]) + a
        # constants are not projected, and samples cannot be
        @test Project(↓, 1)(a) ==ₛ a
        @test_throws ArgumentError Project(↓, 1)(f[(:, 0), (:, 0)])[o...]
        # rules never sample sampled arguments
        @test_throws ArgumentError s[1](f[(:, 0)])[(:, 0)]
        @test sprint(show, Project(↑, 2)(f)) == "Project(↑, 2)(f)"
        # the axis must exist
        @test_throws ArgumentError Project(↓, 0)
        @test_throws ArgumentError Project(↓, 3)(f)[o...]
    end

    @testset "extrusion" begin
        # surfaces at the bottom and the top, interpolated with a normalised vertical coordinate
        @scalars B S c
        Z = Project(:, 1)(c) * Project(↑, 2)(S) + (1 - Project(:, 1)(c)) * Project(↓, 2)(B)
        @test (@inferred Z[o...]) ==ₛ c[:, (:, 0)] * S[(:, 0), ↑] + (1 - c[:, (:, 0)]) * B[(:, 0), ↓]
        @test Z[(:, 1 // 2), (:, 1)] ==ₛ c[:, (:, 1)] * S[(:, 1 // 2), ↑] + (1 - c[:, (:, 1)]) * B[(:, 1 // 2), ↓]
    end

    @testset "coordinates" begin
        # the coordinate of the origin of the frame, constant along the other axes, plus the offset
        @test (@inferred (ξ¹ * f)[(:, 1 // 2)]) ==ₛ (ξ¹[(:, 0)] + 1 // 2) * f[(:, 1 // 2)]
        @test (@inferred (ξ² * f)[(:, 1 // 2), (:, 0)]) ==ₛ ξ²[:, (:, 0)] * f[(:, 1 // 2), (:, 0)]
        # so differences of coordinates are exact
        @test (@inferred s[1](ξ¹)[(:, 0)]) ==ₛ Literal(1)
        @test (@inferred s[2](ξ¹)[(:, 0), (:, 1 // 2)]) ==ₛ Literal(0)
        # the lower wall is at the coordinate 0, and the upper wall at the sampled coordinate of its origin
        @test (@inferred (ξ¹ * f)[(↓, 1 // 2)]) ==ₛ (1 // 2) * f[(↓, 1 // 2)]
        @test (@inferred (ξ¹ * f)[↓, (:, 0)]) ==ₛ Literal(0)
        @test (@inferred ξ¹[(↑, -1 // 2)]) ==ₛ ξ¹[(↑, 0)] - 1 // 2
        @test (@inferred ξ¹[↑, (:, 0)]) ==ₛ ξ¹[(↑, 0), :]
        # coordinates vary along their axis
        @test_throws ArgumentError ξ¹[:, (:, 0)]
    end

    @testset "errors" begin
        @test_throws ArgumentError f[(:, 1 // 3)]
        @test_throws ArgumentError v[o...]
        @test_throws ArgumentError d[3](f)[o...]
        # 1D operators must be lifted to more dimensions
        @test_throws ArgumentError d(f)[o...]
        # a rule must return a scalar of samples
        @test_throws ArgumentError Unindexed()(f)[o...]
        # tensor-valued calls must have a component form
        @test_throws ArgumentError Unexpanded()(f)[1][o...]
        @test_throws ArgumentError (ξ³ * f)[(:, 0)]
    end
end
