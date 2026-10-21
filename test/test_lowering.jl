using Test
using Chmy

# a pointwise operator defined only by its rule, which keeps the call and samples the argument here
struct Pointwise <: Operator end
Chmy.lower_rule(op::Pointwise, args::Tuple{DTerm}, offs, block) = op(args[1][map(zero, offs)...])

# a scalar operator depending on the number of dimensions
struct Scaled <: Operator end
Chmy.lower_rule(::Scaled, args::Tuple{DTerm}, offs, block) = (args[1] / length(offs))[map(zero, offs)...]

# an operator whose rule returns a tensor in component form instead of a scalar
struct Unindexed <: Operator end
Chmy.lower_rule(::Unindexed, args::Tuple{DTerm}, offs, block) = pack([args[1], args[1]])

@testset "lowering" begin
    @scalars f g @uniform(a)
    @vectors u v
    @tensors 2 A B
    d, s = CentralDifference(), StaggeredCentralDifference()
    far1, far2 = (Far(),), (Far(), Far())

    @testset "samples" begin
        # fields are sampled at the evaluation point, uniform terms are kept
        @test (@inferred lower(sin(f) + a * v[1], far2)) ==ₛ sin(f[0, 0]) + a * v[1][0, 0]
        @test (@inferred lower(a, far1)) ==ₛ a
        # the offsets of nested samples accumulate
        @test (@inferred lower(sin(f[1 // 2, 0]), far2)) ==ₛ sin(f[1 // 2, 0])
        @test (@inferred lower(sin(f)[-1 // 2], far1)) ==ₛ sin(f[-1 // 2])
        @test (@inferred lower(f[1 // 2][1], far1)) ==ₛ f[3 // 2]
        # explicit locations are kept until the samples are located
        @test (@inferred lower(f[𝓅, 𝓅][1 // 2, -1], far2)) ==ₛ f[𝓅, 𝓅][1 // 2, -1]
        # uniform calls are expanded as well
        @test (@inferred lower(d[1](a), far1)) ==ₛ (1 // 2) * (a - a)
    end

    @testset "rules" begin
        @test (@inferred lower(d[1](d[2](f)), far2)) ==ₛ
            (1 // 2) * ((1 // 2) * (f[1, 1] - f[1, -1]) - (1 // 2) * (f[-1, 1] - f[-1, -1]))
        @test (@inferred lower(s[1](s[2](f)), far2)) ==ₛ
            (f[1 // 2, 1 // 2] - f[1 // 2, -1 // 2]) - (f[-1 // 2, 1 // 2] - f[-1 // 2, -1 // 2])
        # calls returned by rules are kept, and the operators inside their arguments are expanded
        @test lower(Pointwise()(s[1](f)), far2) ==ₛ Pointwise()(f[1 // 2, 0] - f[-1 // 2, 0])
        @test lower(Pointwise()(Pointwise()(f)), far1) ==ₛ Pointwise()(Pointwise()(f[0]))
        # rules can depend on the number of dimensions
        @test (@inferred lower(Scaled()(f), far2)) ==ₛ f[0, 0] / 2
    end

    @testset "tensor expressions" begin
        divg = Divergence(s)
        @test (@inferred lower(divg(u), far2)) ==ₛ (u[1][1 // 2, 0] - u[1][-1 // 2, 0]) + (u[2][0, 1 // 2] - u[2][0, -1 // 2])
        @test (@inferred lower((A ⋅ B)[1, 2], far2)) ==ₛ A[1, 1][0, 0] * B[1, 2][0, 0] + A[1, 2][0, 0] * B[2, 2][0, 0]
        @test (@inferred lower(f .+ g, far2)) ==ₛ f[0, 0] + g[0, 0]
        # components exceeding the number of dimensions
        @test_throws BoundsError lower(u[3], far2)
        @test_throws BoundsError lower(divg(A)[3], far2)
    end

    @testset "tensors in component form" begin
        gravity = pack([0, -1])
        # a call with all tensor arguments in component form is evaluated right away
        @test ispack(f * gravity)
        @test simplify(lower((f * gravity)[2], far2)) ==ₛ -f[0, 0]
        # mixed with coordinate-free tensors, the call is evaluated during lowering
        @test simplify(lower((u + gravity)[2], far2)) ==ₛ u[2][0, 0] - 1
        @test_throws DimensionMismatch lower((u + gravity)[2], (Far(), Far(), Far()))
    end

    @testset "locations" begin
        @test (@inferred locate(sin(f[0, 0]) + a * v[1][0, 0], (𝓅, 𝓈))) ==ₛ sin(f[𝓅, 𝓈][0, 0]) + a * v[1][𝓅, 𝓈][0, 0]
        @test (@inferred locate(f[0, 0], 𝓅, 𝓈)) ==ₛ locate(f[0, 0], (𝓅, 𝓈))
        # the locations and the grid indices follow from the positions of the samples
        @test (@inferred locate(f[1 // 2, -1], 𝓅, 𝓅)) ==ₛ f[𝓈, 𝓅][0, -1]
        @test (@inferred locate(f[-1 // 2], 𝓅)) ==ₛ f[𝓈][-1]
        @test (@inferred locate(f[3 // 2], 𝓈)) ==ₛ f[𝓅][2]
        @test (@inferred locate(f[𝓅, 𝓅][1 // 2, -1], 𝓈, 𝓅)) ==ₛ f[𝓅, 𝓅][1, -1]
        @test (@inferred locate(lower(s[1](s[2](f)), far2), 𝓅, 𝓈)) ==ₛ
            (f[𝓈, 𝓅][0, 1] - f[𝓈, 𝓅][0, 0]) - (f[𝓈, 𝓅][-1, 1] - f[𝓈, 𝓅][-1, 0])
        # point `1` lies on the lower wall, at the coordinate 0
        @test (@inferred locate(lower(ξ¹ * f, far1), 𝓈)) ==ₛ (𝑖 - 1 // 2) * f[𝓈][0]
        @test (@inferred locate(lower(ξ² * f, far2), 𝓈, 𝓅)) ==ₛ (𝑗 - 1) * f[𝓈, 𝓅][0, 0]
        @test (@inferred locate(lower(s[1](ξ¹ * f), far1), 𝓅)) ==ₛ (𝑖 - 1 // 2) * f[𝓈][0] - (𝑖 - 3 // 2) * f[𝓈][-1]
    end

    @testset "errors" begin
        @test_throws ArgumentError lower(f, ())
        @test_throws ArgumentError lower(v, far2)
        @test_throws ArgumentError lower(𝑖 * f, far1)
        @test_throws ArgumentError lower(f[0], far2)
        @test_throws ArgumentError lower(d[3](f), far2)
        # 1D operators must be lifted to more dimensions
        @test_throws ArgumentError lower(d(f), far2)
        # explicit locations of expressions
        @test_throws ArgumentError lower(sin(f)[𝓈], far1)
        # a rule must return a scalar
        @test_throws ArgumentError lower(Unindexed()(f), far2)
        @test_throws ArgumentError locate(f, 𝓅)
        @test_throws ArgumentError locate(f[0], ())
        @test_throws ArgumentError locate(f[𝓈][0], 𝓅)
        @test_throws ArgumentError locate(f[𝓈][0, 0], 𝓈, 𝓈)
        @test_throws ArgumentError locate(ξ²[0], 𝓅)
    end
end
