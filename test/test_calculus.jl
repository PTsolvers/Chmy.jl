using Test
using Chmy

@testset "calculus" begin
    @scalars a
    @vectors u
    @tensors 2 A
    d = CentralDifference()
    grad, divg, curl = Gradient(d), Divergence(d), Curl(d)
    face, x = (:, :), (0 // 1, 0 // 1)

    # a component is the derivative of a component of the argument at the evaluation point
    @test (@inferred lower_rule(grad, (a,), (2,), face, x)) ==ₛ d[2](a)
    @test (@inferred lower_rule(grad, (u,), (1, 2), face, x)) ==ₛ d[1](u[2])
    @test (@inferred lower_rule(grad, (u,), (2, 1), face, x)) ==ₛ d[2](u[1])
    @test (@inferred lower_rule(divg, (u,), (), face, x)) ==ₛ d[1](u[1]) + d[2](u[2])
    @test (@inferred lower_rule(divg, (A,), (2,), face, x)) ==ₛ d[1](A[1, 2]) + d[2](A[2, 2])
    # nested operators are expanded while lowering
    @test (@inferred lower(divg(grad(a)), (:, :), (0, 0))) ==ₛ lower(d[1](d[1](a)) + d[2](d[2](a)), (:, :), (0, 0))

    face3, x3 = (:, :, :), (0 // 1, 0 // 1, 0 // 1)
    @test (@inferred lower_rule(curl, (u,), (1,), face3, x3)) ==ₛ d[2](u[3]) - d[3](u[2])
    @test (@inferred lower_rule(curl, (u,), (2,), face3, x3)) ==ₛ d[3](u[1]) - d[1](u[3])
    @test (@inferred lower_rule(curl, (u,), (3,), face3, x3)) ==ₛ d[1](u[2]) - d[2](u[1])

    @test_throws ArgumentError divg(a)
    @test_throws ArgumentError curl(A)
    @test_throws ArgumentError lower_rule(curl, (u,), (1,), face, x)
    # tensors in component form must have the same number of dimensions
    @test_throws DimensionMismatch lower(divg(pack([a, a, a])), (:, :), (0, 0))
end
