using Test
using Chmy

@testset "calculus" begin
    @scalars a
    @vectors u
    @tensors 2 A
    d = CentralDifference()
    grad, divg, curl = Gradient(d), Divergence(d), Curl(d)

    # a component of the gradient is the derivative of a component of the argument in any number of dimensions
    @test (@inferred grad(a)[2]) ==ₛ d[2](a)
    @test (@inferred grad(u)[1, 2]) ==ₛ d[1](u[2])
    @test (@inferred grad(u)[2, 1]) ==ₛ d[2](u[1])
    @test (@inferred grad(components(u, 2))) ==ₛ pack([d[1](u[1]) d[1](u[2]); d[2](u[1]) d[2](u[2])])

    # the divergence and the curl depend on the number of dimensions, and are evaluated in component form
    @test (@inferred divg(components(u, 2))) ==ₛ d[1](u[1]) + d[2](u[2])
    @test (@inferred divg(components(A, 2))[2]) ==ₛ d[1](A[1, 2]) + d[2](A[2, 2])
    v = components(u, 3)
    @test (@inferred curl(v)[1]) ==ₛ d[2](u[3]) - d[3](u[2])
    @test (@inferred curl(v)[2]) ==ₛ d[3](u[1]) - d[1](u[3])
    @test (@inferred curl(v)[3]) ==ₛ d[1](u[2]) - d[2](u[1])

    # nested operators are evaluated at once
    @test (@inferred divg(components(grad(a), 2))) ==ₛ d[1](d[1](a)) + d[2](d[2](a))
    @test (@inferred lower(divg(grad(a)), (:, :), (0, 0))) ==ₛ lower(d[1](d[1](a)) + d[2](d[2](a)), (:, :), (0, 0))

    @test_throws ArgumentError divg(a)
    @test_throws ArgumentError curl(A)
    @test_throws ArgumentError curl(components(u, 2))
    @test_throws ArgumentError lower(curl(u)[1], (:, :), (0, 0))
    # tensors in component form must have the same number of dimensions
    @test_throws DimensionMismatch lower(divg(u + pack([a, a, a])), (:, :), (0, 0))
end
