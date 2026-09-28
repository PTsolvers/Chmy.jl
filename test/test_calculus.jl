using Test
using Chmy
using Chmy: TensorComponents

@testset "calculus" begin
    @scalars a
    @vectors u
    @tensors 2 A
    d = CentralDifference()
    grad, divg, curl = Gradient(d), Divergence(d), Curl(d)

    g = @inferred Union{DTerm, TensorComponents} components(grad(a), 2)
    @test g[1] ==ₛ d[1](a)
    @test g[2] ==ₛ d[2](a)
    g = @inferred Union{DTerm, TensorComponents} components(grad(u), 2)
    @test g[1, 2] ==ₛ d[1](u[2])
    @test g[2, 1] ==ₛ d[2](u[1])

    @test (@inferred Union{DTerm, TensorComponents} components(divg(u), 2)) ==ₛ d[1](u[1]) + d[2](u[2])
    div = @inferred Union{DTerm, TensorComponents} components(divg(A), 2)
    @test div[2] ==ₛ d[1](A[1, 2]) + d[2](A[2, 2])
    @test (@inferred Union{DTerm, TensorComponents} components(divg(grad(a)), 2)) ==ₛ d[1](d[1](a)) + d[2](d[2](a))

    c = @inferred Union{DTerm, TensorComponents} components(curl(u), 3)
    @test c[1] ==ₛ d[2](u[3]) - d[3](u[2])
    @test c[2] ==ₛ d[3](u[1]) - d[1](u[3])
    @test c[3] ==ₛ d[1](u[2]) - d[2](u[1])

    @test_throws ArgumentError divg(a)
    @test_throws ArgumentError curl(A)
    @test_throws ArgumentError components(curl(u), 2)
end
