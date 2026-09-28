using Test
using Chmy
using Chmy: BroadcastedFun, TensorComponents, Kind

@testset "broadcasting" begin
    @scalars a b
    @tensors 2 A B @sym(S) @diag(D) @alt(K)

    expr = @inferred Base.Broadcast.broadcasted(sin, A)
    @test operation(expr) === BroadcastedFun(sin)
    @test isequal(arguments(expr), (A,))
    @test tensorrank(expr) == 2

    @test (@inferred Union{DTerm, TensorComponents} components(a .+ b, 2)) ==ₛ a + b
    for (input, expected) in (
            (sin.(A), sin(A[1, 2])),
            (A .* B, A[1, 2] * B[1, 2]),
            (a .+ A, a + A[1, 2]),
            (A .+ 2, A[1, 2] + 2),
            (A .^ 2, A[1, 2]^2),
            (sin.(A .+ a) .* B .^ 2, sin(A[1, 2] + a) * B[1, 2]^2),
        )
        tc = @inferred Union{DTerm, TensorComponents} components(input, 2)
        @test tc[1, 2] ==ₛ expected
    end

    ac, bc = components(A, 2), components(B, 2)
    direct = @inferred Union{DTerm, TensorComponents} Base.Broadcast.broadcasted(sin, ac)
    @test direct[1, 2] ==ₛ sin(A[1, 2])
    for (input, expected) in (
            (ac .* bc, A[1, 2] * B[1, 2]),
            (ac .+ B, A[1, 2] + B[1, 2]),
            (a .+ ac, a + A[1, 2]),
            (ac .^ 2, A[1, 2]^2),
        )
        @test input[1, 2] ==ₛ expected
    end

    symmetric = @inferred Union{DTerm, TensorComponents} components(sin.(S) .+ D, 2)
    @test tensorkind(symmetric) == Kind.Sym()
    diagonal = @inferred Union{DTerm, TensorComponents} components(exp.(D), 2)
    @test tensorkind(diagonal) == Kind.Sym()
    @test simplify(diagonal[1, 2]) ==ₛ Literal(1)
    alternating = @inferred Union{DTerm, TensorComponents} components(K .+ 1, 2)
    @test tensorkind(alternating) == Kind.None()

    @vectors u
    @test_throws ArgumentError A .+ u
    @test_throws DimensionMismatch ac .+ components(B, 3)
end
