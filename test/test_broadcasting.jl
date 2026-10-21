using Test
using Chmy
using Chmy: BroadcastedFun, Kind

@testset "broadcasting" begin
    @scalars a b
    @tensors 2 A B @sym(S) @diag(D) @alt(K)

    expr = @inferred Base.Broadcast.broadcasted(sin, A)
    @test operation(expr) === BroadcastedFun(sin)
    @test isequal(arguments(expr), (A,))
    @test tensorrank(expr) == 2

    # broadcasting over tensors in component form is evaluated right away
    ac, bc = components(A, 2), components(B, 2)
    direct = @inferred Base.Broadcast.broadcasted(sin, ac)
    @test direct[1, 2] ==ₛ sin(A[1, 2])
    for (input, expected) in (
            (ac .* bc, A[1, 2] * B[1, 2]),
            (a .+ ac, a + A[1, 2]),
            (ac .+ 2, A[1, 2] + 2),
            (ac .^ 2, A[1, 2]^2),
            (sin.(ac .+ a) .* bc .^ 2, sin(A[1, 2] + a) * B[1, 2]^2),
        )
        @test input[1, 2] ==ₛ expected
    end
    # broadcasting over scalars, or mixing component and coordinate-free forms, is evaluated while lowering
    far = (Far(), Far())
    @test (@inferred lower(a .+ b, far)) ==ₛ a[0, 0] + b[0, 0]
    mixed = ac .+ B
    @test !ispack(mixed)
    @test (@inferred lower(mixed[1, 2], far)) ==ₛ A[1, 2][0, 0] + B[1, 2][0, 0]

    Sc, Dc, Kc = components(S, 2), components(D, 2), components(K, 2)
    @test tensorkind(sin.(Sc) .+ Dc) == Kind.Sym()
    diagonal = exp.(Dc)
    @test tensorkind(diagonal) == Kind.Sym()
    @test simplify(diagonal[1, 2]) ==ₛ Literal(1)
    @test tensorkind(Kc .+ 1) == Kind.None()

    @vectors u
    @test_throws ArgumentError A .+ u
    @test_throws DimensionMismatch ac .+ components(B, 3)
end
