using Test
using Chmy
using Chmy: Tensor, Kind

@testset "macros" begin
    scalars = @scalars a b @uniform(c)
    @test isequal(scalars, (a, b, c))
    @test isequal(
        scalars, (
            Tensor(0, :a, Kind.None(), false),
            Tensor(0, :b, Kind.None(), false),
            Tensor(0, :c, Kind.None(), true),
        )
    )

    u, v = @vectors p q
    @test isequal((u, v), (p, q))
    @test isequal(
        (p, q), (
            Tensor(1, :p, Kind.None(), false),
            Tensor(1, :q, Kind.None(), false),
        )
    )

    tensors = @tensors 2 A @sym(S, R) @diag(D) @alt(K)
    @test isequal(tensors, (A, S, R, D, K))
    @test isequal(
        tensors, (
            Tensor(2, :A, Kind.None(), false),
            Tensor(2, :S, Kind.Sym(), false),
            Tensor(2, :R, Kind.Sym(), false),
            Tensor(2, :D, Kind.Diag(), false),
            Tensor(2, :K, Kind.Alt(), false),
        )
    )

    @test isequal((@uniform @vectors w), (Tensor(1, :w, Kind.None(), true),))
    uniforms = @uniform begin
        @scalars x
        @tensors 2 @sym(T)
    end
    @test isequal(uniforms, (x, T))
    @test isequal(
        uniforms, (
            Tensor(0, :x, Kind.None(), true),
            Tensor(2, :T, Kind.Sym(), true),
        )
    )

    @test_throws ArgumentError @macroexpand @tensors R A
    @test_throws ArgumentError @macroexpand @scalars @sym(a)
end
