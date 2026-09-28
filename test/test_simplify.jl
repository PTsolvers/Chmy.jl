using Test
using Chmy

@testset "canonicalize" begin
    @scalars a b c
    @vectors u v
    @tensors 2 A @sym(S) @alt(K)
    I, Z = ℐ(2), 𝒪(2)

    @testset "literal values" begin
        for (input, expected) in (
                (0.0, 0), (-0.0, 0), (0 // 1, 0), (false, 0), (0.0 + 0.0im, 0),
                (1.0, 1), (1 // 1, 1), (true, 1), (1.0 + 0.0im, 1),
                (2.0, 2), (-2.0, -2), (3 // 1, 3), (2.0 + 0.0im, 2),
                (Int32(2), 2), (big(2), 2),
            )
            for f in (canonicalize, simplify)
                result = @inferred f(Literal(input))
                @test value(result) === expected
                @test value(@inferred f(result)) === expected
            end
        end
        for input in (0.5, 2 // 3, 2.0 + 1.0im, Inf, -Inf, NaN)
            term = Literal(input)
            @test (@inferred canonicalize(term)) === term
            @test (@inferred simplify(term)) === term
        end
        @test_throws InexactError canonicalize(Literal(big(typemax(Int)) + 1))
    end

    @testset "scalar constants" begin
        for (input, expected) in (
                (𝒪(0), 0), (ℐ(0), 1),
                (sin(Literal(0)), 0), (cos(Literal(0)), 1), (sqrt(Literal(4)), 2),
                (Literal(2.0) * Literal(1), 2),
            )
            @test value(@inferred canonicalize(input)) === expected
            @test value(@inferred simplify(input)) === expected
        end
        for input in (𝒪(1), 𝒪(2), ℐ(1), ℐ(2))
            @test (@inferred canonicalize(input)) === input
            @test (@inferred simplify(input)) === input
        end
        for input in (a^0.5 * a^1.5, (a^0.5)^4.0)
            result = @inferred canonicalize(input)
            @test result ==ₛ a^2
            @test value(arguments(result)[2]) === 2
        end
    end

    @testset "scalar algebra" begin
        for (input, expected) in (
                (Literal(2) + Literal(3), Literal(5)),
                (Literal(2) / Literal(4), Literal(1 // 2)),
                (sin(Literal(0)), Literal(0)),
                (b + (c + a), a + b + c),
                (b * (c * a), a * b * c),
                (a + b + a, 2a + b),
                (a - a, Literal(0)),
                (a + b - a - b, Literal(0)),
                (a + b - a, b),
                (a + b - a - b + c, c),
                (a * b - b * a + 2, Literal(2)),
                (c + a - b, (a - b) + c),
                (c - a - b, -(a + b - c)),
                (b^2 + a * b + a^2, a^2 + a * b + b^2),
                (a^2 + b^2 + b * a, a^2 + a * b + b^2),
                ((2 // 3) * a + (1 // 3) * a, a),
                (a * b * a^2, a^3 * b),
                (a * inv(a), Literal(1)),
                (a / (a * b), inv(b)),
                ((a^b)^c, a^(b * c)),
                (a^b * a^c, a^(b + c)),
                (a^b / a^b, Literal(1)),
                (a^b * a^(-b), Literal(1)),
                ((a^b)^(2c), a^(Literal(2) * b * c)),
                (a^0, Literal(1)),
                (b * a^(-2), b / a^2),
                ((2a)^(-3), Literal(1 // 8) / a^3),
                (-(-a), a),
                (b - a, -(a - b)),
                ((-a)^2, a^2),
                ((-a)^3, -a^3),
            )
            @test (@inferred canonicalize(input)) ==ₛ expected
        end
    end

    @testset "tensor identities" begin
        for (input, expected) in (
                (u - u, 𝒪(1)),
                (inv(A) ⋅ A, I),
                (A ⋅ inv(A), I),
                (A ⋅ adj(A), det(A) * I),
                (adj(A) ⋅ A, det(A) * I),
                (A' ⋅ cof(A), det(A) * I),
                (cof(A) ⋅ A', det(A) * I),
                (cof(S) ⋅ S, det(S) * I),
                (I ⋅ A, A),
                (A ⋅ I, A),
                (A ⊡ I, tr(A)),
                (I ⊡ A, tr(A)),
                (inv(inv(A)), A),
                (inv(I), I),
                (cof(I), I),
                (adj(S), cof(S)),
                (det(I), Literal(1)),
                (𝒪(1) ⋅ v, Literal(0)),
                (Z ⋅ A, Z),
                (A ⋅ Z, Z),
                (inv(Z) ⋅ Z, Z),
                (Z ⋅ inv(Z), Z),
                (adj(Z) ⋅ Z, Z),
                (Z ⋅ adj(Z), Z),
                (Z ⊡ A, Literal(0)),
                (A ⊡ Z, Literal(0)),
                (𝒪(1) ⊗ v, Z),
                (v × v, 𝒪(1)),
                (A'', A),
                (S', S),
                (K', -K),
                (sym(S), S),
                (sym(K), Z),
                (asym(S), Z),
                (asym(K), K),
                (sym(sym(A)), sym(A)),
                (asym(asym(A)), asym(A)),
                (sym(asym(A)), Z),
                (asym(sym(A)), Z),
                (sym(A)', sym(A)),
                (asym(A)', -asym(A)),
                (tr(asym(A)), Literal(0)),
                (diag(asym(A)), 𝒪(1)),
                (tr(K), Literal(0)),
                (diag(K), 𝒪(1)),
                (diag(I), ℐ(1)),
            )
            @test (@inferred canonicalize(input)) ==ₛ expected
        end
    end
end

@testset "simplify" begin
    @scalars a b c
    @tensors 2 A

    expr = sin(b + a)
    @test (@inferred canonicalize(expr)) ==ₛ expr
    @test (@inferred simplify(expr)) ==ₛ sin(a + b)
    @test (@inferred simplify(sin(a + a) - sin(2a))) ==ₛ Literal(0)
    @test value(@inferred simplify(cos(sin(Literal(0))))) === 1
    @test (@inferred simplify(tr((inv(A) ⋅ A) ⋅ A))) ==ₛ tr(A)
    @test (@inferred simplify(adj(ℐ(2)))) ==ₛ ℐ(2)
    @test (@inferred simplify(a * (c + b))) ==ₛ a * (b + c)
    @test (@inferred simplify(c * (b - a))) ==ₛ -(c * (a - b))
    @test (@inferred simplify((a - b) / (b - a))) ==ₛ Literal(-1)

    expr = sin(a + Literal(2.0))
    @test (@inferred canonicalize(expr)) === expr
    result = @inferred simplify(expr)
    @test result ==ₛ sin(a + Literal(2))
    @test value(arguments(only(arguments(result)))[2]) === 2
    @test value(arguments(only(arguments(@inferred simplify(result))))[2]) === 2
    @test (@inferred simplify(a + 𝒪(0))) ==ₛ a
    @test (@inferred simplify(a * ℐ(0))) ==ₛ a
end
