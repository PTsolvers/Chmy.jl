using Test
using Chmy
using Chmy: DExpr, Call, Pack, Kind

# a call built without the rules
raw(f, args...) = DExpr(Call(Fun(f)), args)

# the term and all its subterms
subterms(t) = isexpr(t) ? DTerm[t; mapreduce(subterms, vcat, children(t); init = DTerm[])] : DTerm[t]

# a function with a rule that doesn't apply to literals: sq(sqrt(x)) = x
sq(x) = x^2
Chmy.evaluate(::typeof(sq), (x,)::Tuple{DTerm}) = Chmy.argof(sqrt, x)

@testset "canonical construction" begin
    @scalars a b c
    @vectors u v
    @tensors 2 A @sym(S) @alt(K)
    I, Z = ℐ(2), 𝒪(2)

    @testset "literal values" begin
        for (input, expected) in (
                (0.0, 0), (-0.0, 0), (0 // 1, 0), (0.0 + 0.0im, 0),
                (1.0, 1), (1 // 1, 1), (1.0 + 0.0im, 1),
                (2.0, 2), (-2.0, -2), (3 // 1, 3), (2.0 + 0.0im, 2),
                (Int32(2), 2), (big(2), 2),
                # booleans are kept, so that conditions stay valid
                (false, false), (true, true),
            )
            @test value(@inferred evaluate(Literal(input))) === expected
            # numbers in expressions are canonical
            @test value(Literal(0) + input) == expected
        end
        for input in (0.5, 2 // 3, 2.0 + 1.0im, Inf, -Inf, NaN)
            term = Literal(input)
            @test (@inferred evaluate(term)) === term
        end
        @test_throws InexactError evaluate(Literal(big(typemax(Int)) + 1))
    end

    @testset "scalar constants" begin
        for (input, expected) in (
                (𝒪(0), 0), (ℐ(0), 1),
                (sin(Literal(0)), 0), (cos(Literal(0)), 1), (sqrt(Literal(4)), 2),
                (Literal(2.0) * Literal(1), 2), (Literal(2) / Literal(4), 1 // 2),
            )
            @test value(input) === expected
        end
        for input in (𝒪(1), 𝒪(2), ℐ(1), ℐ(2))
            @test (@inferred evaluate(input)) === input
        end
        # functions of literals are evaluated numerically when their rules don't apply
        @test Fun(sq)(sqrt(a)) ==ₛ a
        @test value(@inferred Fun(sq)(Literal(3))) === 9
        @test (@inferred Fun(sq)(a)) ==ₛ raw(sq, a)
        for input in (a^0.5 * a^1.5, (a^0.5)^4.0)
            @test input ==ₛ a^2
            @test value(last(arguments(input))) === 2
        end
    end

    # different constructions of the same term give the same canonical representative
    @testset "scalar algebra" begin
        for (input, expected) in (
                (Literal(2) + Literal(3), Literal(5)),
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
                # negations and ±1 scalings of sums are distributed, other scalings are kept
                (-(a + b), -a - b),
                (3 * (a + b) - 2 * (a + b) + c, a + b + c),
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
                (sin(a + a) - sin(2a), Literal(0)),
                (c * (b - a), -(c * (a - b))),
                ((a - b) / (b - a), Literal(-1)),
                (a + Literal(2.0), a + 2),
                # real literals are collected into the coefficients
                (2.5a + 0.5a, 3a),
                (0.25 * (4a), a),
                (0.5a - a / 2, Literal(0)),
                (a + 𝒪(0), a),
                (a * ℐ(0), a),
                # a single collected term can be a call of any operator
                (CentralDifference()[1](a) + 0, CentralDifference()[1](a)),
            )
            @test input ==ₛ expected
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
                (K'', K),
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
                (tr((inv(A) ⋅ A) ⋅ A), tr(A)),
                (adj(ℐ(2)), ℐ(2)),
            )
            @test input ==ₛ expected
        end
        # matrix functions of scalars
        for (input, expected) in ((tr(a), a), (det(a), a), (a', a), (sym(a), a), (asym(a), Literal(0)), (cof(a), Literal(1)), (gram(a), a^2))
            @test input ==ₛ expected
        end
    end

    @testset "structure" begin
        @scalars f
        # sums are calls of `+` of their terms in order with the constant last
        @test operation(a + 2b - 1) == Fun(+)
        @test isequal(arguments(a + 2b - 1), (a, 2b, Literal(-1)))
        @test isequal(arguments(b - a), (-a, b))
        # products are calls of `*` with the numeric factor first, and single powers are calls of `^`
        p = -2 * a^2 / b
        @test operation(p) == Fun(*)
        @test isequal(arguments(p), (Literal(-2), a^2, inv(b)))
        @test operation(a^-2) == Fun(^)
        @test isequal(arguments(a^-2), (a, Literal(-2)))
        # products are not distributed, the scaled sum is a factor
        @test 2 * (a + b) !=ₛ 2a + 2b
        @test isequal(arguments(2 * (a + b)), (Literal(2), a + b))
        # but negations are
        @test isequal(arguments(-(a + b)), (-a, -b))
        # integer powers of sums are powers of the sum with a positive leading coefficient
        @test isequal(arguments((b - a)^3), (Literal(-1), (a - b)^3))
        # samples are ordered by descending offsets
        @test isequal(arguments(f[1 // 2] - f[-1 // 2]), (f[1 // 2], -f[-1 // 2]))
        @test isequal(arguments(f[-1] - 2f[0] + f[1]), (f[1], -2f[0], f[-1]))
        # sums and products of tensors in component form are in component form
        @test ispack(components(A, 2) + 2 * components(A, 2))
        @test ispack(a * components(A, 2))
    end

    @testset "evaluate" begin
        # terms assembled from raw parts are rebuilt in canonical form
        @test (@inferred evaluate(raw(+, b, a))) ==ₛ a + b
        @test (@inferred evaluate(raw(-, raw(+, a, b)))) ==ₛ -a - b
        @test (@inferred evaluate(raw(*, Literal(0), a))) ==ₛ Literal(0)
        result = @inferred evaluate(raw(sin, raw(+, a, Literal(2.0))))
        @test result ==ₛ sin(a + 2)
        @test value(last(arguments(only(arguments(result))))) === 2
        @test (@inferred evaluate(raw(tr, raw(⋅, raw(⋅, raw(inv, A), A), A)))) ==ₛ tr(A)
        @test tensorkind(@inferred evaluate(DExpr(Pack(2, 2, Kind.None()), (a, b, b, c)))) == Kind.Sym()
    end

    @testset "fixed points" begin
        # every subterm of a canonical term is canonical
        @scalars f
        D = StaggeredCentralDifference()
        Δ = Divergence(D)(Gradient(D)(override(f, LeftWall => WallValue(a), WallDerivative(b))))
        corpus = DTerm[
            b + (c + a), c - a - b, b^2 + a * b + a^2, a / (a * b), (a^b)^(2c), (2a)^(-3), c * (b - a),
            (a + b)^2 * (a + b)^-3, -(a - b) * -(b - a), 2.5a + 0.5a, A ⋅ adj(A), det(A) * ℐ(2) - A ⋅ adj(A),
            2 * (a + b) + c, 2 * (a + b) - (a + b), (2a)^b, (-2a)^(1 // 2), a^b * 2^b, -(2 * (A + 3 * ℐ(2))),
            sym(asym(A)) + asym(A)', cof(components(A, 3))[1, 2], inv(components(S, 2))[1, 2],
            lower(Δ, (↓, :), (1 // 2, 1 // 2)), lower(Δ, (↓, ↓), (1 // 2, 1 // 2)),
            lower(CentralDifference{5}()[1](sin(f) * f), (:,), (0,)),
        ]
        for t in corpus, s in subterms(t)
            @test (@inferred evaluate(s)) ==ₛ s
        end
    end
end
