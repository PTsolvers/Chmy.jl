using Test
using Chmy
using Chmy: DExpr, AddMul, Call, Comp, Inds, Pack, Kind, TensorKind, Tensor, ZeroTensor, IdTensor
using Moshi.Data: isa_variant

using Random
Random.seed!(1234) # deterministic tests

@testset verbose = true "DTerm" begin
    @testset "Literal" begin
        lit = @inferred Literal(rand())

        @test lit isa DTerm
        @test (@inferred isa_variant(lit, Literal))

        @test (@inferred tensorrank(lit)) == 0

        @test (@inferred isliteral(lit)) == true
        @test (@inferred isuniform(lit)) == true

        @test (@inferred istensor(lit)) == false
        @test (@inferred iscomp(lit)) == false
        @test (@inferred isinds(lit)) == false
        @test (@inferred isexpr(lit)) == false
        @test (@inferred iscall(lit)) == false

        @test (@inferred lit[]) === lit

        @test_throws ArgumentError head(lit)
        @test_throws ArgumentError children(lit)
        @test_throws ArgumentError operation(lit)
        @test_throws ArgumentError arguments(lit)
        @test_throws ArgumentError argument(lit)
        @test_throws ArgumentError components(lit)
        @test_throws ArgumentError indices(lit)
        @test_throws ArgumentError tensorkind(lit)
        @test_throws ArgumentError tensorname(lit)

        @test_throws "Cannot `convert` an object" Literal("123")

    end

    @testset "Literal($T)" for T in (Int, Float64, ComplexF64)
        @test (@inferred isstaticzero(Literal(zero(T))))
        @test (@inferred isstaticone(Literal(oneunit(T))))

        v = rand(T)
        @test value(Literal(v)) == v
    end

    @testset "Coordinate" begin
        ξ = @inferred Coordinate(1)

        @test value(ξ) == 1
        @test ξ === ξ¹
        @test ξ ==ₛ Coordinate(1)
        @test ξ !=ₛ Coordinate(2)

        @test (@inferred tensorrank(ξ)) == 0

        @test (@inferred iscoordinate(ξ)) == true
        @test (@inferred isuniform(ξ)) == false
        @test (@inferred isexpr(ξ)) == false
    end

    KINDS = TensorKind[Kind.None(), Kind.Diag(), Kind.Sym(), Kind.Alt()]

    @testset "Tensor" begin
        t = @inferred Tensor(2, gensym("tensor"), Kind.None(), true)

        @test (@inferred istensor(t)) == true

        @test (@inferred isliteral(t)) == false
        @test (@inferred iscomp(t)) == false
        @test (@inferred isinds(t)) == false
        @test (@inferred isexpr(t)) == false
        @test (@inferred iscall(t)) == false

        @test (@inferred isstaticzero(t)) == false
        @test (@inferred isstaticone(t)) == false

        @test_throws ArgumentError head(t)
        @test_throws ArgumentError children(t)
        @test_throws ArgumentError operation(t)
        @test_throws ArgumentError arguments(t)
        @test_throws ArgumentError argument(t)
        @test_throws ArgumentError components(t)
        @test_throws ArgumentError indices(t)
        @test_throws ArgumentError t[]

        for rank in 1:4, kind in KINDS, uniform in (true, false)
            name = gensym("tensor")
            t = @inferred Tensor(rank, name, kind, uniform)

            @test t isa DTerm
            @test (@inferred isa_variant(t, Tensor))

            @test (@inferred tensorrank(t)) == rank
            @test tensorkind(t) == kind
            @test (@inferred isuniform(t)) == uniform
            @test (@inferred tensorname(t)) == name
        end
    end

    @testset "ZeroTensor" begin
        t = @inferred 𝒪(2)
        @test operation(t) == ZeroTensor(2)
        @test (@inferred t[1, 2]) ==ₛ Literal(0)

        @test (@inferred tensorrank(t)) == 2
        @test (@inferred istensor(t)) == false
        @test (@inferred isuniform(t)) == true
        @test (@inferred isstaticzero(t)) == true
        @test (@inferred isstaticone(t)) == false
    end

    @testset "IdTensor" begin
        t = @inferred ℐ(2)
        @test operation(t) == IdTensor(2)
        @test (@inferred t[1, 1]) ==ₛ Literal(1)
        @test (@inferred t[1, 2]) ==ₛ Literal(0)

        @test (@inferred tensorrank(t)) == 2
        @test (@inferred istensor(t)) == false
        @test (@inferred isuniform(t)) == true
        @test (@inferred isstaticzero(t)) == false
        @test (@inferred isstaticone(t)) == true
    end

    @testset verbose = true "DExpr" begin
        @testset "Call" begin
            expr = @inferred DExpr(Call(Fun(+)), (Literal(1), Literal(2)))

            @test expr isa DTerm
            @test (@inferred isa_variant(expr, DExpr))

            @test (@inferred isexpr(expr)) == true
            @test (@inferred iscall(expr)) == true

            @test (@inferred head(expr)) == Call(Fun(+))

            @test operation(expr) == Fun(+)

            @test children(expr) == (Literal(1), Literal(2))
            @test arguments(expr) == (Literal(1), Literal(2))

            @test (@inferred isliteral(expr)) == false
            @test (@inferred isuniform(expr)) == true

            @test (@inferred tensorrank(expr)) == 0

            @scalars a @uniform(b)

            @test (@inferred isunary(DExpr(Call(Fun(-)), (a,)))) == true
            @test (@inferred isunary(DExpr(Call(Fun(sin)), (a,)))) == true

            @test (@inferred isuniform(DExpr(Call(Fun(+)), (a, Literal(2), b)))) == false
            @test (@inferred isuniform(DExpr(Call(Fun(+)), (b, Literal(2), b)))) == true

            @vectors p q

            @test (@inferred tensorrank(DExpr(Call(Fun(+)), (p, q)))) == 1
        end

        @testset "Comp" begin
            @tensors 2 a @uniform(b)
            expr = @inferred DExpr(Comp((1, 2)), (a,))

            @test expr isa DTerm
            @test (@inferred isa_variant(expr, DExpr))

            @test (@inferred isexpr(expr)) == true
            @test (@inferred iscomp(expr)) == true

            @test (@inferred head(expr)) == Comp((1, 2))
            @test isequal(children(expr), (a,))
            @test (@inferred argument(expr)) ==ₛ a
            @test components(expr) == (1, 2)

            @test (@inferred isliteral(expr)) == false
            @test (@inferred istensor(expr)) == false
            @test (@inferred iscall(expr)) == false
            @test (@inferred isinds(expr)) == false

            @test (@inferred tensorrank(expr)) == 0
            @test (@inferred isuniform(expr)) == false
            @test (@inferred isuniform(DExpr(Comp((1, 2)), (b,)))) == true

            @test_throws ArgumentError operation(expr)
            @test_throws ArgumentError arguments(expr)
            @test_throws ArgumentError indices(expr)
            @test_throws ArgumentError tensorkind(expr)
            @test_throws ArgumentError tensorname(expr)
        end

        @testset "Inds" begin
            @scalars a @uniform(b)
            @vectors v
            expr = @inferred DExpr(Inds((Coord(:, 0), Coord(↓, 1))), (a,))

            @test expr isa DTerm
            @test (@inferred isa_variant(expr, DExpr))

            @test (@inferred isexpr(expr)) == true
            @test (@inferred isinds(expr)) == true

            @test (@inferred head(expr)) == Inds((Coord(:, 0), Coord(↓, 1)))
            @test isequal(children(expr), (a,))
            @test (@inferred argument(expr)) ==ₛ a
            @test indices(expr) === (Coord(:, 0), Coord(↓, 1))

            @test (@inferred isliteral(expr)) == false
            @test (@inferred istensor(expr)) == false
            @test (@inferred iscall(expr)) == false
            @test (@inferred iscomp(expr)) == false

            @test (@inferred tensorrank(expr)) == 0
            @test (@inferred isuniform(expr)) == false
            @test (@inferred isuniform(DExpr(Inds((Coord(:, 1 // 2),)), (b,)))) == true

            # coordinates are written as `(frame, offset)`, and traces as the elements of the axis
            @test (@inferred a[(:, 0), (↓, 1)]) ==ₛ expr
            @test indices(@inferred a[(↑, -1 // 2), Coord(:, 1)]) === (Coord(↑, -1 // 2), Coord(:, 1))
            @test indices(@inferred a[↓, (:, 1 // 2)]) === (Trace(↓), Coord(:, 1 // 2))
            @test indices(@inferred a[↓, :]) === (Trace(↓), Trace(:))
            @test (@inferred iscomp(v[1])) == true
            @test (@inferred isinds(v[1][(:, 1), (:, 2)])) == true
            @test a[↓, (:, 0)] !=ₛ a[(↓, 0), (:, 0)]
            @test a[↓, (:, 0)] !=ₛ a[↑, (:, 0)]
            @test sprint(show, a[↓, (:, -1 // 2)]) == "a[↓, (:, -1//2)]"
            @test sprint(show, a[:, (↑, 1)]) == "a[:, (↑, 1)]"
            @test_throws ArgumentError a[(:, 1 // 3)]
            @test_throws ArgumentError v[(:, 1 // 2)]
            # offsets are not positions
            @test_throws MethodError a[1 // 2]
            @test_throws ArgumentError a[0, 1]
            @test_throws MethodError a[↓, 0]
            # samples cannot be sampled again
            @test_throws ArgumentError a[(:, 1 // 2)][(:, 1)]
            @test_throws ArgumentError a[(:, 1 // 2)][(:, 1), (:, 1)]
            # constants are not sampled, uniform calls are
            @test b[(:, 1 // 2)] ==ₛ b
            @test Literal(2)[(:, 1 // 2)] ==ₛ Literal(2)
            @tensors 1 @uniform(w)
            @test w[1][(:, 0), (:, 1)] ==ₛ w[1]
            @test isinds(sin(b)[(:, 1 // 2)])

            @test_throws ArgumentError operation(expr)
            @test_throws ArgumentError arguments(expr)
            @test_throws ArgumentError components(expr)
            @test_throws ArgumentError tensorkind(expr)
            @test_throws ArgumentError tensorname(expr)
        end

        @testset "Pack" begin
            @scalars a b c
            @vectors v
            expr = @inferred pack([a, b])

            @test expr isa DTerm
            @test (@inferred isa_variant(expr, DExpr))

            @test (@inferred isexpr(expr)) == true
            @test (@inferred ispack(expr)) == true
            @test (@inferred ispack(v)) == false

            @test (@inferred head(expr)) == Pack(1, 2, Kind.None())
            @test isequal(entries(expr), (a, b))
            @test (@inferred tensorrank(expr)) == 1
            @test tensorkind(expr) == Kind.None()
            @test (@inferred isuniform(expr)) == false
            @test (@inferred isuniform(pack([0, -9.81]))) == true

            # entries are compared and hashed by content
            @test expr ==ₛ pack([a, b])
            @test hash(expr) == hash(pack([a, b]))
            @test expr !=ₛ pack([b, a])

            # indexing returns the stored entries, respecting the symmetry of the storage
            @test (@inferred expr[2]) ==ₛ b
            sym = DExpr(Pack(2, 2, Kind.Sym()), (a, b, c))
            @test sym[2, 1] ==ₛ b
            alt = DExpr(Pack(2, 2, Kind.Alt()), (a,))
            @test alt[1, 2] ==ₛ a
            @test alt[2, 1] ==ₛ -a
            @test alt[1, 1] ==ₛ Literal(0)
            diagonal = DExpr(Pack(2, 2, Kind.Diag()), (a, b))
            @test diagonal[2, 2] ==ₛ b
            @test diagonal[1, 2] ==ₛ Literal(0)
            @test_throws ArgumentError expr[1, 1]
            @test_throws BoundsError expr[3]

            @test_throws ArgumentError pack(fill(a))
            @test_throws ArgumentError pack([v, v])
            @test_throws DimensionMismatch pack([a b c; a b c])

            @test_throws ArgumentError operation(expr)
            @test_throws ArgumentError argument(expr)
            @test_throws ArgumentError components(expr)
            @test_throws ArgumentError tensorname(expr)
        end
    end

    @testset "AddMul" begin
        @scalars a @uniform(b)
        @vectors v
        s = a + 2b + 1
        p = 3a^2 * b

        @test s isa DTerm
        @test (@inferred isa_variant(s, AddMul))
        # sums are calls of `+` of their terms in order with the constant last
        @test (@inferred isexpr(s)) == true
        @test (@inferred iscall(s)) == true
        @test (@inferred head(s)) == Call(Fun(+))
        @test operation(s) == Fun(+)
        @test isequal(arguments(s), (a, 2b, Literal(1)))
        @test isequal(children(s), arguments(s))
        @test (@inferred arity(s)) == 3
        @test operation(a - b) == Fun(+)
        @test isequal(arguments(a - b), (a, -b))
        # products are calls of `*` of their numeric factor and the powers of their bases
        @test operation(p) == Fun(*)
        @test isequal(arguments(p), (Literal(3), a^2, b))
        @test operation(-a) == Fun(*)
        @test isequal(arguments(-a), (Literal(-1), a))
        @test (@inferred arity(-a)) == 2
        @test isequal(arguments(a / b), (a, inv(b)))
        # a single power is a call of `^`
        @test operation(a^2) == Fun(^)
        @test isequal(arguments(a^2), (a, Literal(2)))
        @test isequal(arguments(inv(a)), (a, Literal(-1)))
        # the call of the operation with the arguments is the same term
        for t in (s, p, -a, -2a, -a^2 * b, a^2, inv(a), a - b, a / b, 2 * (a + b), -2 * (a + b))
            @test operation(t)(arguments(t)...) ==ₛ t
        end

        @test (@inferred tensorrank(s)) == 0
        @test (@inferred tensorrank(2v)) == 1
        @test (@inferred tensorrank(v - a * v)) == 1
        @test (@inferred isuniform(s)) == false
        @test (@inferred isuniform(2b + 1)) == true
        @test (@inferred isuniform(b^a)) == false

        # compared and hashed by content
        @test s ==ₛ 1 + b + b + a
        @test hash(s) == hash(1 + b + b + a)
        @test s !=ₛ a + 2b
        @test p !=ₛ 2a^2 * b
        @test (@inferred s <ₛ p) == true

        @test_throws ArgumentError value(s)
    end

    @testset "==ₛ" begin
        @scalars a b @uniform(c)
        @vectors P Q
        @tensors 2 τ @sym(σ)

        @test (@inferred a ==ₛ Tensor(0, :a, Kind.None(), false))
        @test (@inferred a !=ₛ Tensor(1, :a, Kind.None(), false))
        @test (@inferred a + b ==ₛ a + b)
        # terms are canonical, but products are not distributed
        @test (@inferred a + b ==ₛ b + a)
        @test (@inferred 2 * (a + b) !=ₛ 2a + 2b)
        @test (@inferred a !=ₛ a + b)
        @test (@inferred 𝒪(2) ==ₛ 𝒪(2))
        @test (@inferred 𝒪(2) !=ₛ 𝒪(3))
        @test (@inferred ℐ(2) ==ₛ ℐ(2))
        @test (@inferred ℐ(2) !=ₛ ℐ(3))
        @test (@inferred 𝒪(2) !=ₛ ℐ(2))
        @test (@inferred ξ² ==ₛ ξ²)
        @test (@inferred ξ¹ !=ₛ ξ²)
        @test (@inferred Literal(1) ==ₛ Literal(1))
        @test (@inferred Literal(1) ==ₛ Literal(1.0))
        @test (@inferred Literal(1) !=ₛ Literal(1.1))
    end

    @testset "<ₛ" begin
        @scalars a b
        @vectors u v

        @test (@inferred 1 <ₛ 2)
        @test (@inferred Literal(1) <ₛ Literal(2))
        @test (@inferred ξ¹ <ₛ ξ²)
        @test (@inferred ξ¹ <ₛ a)
        @test (@inferred a <ₛ b)
        @test (@inferred 𝒪(0) <ₛ ℐ(0))
        @test (@inferred a <ₛ u)
        @test !(@inferred a <ₛ a)

        @test (@inferred cos(a) <ₛ sin(a))
        @test (@inferred sin(a) <ₛ sin(b))
        @test (@inferred a + b <ₛ a + b + b)
        @test (@inferred v[1] <ₛ v[2])
        @test (@inferred u[2] <ₛ v[1])
        @test (@inferred a[(:, 1)] <ₛ a[↓])
        @test (@inferred a[↓] <ₛ a[↑])
        # samples are ordered by the frames, then by descending offsets
        @test (@inferred a[(:, 1)] <ₛ a[(:, 0)])
        @test (@inferred a[(↓, 0)] <ₛ a[(:, 1)])

        terms = DTerm[sin(b), sin(a), cos(a), Literal(2), a]
        expected = DTerm[a, Literal(2), cos(a), sin(a), sin(b)]
        @test isequal(sort(terms; lt = <ₛ), expected)
    end
end
