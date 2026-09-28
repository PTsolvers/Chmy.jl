using Test
using Chmy
using Chmy: DExpr, Call, Comp, Locs, Inds, Kind, TensorKind, Tensor, ZeroTensor, IdTensor
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

        @test (@inferred isindex(lit)) == false
        @test (@inferred istensor(lit)) == false
        @test (@inferred iscomp(lit)) == false
        @test (@inferred isinds(lit)) == false
        @test (@inferred islocs(lit)) == false
        @test (@inferred isexpr(lit)) == false
        @test (@inferred iscall(lit)) == false

        @test_throws ArgumentError head(lit)
        @test_throws ArgumentError children(lit)
        @test_throws ArgumentError operation(lit)
        @test_throws ArgumentError arguments(lit)
        @test_throws ArgumentError argument(lit)
        @test_throws ArgumentError components(lit)
        @test_throws ArgumentError indices(lit)
        @test_throws ArgumentError locations(lit)
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

    @testset "Index" begin
        ind = @inferred Index(1)

        @test value(ind) == 1
        @test ind === (@inferred Index(1.0))

        @test ind isa DTerm
        @test (@inferred isa_variant(ind, Index))

        @test (@inferred tensorrank(ind)) == 0

        @test (@inferred isindex(ind)) == true
        @test (@inferred isuniform(ind)) == true

        @test (@inferred isliteral(ind)) == false
        @test (@inferred istensor(ind)) == false
        @test (@inferred iscomp(ind)) == false
        @test (@inferred isinds(ind)) == false
        @test (@inferred islocs(ind)) == false
        @test (@inferred isexpr(ind)) == false
        @test (@inferred iscall(ind)) == false

        @test_throws ArgumentError head(ind)
        @test_throws ArgumentError children(ind)
        @test_throws ArgumentError operation(ind)
        @test_throws ArgumentError arguments(ind)
        @test_throws ArgumentError argument(ind)
        @test_throws ArgumentError components(ind)
        @test_throws ArgumentError indices(ind)
        @test_throws ArgumentError locations(ind)
        @test_throws ArgumentError tensorkind(ind)
        @test_throws ArgumentError tensorname(ind)

        @test_throws InexactError Index(1.1)
    end

    KINDS = TensorKind[Kind.None(), Kind.Diag(), Kind.Sym(), Kind.Alt()]

    @testset "Tensor" begin
        t = @inferred Tensor(2, gensym("tensor"), Kind.None(), true)

        @test (@inferred istensor(t)) == true

        @test (@inferred isliteral(t)) == false
        @test (@inferred isindex(t)) == false
        @test (@inferred iscomp(t)) == false
        @test (@inferred isinds(t)) == false
        @test (@inferred islocs(t)) == false
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
        @test_throws ArgumentError locations(t)

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
        t = @inferred ZeroTensor(2)
        @test (@inferred 𝒪(2)) === t

        @test (@inferred tensorrank(t)) == 2
        @test (@inferred istensor(t)) == true
        @test (@inferred isuniform(t)) == true
        @test (@inferred isstaticzero(t)) == true
        @test (@inferred isstaticone(t)) == false
    end

    @testset "IdTensor" begin
        t = @inferred IdTensor(2)
        @test (@inferred ℐ(2)) === t

        @test (@inferred tensorrank(t)) == 2
        @test (@inferred istensor(t)) == true
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

            @test (@inferred isunaryminus(DExpr(Call(Fun(-)), (a,)))) == true
            @test (@inferred isunaryminus(DExpr(Call(Fun(sin)), (a,)))) == false

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
            @test (@inferred isindex(expr)) == false
            @test (@inferred istensor(expr)) == false
            @test (@inferred iscall(expr)) == false
            @test (@inferred islocs(expr)) == false
            @test (@inferred isinds(expr)) == false

            @test (@inferred tensorrank(expr)) == 0
            @test (@inferred isuniform(expr)) == false
            @test (@inferred isuniform(DExpr(Comp((1, 2)), (b,)))) == true

            @test_throws ArgumentError operation(expr)
            @test_throws ArgumentError arguments(expr)
            @test_throws ArgumentError indices(expr)
            @test_throws ArgumentError locations(expr)
            @test_throws ArgumentError tensorkind(expr)
            @test_throws ArgumentError tensorname(expr)
        end

        @testset "Locs" begin
            @scalars a @uniform(b)
            expr = @inferred DExpr(Locs((𝓅, 𝓈)), (a,))

            @test expr isa DTerm
            @test (@inferred isa_variant(expr, DExpr))

            @test (@inferred isexpr(expr)) == true
            @test (@inferred islocs(expr)) == true

            @test (@inferred head(expr)) == Locs((𝓅, 𝓈))
            @test isequal(children(expr), (a,))
            @test (@inferred argument(expr)) ==ₛ a
            @test locations(expr) == (𝓅, 𝓈)

            @test (@inferred isliteral(expr)) == false
            @test (@inferred isindex(expr)) == false
            @test (@inferred istensor(expr)) == false
            @test (@inferred iscall(expr)) == false
            @test (@inferred iscomp(expr)) == false
            @test (@inferred isinds(expr)) == false

            @test (@inferred tensorrank(expr)) == 0
            @test (@inferred isuniform(expr)) == false
            @test (@inferred isuniform(DExpr(Locs((𝓅, 𝓈)), (b,)))) == true

            @test_throws ArgumentError operation(expr)
            @test_throws ArgumentError arguments(expr)
            @test_throws ArgumentError components(expr)
            @test_throws ArgumentError indices(expr)
            @test_throws ArgumentError tensorkind(expr)
            @test_throws ArgumentError tensorname(expr)
        end

        @testset "Inds" begin
            @scalars a @uniform(b)
            expr = @inferred DExpr(Inds(), (a, 𝑖, 𝑗 + Literal(1)))

            @test expr isa DTerm
            @test (@inferred isa_variant(expr, DExpr))

            @test (@inferred isexpr(expr)) == true
            @test (@inferred isinds(expr)) == true

            @test (@inferred head(expr)) == Inds()
            @test isequal(children(expr), (a, 𝑖, 𝑗 + Literal(1)))
            @test (@inferred argument(expr)) ==ₛ a
            @test isequal(indices(expr), (𝑖, 𝑗 + Literal(1)))

            @test (@inferred isliteral(expr)) == false
            @test (@inferred isindex(expr)) == false
            @test (@inferred istensor(expr)) == false
            @test (@inferred iscall(expr)) == false
            @test (@inferred iscomp(expr)) == false
            @test (@inferred islocs(expr)) == false

            @test (@inferred tensorrank(expr)) == 0
            @test (@inferred isuniform(expr)) == false
            @test (@inferred isuniform(DExpr(Inds(), (b, 𝑖, 𝑗 + Literal(1))))) == true

            @test (@inferred isuniform(DExpr(Inds(), (b, a, 𝑗)))) == false

            @test_throws ArgumentError operation(expr)
            @test_throws ArgumentError arguments(expr)
            @test_throws ArgumentError components(expr)
            @test_throws ArgumentError locations(expr)
            @test_throws ArgumentError tensorkind(expr)
            @test_throws ArgumentError tensorname(expr)
        end
    end

    @testset "isnegof" begin
        @test (@inferred isnegof(Literal(1), Literal(-1))) == true
        @test (@inferred isnegof(Literal(1), -Literal(1))) == true
        @test (@inferred isnegof(Literal(1), Literal(1))) == false
        @test (@inferred isnegof(Literal(1), Literal(2))) == false

        @scalars a b

        @test (@inferred isnegof(a, -a)) == true
        @test (@inferred isnegof(-a, a)) == true

        @test (@inferred isnegof(a + b, -(a + b))) == true
        @test (@inferred isnegof(a + b, -a - b)) == false

        @test (@inferred isnegof(a, a)) == false
        @test (@inferred isnegof(a, b)) == false
        @test (@inferred isnegof(a, -b)) == false
    end

    @testset "==ₛ" begin
        @scalars a b @uniform(c)
        @vectors P Q
        @tensors 2 τ @sym(σ)

        @test (@inferred a ==ₛ Tensor(0, :a, Kind.None(), false))
        @test (@inferred a !=ₛ Tensor(1, :a, Kind.None(), false))
        @test (@inferred a + b ==ₛ a + b)
        @test (@inferred a + b !=ₛ b + a)
        @test (@inferred a !=ₛ a + b)
        @test (@inferred ZeroTensor(2) ==ₛ ZeroTensor(2))
        @test (@inferred ZeroTensor(2) !=ₛ ZeroTensor(3))
        @test (@inferred IdTensor(2) ==ₛ IdTensor(2))
        @test (@inferred IdTensor(2) !=ₛ IdTensor(3))
        @test (@inferred Index(2) ==ₛ Index(2))
        @test (@inferred Index(1) !=ₛ Index(2))
        @test (@inferred Literal(1) ==ₛ Literal(1))
        @test (@inferred Literal(1) ==ₛ Literal(1.0))
        @test (@inferred Literal(1) !=ₛ Literal(1.1))
    end

    @testset "<ₛ" begin
        @scalars a b
        @vectors u v

        @test (@inferred 1 <ₛ 2)
        @test (@inferred Literal(1) <ₛ Literal(2))
        @test (@inferred 𝑖 <ₛ 𝑗)
        @test (@inferred 𝑖 <ₛ a)
        @test (@inferred a <ₛ b)
        @test (@inferred 𝒪(0) <ₛ ℐ(0))
        @test (@inferred a <ₛ u)
        @test !(@inferred a <ₛ a)

        @test (@inferred cos(a) <ₛ sin(a))
        @test (@inferred sin(a) <ₛ sin(b))
        @test (@inferred a + b <ₛ a + b + b)
        @test (@inferred v[1] <ₛ v[2])
        @test (@inferred u[2] <ₛ v[1])
        @test (@inferred a[𝓅] <ₛ a[𝓈])
        @test (@inferred a[𝑖] <ₛ a[𝑗])

        terms = DTerm[sin(b), sin(a), cos(a), Literal(2), a]
        expected = DTerm[a, Literal(2), cos(a), sin(a), sin(b)]
        @test isequal(sort(terms; lt = <ₛ), expected)
    end
end
