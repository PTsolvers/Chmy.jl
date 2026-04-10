@testset "isless_lex" begin
    @scalars a b c
    @vectors u v

    @testset "leaf terms" begin
        @test isless_lex(SIndex(1), SIndex(2))
        @test !isless_lex(SIndex(2), SIndex(1))

        @test isless_lex(SRef(:a), SRef(:b))
        @test !isless_lex(SRef(:b), SRef(:a))

        @test isless_lex(SFun(cos), SFun(sin))
        @test !isless_lex(SFun(sin), SFun(cos))

        @test isless_lex(Point(), Segment())
        @test !isless_lex(Segment(), Point())

        @test isless_lex(SLiteral(2), SLiteral(3))
        @test !isless_lex(SLiteral(3), SLiteral(2))

        @test isless_lex(SIndex(1), a)
        @test !isless_lex(a, SIndex(1))
    end

    @testset "tensors" begin
        @test isless_lex(a, b)
        @test !isless_lex(b, a)

        @test isless_lex(a, u)
        @test !isless_lex(u, a)

        @test_throws ArgumentError isless_lex(SSymTensor{2}(:T), SAltTensor{2}(:T))
    end

    @testset "derivatives and call expressions" begin
        d = CentralDifference()
        sd = StaggeredCentralDifference()

        @test isless_lex(PartialDerivative(SRef(:a))(a, 2), PartialDerivative(SRef(:b))(a, 1))
        @test !isless_lex(PartialDerivative(SRef(:b))(a, 1), PartialDerivative(SRef(:a))(a, 2))

        @test isless_lex(a + b, sin(a))
        @test isless_lex(sin(a), d(a))
        @test isless_lex(Divergence(d)(u), Curl(d)(u))

        @test isless_lex(sin(a), sin(b))
        @test !isless_lex(sin(b), sin(a))

        @test isless_lex(makeop(:+, a, c), makeop(:+, b, a))
        @test !isless_lex(makeop(:+, b, a), makeop(:+, a, c))

        @test isless_lex(v[1], v[2])
        @test !isless_lex(v[2], v[1])

        @test isless_lex(a[Point()], a[Segment()])
        @test !isless_lex(a[Segment()], a[Point()])

        @test isless_lex(a[SIndex(1)], a[Point()])
        @test !isless_lex(a[Point()], a[SIndex(1)])

        @test isless_lex(d, sd) != isless_lex(sd, d)
    end

    @testset "sorting contract" begin
        terms = [sin(b), sin(a), cos(a), SLiteral(2), a]
        sorted = sort(terms; lt=isless_lex)
        expected = [a, SLiteral(2), cos(a), sin(a), sin(b)]
        @test length(sorted) == length(expected)
        @test all(((x, y),) -> x === y, zip(sorted, expected))
    end
end
