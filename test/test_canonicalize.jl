using Test
using Chmy
import Chmy: makeop

@testset "canonicalize" begin
    @scalars a b c

    @testset "commutativity" begin
        @test b + c + a === makeop(:+, a, b, c)
        @test c + b - a === makeop(:+, makeop(:-, a), b, c)
        @test b * c * a === makeop(:*, a, b, c)
    end

    @testset "associativity" begin
        @test (a + b) + c === makeop(:+, a, b, c)
        @test a + (b + c) === makeop(:+, a, b, c)
        @test (a * b) * c === makeop(:*, a, b, c)
        @test a * (b * c) === makeop(:*, a, b, c)
    end

    @testset "constant folding" begin
        @test SLiteral(2) + SLiteral(3) === SLiteral(5)
        @test SLiteral(2) * SLiteral(3) === SLiteral(6)
        @test SLiteral(2) - SLiteral(3) === SLiteral(-1)
        @test SLiteral(2) / SLiteral(4) === SLiteral(1 // 2)
    end

    @testset "monomials" begin
        @test a * a === makeop(:^, a, SLiteral(2))
        @test a * b * a^SLiteral(2) === makeop(:*, makeop(:^, a, SLiteral(3)), b)
        @test a * inv(a) === SLiteral(1)
        @test b * a^(-SLiteral(2)) === makeop(:/, b, makeop(:^, a, SLiteral(2)))
        @test SLiteral(1) / a === makeop(:inv, a)
        @test a * b^(-SLiteral(1)) === makeop(:/, a, b)
        @test (-a)^SLiteral(2) === makeop(:^, a, SLiteral(2))
        @test (-a)^SLiteral(3) === makeop(:-, makeop(:^, a, SLiteral(3)))
        @test (-a)^SLiteral(3 // 2) === makeop(:^, makeop(:-, a), SLiteral(3 // 2))
        @test a^b * a^c === makeop(:^, a, makeop(:+, b, c))
        @test a^(-SLiteral(2)) * a^(-b) === makeop(:^, a, makeop(:-, makeop(:-, b), SLiteral(2)))
        @test (SLiteral(2) * a)^SLiteral(3) === makeop(:*, SLiteral(8), makeop(:^, a, SLiteral(3)))
        @test (SLiteral(2) * a)^(-SLiteral(3)) === makeop(:/, SLiteral(1 // 8), makeop(:^, a, SLiteral(3)))
    end

    @testset "sums" begin
        @test a + a === makeop(:*, SLiteral(2), a)
        @test a + b + a === makeop(:+, makeop(:*, SLiteral(2), a), b)
        @test a - a === SLiteral(0)
    end

    @testset "simplify" begin
        @test simplify(sin(makeop(:+, b, a))) === sin(makeop(:+, a, b))
    end
end
