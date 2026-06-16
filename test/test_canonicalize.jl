using Test
using Chmy

@testset "canonicalize" begin
    @scalars a b c

    @testset "commutativity" begin
        @test canonicalize(b + c + a) === a + b + c
        @test canonicalize(c + b - a) === -a + b + c
        @test canonicalize(b * c * a) === a * b * c
    end

    @testset "associativity" begin
        @test canonicalize((a + b) + c) === a + b + c
        @test canonicalize(a + (b + c)) === a + b + c
        @test canonicalize((a * b) * c) === a * b * c
        @test canonicalize(a * (b * c)) === a * b * c
    end

    @testset "constant folding" begin
        @test canonicalize(SLiteral(2) + SLiteral(3)) === SLiteral(5)
        @test canonicalize(SLiteral(2) * SLiteral(3)) === SLiteral(6)
        @test canonicalize(SLiteral(2) - SLiteral(3)) === SLiteral(-1)
        @test canonicalize(SLiteral(2) / SLiteral(4)) === SLiteral(1 // 2)
    end

    @testset "sums" begin
        @test canonicalize(a + a) === SLiteral(2) * a
        @test canonicalize(a + b + a) === SLiteral(2) * a + b
        @test canonicalize(a - a) === SLiteral(0)
    end

    @testset "products" begin
        @test canonicalize(a * a) === a^SLiteral(2)
        @test canonicalize(a * b * a^SLiteral(2)) === a^SLiteral(3) * b
        @test canonicalize(a * inv(a)) === SLiteral(1)
        @test canonicalize(b * a^(-SLiteral(2))) === b / a^SLiteral(2)
        @test canonicalize(SLiteral(1) / a) === inv(a)
        @test canonicalize(a * b^(-SLiteral(1))) === a / b
        @test canonicalize((-a)^SLiteral(2)) === a^SLiteral(2)
        @test canonicalize((-a)^SLiteral(3)) === -a^SLiteral(3)
        @test canonicalize((-a)^SLiteral(3 // 2)) === (-a)^SLiteral(3 // 2)
        @test canonicalize(a^b * a^c) === a^(b + c)
        @test canonicalize(a^(-SLiteral(2)) * a^(-b)) === a^(-b - SLiteral(2))
        @test canonicalize((SLiteral(2) * a)^SLiteral(3)) === SLiteral(8) * a^SLiteral(3)
        @test canonicalize((SLiteral(2) * a)^(-SLiteral(3))) === SLiteral(1 // 8) / a^SLiteral(3)
    end
end
