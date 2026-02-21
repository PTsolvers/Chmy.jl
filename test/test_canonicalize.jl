@testset "canonicalize" begin
    a = SScalar(:a)
    b = SScalar(:b)
    c = SScalar(:c)

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
        @test SUniform(2) + SUniform(3) === SUniform(5)
        @test SUniform(2) * SUniform(3) === SUniform(6)
        @test SUniform(2) - SUniform(3) === SUniform(-1)
        @test SUniform(2) / SUniform(4) === SUniform(1 // 2)
    end

    @testset "monomials" begin
        @test a * a === makeop(:^, a, SUniform(2))
        @test a * b * a^SUniform(2) === makeop(:*, makeop(:^, a, SUniform(3)), b)
        @test a * inv(a) === SUniform(1)
        @test b * a^(-SUniform(2)) === makeop(:/, b, makeop(:^, a, SUniform(2)))
        @test SUniform(1) / a === makeop(:inv, a)
        @test a * b^(-SUniform(1)) === makeop(:/, a, b)
        @test (-a)^SUniform(2) === makeop(:^, a, SUniform(2))
        @test (-a)^SUniform(3) === makeop(:-, makeop(:^, a, SUniform(3)))
        @test a^b * a^c === makeop(:^, a, makeop(:+, b, c))
        @test a^(-SUniform(2)) * a^(-b) === makeop(:^, a, makeop(:-, makeop(:-, b), SUniform(2)))
        @test (SUniform(2) * a)^SUniform(3) === makeop(:*, SUniform(8), makeop(:^, a, SUniform(3)))
        @test (SUniform(2) * a)^(-SUniform(3)) === makeop(:/, SUniform(1 // 8), makeop(:^, a, SUniform(3)))
    end

    @testset "sums" begin
        @test a + a === makeop(:*, SUniform(2), a)
        @test a + b + a === makeop(:+, makeop(:*, SUniform(2), a), b)
        @test a - a === SUniform(0)
    end

    @testset "simplify" begin
        @test simplify(sin(makeop(:+, b, a))) === sin(makeop(:+, a, b))
    end
end
