@testset "Canonicalize" begin
    a = SScalar(:a)
    b = SScalar(:b)
    c = SScalar(:c)
    d = SScalar(:d)
    e = SScalar(:e)
    f = SScalar(:f)

    @testset "SIndex constraint" begin
        @test SIndex(1) === SIndex{1}()
        @test_throws MethodError SIndex(:i)
    end

    @testset "product canonicalization" begin
        @test canonicalize(b * a) === *(a, b)
        @test canonicalize(a * a) === a^SUniform(2)
        @test canonicalize(a * (-a)) === -(a^SUniform(2))
        @test canonicalize((-a) * (-a)) === a^SUniform(2)
        @test canonicalize(SUniform(2) * b * SUniform(3) * a) === *(SUniform(6), a, b)

        expr = a * (a * b) / (inv(b) * (SUniform(1 // 2) * a)^(-SUniform(1)))
        @test canonicalize(expr) === *(SUniform(1 // 2), a^SUniform(3), b^SUniform(2))

        @test canonicalize(inv(a^b)) === a^(-b)
        @test canonicalize(SUniform(1) / (a^b)) === a^(-b)
        @test canonicalize(a * a^(-b)) === a^(-b + SUniform(1))
    end

    @testset "sum canonicalization" begin
        @test canonicalize((a + b) + (c + a)) === +(SUniform(2) * a, b, c)
        @test canonicalize(a + b - a) === b
        @test canonicalize(SUniform(2) + a + SUniform(1)) === a + SUniform(3)

        @test canonicalize(-d + a - e + b + c - f) === a + b + c - d - e - f
        @test canonicalize(-d + a - e - b + c + f) === a - b + c - d - e + f
    end

    @testset "recursive canonicalization" begin
        expr = sin(b + a) * sin(a + b)
        @test canonicalize(expr) === sin(a + b)^SUniform(2)
    end

    @testset "ordering" begin
        @test canonicalize(sin(a) * a) === *(a, sin(a))
        @test canonicalize(sin(a) + a) === +(a, sin(a))
        @test canonicalize(a^SUniform(2) + a^(b + SUniform(1))) === +(a^(b + SUniform(1)), a^SUniform(2))
        @test_throws ArgumentError canonicalize(SScalar(:q) * SSymTensor{0}(:q))
    end

    @testset "opaque division operators" begin
        @test canonicalize((a // b) / c) === (a // b) / c
        @test canonicalize((a ÷ b) * c) === *(c, a ÷ b)
    end

    @testset "idempotence" begin
        expr = sin(b + a) * sin(a + b) + a * (a * b) / (inv(b) * (SUniform(1 // 2) * a)^(-SUniform(1))) - (a + b - a)
        once = canonicalize(expr)
        twice = canonicalize(once)
        @test once === twice
    end
end
