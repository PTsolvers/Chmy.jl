@testset "Canonicalize and Seval" begin
    a = SScalar(:a)
    b = SScalar(:b)
    c = SScalar(:c)
    d = SScalar(:d)
    e = SScalar(:e)
    f = SScalar(:f)
    n = SScalar(:n)
    x = SScalar(:x)
    y = SScalar(:y)
    z = SScalar(:z)

    @testset "SIndex constraint" begin
        @test SIndex(1) === SIndex{1}()
        @test_throws MethodError SIndex(:i)
    end

    @testset "seval" begin
        @test seval(SUniform(1 // 2) + sin(SUniform(π))) === SUniform(1 // 2)
        @test seval(log(sin(SUniform(π)) + SUniform(1))) === SUniform(0)
    end

    @testset "product canonicalization" begin
        @test canonicalize(y * x) === *(x, y)
        @test canonicalize(a * a) === a^SUniform(2)
        @test canonicalize(a * (-a)) === -(a^SUniform(2))
        @test canonicalize((-a) * (-a)) === a^SUniform(2)
        @test canonicalize(2 * b * 3 * a) === *(SUniform(6), a, b)

        expr = x * (x * y) / (inv(y) * (SUniform(1 // 2) * x)^(-SUniform(1)))
        @test canonicalize(expr) === *(SUniform(1 // 2), x^SUniform(3), y^SUniform(2))

        @test canonicalize(inv(a^b)) === a^(-b)
        @test canonicalize(SUniform(1) / (a^b)) === a^(-b)
        @test canonicalize(a * a^(-b)) === a^(-b + SUniform(1))
    end

    @testset "sum canonicalization" begin
        @test canonicalize((a + b) + (c + a)) === +(2 * a, b, c)
        @test canonicalize(a + b - a) === b
        @test canonicalize(SUniform(2) + x + SUniform(1)) === x + SUniform(3)

        expr = -d + a - e + b + c - f
        expected = (((a + b + c) - d) - e) - f
        @test canonicalize(expr) === expected

        expr2 = -d + a - e - b + c + f
        expected2 = ((((a - b) + c) - d) - e) + f
        @test canonicalize(expr2) === expected2
    end

    @testset "recursive canonicalization" begin
        expr = sin(y + x) * sin(x + y)
        @test canonicalize(expr) === sin(x + y)^SUniform(2)
    end

    @testset "ordering" begin
        @test canonicalize(sin(x) * x) === *(x, sin(x))
        @test canonicalize(sin(x) + x) === +(x, sin(x))
        @test canonicalize(x^SUniform(2) + x^(n + SUniform(1))) === +(x^(n + SUniform(1)), x^SUniform(2))
        @test_throws ArgumentError canonicalize(SScalar(:q) * SSymTensor{0}(:q))
    end

    @testset "opaque division operators" begin
        @test canonicalize((x // y) / z) === (x // y) / z
        @test canonicalize((x ÷ y) * z) === *(z, x ÷ y)
    end

    @testset "idempotence" begin
        expr = sin(y + x) * sin(x + y) + x * (x * y) / (inv(y) * (SUniform(1 // 2) * x)^(-SUniform(1))) - (a + b - a)
        once = canonicalize(expr)
        twice = canonicalize(once)
        @test once === twice
    end
end
