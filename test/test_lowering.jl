using Test
using Chmy

@testset "lowering" begin
    @scalars f g @uniform(a)
    @vectors u v
    @tensors 2 A B
    i, j = 𝑖, 𝑗
    d, s = CentralDifference(), StaggeredCentralDifference()

    @testset "sampling" begin
        expr = sin(f) + a * v[1]
        @test (@inferred lower(expr, (𝓅, 𝓈))) ==ₛ sin(f[𝓅, 𝓈][i, j]) + a * v[1][𝓅, 𝓈][i, j]
        @test (@inferred lower(expr, 𝓅, 𝓈)) ==ₛ lower(expr, (𝓅, 𝓈))
        @test (@inferred lower(a, 𝓈)) ==ₛ a
        @test (@inferred lower(f[𝓈, 𝓅], 𝓈, 𝓅)) ==ₛ f[𝓈, 𝓅][i, j]
        @test (@inferred lower(sin(f[ξ¹ + 1 // 2, ξ²]), 𝓅, 𝓅)) ==ₛ sin(f[𝓈, 𝓅][i, j])
        @test (@inferred lower(f[𝓅, 𝓅][ξ¹ + 1 // 2, ξ² - 1], 𝓈, 𝓅)) ==ₛ f[𝓅, 𝓅][i + 1, j - 1]
        @test (@inferred lower(sin(f)[ξ¹ - 1 // 2], 𝓅)) ==ₛ sin(f[𝓈][i - 1])
    end

    @testset "free coordinates" begin
        @test (@inferred lower(ξ¹ * f, 𝓈)) ==ₛ (i + 1 // 2) * f[𝓈][i]
        @test (@inferred lower(ξ² * f, 𝓈, 𝓅)) ==ₛ j * f[𝓈, 𝓅][i, j]
        @test (@inferred lower(s[1](ξ¹ * f), 𝓅)) ==ₛ (i + 1 // 2) * f[𝓈][i] - (i - 1 // 2) * f[𝓈][i - 1]
    end

    @testset "nested derivatives" begin
        @test (@inferred lower(d[1](d[2](f)), 𝓅, 𝓅)) ==ₛ (1 // 2) * (
            (1 // 2) * (f[𝓅, 𝓅][i + 1, j + 1] - f[𝓅, 𝓅][i + 1, j - 1]) - (1 // 2) * (f[𝓅, 𝓅][i - 1, j + 1] - f[𝓅, 𝓅][i - 1, j - 1])
        )
        @test (@inferred lower(s[1](s[2](f)), 𝓅, 𝓈)) ==ₛ
            (f[𝓈, 𝓅][i, j + 1] - f[𝓈, 𝓅][i, j]) -
            (f[𝓈, 𝓅][i - 1, j + 1] - f[𝓈, 𝓅][i - 1, j])
    end

    @testset "tensor expressions" begin
        divg = Divergence(s)
        @test (@inferred lower(divg(u), 𝓅, 𝓅)) ==ₛ lower(components(divg(u), 2), 𝓅, 𝓅)
        @test (@inferred lower(divg(u), 𝓅, 𝓅)) ==ₛ
            (u[1][𝓈, 𝓅][i, j] - u[1][𝓈, 𝓅][i - 1, j]) + (u[2][𝓅, 𝓈][i, j] - u[2][𝓅, 𝓈][i, j - 1])
        @test (@inferred lower((A ⋅ B)[1, 2], 𝓅, 𝓅)) ==ₛ
            A[1, 1][𝓅, 𝓅][i, j] * B[1, 2][𝓅, 𝓅][i, j] + A[1, 2][𝓅, 𝓅][i, j] * B[2, 2][𝓅, 𝓅][i, j]
        @test (@inferred lower(f .+ g, 𝓈, 𝓈)) ==ₛ f[𝓈, 𝓈][i, j] + g[𝓈, 𝓈][i, j]
    end

    @testset "errors" begin
        @test_throws ArgumentError lower(f)
        @test_throws ArgumentError lower(v, 𝓅, 𝓅)
        @test_throws ArgumentError lower(f[i, j], 𝓅, 𝓅)
        @test_throws ArgumentError lower(i * f, 𝓅)
        @test_throws ArgumentError lower(f[𝓈], 𝓅)
        @test_throws ArgumentError lower(f[𝓈], 𝓈, 𝓈)
        @test_throws ArgumentError lower(f[2ξ¹], 𝓅)
        @test_throws ArgumentError lower(f[ξ¹ + 1 // 3], 𝓅)
        @test_throws ArgumentError lower(f[ξ², ξ¹], 𝓅, 𝓅)
        @test_throws ArgumentError lower(f[ξ¹], 𝓅, 𝓅)
        @test_throws ArgumentError lower(ξ² * f, 𝓅)
        @test_throws ArgumentError lower(d[3](f), 𝓅, 𝓅)
    end
end
