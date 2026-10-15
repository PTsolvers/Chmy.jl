using Test
using Chmy

@testset "lowering" begin
    @scalars f g @uniform(a)
    @vectors u v
    @tensors 2 A B
    d, s = CentralDifference(), StaggeredCentralDifference()

    @testset "sampling" begin
        expr = sin(f) + a * v[1]
        @test (@inferred lower(expr, (𝓅, 𝓈))) ==ₛ sin(f[𝓅, 𝓈][0, 0]) + a * v[1][𝓅, 𝓈][0, 0]
        @test (@inferred lower(expr, 𝓅, 𝓈)) ==ₛ lower(expr, (𝓅, 𝓈))
        @test (@inferred lower(a, 𝓈)) ==ₛ a
        @test (@inferred lower(f[𝓈, 𝓅], 𝓈, 𝓅)) ==ₛ f[𝓈, 𝓅][0, 0]
        @test (@inferred lower(sin(f[1 // 2, 0]), 𝓅, 𝓅)) ==ₛ sin(f[𝓈, 𝓅][0, 0])
        @test (@inferred lower(f[𝓅, 𝓅][1 // 2, -1], 𝓈, 𝓅)) ==ₛ f[𝓅, 𝓅][1, -1]
        @test (@inferred lower(sin(f)[-1 // 2], 𝓅)) ==ₛ sin(f[𝓈][-1])
        @test (@inferred lower(f[1 // 2][1], 𝓈)) ==ₛ f[𝓅][2]
    end

    @testset "free coordinates" begin
        @test (@inferred lower(ξ¹ * f, 𝓈)) ==ₛ (𝑖 + 1 // 2) * f[𝓈][0]
        @test (@inferred lower(ξ² * f, 𝓈, 𝓅)) ==ₛ 𝑗 * f[𝓈, 𝓅][0, 0]
        @test (@inferred lower(s[1](ξ¹ * f), 𝓅)) ==ₛ (𝑖 + 1 // 2) * f[𝓈][0] - (𝑖 - 1 // 2) * f[𝓈][-1]
    end

    @testset "nested derivatives" begin
        @test (@inferred lower(d[1](d[2](f)), 𝓅, 𝓅)) ==ₛ (1 // 2) * (
            (1 // 2) * (f[𝓅, 𝓅][1, 1] - f[𝓅, 𝓅][1, -1]) - (1 // 2) * (f[𝓅, 𝓅][-1, 1] - f[𝓅, 𝓅][-1, -1])
        )
        @test (@inferred lower(s[1](s[2](f)), 𝓅, 𝓈)) ==ₛ
            (f[𝓈, 𝓅][0, 1] - f[𝓈, 𝓅][0, 0]) -
            (f[𝓈, 𝓅][-1, 1] - f[𝓈, 𝓅][-1, 0])
    end

    @testset "tensor expressions" begin
        divg = Divergence(s)
        @test (@inferred lower(divg(u), 𝓅, 𝓅)) ==ₛ lower(components(divg(u), 2), 𝓅, 𝓅)
        @test (@inferred lower(divg(u), 𝓅, 𝓅)) ==ₛ
            (u[1][𝓈, 𝓅][0, 0] - u[1][𝓈, 𝓅][-1, 0]) + (u[2][𝓅, 𝓈][0, 0] - u[2][𝓅, 𝓈][0, -1])
        @test (@inferred lower((A ⋅ B)[1, 2], 𝓅, 𝓅)) ==ₛ
            A[1, 1][𝓅, 𝓅][0, 0] * B[1, 2][𝓅, 𝓅][0, 0] + A[1, 2][𝓅, 𝓅][0, 0] * B[2, 2][𝓅, 𝓅][0, 0]
        @test (@inferred lower(f .+ g, 𝓈, 𝓈)) ==ₛ f[𝓈, 𝓈][0, 0] + g[𝓈, 𝓈][0, 0]
    end

    @testset "errors" begin
        @test_throws ArgumentError lower(f)
        @test_throws ArgumentError lower(v, 𝓅, 𝓅)
        @test_throws ArgumentError lower(𝑖 * f, 𝓅)
        @test_throws ArgumentError lower(f[𝓈], 𝓅)
        @test_throws ArgumentError lower(f[𝓈], 𝓈, 𝓈)
        @test_throws ArgumentError lower(f[0], 𝓅, 𝓅)
        @test_throws ArgumentError lower(ξ² * f, 𝓅)
        @test_throws ArgumentError lower(d[3](f), 𝓅, 𝓅)
    end
end
