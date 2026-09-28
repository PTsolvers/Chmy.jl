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
        @test (@inferred lower(expr, (i, j))) ==ₛ sin(f[i, j]) + a * v[1][i, j]
        @test (@inferred lower(expr, (𝓅, 𝓈), (i, j))) ==ₛ sin(f[𝓅, 𝓈][i, j]) + a * v[1][𝓅, 𝓈][i, j]
        @test (@inferred lower(g, (𝓅, 𝓈))) ==ₛ g[𝓅, 𝓈][i, j]
        @test (@inferred lower(sin(f)[i, j])) ==ₛ sin(f[i, j])
        @test (@inferred lower(f[𝓈, 𝓅], (𝓅, 𝓈), (i, j))) ==ₛ f[𝓈, 𝓅][i, j]
        @test (@inferred lower(sin(f[i + 1, j]) + g, (i, j))) ==ₛ sin(f[i + 1, j]) + g[i, j]
        @test (@inferred lower(subs(f[i, j], f => sin(g)))) ==ₛ sin(g[i, j])
    end

    @testset "nested derivatives" begin
        @test (@inferred lower(d[1](d[2](f)), (i, j))) ==ₛ (1 // 2) * (
            (1 // 2) * (f[i + 1, j + 1] - f[i + 1, j - 1]) - (1 // 2) * (f[i - 1, j + 1] - f[i - 1, j - 1])
        )
        @test (@inferred lower(s[1](s[2](f)), (𝓅, 𝓈), (i, j))) ==ₛ
            (f[𝓈, 𝓅][i, j + 1] - f[𝓈, 𝓅][i, j]) -
            (f[𝓈, 𝓅][i - 1, j + 1] - f[𝓈, 𝓅][i - 1, j])
    end

    @testset "tensor expressions" begin
        divg = Divergence(d)
        @test (@inferred lower(divg(u), (i, j))) ==ₛ lower(components(divg(u), 2), (i, j))
        @test (@inferred lower(Divergence(s)(u), (𝓅, 𝓈), (i, j))) ==ₛ lower(components(Divergence(s)(u), 2), (𝓅, 𝓈), (i, j))
        @test (@inferred lower((A ⋅ B)[1, 2], (i, j))) ==ₛ A[1, 1][i, j] * B[1, 2][i, j] + A[1, 2][i, j] * B[2, 2][i, j]
        @test (@inferred lower(f .+ g, (i, j))) ==ₛ f[i, j] + g[i, j]
        @test (@inferred lower(divg(u)[i, j])) ==ₛ lower(divg(u), (i, j))
        @test (@inferred lower(divg(u)[𝓅, 𝓈][i, j])) ==ₛ lower(divg(u), (𝓅, 𝓈), (i, j))
    end

    @testset "errors" begin
        @test_throws ArgumentError lower(v, (i, j))
        @test_throws ArgumentError lower(sin(f))
        @test_throws ArgumentError lower(f[𝓅, 𝓈][i])
        @test_throws ArgumentError lower(f, (𝓅, 𝓈), (i,))
        @test_throws ArgumentError lower(g[i], (i, j))
        @test_throws ArgumentError lower(f + tr(A)[i], (i, j))
    end
end
