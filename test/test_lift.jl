using Test
using Chmy

struct WeightedStencil <: Operator end
Chmy.stencil_rule(::WeightedStencil, args::Tuple{DTerm, DTerm}, inds::Tuple{DTerm}) = (1 // 3) * args[1][inds[1]] + args[2][inds[1] + 1]

@testset "lifting" begin
    @scalars f g
    d, s = CentralDifference(), StaggeredCentralDifference()
    i, j = 𝑖, 𝑗

    @test (@inferred d[2]) == Lifted(d, 2)
    @test (@inferred stencil_rule(d[2], (f,), (i, j))) ==ₛ (1 // 2) * (f[i, j + 1] - f[i, j - 1])
    @test (@inferred stencil_rule(d[2], (f,), (𝓅, 𝓈), (i, j))) ==ₛ (1 // 2) * (f[𝓅, 𝓈][i, j + 1] - f[𝓅, 𝓈][i, j - 1])
    @test (@inferred stencil_rule(s[1], (f,), (𝓅, 𝓈), (i, j))) ==ₛ f[𝓈, 𝓈][i, j] - f[𝓈, 𝓈][i - 1, j]
    @test (@inferred stencil_rule(s[2], (f,), (𝓅, 𝓈), (i, j))) ==ₛ f[𝓅, 𝓅][i, j + 1] - f[𝓅, 𝓅][i, j]
    @test (@inferred stencil_rule(Lifted(WeightedStencil(), 2), (f, g), (i, j))) ==ₛ (1 // 3) * f[i, j] + g[i, j + 1]

    @test_throws ArgumentError Lifted(d, 0)
    @test_throws ArgumentError stencil_rule(d[3], (f,), (i, j))
end
