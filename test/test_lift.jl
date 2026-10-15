using Test
using Chmy

struct WeightedStencil <: Operator end
Chmy.stencil_rule(::WeightedStencil, args::Tuple{DTerm, DTerm}, coords::Tuple{DTerm}) = (1 // 3) * args[1][coords[1]] + args[2][coords[1] + 1]

@testset "lifting" begin
    @scalars f g
    d, s = CentralDifference(), StaggeredCentralDifference()

    @test (@inferred d[2]) == Lifted(d, 2)
    @test (@inferred stencil_rule(d[2], (f,), (ξ¹, ξ²))) ==ₛ (1 // 2) * (f[ξ¹, ξ² + 1] - f[ξ¹, ξ² - 1])
    @test (@inferred stencil_rule(s[1], (f,), (ξ¹, ξ²))) ==ₛ f[ξ¹ + 1 // 2, ξ²] - f[ξ¹ - 1 // 2, ξ²]
    @test (@inferred stencil_rule(s[2], (f,), (ξ¹ + 1 // 2, ξ²))) ==ₛ f[ξ¹ + 1 // 2, ξ² + 1 // 2] - f[ξ¹ + 1 // 2, ξ² - 1 // 2]
    @test (@inferred stencil_rule(Lifted(WeightedStencil(), 2), (f, g), (ξ¹, ξ²))) ==ₛ (1 // 3) * f[ξ¹, ξ²] + g[ξ¹, ξ² + 1]

    @test_throws ArgumentError Lifted(d, 0)
    @test_throws ArgumentError stencil_rule(d[3], (f,), (ξ¹, ξ²))
end
