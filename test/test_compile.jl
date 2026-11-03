using Test
using Chmy

@testset "compilation" begin
    @scalars a g
    @tensors 2 A B

    sampled = lower((sin.(A .+ a) .* B .^ 2)[1, 2], (:, :), (0, 0))
    binding = Binding(A[1, 2][𝓅, 𝓅] => fill(0.2, 2, 2), a[𝓅, 𝓅] => fill(0.3, 2, 2), B[1, 2][𝓅, 𝓅] => fill(2.0, 2, 2))
    f = compile(sampled, binding)
    @test (@inferred f(binding.data, (1, 2))) ≈ sin(0.5) * 4

    staggered = lower(StaggeredCentralDifference()[1](a), (:,), (0,))
    binding = Binding(a[𝓈] => [1.0, 3.0, 7.0])
    f = compile(staggered, binding)
    @test (@inferred f(binding.data, (2,))) == 2.0

    # boundary data stored along the left wall are indexed along the second axis only
    D = StaggeredCentralDifference()
    ã = override(a, LeftWall => WallValue(𝒷(g)))
    left = simplify(lower(D[1](ã), (↓, :), (0, 1 // 2)))
    binding = Binding(a[𝓈, 𝓈] => [1.0 2.0; 3.0 4.0], g[↓, 𝓈] => [5.0, 7.0])
    f = compile(left, binding)
    @test (@inferred f(binding.data, (1, 2))) == 2.0 - 2 * 7.0 + 2.0
    # a single value for the whole wall
    ã = override(a, LeftWall => WallValue(𝒷(g; span = true)))
    left = simplify(lower(D[1](ã), (↓, :), (0, 1 // 2)))
    f = compile(left, Binding(a[𝓈, 𝓈] => [1.0 2.0; 3.0 4.0], g[↓, :] => 5.0))
    @test f((([1.0 2.0; 3.0 4.0]), 5.0), (1, 2)) == 2.0 - 2 * 5.0 + 2.0
    f = compile(left, Binding(a[𝓈, 𝓈] => [1.0 2.0; 3.0 4.0], g[↓, :] => fill(5.0)))
    @test f((([1.0 2.0; 3.0 4.0]), fill(5.0)), (1, 2)) == 2.0 - 2 * 5.0 + 2.0
end
