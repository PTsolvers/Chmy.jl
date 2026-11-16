using Test
using Chmy

@testset "bindings" begin
    @scalars u g
    @uniform @scalars a
    # samples are bound by the location of their positions, whatever the frame, traces by their element
    @test isequal(bindings(a * u[(:, 1 // 2), (↓, 3 // 2)] + u[(↑, -1 // 2), (:, 1 // 2)]), (u[𝓈, 𝓈], a))
    # in the order of the canonical form
    @test isequal(bindings(u[(:, 0), (:, 1 // 2)] * g[↓, (:, 1)]), (g[↓, 𝓅], u[𝓅, 𝓈]))
    # samples at traces only are their own keys, and coordinates are not bound
    @test isequal(bindings(g[↓, :] + u[(:, 1), (:, 1)]), (g[↓, :], u[𝓅, 𝓅]))
    @test isequal(bindings((ξ¹ * u)[(:, 0)]), (u[𝓅],))
    @test u[𝓈, ↓] isa Chmy.Located
    @test isequal(u[𝓈, ↓], u[𝓈, ↓])
    @test !isequal(u[𝓈, ↓], u[𝓈, ↑])
    @test sprint(show, u[𝓈, :]) == "u[𝓈, :]"
end

@testset "compilation" begin
    @scalars a g u
    @tensors 2 A B
    N = (4, 4)
    o = ((:, 0), (:, 0))

    sampled = (sin.(A .+ a) .* B .^ 2)[1, 2][o...]
    binding = Binding(A[1, 2][𝓅, 𝓅] => fill(0.2, 2, 2), a[𝓅, 𝓅] => fill(0.3, 2, 2), B[1, 2][𝓅, 𝓅] => fill(2.0, 2, 2))
    f = compile(sampled, binding)
    @test (@inferred f(binding.data, (1, 2), N)) ≈ sin(0.5) * 4

    staggered = StaggeredCentralDifference()[1](a)[(:, 0)]
    binding = Binding(a[𝓈] => [1.0, 3.0, 7.0])
    f = compile(staggered, binding)
    @test (@inferred f(binding.data, (2,), (4,))) == 2.0

    # the laplacian with a zero outward derivative, indexed from the origin of the frame
    D = StaggeredCentralDifference()
    Δ = Divergence(D)(Gradient(D)(override(u, WallDerivative(0))))
    U = [1.0, 2.0, 4.0, 8.0]
    data = (U,)
    for (x, expected) in (
            ((:, 1 // 2), U[1] - 2U[2] + U[3]),
            ((↓, 1 // 2), U[2] - U[1]),
            ((↓, 3 // 2), U[1] - 2U[2] + U[3]),
            ((↑, -1 // 2), U[3] - U[4]),
        )
        f = compile(Δ[x], Binding(u[𝓈] => U))
        @test (@inferred f(data, (2,), (5,))) == expected
    end

    # boundary data stored along the left wall are indexed along the second axis only
    ã = override(a, LeftWall => WallValue(𝒷(g)))
    left = D[1](ã)[(↓, 0), (:, 1 // 2)]
    @test isequal(bindings(left), (a[𝓈, 𝓈], g[↓, 𝓈]))
    binding = Binding(a[𝓈, 𝓈] => [1.0 2.0; 3.0 4.0], g[↓, 𝓈] => [5.0, 7.0])
    f = compile(left, binding)
    @test (@inferred f(binding.data, (99, 2), N)) == 2.0 - 2 * 7.0 + 2.0
    # a single value for the whole wall
    ã = override(a, LeftWall => WallValue(𝒷(g; span = true)))
    left = D[1](ã)[(↓, 0), (:, 1 // 2)]
    f = compile(left, Binding(a[𝓈, 𝓈] => [1.0 2.0; 3.0 4.0], g[↓, :] => 5.0))
    @test f(([1.0 2.0; 3.0 4.0], 5.0), (1, 2), N) == 2.0 - 2 * 5.0 + 2.0
    f = compile(left, Binding(a[𝓈, 𝓈] => [1.0 2.0; 3.0 4.0], g[↓, :] => fill(5.0)))
    @test f(([1.0 2.0; 3.0 4.0], fill(5.0)), (1, 2), N) == 2.0 - 2 * 5.0 + 2.0

    # surfaces at the bottom and the top extruded with a vertical coordinate
    @scalars B S c
    Z = Project(:, 1)(c) * Project(↑, 2)(S) + (1 - Project(:, 1)(c)) * Project(↓, 2)(B)
    binding = Binding(c[:, 𝓅] => [0.0, 0.5, 1.0], S[𝓅, ↑] => [10.0, 20.0], B[𝓅, ↓] => [1.0, 2.0])
    f = compile(Z[o...], binding)
    @test f(binding.data, (2, 2), (2, 3)) == 0.5 * 20.0 + 0.5 * 2.0

    # coordinates count from the lower wall, and the upper wall is the point `N`
    f = compile((ξ¹ * a)[(↑, -1 // 2)], Binding(a[𝓈] => [1.0, 1.0, 1.0]))
    @test f(([1.0, 1.0, 1.0],), (1,), (4,)) == 5 // 2
    f = compile((ξ¹ * a)[(:, 1 // 2)], Binding(a[𝓈] => [1.0, 1.0, 1.0]))
    @test f(([1.0, 1.0, 1.0],), (2,), (4,)) == 3 // 2
    f = compile(ξ¹[↑], Binding())
    @test f((), (1,), (4,)) == 3

    # missing keys
    @test_throws ArgumentError compile(left, Binding(a[𝓈, 𝓈] => [1.0 2.0; 3.0 4.0]))
end
