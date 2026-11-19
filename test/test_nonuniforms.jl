using Test
using Chmy

@testset "nonuniforms" begin
    @scalars f g
    @uniform @scalars a
    @vectors u
    s = StaggeredCentralDifference()
    divg, grad = Divergence(s), Gradient(s)
    o = ((:, 0), (:, 0))

    @testset "samples" begin
        nu = @inferred nonuniforms(divg(grad(f)), o...)
        @test nu.position == (Coord(:, 0), Coord(:, 0))
        @test isequal(collect(keys(nu.samples)), [f])
        @test issetequal(
            nu.samples[f],
            [(Coord(:, x), Coord(:, y)) for (x, y) in ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))],
        )
        # each component is a field of its own
        nu = nonuniforms(divg(u), (:, 1 // 2), (:, 1 // 2))
        @test nu.samples[u[1]] == [(Coord(:, 1), Coord(:, 1 // 2)), (Coord(:, 0), Coord(:, 1 // 2))]
        @test nu.samples[u[2]] == [(Coord(:, 1 // 2), Coord(:, 1)), (Coord(:, 1 // 2), Coord(:, 0))]
        # coordinates and uniform terms are not fields
        @test isequal(collect(keys(nonuniforms(a * ξ¹ * f, (:, 0)).samples)), [f])
        # samples next to a wall are in the frame of the wall
        @test issetequal(nonuniforms(divg(grad(f)), (↓, 1 // 2)).samples[f], [(Coord(↓, x),) for x in (-1 // 2, 1 // 2, 3 // 2)])
        @test sprint(show, nonuniforms(f + g, o...)) == "Nonuniforms(f, g) at [(:, 0), (:, 0)]"
    end
end
