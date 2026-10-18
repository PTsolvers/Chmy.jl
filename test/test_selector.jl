using Test
using Chmy
using Chmy: LowerContext, select, Next, Add, Scale

# lower `expr` at the block `block`, with the evaluation point at zero offsets
lowerat(expr, locs, block) = simplify(lower(expr, LowerContext(locs, map(_ -> 0 // 1, locs), block)))

@testset "selector" begin
    @scalars u
    @uniform @scalars p q pc
    L, U = Lower(), Upper()
    half = Literal(1 // 2)
    corner = (Near(L, 0), Near(L, 0))
    r = lowerat(u, (𝓅, 𝓅), corner)

    @testset "unselected value" begin
        # far from every wall, outside the region, or a rule that imposes nothing
        @test lowerat(select(u, Left() => OnBoundary() => Add(p)), (𝓅, 𝓅), (Far(), Far())) ==ₛ r
        @test lowerat(select(u, Left() => OnBoundary() => Add(p)), (𝓅, 𝓅), (Near(L, 1), Far())) ==ₛ r
        @test lowerat(select(u, Left() => OnBoundary() => Next()), (𝓅, 𝓅), corner) ==ₛ r
    end

    @testset "one wall" begin
        # only the bottom wall is near, so the left clause never applies
        rules = (Left() => OnBoundary() => Add(p), Bottom() => OnBoundary() => Add(q))
        @test lowerat(select(u, rules...), (𝓅, 𝓅), (Far(), Near(L, 0))) ==ₛ simplify(r + q)
    end

    @testset "composition at a corner" begin
        # walls compose through next, the first declared wall is the outermost wrapper
        @test lowerat(select(u, Left() => OnBoundary() => Add(p), Bottom() => OnBoundary() => Add(q)), (𝓅, 𝓅), corner) ==ₛ simplify(r + q + p)
        @test lowerat(select(u, Left() => OnBoundary() => Scale(half), Bottom() => OnBoundary() => Scale(half)), (𝓅, 𝓅), corner) ==ₛ simplify(Literal(1 // 4) * r)
        @test lowerat(select(u, Left() => OnBoundary() => Add(p), Bottom() => OnBoundary() => Scale(Literal(2))), (𝓅, 𝓅), corner) ==ₛ simplify(2 * r + p)
        @test lowerat(select(u, Left() => OnBoundary() => Scale(Literal(2)), Bottom() => OnBoundary() => Add(q)), (𝓅, 𝓅), corner) ==ₛ simplify(2 * (r + q))
        # the corner clause wraps both walls, whatever the declaration order
        @test lowerat(select(u, AnyWall() => OnBoundary() => Add(p), AnyCorner() => OnBoundary() => Add(pc)), (𝓅, 𝓅), corner) ==ₛ simplify(r + 2 * p + pc)
    end

    @testset "positions through stencils" begin
        D = StaggeredCentralDifference()
        # cell centre next to the lower wall: the sample at -1//2 lies on the wall
        lo = lowerat(D[2](select(u, Bottom() => OnBoundary() => Add(q))), (𝓈, 𝓈), (Far(), Near(L, 1 // 2)))
        @test lo ==ₛ simplify(u[𝓈, 𝓅][0, 1] - (u[𝓈, 𝓅][0, 0] + q))
        # next to the upper wall: the sample at +1//2 lies on the wall
        hi = lowerat(D[2](select(u, Top() => OnBoundary() => Add(q))), (𝓈, 𝓈), (Far(), Near(U, 1 // 2)))
        @test hi ==ₛ simplify((u[𝓈, 𝓅][0, 1] + q) - u[𝓈, 𝓅][0, 0])
    end
end
