using Test
using Chmy
using Chmy: LowerContext, select, Next, Add, Scale, Dirichlet, Neumann, SiteSelector

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
        @test lowerat(select(u, Left() => OnBoundary() => Add()(p)), (𝓅, 𝓅), (Far(), Far())) ==ₛ r
        @test lowerat(select(u, Left() => OnBoundary() => Add()(p)), (𝓅, 𝓅), (Near(L, -1), Far())) ==ₛ r
        @test lowerat(select(u, Left() => OnBoundary() => Next()), (𝓅, 𝓅), corner) ==ₛ r
    end

    @testset "one wall" begin
        # only the bottom wall is near, so the left clause never applies
        rules = (Left() => OnBoundary() => Add()(p), Bottom() => OnBoundary() => Add()(q))
        @test lowerat(select(u, rules...), (𝓅, 𝓅), (Far(), Near(L, 0))) ==ₛ simplify(r + q)
    end

    @testset "composition at a corner" begin
        # walls compose through next, the first declared wall is the outermost wrapper
        @test lowerat(select(u, Left() => OnBoundary() => Add()(p), Bottom() => OnBoundary() => Add()(q)), (𝓅, 𝓅), corner) ==ₛ simplify(r + q + p)
        @test lowerat(select(u, Left() => OnBoundary() => Scale()(half), Bottom() => OnBoundary() => Scale()(half)), (𝓅, 𝓅), corner) ==ₛ simplify(Literal(1 // 4) * r)
        @test lowerat(select(u, Left() => OnBoundary() => Add()(p), Bottom() => OnBoundary() => Scale()(Literal(2))), (𝓅, 𝓅), corner) ==ₛ simplify(2 * r + p)
        @test lowerat(select(u, Left() => OnBoundary() => Scale()(Literal(2)), Bottom() => OnBoundary() => Add()(q)), (𝓅, 𝓅), corner) ==ₛ simplify(2 * (r + q))
        # the corner clause wraps both walls, whatever the declaration order
        @test lowerat(select(u, AnyWall() => OnBoundary() => Add()(p), AnyCorner() => OnBoundary() => Add()(pc)), (𝓅, 𝓅), corner) ==ₛ simplify(r + 2 * p + pc)
    end

    @testset "positions through stencils" begin
        D = StaggeredCentralDifference()
        # cell centre next to the lower wall: the sample at -1//2 lies on the wall
        lo = lowerat(D[2](select(u, Bottom() => OnBoundary() => Add()(q))), (𝓈, 𝓈), (Far(), Near(L, -1 // 2)))
        @test lo ==ₛ simplify(u[𝓈, 𝓅][0, 1] - (u[𝓈, 𝓅][0, 0] + q))
        # next to the upper wall: the sample at +1//2 lies on the wall
        hi = lowerat(D[2](select(u, Top() => OnBoundary() => Add()(q))), (𝓈, 𝓈), (Far(), Near(U, -1 // 2)))
        @test hi ==ₛ simplify((u[𝓈, 𝓅][0, 1] + q) - u[𝓈, 𝓅][0, 0])
    end

    @testset "upper walls" begin
        # rules are written for the upper wall, so they are lifted with positive axes there
        @test lowerat(select(u, Right() => OnBoundary() => Add()(p)), (𝓅,), (Near(U, 0),)) ==ₛ simplify(u[𝓅][0] + p)
        rules = (Right() => OnBoundary() => Add()(p), Top() => OnBoundary() => Scale()(Literal(2)))
        @test lowerat(select(u, rules...), (𝓅, 𝓅), (Near(U, 0), Near(U, 0))) ==ₛ simplify(2 * r + p)
        # every wall of a 3D corner on mixed sides contributes once
        corner3 = (Near(U, 0), Near(L, 0), Near(U, 0))
        @test lowerat(select(u, AnyWall() => OnBoundary() => Add()(p)), (𝓅, 𝓅, 𝓅), corner3) ==ₛ simplify(u[𝓅, 𝓅, 𝓅][0, 0, 0] + 3 * p)
    end

    @testset "data of boundary operators" begin
        # the data are arguments of the selector call, so rewriters reach them
        @test lowerat(subs(select(u, Left() => OnBoundary() => Add()(p)), p => q), (𝓅, 𝓅), corner) ==ₛ simplify(r + q)
    end

    @testset "site instances" begin
        # instances built independently at the same site are equal, so they compare and hash by content
        sel = operation(select(u, Left() => OnBoundary() => Add()(p)))
        a = SiteSelector(sel, (0 // 1, 0 // 1), (((1,) => 1),))
        b = SiteSelector(sel, (0 // 1, 0 // 1), (((1,) => 1),))
        @test a == b
        @test hash(a) == hash(b)
    end

    @testset "unsupported codimension" begin
        # a wall rule declared on a corner pattern
        @uniform @scalars g
        ghost = (Near(L, 1 // 2), Near(L, 1 // 2))
        @test_throws ArgumentError lowerat(select(u, AnyCorner() => Ghost() => Dirichlet()(g)), (𝓈, 𝓈), ghost)
    end

    @testset "wall rules" begin
        @uniform @scalars g gL gB
        uL = Left() => Ghost() => Dirichlet()(g)
        # a ghost segment: the wall is half a cell away, the mirror sample one cell inside
        @test lowerat(select(u, uL), (𝓈,), (Near(L, 1 // 2),)) ==ₛ simplify(2g - u[𝓈][1])
        # the same at the upper wall, mirrored
        @test lowerat(select(u, Right() => Ghost() => Dirichlet()(g)), (𝓈,), (Near(U, 1 // 2),)) ==ₛ simplify(2g - u[𝓈][-1])
        # a ghost point extrapolates from the boundary value as well, and the wall node takes the boundary value
        rules = (Left() => OnBoundary() => Dirichlet()(g), uL)
        @test lowerat(select(u, rules...), (𝓅,), (Near(L, 1),)) ==ₛ simplify(2g - u[𝓅][2])
        @test lowerat(select(u, rules...), (𝓅,), (Near(L, 0),)) ==ₛ g
        # the boundary value is read at the wall, at both walls
        @test lowerat(select(u, Left() => Ghost() => Dirichlet()(sin(ξ¹))), (𝓈,), (Near(L, 1 // 2),)) ==ₛ simplify(2sin(𝑖) - u[𝓈][1])
        @test lowerat(select(u, Right() => Ghost() => Dirichlet()(sin(ξ¹))), (𝓈,), (Near(U, 1 // 2),)) ==ₛ simplify(2sin(𝑖 - 1) - u[𝓈][-1])
        # an outward derivative gives the same expression at both walls
        @test lowerat(select(u, AnyWall() => Ghost() => Neumann()(g)), (𝓈,), (Near(L, 1 // 2),)) ==ₛ simplify(u[𝓈][1] + g)
        @test lowerat(select(u, AnyWall() => Ghost() => Neumann()(g)), (𝓈,), (Near(U, 1 // 2),)) ==ₛ simplify(u[𝓈][-1] + g)
        @test lowerat(select(u, Left() => Ghost() => Neumann()(g)), (𝓅,), (Near(L, 1),)) ==ₛ simplify(u[𝓅][2] + 2g)
        # outside of their range the rules throw
        @test_throws ArgumentError lowerat(select(u, Left() => NearBoundary(1) => Dirichlet()(g)), (𝓈,), (Near(L, -1 // 2),))
        @test_throws ArgumentError lowerat(select(u, Left() => OnBoundary() => Neumann()(g)), (𝓅,), (Near(L, 0),))
        # a diagonal ghost: the left rule moves the sample inside, where the bottom rule applies
        rules = (Left() => Ghost() => Dirichlet()(gL), Bottom() => Ghost() => Dirichlet()(gB))
        ghosts = (Near(L, 1 // 2), Near(L, 1 // 2))
        @test lowerat(select(u, rules...), (𝓈, 𝓈), ghosts) ==ₛ simplify(2gL - (2gB - u[𝓈, 𝓈][1, 1]))
    end

    @testset "laplacian at a corner" begin
        @uniform @scalars uL ĝ
        D = StaggeredCentralDifference()
        ũ = select(u, Left() => Ghost() => Dirichlet()(uL), AnyWall() => Ghost() => Neumann()(ĝ))
        Δũ = D[1](D[1](ũ)) + D[2](D[2](ũ))
        expected = u[𝓈, 𝓈][1, 0] + u[𝓈, 𝓈][0, 1] - 4 * u[𝓈, 𝓈][0, 0] + 2uL + ĝ
        @test lowerat(Δũ, (𝓈, 𝓈), (Near(L, -1 // 2), Near(L, -1 // 2))) ==ₛ simplify(expected)
    end
end
