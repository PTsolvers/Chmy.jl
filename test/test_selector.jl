using Test
using Chmy
using Chmy: select, Next, Add, Scale, Dirichlet, Neumann, SiteSelector

# lower `expr` evaluated in the block `block`, and locate it at `locs`
lowerat(expr, locs, block) = simplify(locate(lower(expr, block), locs))

@testset "selector" begin
    @scalars u v
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

    @testset "replacement" begin
        # a term that is not a boundary operator call replaces the selected term
        @test lowerat(select(u, AnyWall() => OnBoundary() => v), (𝓅, 𝓅), (Far(), Near(L, 0))) ==ₛ v[𝓅, 𝓅][0, 0]
        @test lowerat(select(u, Left() => OnBoundary() => p + q), (𝓅, 𝓅), (Near(L, 0), Far())) ==ₛ simplify(p + q)
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
        a = SiteSelector(sel, (), (0 // 1, 0 // 1), (((1,) => 1),))
        b = SiteSelector(sel, (), (0 // 1, 0 // 1), (((1,) => 1),))
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

@testset "tensor slices" begin
    @vectors V t
    @tensors 2 σ
    @uniform @scalars g h
    L, U = Lower(), Upper()
    bottom, top = (Far(), Near(L, 0)), (Far(), Near(U, 0))

    @testset "vectors" begin
        # the normal component is taken along the outward normal, so it changes sign at the lower wall
        V̄ = select(V, AnyWall() => OnBoundary() => Normal() => Dirichlet()(g))
        @test lowerat(V̄[2], (𝓅, 𝓅), bottom) ==ₛ simplify(-g)
        @test lowerat(V̄[2], (𝓅, 𝓅), top) ==ₛ g
        @test lowerat(V̄[1], (𝓅, 𝓅), bottom) ==ₛ V[1][𝓅, 𝓅][0, 0]
        # a later clause applies to the components that the earlier one does not match
        V̄ = select(V, Bottom() => OnBoundary() => Normal() => Dirichlet()(g), Bottom() => OnBoundary() => Tangential() => Add()(h))
        @test lowerat(V̄[1], (𝓅, 𝓅), bottom) ==ₛ simplify(V[1][𝓅, 𝓅][0, 0] + h)
        # fixed and all components are in the local frame as well
        @test lowerat(select(V, Bottom() => OnBoundary() => 2 => Dirichlet()(g))[2], (𝓅, 𝓅), bottom) ==ₛ simplify(-g)
        @test lowerat(select(V, Bottom() => OnBoundary() => (:) => Dirichlet()(g))[1], (𝓅, 𝓅), bottom) ==ₛ g
    end

    @testset "rank 2" begin
        # vector data are indexed by the free slots
        σ̄ = select(σ, AnyWall() => Ghost() => (Normal(), :) => Dirichlet()(t))
        @test lowerat(σ̄[2, 1], (𝓈, 𝓈), (Far(), Near(U, 1 // 2))) ==ₛ simplify(2 * t[1][𝓈, 𝓅][0, 0] - σ[2, 1][𝓈, 𝓈][0, -1])
        @test lowerat(σ̄[2, 1], (𝓈, 𝓈), (Far(), Near(L, 1 // 2))) ==ₛ simplify(-2 * t[1][𝓈, 𝓅][0, 1] - σ[2, 1][𝓈, 𝓈][0, 1])
        # two indices along the normal keep the sign, while the vector data are global and change sign along the normal
        @test lowerat(σ̄[2, 2], (𝓈, 𝓈), (Far(), Near(L, 1 // 2))) ==ₛ simplify(-2 * t[2][𝓈, 𝓅][0, 1] - σ[2, 2][𝓈, 𝓈][0, 1])
        @test lowerat(σ̄[1, 2], (𝓈, 𝓈), (Far(), Near(L, 1 // 2))) ==ₛ σ[1, 2][𝓈, 𝓈][0, 0]
    end

    @testset "stokes" begin
        @scalars P
        @uniform @scalars η
        grad = Gradient(StaggeredCentralDifference())
        # no penetration on the bottom wall
        V̄ = select(V, Bottom() => OnBoundary() => Normal() => Dirichlet()(Literal(0)))
        τ = -P * ℐ(2) + 2η * sym(grad(V̄))
        # free slip on the bottom wall, free surface on the top wall
        τ̄ = select(
            τ, Bottom() => OnBoundary() => (Normal(), Tangential()) => Dirichlet()(Literal(0)),
            Top() => Ghost() | OnBoundary() => (Normal(), :) => Dirichlet()(𝒪(1))
        )
        @test lowerat(τ̄[2, 1], (𝓅, 𝓅), bottom) ==ₛ Literal(0)
        @test lowerat(τ̄[1, 1], (𝓅, 𝓅), bottom) ==ₛ lowerat(τ[1, 1], (𝓅, 𝓅), bottom)
        # the ghost value mirrors the value inside the domain
        ghost = (Far(), Near(U, 1 // 2))
        @test lowerat(τ̄[2, 2], (𝓈, 𝓈), ghost) ==ₛ simplify(-lowerat(τ[2, 2][0, -1], (𝓈, 𝓈), ghost))
    end

    @testset "wall vectors" begin
        @uniform @scalars β
        @uniform @vectors W
        @vectors v
        # the normal points outwards, so the normal component of a global vector is the same at both walls
        V̄ = select(V, AnyWall() => OnBoundary() => Normal() => Dirichlet()(W ⋅ 𝐧))
        @test lowerat(V̄[2], (𝓅, 𝓅), bottom) ==ₛ W[2]
        @test lowerat(V̄[2], (𝓅, 𝓅), top) ==ₛ W[2]
        # a global vector along the normal
        V̄ = select(V, AnyWall() => OnBoundary() => (:) => Dirichlet()(g * 𝐧))
        @test lowerat(V̄[2], (𝓅, 𝓅), bottom) ==ₛ simplify(-g)
        @test lowerat(V̄[2], (𝓅, 𝓅), top) ==ₛ g
        @test lowerat(V̄[1], (𝓅, 𝓅), bottom) ==ₛ Literal(0)
        # partial slip, the tangential traction is proportional to the tangential velocity
        vᵗ = v - (v ⋅ 𝐧) * 𝐧
        σ̄ = select(σ, AnyWall() => Ghost() => (Normal(), Tangential()) => Dirichlet()(β * vᵗ))
        @test lowerat(σ̄[2, 1], (𝓈, 𝓈), (Far(), Near(L, 1 // 2))) ==ₛ simplify(-2β * v[1][𝓈, 𝓅][0, 1] - σ[2, 1][𝓈, 𝓈][0, 1])
        @test lowerat(σ̄[2, 1], (𝓈, 𝓈), (Far(), Near(U, 1 // 2))) ==ₛ simplify(2β * v[1][𝓈, 𝓅][0, 0] - σ[2, 1][𝓈, 𝓈][0, -1])
        @test lowerat(σ̄[1, 2], (𝓈, 𝓈), (Near(L, 1 // 2), Far())) ==ₛ simplify(-2β * v[2][𝓅, 𝓈][1, 0] - σ[1, 2][𝓈, 𝓈][1, 0])
        # every tangential component has its own tangent
        σ̄ = select(σ, Bottom() => Ghost() => (Normal(), Tangential()) => Dirichlet()(β * (v ⋅ 𝐭)))
        bottom3 = (Far(), Near(L, 1 // 2), Far())
        for c in (1, 3)
            @test lowerat(σ̄[2, c], (𝓈, 𝓈, 𝓈), bottom3) ==ₛ simplify(-2β * v[c][𝓈, 𝓅, 𝓈][0, 1, 0] - σ[2, c][𝓈, 𝓈, 𝓈][0, 1, 0])
        end
        # the normal is only defined on walls, the tangent only for tangential slots, and both only in the rule data
        corner = (Near(L, 0), Near(L, 0))
        @test_throws ArgumentError lowerat(select(V, AnyCorner() => OnBoundary() => (:) => Dirichlet()(𝐧))[1], (𝓅, 𝓅), corner)
        @test_throws ArgumentError lowerat(select(V, AnyWall() => OnBoundary() => Normal() => Dirichlet()(v ⋅ 𝐭))[2], (𝓅, 𝓅), bottom)
        @test_throws ArgumentError lowerat(𝐧[1], (𝓅, 𝓅), bottom)
    end

    @testset "malformed slices" begin
        # the slice has one slot per index, and tensor data have the rank of its free slots
        @test_throws ArgumentError select(V, Left() => Ghost() => Dirichlet()(g))
        @test_throws ArgumentError select(σ, Left() => Ghost() => Normal() => Dirichlet()(g))
        @test_throws ArgumentError select(σ, Left() => Ghost() => (Normal(), 1) => Dirichlet()(t))
    end
end
