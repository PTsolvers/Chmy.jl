using Test
using Chmy
using Chmy: OverrideComponent, OverrideSample, candidates, match

@testset "predicates" begin
    # a ghost segment by the left wall, on the bottom wall
    x = (Coord(↓, -1 // 2), Coord(↓, 0))
    # a wall predicate matches the wall at the outward distance of the sample
    @test match(WallPredicate(LeftWall, Ghost, ()), (↓, :), (), x) == FaceMatch((-1,), (1 // 2,))
    @test isnothing(match(WallPredicate(LeftWall, Ghost, ()), (:, ↓), (), x))
    @test isnothing(match(WallPredicate(BottomWall, Ghost, ()), (:, ↓), (), x))
    @test isnothing(match(WallPredicate(LeftWall, Ghost, ()), (↓, ↓), (), x))
    @test match(WallPredicate(LeftWall, Boundary, ()), (↓, :), (), (Coord(↓, 0), Coord(:, 0))) == FaceMatch((-1,), (0 // 1,))
    # and the component must match the tensor pattern
    @test isnothing(match(WallPredicate(LeftWall, Ghost, (𝐧,)), (↓, :), (2,), x))
    @test match(WallPredicate(LeftWall, Ghost, (𝐧,)), (↓, :), (1,), x) == FaceMatch((-1,), (1 // 2,))
    # traces have no distance from the wall
    @test isnothing(match(WallPredicate(LeftWall, Ghost | Boundary, ()), (↓, :), (), (Trace(↓), Coord(:, 0))))
    # a face predicate matches the region on every wall of the face
    m = match(FacePredicate(AnyCorner, Ghost | Boundary), (↓, ↓), (), x)
    @test m == FaceMatch((-1, -2), (1 // 2, 0 // 1))
    @test signed_axes(m) == (-1, -2)
    @test signed_distances(m) == (1 // 2, 0 // 1)
    @test isnothing(match(FacePredicate(AnyCorner, Ghost), (↓, ↓), (), x))
    @test isnothing(match(FacePredicate(AnyCorner, Ghost | Boundary), (↓, ↓), (), (Trace(↓), Coord(↓, 0))))
    @test match(FacePredicate(LeftWall, Ghost), (↓, :), (), x) == FaceMatch((-1,), (1 // 2,))
end

@testset "candidates" begin
    x = (Coord(↓, 0), Coord(↑, 0))
    walls, corners = WallPredicate(AnyWall, Boundary, ()), FacePredicate(AnyCorner, Boundary)
    # every face next to the sample takes its first matching predicate, ordered by the predicate
    wall1, wall2, corner = FaceMatch((-1,), (0 // 1,)), FaceMatch((2,), (0 // 1,)), FaceMatch((-1, 2), (0 // 1, 0 // 1))
    @test candidates((walls, corners), (), x) == (1 => wall1, 1 => wall2, 2 => corner)
    @test candidates((corners, walls), (), x) == (1 => corner, 2 => wall1, 2 => wall2)
    # a later predicate of the same face is shadowed
    @test candidates((walls, WallPredicate(LeftWall, Boundary, ())), (), x) == (1 => wall1, 1 => wall2)
    # no faces next to coordinates relative to the grid point, or to traces
    @test candidates((walls,), (), (Coord(:, 0), Coord(:, 0))) == ()
    @test candidates((walls,), (), (Trace(↓), Trace(↑))) == ()
end

@testset "override" begin
    @scalars u v
    @uniform @scalars p q pc

    @testset "not overridden" begin
        # in the interior, or outside the region
        @test (@inferred override(u, LeftWall => Add(p))[(:, 0), (:, 0)]) ==ₛ u[(:, 0), (:, 0)]
        @test override(u, (LeftWall, Boundary) => Add(p))[(↓, 1), (:, 0)] ==ₛ u[(↓, 1), (:, 0)]
        # and at traces, which have no distance from the wall
        @test override(u, (LeftWall, Boundary) => Add(p))[↓, (:, 0)] ==ₛ u[↓, (:, 0)]
    end

    @testset "one wall" begin
        # only the bottom wall is next to the sample, so the left predicate never applies
        x = ((:, 0), (↓, 0))
        @test override(u, (LeftWall, Boundary) => Add(p), (BottomWall, Boundary) => Add(q))[x...] ==ₛ u[x...] + q
        # a sample on the wall is overridden in an expression of samples anywhere
        w = ((↓, 0), (:, 0))
        @test v[(:, 1), (:, 1)] * override(u, (LeftWall, Boundary) => Add(p))[w...] ==ₛ v[(:, 1), (:, 1)] * (u[w...] + p)
    end

    @testset "replacement" begin
        # a term that is not a rule call replaces the overridden term
        @test override(u, Boundary => v)[(:, 0), (↓, 0)] ==ₛ v[(:, 0), (↓, 0)]
        @test override(u, (LeftWall, Boundary) => p + q)[(↓, 0), (:, 0)] ==ₛ p + q
        @test override(u, Boundary => 0)[(↓, 0), (:, 0)] ==ₛ Literal(0)
        # also a number
        @test override(1, Boundary => Scale(1 // 2))[(↓, 0), (:, 0)] ==ₛ Literal(1 // 2)
    end

    @testset "composition at a corner" begin
        c = ((↓, 0), (↓, 0))
        r = u[c...]
        # the rules compose in the order of the predicates, the first one is the outermost
        @test override(u, (LeftWall, Boundary) => Add(p), (BottomWall, Boundary) => Scale(2))[c...] ==ₛ 2r + p
        @test override(u, (BottomWall, Boundary) => Scale(2), (LeftWall, Boundary) => Add(p))[c...] ==ₛ 2 * (r + p)
        @test override(u, (AnyCorner, Boundary) => Scale(1 // 2), Boundary => Add(p))[c...] ==ₛ (1 // 2) * (r + p + p)
        @test override(u, Boundary => Add(p), (AnyCorner, Boundary) => Add(pc))[c...] ==ₛ r + p + p + pc
        # every wall of a 3D corner on mixed sides contributes once
        c3 = ((↑, 0), (↓, 0), (↑, 0))
        @test override(u, Boundary => Add(p))[c3...] ==ₛ u[c3...] + 3p
    end

    @testset "positions through stencils" begin
        D = StaggeredCentralDifference()
        # cell centre next to the lower wall: the sample at 0 lies on the wall
        lo = D[2](override(u, (BottomWall, Boundary) => Add(q)))[(:, 1 // 2), (↓, 1 // 2)]
        @test lo ==ₛ u[(:, 1 // 2), (↓, 1)] - (u[(:, 1 // 2), (↓, 0)] + q)
        # next to the upper wall: the sample at 0 lies on the wall
        hi = D[2](override(u, (TopWall, Boundary) => Add(q)))[(:, 1 // 2), (↑, -1 // 2)]
        @test hi ==ₛ (u[(:, 1 // 2), (↑, 0)] + q) - u[(:, 1 // 2), (↑, -1)]
    end

    @testset "data of rules" begin
        # the data are arguments of the override call, so rewriters reach them
        c = ((↓, 0), (↓, 0))
        @test subs(override(u, Boundary => Add(p)), p => q)[c...] ==ₛ u[c...] + q + q
    end

    @testset "samples" begin
        # samples built independently at the same site are equal, so they compare and hash by content
        c = OverrideComponent(operation(override(u, LeftWall => Add(p))), ())
        x = (Coord(↓, 0), Coord(:, 0))
        a = OverrideSample(c, x, (1 => FaceMatch((-1,), (0 // 1,)),))
        b = OverrideSample(c, x, (1 => FaceMatch((-1,), (0 // 1,)),))
        @test a == b
        @test hash(a) == hash(b)
    end
end

@testset "wall rules" begin
    @scalars u
    @uniform @scalars g gL gB uL ĝ α β

    # a ghost segment: the wall is half a cell away, the mirror sample half a cell inside
    @test override(u, LeftWall => WallValue(g))[(↓, -1 // 2)] ==ₛ 2g - u[(↓, 1 // 2)]
    @test override(u, RightWall => WallValue(g))[(↑, 1 // 2)] ==ₛ 2g - u[(↑, -1 // 2)]
    # a ghost point, and the wall point
    @test override(u, LeftWall => WallValue(g))[(↓, -1)] ==ₛ 2g - u[(↓, 1)]
    @test override(u, LeftWall => WallValue(g))[(↓, 0)] ==ₛ g
    # deeper ghosts extrapolate from their mirror image
    @test override(u, LeftWall => WallValue(g))[(↓, -3 // 2)] ==ₛ 2g - u[(↓, 3 // 2)]
    # and from more points inside the domain
    @test override(u, LeftWall => WallValue(g; npoints = 2))[(↓, -1 // 2)] ==ₛ
        (8 // 3) * g - 2u[(↓, 1 // 2)] + (1 // 3) * u[(↓, 3 // 2)]
    # the boundary value is read at the wall, at both walls
    @test override(u, LeftWall => WallValue(cos(ξ¹)))[(↓, -1 // 2)] ==ₛ 2 - u[(↓, 1 // 2)]
    @test override(u, RightWall => WallValue(cos(ξ¹)))[(↑, 1 // 2)] ==ₛ 2cos(ξ¹[(↑, 0)]) - u[(↑, -1 // 2)]
    # an outward derivative gives the same expression at both walls, and is one-sided on the wall
    @test override(u, WallDerivative(g))[(↓, -1 // 2)] ==ₛ u[(↓, 1 // 2)] + g
    @test override(u, WallDerivative(g))[(↑, 1 // 2)] ==ₛ u[(↑, -1 // 2)] + g
    @test override(u, WallDerivative(g))[(↓, -1)] ==ₛ u[(↓, 1)] + 2g
    @test override(u, WallDerivative(g))[(↓, 0)] ==ₛ u[(↓, 1)] + g
    # a Robin condition `αu + β∂ₙu = g`, the same at both walls
    robin(v) = g / (α / 2 + β) - (α / 2 - β) / (α / 2 + β) * v
    @test override(u, WallData(g, α, β))[(↓, -1 // 2)] ==ₛ robin(u[(↓, 1 // 2)])
    @test override(u, WallData(g, α, β))[(↑, 1 // 2)] ==ₛ robin(u[(↑, -1 // 2)])
    # one point determines a constraint up to the first derivative
    @test_throws ArgumentError override(u, WallData(g, 0, 0, 1))
    @test_throws ArgumentError WallData(g, 1; npoints = 0)
    # inside the domain the rules throw
    @test_throws ArgumentError override(u, (LeftWall, Interior) => WallValue(g))[(↓, 1 // 2)]
    @test_throws ArgumentError override(u, (LeftWall, Interior) => WallDerivative(g))[(↓, 1 // 2)]
    # wall rules are only defined on walls
    @test_throws ArgumentError override(u, (AnyCorner, Ghost) => WallValue(g))[(↓, -1 // 2), (↓, -1 // 2)]
    # a diagonal ghost: the left rule moves the sample inside, where the bottom rule applies
    ũ = override(u, LeftWall => WallValue(gL), BottomWall => WallValue(gB))
    @test ũ[(↓, -1 // 2), (↓, -1 // 2)] ==ₛ 2gL - (2gB - u[(↓, 1 // 2), (↓, 1 // 2)])

    # the laplacian next to the wall with a zero outward derivative, at both walls
    D = StaggeredCentralDifference()
    Δ(x) = Divergence(D)(Gradient(D)(x))
    ũ = override(u, WallDerivative(0))
    @test Δ(ũ)[(↓, 1 // 2)] ==ₛ u[(↓, 3 // 2)] - u[(↓, 1 // 2)]
    @test Δ(ũ)[(↑, -1 // 2)] ==ₛ u[(↑, -3 // 2)] - u[(↑, -1 // 2)]
    # or with a zero normal flux on the wall
    q̃ = override(Gradient(D)(u), (Boundary, 𝐧) => 0)
    @test Divergence(D)(q̃)[(↓, 1 // 2)] ==ₛ u[(↓, 3 // 2)] - u[(↓, 1 // 2)]

    # the laplacian in the corner cell
    ũ = override(u, LeftWall => WallValue(uL), WallDerivative(ĝ))
    Δũ = D[1](D[1](ũ)) + D[2](D[2](ũ))
    c = ((↓, 1 // 2), (↓, 1 // 2))
    expected = u[(↓, 3 // 2), (↓, 1 // 2)] + u[(↓, 1 // 2), (↓, 3 // 2)] - 4 * u[c...] + 2uL + ĝ
    @test Δũ[c...] ==ₛ expected
    @test (@inferred Δũ[c...]) isa DTerm
end

@testset "tensors" begin
    @vectors V t
    @tensors 2 σ
    @uniform @scalars g h
    # the bottom and the top wall
    x, y = ((:, 0), (↓, 0)), ((:, 0), (↑, 0))

    @testset "vectors" begin
        # the normal component is taken along the outward normal, so it changes sign at the lower wall
        V̄ = override(V, (Boundary, 𝐧) => WallValue(g))
        @test V̄[2][x...] ==ₛ -g
        @test V̄[2][y...] ==ₛ g
        @test V̄[1][x...] ==ₛ V[1][x...]
        # pointwise rules see the outward frame as well
        @test override(V, (Boundary, 𝐧) => g)[2][x...] ==ₛ -g
        # a later predicate applies to the components that the earlier one does not match
        V̄ = override(V, (BottomWall, Boundary, 𝐧) => WallValue(g), (BottomWall, Boundary, 𝐭) => Add(h))
        @test V̄[1][x...] ==ₛ V[1][x...] + h
        # every component
        V̄ = override(V, (BottomWall, Boundary) => WallValue(g))
        @test V̄[1][x...] ==ₛ g
        @test V̄[2][x...] ==ₛ -g
    end

    @testset "rank 2" begin
        # vector data are indexed by the free indices
        σ̄ = override(σ, (Ghost, (𝐧, :)) => WallValue(t))
        @test σ̄[2, 1][(:, 1 // 2), (↑, 1 // 2)] ==ₛ 2 * t[1][(:, 1 // 2), (↑, 0)] - σ[2, 1][(:, 1 // 2), (↑, -1 // 2)]
        @test σ̄[2, 1][(:, 1 // 2), (↓, -1 // 2)] ==ₛ -2 * t[1][(:, 1 // 2), (↓, 0)] - σ[2, 1][(:, 1 // 2), (↓, 1 // 2)]
        # two indices along the normal keep the sign, while the vector data are global and change sign along the normal
        @test σ̄[2, 2][(:, 1 // 2), (↓, -1 // 2)] ==ₛ -2 * t[2][(:, 1 // 2), (↓, 0)] - σ[2, 2][(:, 1 // 2), (↓, 1 // 2)]
        @test σ̄[1, 2][(:, 1 // 2), (↓, -1 // 2)] ==ₛ σ[1, 2][(:, 1 // 2), (↓, -1 // 2)]
    end

    @testset "stokes" begin
        @scalars P
        @uniform @scalars η
        grad = Gradient(StaggeredCentralDifference())
        # no penetration on the bottom wall
        V̄ = override(V, (BottomWall, Boundary, 𝐧) => 0)
        τ = -P * ℐ(2) + 2η * sym(grad(V̄))
        # free slip on the bottom wall, free surface on the top wall
        τ̄ = override(τ, (BottomWall, Boundary, (𝐧, 𝐭)) => WallValue(0), (TopWall, (𝐧, :)) => WallValue(𝒪(1)))
        @test τ̄[2, 1][x...] ==ₛ Literal(0)
        @test τ̄[1, 1][x...] ==ₛ τ[1, 1][x...]
        # the ghost value mirrors the value inside the domain
        @test τ̄[2, 2][(:, 1 // 2), (↑, 1 // 2)] ==ₛ -τ[2, 2][(:, 1 // 2), (↑, -1 // 2)]
    end

    @testset "wall vectors" begin
        @uniform @scalars β
        @uniform @vectors W
        @vectors v
        # the normal points outwards, so the normal component of a global vector is the same at both walls
        V̄ = override(V, (Boundary, 𝐧) => WallValue(W ⋅ 𝐧))
        @test V̄[2][x...] ==ₛ W[2]
        @test V̄[2][y...] ==ₛ W[2]
        # a global vector along the normal
        V̄ = override(V, Boundary => WallValue(g * 𝐧))
        @test V̄[2][x...] ==ₛ -g
        @test V̄[2][y...] ==ₛ g
        @test V̄[1][x...] ==ₛ Literal(0)
        # partial slip, the tangential traction is proportional to the tangential velocity
        σ̄ = override(σ, (Ghost, (𝐧, 𝐭)) => WallValue(β * (v ⋅ 𝐭)))
        @test σ̄[2, 1][(:, 1 // 2), (↓, -1 // 2)] ==ₛ -2β * v[1][(:, 1 // 2), (↓, 0)] - σ[2, 1][(:, 1 // 2), (↓, 1 // 2)]
        @test σ̄[2, 1][(:, 1 // 2), (↑, 1 // 2)] ==ₛ 2β * v[1][(:, 1 // 2), (↑, 0)] - σ[2, 1][(:, 1 // 2), (↑, -1 // 2)]
        @test σ̄[1, 2][(↓, -1 // 2), (:, 1 // 2)] ==ₛ -2β * v[2][(↓, 0), (:, 1 // 2)] - σ[1, 2][(↓, 1 // 2), (:, 1 // 2)]
        # the tangential part of a vector
        vᵗ = v - (v ⋅ 𝐧) * 𝐧
        V̄ = override(V, (Boundary, :) => vᵗ)
        @test V̄[1][x...] ==ₛ v[1][x...]
        @test V̄[2][x...] ==ₛ Literal(0)
        # every tangential component has its own tangent, the bottom wall is normal to the last axis
        σ̄ = override(σ, (BottomWall, Ghost, (𝐧, 𝐭)) => WallValue(β * (v ⋅ 𝐭)))
        for c in (1, 2)
            @test σ̄[3, c][(:, 1 // 2), (:, 1 // 2), (↓, -1 // 2)] ==ₛ
                -2β * v[c][(:, 1 // 2), (:, 1 // 2), (↓, 0)] - σ[3, c][(:, 1 // 2), (:, 1 // 2), (↓, 1 // 2)]
        end
        # the normal is only defined on walls, the tangent only with tangential indices, and both only in the rule data
        @test_throws ArgumentError override(V, (AnyCorner, Boundary) => 𝐧)[1][(↓, 0), (↓, 0)]
        @test_throws ArgumentError override(V, (Boundary, 𝐧) => WallValue(v ⋅ 𝐭))[2][x...]
        @test_throws ArgumentError (g * 𝐧)[1][x...]
        # the components of the wall vectors are only known at the matched face
        @test (@inferred components(g * 𝐧, 2)[2]) ==ₛ g * 𝐧()[2]
    end

    @testset "malformed patterns" begin
        # the tensor pattern has one index per tensor index, and tensor data have the rank of its free indices
        @test_throws ArgumentError override(V, (LeftWall, ()) => WallValue(g))
        @test_throws ArgumentError override(σ, (LeftWall, 𝐧) => WallValue(g))
        @test_throws ArgumentError override(σ, (LeftWall, (𝐧, 𝐭)) => WallValue(t))
        # tensor patterns need a wall
        @test_throws ArgumentError override(V, (AnyCorner, 𝐧) => 0)
        @test_throws ArgumentError override(V, 1 => 0)
    end
end

@testset "boundary data" begin
    @scalars u v g
    @vectors V W
    @uniform @scalars a
    D = StaggeredCentralDifference()
    # boundary data are sampled at the trace of the wall, and along the wall at the sample
    @test override(u, LeftWall => WallValue(𝒷(g)))[(↓, -1 // 2), (:, 1 // 2)] ==ₛ 2g[↓, (:, 1 // 2)] - u[(↓, 1 // 2), (:, 1 // 2)]
    @test override(u, RightWall => WallValue(𝒷(g)))[(↑, 1 // 2), (:, 0)] ==ₛ 2g[↑, (:, 0)] - u[(↑, -1 // 2), (:, 0)]
    @test override(u, LeftWall => WallValue(𝒷(g)))[(↓, -1 // 2)] ==ₛ 2g[↓] - u[(↓, 1 // 2)]
    # at a corner point, the data of a wall predicate are read from the wall, and of a corner predicate at the corner
    c = ((↓, 0), (↓, 0))
    @test override(u, (LeftWall, Boundary) => 𝒷(g))[c...] ==ₛ g[↓, (↓, 0)]
    @test override(u, (AnyCorner, Boundary) => 𝒷(g))[c...] ==ₛ g[↓, ↓]
    # a single value for the whole wall
    @test override(u, (LeftWall, Boundary) => 𝒷(g; span = true))[(↓, 0), (:, 1 // 2)] ==ₛ g[↓, :]
    # boundary data in expressions, and stencils along the wall
    w = ((↓, 0), (:, 0))
    @test override(u, (LeftWall, Boundary) => 𝒷(g) * v)[w...] ==ₛ g[↓, (:, 0)] * v[w...]
    @test override(u, (LeftWall, Boundary) => D[2](𝒷(g)))[(↓, 0), (:, 1 // 2)] ==ₛ g[↓, (:, 1)] - g[↓, (:, 0)]
    # expressions of boundary data are traces as a whole, constant along the normal
    @test override(u, (LeftWall, Boundary) => 𝒷(2g))[w...] ==ₛ 2g[↓, (:, 0)]
    @test override(u, (LeftWall, Boundary) => 𝒷(D[2](g)))[(↓, 0), (:, 1 // 2)] ==ₛ g[↓, (:, 1)] - g[↓, (:, 0)]
    @test override(u, (LeftWall, Boundary) => 𝒷(D[1](g)))[(↓, 0), (:, 1 // 2)] ==ₛ Literal(0)
    # tensor data, and uniform data
    V̄ = override(V, (LeftWall, Boundary) => 𝒷(W))
    @test V̄[1][w...] ==ₛ W[1][↓, (:, 0)]
    @test V̄[2][w...] ==ₛ W[2][↓, (:, 0)]
    @test override(u, (LeftWall, Boundary) => 𝒷(a))[w...] ==ₛ a
    # boundary data are only defined in the data of rules
    @test_throws ArgumentError 𝒷(g)[(:, 0)]
end

@testset "predicates from patterns" begin
    @scalars u
    @vectors V
    @tensors 2 σ
    predicates(term) = operation(term).predicates
    @test predicates(override(u, LeftWall => 0)) == (WallPredicate(LeftWall, Ghost | Boundary, ()),)
    @test predicates(override(u, Boundary => 0)) == (WallPredicate(AnyWall, Boundary, ()),)
    @test predicates(override(u, WallDerivative(0))) == (WallPredicate(AnyWall, Ghost | Boundary, ()),)
    @test predicates(override(V, (LeftWall, Boundary) => 0)) == (WallPredicate(LeftWall, Boundary, (:,)),)
    @test predicates(override(V, 𝐧 => 0)) == (WallPredicate(AnyWall, Ghost | Boundary, (𝐧,)),)
    @test predicates(override(σ, (𝐧, 𝐭) => 0)) == (WallPredicate(AnyWall, Ghost | Boundary, (𝐧, 𝐭)),)
    @test predicates(override(σ, (TopWall, Ghost, (𝐧, :)) => 0)) == (WallPredicate(TopWall, Ghost, (𝐧, :)),)
    @test predicates(override(u, AnyCorner => 0)) == (FacePredicate(AnyCorner, Ghost | Boundary),)
    @test predicates(override(u, FacePredicate(AnyCorner, Boundary) => 0)) == (FacePredicate(AnyCorner, Boundary),)
end
