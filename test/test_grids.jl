using Test
using Chmy

@testset "locations" begin
    # on the MAC grid, a component is at points along the axes appearing an odd number of times in its indices
    @test locations(Grids.MAC, (), 3) == ((𝓈, 𝓈, 𝓈),)
    @test locations(Grids.MAC, (2,), 3) == ((𝓈, 𝓅, 𝓈),)
    @test locations(Grids.MAC, (1, 1), 2) == ((𝓈, 𝓈),)
    @test locations(Grids.MAC, (1, 2), 3) == ((𝓅, 𝓅, 𝓈),)
    # on the fully staggered grid, at all locations whose number of points has the parity of the rank
    @test locations(Grids.FullyStaggered, (), 1) == ((𝓈,),)
    @test locations(Grids.FullyStaggered, (1,), 1) == ((𝓅,),)
    @test Set(locations(Grids.FullyStaggered, (), 2)) == Set(((𝓈, 𝓈), (𝓅, 𝓅)))
    @test Set(locations(Grids.FullyStaggered, (1, 2), 2)) == Set(((𝓈, 𝓈), (𝓅, 𝓅)))
    @test Set(locations(Grids.FullyStaggered, (2,), 3)) == Set(((𝓅, 𝓈, 𝓈), (𝓈, 𝓅, 𝓈), (𝓈, 𝓈, 𝓅), (𝓅, 𝓅, 𝓅)))
    # a location stores every component there along every axis, and a tuple of locations one per axis
    @test locations(𝓅, (1, 2), 2) == ((𝓅, 𝓅),)
    @test locations((𝓅, 𝓈), (), 2) == ((𝓅, 𝓈),)
    @test_throws DimensionMismatch locations((𝓅, 𝓈), (), 3)
end

@testset "interpolation" begin
    @scalars η p u
    @vectors v ρg
    @tensors 2 τ
    @uniform @scalars α g
    avg(a, b) = (1 // 2) * (a + b)

    @testset "scalars" begin
        η̃ = @inferred interpolate(η, Grids.MAC)
        # at its locations the term is read as it is, and elsewhere it is interpolated along every axis
        @test (@inferred η̃[(:, 1 // 2), (↓, 1 // 2)]) ==ₛ η[(:, 1 // 2), (↓, 1 // 2)]
        @test η̃[(:, 0), (:, 1 // 2)] ==ₛ avg(η[(:, 1 // 2), (:, 1 // 2)], η[(:, -1 // 2), (:, 1 // 2)])
        @test η̃[(:, 0), (:, 0)] ==ₛ avg(
            avg(η[(:, 1 // 2), (:, 1 // 2)], η[(:, -1 // 2), (:, 1 // 2)]),
            avg(η[(:, 1 // 2), (:, -1 // 2)], η[(:, -1 // 2), (:, -1 // 2)])
        )
        # nodes outside the domain are dropped, see `InterpolateFrom`
        @test (@inferred η̃[(↓, 0), (:, 0)]) ==ₛ avg(η[(↓, 1 // 2), (:, 1 // 2)], η[(↓, 1 // 2), (:, -1 // 2)])
        # traces are read as they are along their axis
        @test η̃[↓, (:, 0)] ==ₛ avg(η[↓, (:, 1 // 2)], η[↓, (:, -1 // 2)])
        # collocated at a location, or at one location per axis
        @test interpolate(u, 𝓅)[(:, 1 // 2)] ==ₛ avg(u[(:, 1)], u[(:, 0)])
        @test interpolate(u, (𝓅, 𝓈))[(:, 1 // 2), (:, 1 // 2)] ==ₛ avg(u[(:, 1), (:, 1 // 2)], u[(:, 0), (:, 1 // 2)])
        @test_throws DimensionMismatch interpolate(u, (𝓅, 𝓈))[(:, 0)]
        # constants are not interpolated
        @test interpolate(1, Grids.MAC) ==ₛ Literal(1)
        @test interpolate(α, 𝓅) ==ₛ α
        @test_throws ArgumentError interpolate(η, Grids.MAC; order = 3)
    end

    @testset "tensors" begin
        # every component is interpolated from its own location
        ṽ = @inferred interpolate(v, Grids.MAC)
        @test (@inferred ṽ[1][(:, 1 // 2), (:, 1 // 2)]) ==ₛ avg(v[1][(:, 1), (:, 1 // 2)], v[1][(:, 0), (:, 1 // 2)])
        @test ṽ[2][(:, 1 // 2), (:, 0)] ==ₛ v[2][(:, 1 // 2), (:, 0)]
        @test sqrt(ṽ ⋅ ṽ)[(:, 1 // 2), (:, 1 // 2)] ==ₛ sqrt(
            avg(v[1][(:, 1), (:, 1 // 2)], v[1][(:, 0), (:, 1 // 2)])^2 + avg(v[2][(:, 1 // 2), (:, 1)], v[2][(:, 1 // 2), (:, 0)])^2
        )
        τ̃ = interpolate(τ, Grids.MAC)
        @test τ̃[1, 1][(:, 1 // 2), (:, 1 // 2)] ==ₛ τ[1, 1][(:, 1 // 2), (:, 1 // 2)]
        @test τ̃[1, 2][(:, 1 // 2), (:, 1 // 2)] ==ₛ avg(
            avg(τ[1, 2][(:, 1), (:, 1)], τ[1, 2][(:, 0), (:, 1)]),
            avg(τ[1, 2][(:, 1), (:, 0)], τ[1, 2][(:, 0), (:, 0)])
        )
        # unsampled components print as the interpolation of the component
        @test string(ṽ[1]) == "interpolate(v[1])"
    end

    @testset "Stokes" begin
        divg = Divergence(StaggeredCentralDifference())
        grad = Gradient(StaggeredCentralDifference())
        residual(η) = divg(-p * ℐ(2) + 2η * sym(grad(v))) + ρg
        r_v = residual(interpolate(η, Grids.MAC))
        # the locations at which a field is read
        readat(u, term) = [k.locations for k in bindings(term) if k isa Chmy.Located && isequal(k.term, u)]
        # in 1D the viscosity is only needed at the cell centres
        @test r_v[1][(:, 0)] ==ₛ residual(η)[1][(:, 0)]
        # the same expression reads the fields only at their locations in every dimension, also next to the walls
        for x in (((:, 0), (:, 1 // 2)), ((:, 0), (↓, 1 // 2)), ((:, 0), (:, 1 // 2), (:, 1 // 2)))
            r = r_v[1][x...]
            @test readat(η, r) == collect(locations(Grids.MAC, (), length(x)))
            @test readat(v[1], r) == collect(locations(Grids.MAC, (1,), length(x)))
        end
    end

    @testset "several locations" begin
        η̂ = interpolate(η, Grids.FullyStaggered)
        # the term is read at any of its locations
        @test η̂[(:, 1 // 2), (:, 1 // 2)] ==ₛ η[(:, 1 // 2), (:, 1 // 2)]
        @test (@inferred η̂[(:, 0), (↓, 0)]) ==ₛ η[(:, 0), (↓, 0)]
        @test η̂[↓, (:, 1 // 2)] ==ₛ η[↓, (:, 1 // 2)]
        # but not interpolated between them
        @test_throws ArgumentError η̂[(:, 0), (:, 1 // 2)]
        # in 1D, every component has a single location
        @test η̂[(:, 0)] ==ₛ avg(η[(:, 1 // 2)], η[(:, -1 // 2)])
    end

    @testset "overrides" begin
        ũ = override(u, LeftWall => WallValue(g))
        # ghosts read from an override, here the value on the wall
        @test interpolate(ũ, 𝓈; wall = ReadGhosts)[(↓, 0)] ==ₛ g
        @test interpolate(ũ, 𝓈)[(↓, 0)] ==ₛ u[(↓, 1 // 2)]
    end
end
