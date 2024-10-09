include("common.jl")

using LinearAlgebra

for backend in TEST_BACKENDS, T in TEST_TYPES
    if !compatible(backend, T)
        continue
    end

    @testset "$(basename(@__FILE__)) (backend: $backend, type: $T)" begin
        # test setup
        arch = Arch(backend)
        grid = UniformGrid(arch; origin=(T(0.0), T(0.0), T(0.0)), extent=(T(1.0), T(1.0), T(1.0)), dims=(2, 2, 2))
        loc = (Center(), Vertex(), Center())
        @testset "location" begin
            @test location(Field(backend, grid, Center())) == (Center(), Center(), Center())
            @test location(Field(backend, grid, loc)) == loc
        end
        @testset "set" begin
            f = Field(backend, grid, (Center(), Vertex(), Center()); halo=(1, 0, 1))
            @testset "discrete" begin
                # no parameters vertex
                fill!(parent(f), NaN)
                set!(f, grid, (grid, loc, ix, iy, iz) -> ycoord(grid, loc, iy); discrete=true)
                @test Array(interior(f)) == [0.0; 0.0;; 0.5; 0.5;; 1.0; 1.0;;;
                                                0.0; 0.0;; 0.5; 0.5;; 1.0; 1.0]
                # no parameters center
                fill!(parent(f), NaN)
                set!(f, grid, (grid, loc, ix, iy, iz) -> xcoord(grid, loc, ix); discrete=true)
                @test Array(interior(f)) == [0.25; 0.75;; 0.25; 0.75;; 0.25; 0.75;;;
                                                0.25; 0.75;; 0.25; 0.75;; 0.25; 0.75]
                # with parameters
                fill!(parent(f), NaN)
                set!(f, grid, (grid, loc, ix, iy, iz, sc) -> ycoord(grid, loc, iy) * sc; discrete=true, parameters=(T(2.0),))
                @test Array(interior(f)) == [0.0; 0.0;; 1.0; 1.0;; 2.0; 2.0;;;
                                                0.0; 0.0;; 1.0; 1.0;; 2.0; 2.0]
            end
            @testset "continuous" begin
                # no parameters vertex
                fill!(parent(f), NaN)
                set!(f, grid, (x, y, z) -> y)
                @test Array(interior(f)) == [0.0; 0.0;; 0.5; 0.5;; 1.0; 1.0;;;
                                                0.0; 0.0;; 0.5; 0.5;; 1.0; 1.0]
                # no parameters center
                fill!(parent(f), NaN)
                set!(f, grid, (x, y, z) -> x)
                @test Array(interior(f)) == [0.25; 0.75;; 0.25; 0.75;; 0.25; 0.75;;;
                                                0.25; 0.75;; 0.25; 0.75;; 0.25; 0.75]
                # with parameters
                fill!(parent(f), NaN)
                set!(f, grid, (x, y, z, sc) -> y * sc; parameters=(T(2.0),))
                @test Array(interior(f)) == [0.0; 0.0;; 1.0; 1.0;; 2.0; 2.0;;;
                                                0.0; 0.0;; 1.0; 1.0;; 2.0; 2.0]
            end
        end
        @testset "constant field" begin
            @testset "zero" begin
                field = ZeroField{Float64}()
                @test field[1, 1, 1] ≈ 0.0
                @test field[2, 2, 2] ≈ 0.0
                @test size(field) == ()
            end
            @testset "one" begin
                field = OneField{Float64}()
                @test field[1, 1, 1] ≈ 1.0
                @test field[2, 2, 2] ≈ 1.0
                @test size(field) == ()
            end
            @testset "const" begin
                field = ValueField(2.0)
                @test field[1, 1, 1] ≈ 2.0
                @test field[2, 2, 2] ≈ 2.0
                @test size(field) == ()
            end
        end
    end
end
