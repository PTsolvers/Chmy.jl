include("common.jl")

using Chmy.Architectures
using Chmy.Fields
using Chmy.Grids

for backend in backends
    @testset "$(basename(@__FILE__)) (backend: $backend)" begin
        arch = Arch(backend)
        grid = UniformGrid(arch; origin=(0.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0), dims=(2, 2, 2))
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
                set!(f, grid, (grid, loc, I) -> ycoord(grid, loc, I[2]); discrete=true)
                @test Array(interior(f)) == [0.0; 0.0;; 0.5; 0.5;; 1.0; 1.0;;;
                                             0.0; 0.0;; 0.5; 0.5;; 1.0; 1.0]
                # no parameters center
                fill!(parent(f), NaN)
                set!(f, grid, (grid, loc, I) -> xcoord(grid, loc, I[1]); discrete=true)
                @test Array(interior(f)) == [0.25; 0.75;; 0.25; 0.75;; 0.25; 0.75;;;
                                             0.25; 0.75;; 0.25; 0.75;; 0.25; 0.75]
                # with parameters
                fill!(parent(f), NaN)
                set!(f, grid, (grid, loc, I, sc) -> ycoord(grid, loc, I[2]) * sc; discrete=true, parameters=(2.0,))
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
                set!(f, grid, (x, y, z, sc) -> y * sc; parameters=(2.0,))
                @test Array(interior(f)) == [0.0; 0.0;; 1.0; 1.0;; 2.0; 2.0;;;
                                             0.0; 0.0;; 1.0; 1.0;; 2.0; 2.0]
            end
        end
    end
end
