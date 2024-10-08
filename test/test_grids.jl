include("common.jl")

using Chmy.Grids

for T in TEST_TYPES
    @testset "$(basename(@__FILE__)) (backend: CPU, type: $T)" begin
        @testset "common" begin
            @test flip(Center()) == Vertex()
            @test flip(Vertex()) == Center()
        end

        @testset "grids" begin
            arch = Arch(CPU())
            nx, ny = 5, 20
            @testset "uniform grids" begin
                grid = UniformGrid(arch; origin=(T(-1), T(-2)), extent=(T(2), T(4)), dims=(nx, ny))
                @test grid isa UniformGrid

                @testset "type" begin
                    @test eltype(grid) == T
                end

                # connectivity
                @test connectivity(grid, Dim(1), Side(1)) isa Bounded
                @test connectivity(grid, Dim(1), Side(2)) isa Bounded
                @test connectivity(grid, Dim(2), Side(1)) isa Bounded
                @test connectivity(grid, Dim(2), Side(2)) isa Bounded

                # axes
                @test axis(grid, Dim(1)) == grid.axes[1]
                @test axis(grid, Dim(2)) == grid.axes[2]

                @testset "size" begin
                    @test size(grid, Center()) == (nx, ny)
                    @test size(grid, Vertex()) == (nx + 1, ny + 1)
                    @test size(grid, (Center(), Vertex())) == (nx, ny + 1)
                    @test size(grid, (Vertex(), Center())) == (nx + 1, ny)
                    # repeating locations
                    @test size(grid, Center()) == size(grid, (Center(), Center()))
                    @test size(grid, Vertex()) == size(grid, (Vertex(), Vertex()))
                end

                @testset "bounds" begin
                    @test all(bounds(grid, Vertex(), Dim(1)) .≈ (-1.0, 1.0))
                    @test all(bounds(grid, Vertex(), Dim(2)) .≈ (-2.0, 2.0))
                    @test all(bounds(grid, Center(), Dim(1)) .≈ (-0.8, 0.8))
                    @test all(bounds(grid, Center(), Dim(2)) .≈ (-1.9, 1.9))
                end

                @testset "extent" begin
                    # one location
                    @test extent(grid, Vertex(), Dim(1)) ≈ 2.0
                    @test extent(grid, Vertex(), Dim(2)) ≈ 4.0
                    @test extent(grid, Center(), Dim(1)) ≈ 1.6
                    @test extent(grid, Center(), Dim(2)) ≈ 3.8
                    # many locations
                    @test all(extent(grid, (Vertex(), Vertex())) .≈ (2.0, 4.0))
                    @test all(extent(grid, (Center(), Center())) .≈ (1.6, 3.8))
                    @test all(extent(grid, (Center(), Vertex())) .≈ (1.6, 4.0))
                    @test all(extent(grid, (Vertex(), Center())) .≈ (2.0, 3.8))
                    # repeating locations
                    @test extent(grid, Vertex()) == extent(grid, (Vertex(), Vertex()))
                    @test extent(grid, Center()) == extent(grid, (Center(), Center()))
                end

                @testset "origin" begin
                    # one location
                    @test origin(grid, Vertex(), Dim(1)) ≈ -1
                    @test origin(grid, Vertex(), Dim(2)) ≈ -2
                    @test origin(grid, Center(), Dim(1)) ≈ -0.8
                    @test origin(grid, Center(), Dim(2)) ≈ -1.9
                    # many locations
                    @test all(origin(grid, (Vertex(), Vertex())) .≈ (-1.0, -2.0))
                    @test all(origin(grid, (Center(), Center())) .≈ (-0.8, -1.9))
                    @test all(origin(grid, (Center(), Vertex())) .≈ (-0.8, -2.0))
                    @test all(origin(grid, (Vertex(), Center())) .≈ (-1.0, -1.9))
                    # repeating locations
                    @test origin(grid, (Vertex(), Vertex())) == origin(grid, Vertex())
                    @test origin(grid, (Center(), Center())) == origin(grid, Center())
                end

                @testset "spacing" begin
                    @test Δ == spacing
                    # one location
                    @test spacing(grid, Vertex(), Dim(1), 1) ≈ 0.4
                    @test spacing(grid, Vertex(), Dim(2), 1) ≈ 0.2
                    @test spacing(grid, Center(), Dim(1), 1) ≈ 0.4
                    @test spacing(grid, Center(), Dim(2), 1) ≈ 0.2
                    # many locations
                    @test all(spacing(grid, (Center(), Center()), 1, 1) .≈ (0.4, 0.2))
                    @test all(spacing(grid, (Vertex(), Vertex()), 1, 1) .≈ (0.4, 0.2))
                    @test all(spacing(grid, (Center(), Vertex()), 1, 1) .≈ (0.4, 0.2))
                    @test all(spacing(grid, (Vertex(), Center()), 1, 1) .≈ (0.4, 0.2))
                    # repeating locations
                    @test spacing(grid, Vertex(), 1, 1) == spacing(grid, (Vertex(), Vertex()), 1, 1)
                    @test spacing(grid, Center(), 1, 1) == spacing(grid, (Center(), Center()), 1, 1)
                end

                @testset "inverse spacing" begin
                    @test iΔ == inv_spacing
                    # one location
                    @test inv_spacing(grid, Vertex(), Dim(1), 1) ≈ 2.5
                    @test inv_spacing(grid, Vertex(), Dim(2), 1) ≈ 5.0
                    @test inv_spacing(grid, Center(), Dim(1), 1) ≈ 2.5
                    @test inv_spacing(grid, Center(), Dim(2), 1) ≈ 5.0
                    # many locations
                    @test all(inv_spacing(grid, (Center(), Center()), 1, 1) .≈ (2.5, 5.0))
                    @test all(inv_spacing(grid, (Vertex(), Vertex()), 1, 1) .≈ (2.5, 5.0))
                    @test all(inv_spacing(grid, (Center(), Vertex()), 1, 1) .≈ (2.5, 5.0))
                    @test all(inv_spacing(grid, (Vertex(), Center()), 1, 1) .≈ (2.5, 5.0))
                    # repeating locations
                    @test inv_spacing(grid, Vertex(), 1, 1) == inv_spacing(grid, (Vertex(), Vertex()), 1, 1)
                    @test inv_spacing(grid, Center(), 1, 1) == inv_spacing(grid, (Center(), Center()), 1, 1)
                end

                @testset "uniform spacing" begin
                    # spacing
                    @test all(spacing(grid) .≈ (0.4, 0.2))
                    # inverse
                    @test all(inv_spacing(grid) .≈ (2.5, 5.0))
                    # cartesian
                    @test Δx(grid) ≈ 0.4
                    @test Δy(grid) ≈ 0.2
                end

                @testset "coords" begin
                    # one index
                    @test coord(grid, Vertex(), Dim(1), 1) ≈ -1.0
                    @test coord(grid, Vertex(), Dim(2), 1) ≈ -2.0
                    @test coord(grid, Vertex(), Dim(1), nx + 1) ≈ 1.0
                    @test coord(grid, Vertex(), Dim(2), ny + 1) ≈ 2.0
                    @test coord(grid, Center(), Dim(1), 1) ≈ -0.8
                    @test coord(grid, Center(), Dim(2), 1) ≈ -1.9
                    @test coord(grid, Center(), Dim(1), nx) ≈ 0.8
                    @test coord(grid, Center(), Dim(2), ny) ≈ 1.9
                    # many indices
                    for loc in (Center(), Vertex())
                        @test coord(grid, loc, Dim(1), 1, 1) == coord(grid, loc, Dim(1), 1)
                        @test coord(grid, loc, Dim(2), 1, 1) == coord(grid, loc, Dim(2), 1)
                        @test coord(grid, loc, Dim(1), nx + 1, 1) == coord(grid, loc, Dim(1), nx + 1)
                        @test coord(grid, loc, Dim(2), 1, ny + 1) == coord(grid, loc, Dim(2), ny + 1)
                    end
                    # many locations
                    @test all(coord(grid, (Vertex(), Center()), 1, 1) .≈ (-1.0, -1.9))
                    @test all(coord(grid, (Vertex(), Center()), nx + 1, 1) .≈ (1.0, -1.9))
                    @test all(coord(grid, (Vertex(), Center()), 1, ny) .≈ (-1.0, 1.9))
                    # repeated locations
                    @test coord(grid, (Vertex(), Center()), Dim(1), 1) == coord(grid, Vertex(), Dim(1), 1) == coord(grid, Vertex(), Dim(1), 1, 1)
                    @test coord(grid, (Vertex(), Center()), Dim(2), 1) == coord(grid, Center(), Dim(2), 1) == coord(grid, Center(), Dim(2), 1, 1)
                end

                @testset "shortcut coords" begin
                    # short coords
                    @test vertex(grid, Dim(1), 1) == vertex(grid, Dim(1), 1, 1) == coord(grid, Vertex(), Dim(1), 1)
                    @test vertex(grid, Dim(2), 1) == vertex(grid, Dim(2), 1, 1) == coord(grid, Vertex(), Dim(2), 1)
                    @test center(grid, Dim(1), 1) == center(grid, Dim(1), 1, 1) == coord(grid, Center(), Dim(1), 1)
                    @test center(grid, Dim(2), 1) == center(grid, Dim(2), 1, 1) == coord(grid, Center(), Dim(2), 1)
                end

                @testset "cartesian" begin
                    # coords
                    @test xvertex(grid, 1) == vertex(grid, Dim(1), 1)
                    @test xcenter(grid, 1) == center(grid, Dim(1), 1)
                    @test yvertex(grid, 1) == vertex(grid, Dim(2), 1)
                    @test ycenter(grid, 1) == center(grid, Dim(2), 1)
                    # spacing
                    for loc in (Center(), Vertex())
                        @test Δx(grid, loc, 1) == spacing(grid, loc, Dim(1), 1)
                        @test Δx(grid, loc, 1) == spacing(grid, loc, Dim(1), 1)
                        @test Δy(grid, loc, 1) == spacing(grid, loc, Dim(2), 1)
                        @test Δy(grid, loc, 1) == spacing(grid, loc, Dim(2), 1)
                    end
                end
            end
        end
    end
end
