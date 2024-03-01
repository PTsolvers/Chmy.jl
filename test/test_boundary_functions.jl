include("common.jl")

using Chmy.Architectures
using Chmy.Grids
using Chmy.BoundaryConditions

@testset "$(basename(@__FILE__)) (backend: CPU)" begin
    @testset "boundary functions" begin
        arch = Arch(CPU())
        nx, ny = 8, 8
        grid = UniformGrid(arch; origin=(-π, -π), extent=(2π, 2π), dims=(nx, ny))

        @testset "continous" begin
            @testset "reduced dimensions" begin
                bf = BoundaryFunction(ξ -> cos(ξ))
                @test bf(grid, Vertex(), Dim(1), 1, 1) ≈ -1.0
                @test bf(grid, Vertex(), Dim(1), 1, ny + 1) ≈ -1.0
                @test bf(grid, Vertex(), Dim(1), 1, ny ÷ 2 + 1) ≈ 1.0

                @test bf(grid, Vertex(), Dim(2), 1, 1) ≈ -1.0
                @test bf(grid, Vertex(), Dim(2), nx + 1, 1) ≈ -1.0
                @test bf(grid, Vertex(), Dim(2), nx ÷ 2 + 1, 1) ≈ 1.0

                # changing index along other dimension shouldn't affect bc value
                @test bf(grid, Vertex(), Dim(1), ny + 1, 1) ≈ bf(grid, Vertex(), Dim(1), 1, 1)
                @test bf(grid, Vertex(), Dim(1), ny ÷ 2 + 1, 1) ≈ bf(grid, Vertex(), Dim(1), 1, 1)

                @test bf(grid, Vertex(), Dim(2), 1, 1) ≈ bf(grid, Vertex(), Dim(2), 1, ny + 1)
                @test bf(grid, Vertex(), Dim(2), 1, 1) ≈ bf(grid, Vertex(), Dim(2), 1, ny ÷ 2 + 1)
            end

            @testset "full dimensions" begin
                bf = BoundaryFunction((ξ, η) -> cos(ξ) * η; reduce_dims=false)
                @test bf(grid, Vertex(), Dim(1), 1, 1) ≈ π
                @test bf(grid, Vertex(), Dim(1), 1, ny + 1) ≈ -π
                @test bf(grid, Vertex(), Dim(1), 1, ny ÷ 2 + 1) ≈ 0.0

                @test bf(grid, Vertex(), Dim(2), 1, 1) ≈ π
                @test bf(grid, Vertex(), Dim(2), nx + 1, 1) ≈ π
                @test bf(grid, Vertex(), Dim(2), nx ÷ 2 + 1, 1) ≈ -π
            end

            @testset "with parameters" begin
                bf = BoundaryFunction((ξ, η) -> cos(ξ) * η; parameters=(η = π))
                @test bf(grid, Vertex(), Dim(1), 1, 1) ≈ -π
                @test bf(grid, Vertex(), Dim(1), 1, ny + 1) ≈ -π
                @test bf(grid, Vertex(), Dim(1), 1, ny ÷ 2 + 1) ≈ π

                @test bf(grid, Vertex(), Dim(2), 1, 1) ≈ -π
                @test bf(grid, Vertex(), Dim(2), nx + 1, 1) ≈ -π
                @test bf(grid, Vertex(), Dim(2), nx ÷ 2 + 1, 1) ≈ π
            end
        end

        @testset "discrete" begin
            @testset "reduced dimensions" begin
                bf = BoundaryFunction((grid, loc, dim, i) -> cos(coord(grid, loc, dim, i)); discrete=true)
                @test bf(grid, Vertex(), Dim(1), 1, 1) ≈ -1.0
                @test bf(grid, Vertex(), Dim(1), 1, ny + 1) ≈ -1.0
                @test bf(grid, Vertex(), Dim(1), 1, ny ÷ 2 + 1) ≈ 1.0

                @test bf(grid, Vertex(), Dim(2), 1, 1) ≈ -1.0
                @test bf(grid, Vertex(), Dim(2), nx + 1, 1) ≈ -1.0
                @test bf(grid, Vertex(), Dim(2), nx ÷ 2 + 1, 1) ≈ 1.0

                # changing index along other dimension shouldn't affect bc value
                @test bf(grid, Vertex(), Dim(1), ny + 1, 1) ≈ bf(grid, Vertex(), Dim(1), 1, 1)
                @test bf(grid, Vertex(), Dim(1), ny ÷ 2 + 1, 1) ≈ bf(grid, Vertex(), Dim(1), 1, 1)

                @test bf(grid, Vertex(), Dim(2), 1, 1) ≈ bf(grid, Vertex(), Dim(2), 1, ny + 1)
                @test bf(grid, Vertex(), Dim(2), 1, 1) ≈ bf(grid, Vertex(), Dim(2), 1, ny ÷ 2 + 1)
            end

            @testset "full dimensions" begin
                bf_fun(grid, loc, dim, ix, iy) = cos(coord(grid, loc, ix, iy)[1]) * coord(grid, loc, ix, iy)[2]
                bf = BoundaryFunction(bf_fun; discrete=true, reduce_dims=false)
                @test bf(grid, Vertex(), Dim(1), 1, 1) ≈ π
                @test bf(grid, Vertex(), Dim(1), 1, ny + 1) ≈ -π
                @test bf(grid, Vertex(), Dim(1), 1, ny ÷ 2 + 1) ≈ 0.0

                @test bf(grid, Vertex(), Dim(2), 1, 1) ≈ π
                @test bf(grid, Vertex(), Dim(2), nx + 1, 1) ≈ π
                @test bf(grid, Vertex(), Dim(2), nx ÷ 2 + 1, 1) ≈ -π
            end

            @testset "with parameters" begin
                bf = BoundaryFunction((grid, loc, dim, i, η) -> cos(coord(grid, loc, dim, i)) * η; discrete=true, parameters=(η = π))
                @test bf(grid, Vertex(), Dim(1), 1, 1) ≈ -π
                @test bf(grid, Vertex(), Dim(1), 1, ny + 1) ≈ -π
                @test bf(grid, Vertex(), Dim(1), 1, ny ÷ 2 + 1) ≈ π

                @test bf(grid, Vertex(), Dim(2), 1, 1) ≈ -π
                @test bf(grid, Vertex(), Dim(2), nx + 1, 1) ≈ -π
                @test bf(grid, Vertex(), Dim(2), nx ÷ 2 + 1, 1) ≈ π
            end
        end
    end
end
