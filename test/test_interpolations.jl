include("common.jl")

using Chmy.Grids
using Chmy.GridOperators
using Chmy.Fields
using Chmy.Architectures

@testset "$(basename(@__FILE__)) (backend: CPU)" begin
    arch = Arch(CPU())
    grid = UniformGrid(arch; origin=(0, 0), extent=(1, 1), dims=(2, 2))
    r    = Linear()
    @testset "center" begin
        field = Field(arch, grid, Center())
        src = reshape(1:4, size(grid, Center())) |> collect
        set!(field, src)
        @testset "c2v" begin
            @test itp(field, (Vertex(), Vertex()), r, grid, 2, 2) ≈ 2.5
        end
        @testset "c2c" for ix in 1:2, iy in 1:2
            @test itp(field, (Center(), Center()), r, grid, ix, iy) == src[ix, iy]
        end
        @testset "c2cv" begin
            @test itp(field, (Center(), Vertex()), r, grid, 1, 2) ≈ 2.0
            @test itp(field, (Center(), Vertex()), r, grid, 2, 2) ≈ 3.0
        end
        @testset "c2vc" begin
            @test itp(field, (Vertex(), Center()), r, grid, 2, 1) ≈ 1.5
            @test itp(field, (Vertex(), Center()), r, grid, 2, 2) ≈ 3.5
        end
    end
    @testset "vertex" begin
        field = Field(arch, grid, Vertex())
        src = reshape(1:9, size(grid, Vertex())) |> collect
        set!(field, src)
        @testset "v2c" begin
            @test itp(field, (Center(), Center()), r, grid, 1, 1) ≈ 3.0
        end
        @testset "v2v" for ix in 1:3, iy in 1:3
            @test itp(field, (Vertex(), Vertex()), r, grid, ix, iy) == src[ix, iy]
        end
        @testset "v2cv" begin
            @test itp(field, (Center(), Vertex()), r, grid, 1, 1) ≈ 1.5
            @test itp(field, (Center(), Vertex()), r, grid, 2, 1) ≈ 2.5
            @test itp(field, (Center(), Vertex()), r, grid, 1, 2) ≈ 4.5
            @test itp(field, (Center(), Vertex()), r, grid, 2, 2) ≈ 5.5
            @test itp(field, (Center(), Vertex()), r, grid, 1, 3) ≈ 7.5
            @test itp(field, (Center(), Vertex()), r, grid, 2, 3) ≈ 8.5
        end
        @testset "v2vc" begin
            @test itp(field, (Vertex(), Center()), r, grid, 1, 1) ≈ 2.5
            @test itp(field, (Vertex(), Center()), r, grid, 1, 2) ≈ 5.5
            @test itp(field, (Vertex(), Center()), r, grid, 2, 1) ≈ 3.5
            @test itp(field, (Vertex(), Center()), r, grid, 2, 2) ≈ 6.5
            @test itp(field, (Vertex(), Center()), r, grid, 3, 1) ≈ 4.5
            @test itp(field, (Vertex(), Center()), r, grid, 3, 2) ≈ 7.5
        end
    end
end
