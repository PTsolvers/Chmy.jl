include("common.jl")

using Chmy.Architectures
using Chmy.Fields
using Chmy.Grids
using Chmy.BoundaryConditions

for backend in backends
    @testset "$(basename(@__FILE__)) (backend: $backend)" begin
        arch = Arch(backend)
        grid = UniformGrid(arch; origin=(0.0, 0.0), extent=(1.0, 1.0), dims=(8, 8))
        field = Field(arch, grid, (Center(), Vertex()))

        @testset "default Dirichlet" begin
            set!(field, 1)
            bc!(arch, grid, field => Dirichlet())
            field_i = interior(field; with_halo=true) |> Array
            @test all(field_i[1, 2:end-1] .≈ .-field_i[2, 2:end-1])
            @test all(field_i[end, 2:end-1] .≈ .-field_i[end-1, 2:end-1])

            @test all(field_i[2:end-1, 2] .≈ 0.0)
            @test all(field_i[2:end-1, end-1] .≈ 0.0)
        end

        @testset "default Neumann" begin
            set!(field, 1)
            bc!(arch, grid, field => Neumann())
            field_i = interior(field; with_halo=true) |> Array
            @test all(field_i[1, 2:end-1] .≈ field_i[2, 2:end-1])
            @test all(field_i[end, 2:end-1] .≈ field_i[end-1, 2:end-1])

            @test all(field_i[2:end-1, 1] .≈ field_i[2:end-1, 2])
            @test all(field_i[2:end-1, end] .≈ field_i[2:end-1, end-1])
        end

        @testset "non-homogenous Dirichlet" begin
            set!(field, 1)
            v = 2.0
            bc!(arch, grid, field => Dirichlet(v))
            field_i = interior(field; with_halo=true) |> Array
            @test all(field_i[1, 2:end-1] .≈ .-field_i[2, 2:end-1] .+ 2v)
            @test all(field_i[end, 2:end-1] .≈ .-field_i[end-1, 2:end-1] .+ 2v)
            
            @test all(field_i[2:end-1, 2] .≈ v)
            @test all(field_i[2:end-1, end-1] .≈ v)
        end

        @testset "non-homogenous Neumann" begin
            set!(field, 1)
            q = 2.0
            bc!(arch, grid, field => Neumann(q))
            field_i = interior(field; with_halo=true) |> Array
            @test all((field_i[2, 2:end-1] .- field_i[1, 2:end-1]) ./ Δx(grid, Vertex(), 1, 1) .≈ q)
            @test all((field_i[end, 2:end-1] .- field_i[end-1, 2:end-1]) ./ Δx(grid, Vertex(), 8, 1) .≈ q)

            @test all((field_i[2:end-1, 2] .- field_i[2:end-1, 1]) ./ Δy(grid, Center(), 1, 1) .≈ q)
            @test all((field_i[2:end-1, end] .- field_i[2:end-1, end-1]) ./ Δy(grid, Center(), 1, 8) .≈ q)
        end
    end
end
