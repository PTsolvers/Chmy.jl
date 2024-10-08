include("common.jl")

using Chmy.Grids
using Chmy.GridOperators

@views av4(A) = 0.25 .* (A[1:end-1, 1:end-1] .+ A[2:end, 1:end-1] .+ A[2:end, 2:end] .+ A[1:end-1, 2:end])
@views avx(A) = 0.5 .* (A[1:end-1, :] .+ A[2:end, :])
@views avy(A) = 0.5 .* (A[:, 1:end-1] .+ A[:, 2:end])

for backend in TEST_BACKENDS, T in TEST_TYPES
    if !compatible(backend, T)
        continue
    end

    @testset "$(basename(@__FILE__)) (backend: $backend, type: $T)" begin
        arch = Arch(backend)
        grid = UniformGrid(arch; origin=(T(0.0), T(0.0)), extent=(T(1.0), T(1.0)), dims=(2, 2))
        @testset "center" begin
            f_c = Field(arch, grid, Center())
            src = reshape(1:4, size(grid, Center())) |> collect
            set!(f_c, src)
            f_c_i = interior(f_c) |> Array
            @testset "c2v" begin
                f_v = Field(arch, grid, Vertex())
                set!(f_v, grid, (grid, loc, ix, iy, f_c) -> lerp(f_c, loc, grid, ix, iy); discrete=true, parameters=(f_c,))
                f_v_i = interior(f_v) |> Array
                @test f_v_i[2:end-1, 2:end-1] ≈ av4(f_c_i)
            end
            @testset "c2c" begin
                f_c2 = Field(arch, grid, Center())
                set!(f_c2, grid, (grid, loc, ix, iy, f_c) -> lerp(f_c, loc, grid, ix, iy); discrete=true, parameters=(f_c,))
                f_c2_i = interior(f_c2) |> Array
                @test f_c_i ≈ f_c2_i
            end
            @testset "c2cv" begin
                f_cv = Field(arch, grid, (Center(), Vertex()))
                set!(f_cv, grid, (grid, loc, ix, iy, f_c) -> lerp(f_c, loc, grid, ix, iy); discrete=true, parameters=(f_c,))
                f_cv_i = interior(f_cv) |> Array
                @test f_cv_i[:, 2:end-1] ≈ avy(f_c_i)
            end
            @testset "c2vc" begin
                f_vc = Field(arch, grid, (Vertex(), Center()))
                set!(f_vc, grid, (grid, loc, ix, iy, f_c) -> lerp(f_c, loc, grid, ix, iy); discrete=true, parameters=(f_c,))
                f_vc_i = interior(f_vc) |> Array
                @test f_vc_i[2:end-1, :] ≈ avx(f_c_i)
            end
        end
        @testset "vertex" begin
            f_v = Field(arch, grid, Vertex())
            src = reshape(1:9, size(grid, Vertex())) |> collect
            set!(f_v, src)
            f_v_i = interior(f_v) |> Array
            @testset "v2c" begin
                f_c = Field(arch, grid, Center())
                set!(f_c, grid, (grid, loc, ix, iy, f_v) -> lerp(f_v, loc, grid, ix, iy); discrete=true, parameters=(f_v,))
                f_c_i = interior(f_c) |> Array
                @test f_c_i ≈ av4(f_v_i)
            end
            @testset "v2v" begin
                f_v2 = Field(arch, grid, Vertex())
                set!(f_v2, grid, (grid, loc, ix, iy, f_v) -> lerp(f_v, loc, grid, ix, iy); discrete=true, parameters=(f_v,))
                f_v2_i = interior(f_v2) |> Array
                @test f_v2_i ≈ f_v_i
            end
            @testset "v2cv" begin
                f_cv = Field(arch, grid, (Center(), Vertex()))
                set!(f_cv, grid, (grid, loc, ix, iy, f_v) -> lerp(f_v, loc, grid, ix, iy); discrete=true, parameters=(f_v,))
                f_cv_i = interior(f_cv) |> Array
                @test f_cv_i ≈ avx(f_v_i)
            end
            @testset "v2vc" begin
                f_vc = Field(arch, grid, (Vertex(), Center()))
                set!(f_vc, grid, (grid, loc, ix, iy, f_v) -> lerp(f_v, loc, grid, ix, iy); discrete=true, parameters=(f_v,))
                f_vc_i = interior(f_vc) |> Array
                @test f_vc_i ≈ avy(f_v_i)
            end
        end
    end
end
