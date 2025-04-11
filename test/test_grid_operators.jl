include("common.jl")

for backend in TEST_BACKENDS, T in TEST_TYPES
    if !compatible(backend, T)
        continue
    end

    @testset "$(basename(@__FILE__)) (backend: $backend, type: $T)" begin
        # test setup
        arch = Arch(backend)
        grid = UniformGrid(arch; origin=(T(-5), T(-5), T(-5)), extent=(T(10.0), T(10.0), T(10.0)), dims=(12, 10, 8))
        launch = Launcher(arch, grid)

        @testset "divg" begin
            Ci = Field(backend, grid, Center())
            C1 = Field(backend, grid, Center())
            C2 = Field(backend, grid, Center())
            V  = VectorField(backend, grid)

            set!(Ci, grid, (x, y, z) -> exp(-x^2 - y^2 - z^2))

            @kernel function divg1!(V, Ci, g::StructuredGrid, O)
                I = @index(Global, NTuple)
                I = I + O
                V.x[I...] = ∂x(Ci, g, I...)
                V.y[I...] = ∂y(Ci, g, I...)
                V.z[I...] = ∂z(Ci, g, I...)
            end

            @kernel function divg2!(C1, C2, V, g::StructuredGrid, O)
                I = @index(Global, NTuple)
                I = I + O
                C1[I...] = ∂x(V.x, g, I...) + ∂y(V.y, g, I...) + ∂z(V.z, g, I...)
                C2[I...] = divg(V, g, I...)
            end

            launch(arch, grid, divg1! => (V, Ci, grid))
            launch(arch, grid, divg2! => (C1, C2, V, grid))

            KernelAbstractions.synchronize(backend)
            @test interior(C2) == interior(C1)
        end

        @testset "lapl" begin
            Ci = Field(backend, grid, Center())
            C1 = Field(backend, grid, Center())
            C2 = Field(backend, grid, Center())

            set!(Ci, grid, (x, y, z) -> exp(-x^2 - y^2 - z^2))

            @kernel function lapl!(C1, C2, Ci, g::StructuredGrid, O)
                I = @index(Global, NTuple)
                I = I + O
                C1[I...] = ∂²x(Ci, g, I...) + ∂²y(Ci, g, I...) + ∂²z(Ci, g, I...)
                C2[I...] = lapl(Ci, g, I...)
            end

            launch(arch, grid, lapl! => (C1, C2, Ci, grid))

            KernelAbstractions.synchronize(backend)
            @test interior(C2) == interior(C1)
        end

        @testset "divg_grad" begin
            Ci = Field(backend, grid, Center())
            C1 = Field(backend, grid, Center())
            C2 = Field(backend, grid, Center())
            χc = Field(backend, grid, Center())
            χv = Field(backend, grid, Vertex())
            V  = VectorField(backend, grid)

            set!(Ci, grid, (x, y, z) -> exp(-x^2 - y^2 - z^2))

            @kernel function divg_grad1!(V, Ci, χ, g::StructuredGrid, O)
                I = @index(Global, NTuple)
                I = I + O
                V.x[I...] = lerp(χ, location(V.x), g, I...) * ∂x(Ci, g, I...)
                V.y[I...] = lerp(χ, location(V.y), g, I...) * ∂y(Ci, g, I...)
                V.z[I...] = lerp(χ, location(V.z), g, I...) * ∂z(Ci, g, I...)
            end

            @kernel function divg_grad2!(C1, C2, V, χ, g::StructuredGrid, O)
                I = @index(Global, NTuple)
                I = I + O
                C1[I...] = ∂x(V.x, g, I...) + ∂y(V.y, g, I...) + ∂z(V.z, g, I...)
                C2[I...] = divg_grad(Ci, χ, g, I...)
            end

            @testset "χ Center()" begin
                set!(χc, grid, (x, y, z) -> exp(-x^2 - y^2 - z^2))

                launch(arch, grid, divg_grad1! => (V, Ci, χc, grid))
                launch(arch, grid, divg_grad2! => (C1, C2, V, χc, grid))

                KernelAbstractions.synchronize(backend)
                @test all(Array(interior(C2)) .≈ Array(interior(C1)))
            end

            @testset "χ Vertex()" begin
                set!(χv, grid, (x, y, z) -> exp(-x^2 - y^2 - z^2))

                launch(arch, grid, divg_grad1! => (V, Ci, χv, grid))
                launch(arch, grid, divg_grad2! => (C1, C2, V, χv, grid))

                KernelAbstractions.synchronize(backend)
                @test all(Array(interior(C2)) .≈ Array(interior(C1)))
            end
        end

        @testset "vmag" begin
            V  = VectorField(backend, grid)
            C1 = Field(backend, grid, Center())

            for comp in eachindex(Tuple(V))
                set!(Tuple(V)[comp], T(2.0))
            end

            @kernel function lapl!(C1, V, g::StructuredGrid, O)
                I = @index(Global, NTuple)
                I = I + O
                C1[I...] = vmag(V, g, I...)
            end

            launch(arch, grid, lapl! => (C1, V, grid))

            KernelAbstractions.synchronize(backend)
            # @show interior(C1)
            @test all(round.(Array(interior(C1)), sigdigits=5) .≈ 3.4641)
        end
    end
end
