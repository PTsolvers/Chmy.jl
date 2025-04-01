include("common.jl")

for backend in TEST_BACKENDS, T in TEST_TYPES
    if !compatible(backend, T)
        continue
    end

    @testset "$(basename(@__FILE__)) (backend: $backend, type: $T)" begin

        return_type(x) = typeof(x)

        if (backend isa CPU)
            @testset "modify_sync! on CPU backend" begin

                @testset "Tuples" begin
                    x = (1, [3e3])
                    @test modify_sync!(x, return_type) == Vector{Float64}

                    y = (2, x)
                    @test modify_sync!(y, return_type) == Vector{Float64}
                end

                @testset "NamedTuples" begin
                    t1 = (; x=1, y=2)
                    @test isnothing(modify_sync!(t1, return_type))

                    t2 = (; x=1, y=[2.1])
                    @test modify_sync!(t2, return_type) == Vector{Float64}
                end

                @testset "Structs" begin
                    struct Foo{A,B}
                        x::A
                        y::B
                    end
                    foo1 = Foo(1, 2)
                    @test isnothing(modify_sync!(foo1, return_type))

                    foo2 = Foo(1, [2.1])
                    @test modify_sync!(foo2, return_type) == Vector{Float64}
                end
            end
        end

        @kernel function update_C!(C, q, Δt, g::StructuredGrid, O)
            I = @index(Global, NTuple)
            I = I + O
            C[I...] += Δt * divg(q, g, I...)
        end

        arch = Arch(backend)
        grid = UniformGrid(arch; origin=(T(0.0), T(0.0)), extent=(T(1.0), T(1.0)), dims=(6, 4))

        Δt   = T(1.0)
        C    = Field(backend, grid, Center())
        q    = VectorField(backend, grid)

        set!(q.x, grid, (x, y) -> x)
        launch = Launcher(arch, grid; outer_width=(2, 2))

        # @testset "Recurse into Field" begin
        #     args = (C, q, Δt, grid)
        #     modify_sync!(args, return_type)
        # end
        # @testset "Kernel" begin
        #     launch(arch, grid, update_C! => (C, q, Δt, grid); bc=batch(grid, C => Neumann()))
        # end
    end
end
