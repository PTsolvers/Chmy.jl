include("common.jl")

for backend in TEST_BACKENDS, T in TEST_TYPES
    if !compatible(backend, T)
        continue
    end

    @testset "$(basename(@__FILE__)) (backend: $backend, type: $T)" begin
        buff = Any[T]
        myfun!(::T) where {T} = buff[1] = T

        if (backend isa CPU)
            @testset "modify_sync! on CPU backend" begin
                @testset "Tuples" begin
                    buff[1] = nothing
                    x = (T(1), T[3e3])
                    modify_sync!(x, myfun!)
                    @test buff[1] == Vector{T}

                    buff[1] = nothing
                    y = (T(2), x)
                    modify_sync!(y, myfun!)
                    @test buff[1] == Vector{T}
                end

                @testset "NamedTuples" begin
                    buff[1] = nothing
                    t1 = (; x=T(1), y=T(2))
                    modify_sync!(t1, myfun!)
                    @test isnothing(buff[1])

                    buff[1] = nothing
                    t2 = (; x=1, y=T[2.1])
                    modify_sync!(t2, myfun!)
                    @test buff[1] == Vector{T}
                end

                @testset "Structs" begin
                    buff[1] = nothing
                    struct Foo{A,B}
                        x::A
                        y::B
                    end
                    foo1 = Foo(T(1), T(2))
                    modify_sync!(foo1, myfun!)
                    @test isnothing(buff[1])

                    buff[1] = nothing
                    foo2 = Foo(T(1), T[2.1])
                    modify_sync!(foo2, myfun!)
                    @test buff[1] == Vector{T}
                end
            end
        end

        arch = Arch(backend)
        grid = UniformGrid(arch; origin=(T(0.0), T(0.0)), extent=(T(1.0), T(1.0)), dims=(6, 4))

        Δt = T(1.0)
        C  = Field(backend, grid, Center())
        q  = VectorField(backend, grid)

        @testset "modify_sync! recurse into Field" begin
            buff[1] = nothing
            modify_sync!(C, myfun!)
            @test buff[1] == typeof(C.data)

            args = (q, Δt, grid)
            modify_sync!(args, myfun!)
            @test buff[1] == typeof(q.y.data)
        end

        if haskey(ENV, "JULIA_CHMY_BACKEND_CUDA") && (backend isa CUDABackend)
            @testset "CUDA task_sync" begin
                @testset "Single Field" begin
                    @testset "disable_task_sync!" begin
                        @assert C.data.task_sync == true
                        modify_sync!(C, disable_task_sync!)
                        @test C.data.task_sync == false
                    end

                    @testset "enable_task_sync!" begin
                        @assert C.data.task_sync == false
                        modify_sync!(C, enable_task_sync!)
                        @test C.data.task_sync == true
                    end
                end

                @testset "Multiple args" begin
                    args = (C, q, Δt, grid)
                    @testset "disable_task_sync!" begin
                        @assert C.data.task_sync == true
                        @assert q.x.data.task_sync == true
                        @assert q.y.data.task_sync == true
                        modify_sync!(args, disable_task_sync!)
                        @test C.data.task_sync == false
                        @test q.x.data.task_sync == false
                        @test q.y.data.task_sync == false
                    end

                    @testset "enable_task_sync!" begin
                        @assert C.data.task_sync == false
                        @assert q.x.data.task_sync == false
                        @assert q.y.data.task_sync == false
                        modify_sync!(args, enable_task_sync!)
                        @test C.data.task_sync == true
                        @test q.x.data.task_sync == true
                        @test q.y.data.task_sync == true
                    end
                end
            end
        end
    end
end
