include("common.jl")

import Chmy.Architectures: deepmap!

for backend in TEST_BACKENDS, T in TEST_TYPES
    if !compatible(backend, T)
        continue
    end

    @testset "$(basename(@__FILE__)) (backend: $backend, type: $T)" begin
        if (backend isa CPU)
            @testset "deepmap! on CPU backend" begin
                struct CuArray end # in reality this is replaced by "using CUDA"
                calls = Symbol[]

                Chmy.Architectures.disable_task_sync!(::Any) = nothing
                Chmy.Architectures.enable_task_sync!(::Any) = nothing
                Chmy.Architectures.disable_task_sync!(::CuArray) = push!(calls, :disable) # actual unsafe_disable_task_sync! call
                Chmy.Architectures.enable_task_sync!(::CuArray) = push!(calls, :enable) # actual unsafe_enable_task_sync! call

                empty!(calls)
                foo1 = (1, 2, 3)
                deepmap!(disable_task_sync!, foo1) # prints nothing
                deepmap!(enable_task_sync!, foo1)  # prints nothing
                @test isempty(calls)

                empty!(calls)
                foo2 = ()
                deepmap!(disable_task_sync!, foo2) # prints nothing
                deepmap!(enable_task_sync!, foo2)  # prints nothing
                @test isempty(calls)

                empty!(calls)
                foo3 = (CuArray(), CuArray(), (1, ()))
                deepmap!(disable_task_sync!, foo3) # prints "disable" twice
                deepmap!(enable_task_sync!, foo3) # prints "enable" twice
                @test calls == [:disable, :disable, :enable, :enable]

                empty!(calls)
                foo4 = (a=1, b=CuArray(), c=((), 1))
                deepmap!(disable_task_sync!, foo4) # prints "disable" once
                deepmap!(enable_task_sync!, foo4) # prints "enable" once
                @test calls == [:disable, :enable]

                empty!(calls)
                struct Foo{T}
                    a::Int
                    b::CuArray
                    c::T
                end
                foo5 = Foo(1, CuArray(), foo4)

                with_no_task_sync!(foo5) do
                    push!(calls, :in_out) # kernel execution for inner and outer regions goes here
                end # will print "disable" twice, then "in_out" and "enable" twice
                @test calls == [:disable, :disable, :in_out, :enable, :enable]
            end
        end

        @testset "deepmap! on Field" begin
            myfun!(::Any) = nothing
            count = DataType[]

            arch = Arch(backend)
            grid = UniformGrid(arch; origin=(T(0.0), T(0.0)), extent=(T(1.0), T(1.0)), dims=(6, 4))

            Δt = T(1.0)
            C  = Field(backend, grid, Center())
            q  = VectorField(backend, grid)

            M = typeof(C.data)
            myfun!(x::M) where {M} = push!(count, typeof(x))
            # # myfun!(x::M) = push!(count, typeof(x))

            # @testset "deepmap! on single Field" begin
            #     empty!(count)
            #     deepmap!(myfun!, C)
            #     @test length(count) == 1
            #     @test count[1] == M
            # end

            # @testset "deepmap! on multiple args" begin
            #     empty!(count)
            #     args = (q, Δt, grid)
            #     deepmap!(myfun!, args)
            #     @test length(count) == 2
            #     @test count[1] == M
            #     @test count[2] == M
            # end

            if haskey(ENV, "JULIA_CHMY_BACKEND_CUDA") && (backend isa CUDABackend)
                @testset "CUDA task_sync" begin
                    @testset "Single Field" begin
                        @testset "disable_task_sync!" begin
                            @assert C.data.data[].task_sync == true
                            deepmap!(disable_task_sync!, C)
                            @test C.data.data[].task_sync == false
                        end

                        @testset "enable_task_sync!" begin
                            @assert C.data.data[].task_sync == false
                            deepmap!(enable_task_sync!, C)
                            @test C.data.data[].task_sync == true
                        end
                    end

                    @testset "Multiple args" begin
                        args = (C, q, Δt, grid)
                        @testset "disable_task_sync!" begin
                            @assert C.data.data[].task_sync == true
                            @assert q.x.data.data[].task_sync == true
                            @assert q.y.data.data[].task_sync == true
                            deepmap!(disable_task_sync!, args)
                            @test C.data.data[].task_sync == false
                            @test q.x.data.data[].task_sync == false
                            @test q.y.data.data[].task_sync == false
                        end

                        @testset "enable_task_sync!" begin
                            @assert C.data.data[].task_sync == false
                            @assert q.x.data.data[].task_sync == false
                            @assert q.y.data.data[].task_sync == false
                            deepmap!(enable_task_sync!, args)
                            @test C.data.data[].task_sync == true
                            @test q.x.data.data[].task_sync == true
                            @test q.y.data.data[].task_sync == true
                        end
                    end
                end
            end
        end
    end
end
