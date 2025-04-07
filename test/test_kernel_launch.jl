include("common.jl")

import Chmy.Architectures: disable_task_sync!, enable_task_sync!
import Chmy.KernelLaunch: deepmap!, with_no_task_sync!

myfun!(::Any) = nothing

const buff = DataType[]

for backend in TEST_BACKENDS, T in TEST_TYPES
    if !compatible(backend, T)
        continue
    end

    @testset "$(basename(@__FILE__)) (backend: $backend, type: $T)" begin
        if (backend isa CPU)
            @testset "deepmap! on CPU backend" begin
                calls = Symbol[]

                struct CuArray end # Dummy struct to simulate CUDA array

                Chmy.Architectures.disable_task_sync!(::Any) = nothing
                Chmy.Architectures.enable_task_sync!(::Any) = nothing
                Chmy.Architectures.disable_task_sync!(::CuArray) = push!(calls, :disable)
                Chmy.Architectures.enable_task_sync!(::CuArray) = push!(calls, :enable)

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
            arch = Arch(backend)
            grid = UniformGrid(arch; origin=(T(0.0), T(0.0)), extent=(T(1.0), T(1.0)), dims=(6, 4))

            Δt = T(1.0)
            C  = Field(backend, grid, Center())
            q  = VectorField(backend, grid)

            M = typeof(C.data)

            @eval myfun!(x::$M) = push!(buff, typeof(x))

            @testset "deepmap! on single Field" begin
                empty!(buff)
                deepmap!(myfun!, C)
                @test length(buff) == 1
                @test buff[1] == M
            end
            @testset "deepmap! on multiple args" begin
                empty!(buff)
                args = (q, Δt, grid)
                deepmap!(myfun!, args)
                @test length(buff) == 2
                @test buff[1] == M
                @test buff[2] == M
            end

            if haskey(ENV, "JULIA_CHMY_BACKEND_CUDA") && (backend isa CUDABackend)
                @testset "CUDA implicit synchronization" begin
                    @testset "Single Field" begin
                        @testset "disable_task_sync!" begin
                            @assert C.data.data[].synchronizing == true
                            deepmap!(disable_task_sync!, C)
                            @test C.data.data[].synchronizing == false
                        end

                        @testset "enable_task_sync!" begin
                            @assert C.data.data[].synchronizing == false
                            deepmap!(enable_task_sync!, C)
                            @test C.data.data[].synchronizing == true
                        end
                    end

                    @testset "Multiple args" begin
                        args = (C, q, Δt, grid)
                        @testset "disable_task_sync!" begin
                            @assert C.data.data[].synchronizing == true
                            @assert q.x.data.data[].synchronizing == true
                            @assert q.y.data.data[].synchronizing == true
                            deepmap!(disable_task_sync!, args)
                            @test C.data.data[].synchronizing == false
                            @test q.x.data.data[].synchronizing == false
                            @test q.y.data.data[].synchronizing == false
                        end

                        @testset "enable_task_sync!" begin
                            @assert C.data.data[].synchronizing == false
                            @assert q.x.data.data[].synchronizing == false
                            @assert q.y.data.data[].synchronizing == false
                            deepmap!(enable_task_sync!, args)
                            @test C.data.data[].synchronizing == true
                            @test q.x.data.data[].synchronizing == true
                            @test q.y.data.data[].synchronizing == true
                        end
                    end
                end
            end
        end
    end
end
