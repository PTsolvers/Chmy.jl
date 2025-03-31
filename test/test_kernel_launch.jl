include("common.jl")

import Chmy.KernelLaunch

@testset "$(basename(@__FILE__)) (backend: CPU)" begin
    @testset "modify_task_sync! function" begin
        return_type(x) = typeof(x)
        modify_task_sync!(x, return_type)

        @testset "Tuples" begin
            x = (1, [3e3])
            @test modify_task_sync!(x, return_type) == Vector{Float64}

            y = (2, x)
            @test modify_task_sync!(y, return_type) == Vector{Float64}
        end

        @testset "NamedTuples" begin
            t1 = (; x=1, y=2)
            @test isnothing(modify_task_sync!(t1, return_type))

            t2 = (; x=1, y=[2.1])
            @test modify_task_sync!(t2, return_type) == Vector{Float64}
        end

        @testset "Structs" begin
            struct Foo1
                x::T
                y::T
            end
            foo1 = Foo1(1, 2)
            @test isnothing(modify_task_sync!(foo1, return_type))

            struct Foo2
                x::Int
                y::AbstractArray{Float64}
            end
            foo2 = Foo2(1, [2.1])
            @test modify_task_sync!(foo2, return_type) == Vector{T}
        end
    end
end
