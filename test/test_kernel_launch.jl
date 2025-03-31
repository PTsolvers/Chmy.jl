include("common.jl")

import Chmy.KernelLaunch

@testset "$(basename(@__FILE__)) (backend: CPU)" begin
    @testset "modify_task_sync! function" begin
        return_type(x) = typeof(x)
        modify_task_sync!(x, return_type)

        @testset "Tuples" begin
            x = (T(1), T[3e3])
            @test modify_task_sync!(x, return_type) == Vector{T}

            y = (T(2), x)
            @test modify_task_sync!(y, return_type) == Vector{T}
        end

        @testset "NamedTuples" begin
            t1 = (; x=1, y=2)
            @test isnothing(modify_task_sync!(t1, return_type))

            t2 = (; x=1, y=T[2])
            @test modify_task_sync!(t2, return_type) == Vector{T}
        end

        @testset "Structs" begin
            struct Foo1
                x::T
                y::T
            end
            foo1 = Foo1(T(1), T(2))
            @test isnothing(modify_task_sync!(foo1, return_type))

            struct Foo2
                x::T
                y::AbstractArray{T}
            end
            foo2 = Foo2(1, [2])
            @test modify_task_sync!(foo2, return_type) == Vector{T}
        end
    end
end
