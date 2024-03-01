include("common.jl")

using Chmy.Workers

@testset "$(basename(@__FILE__)) (backend: CPU)" begin
    @testset "workers" begin
        @testset "setup" begin
            a = 0
            worker = Worker(; setup=() -> a += 1)
            put!(() -> nothing, worker)
            wait(worker)
            @test a == 1
            close(worker)
        end
        @testset "teardown" begin
            a = 0
            worker = Worker(; teardown=() -> a += 2)
            put!(worker) do
                a -= 1
            end
            wait(worker)
            close(worker)
            @test a == 1
        end
        @testset "do work" begin
            a = 0
            worker = Worker()
            put!(worker) do
                a += 1
            end
            wait(worker)
            close(worker)
            @test a == 1
        end
        @testset "not running" begin
            worker = Worker()
            close(worker)
            @test_throws ErrorException wait(worker)
        end
    end
end
