using Test
using Chmy
import Chmy: makeop

@testset "show" begin
    @scalars a b c
    n = node(a + b)

    @test sprint(show, -a + b) == "-a + b"
    @test sprint(show, (-a)^b) == "(-a) ^ b"
    @test sprint(show, a + b - a / b) == "a + b - a / b"
    @test sprint(show, makeop(:-, a, makeop(:+, b, a^b))) == "a - (b + a ^ b)"
    @test sprint(show, a .+ b) == "a .+ b"
    @test sprint(show, sin.(a)) == "sin.(a)"
    @test sprint(show, n) == "(a + b)"
    @test sprint(show, n * c) == "c * (a + b)"

    colored = sprint(show, n; context=:color => true)
    @test occursin("\e[31m(\e[39m", colored)
    @test occursin("\e[31m)\e[39m", colored)

    stencil2d = sprint(show, MIME"text/plain"(), Stencil(δ(0, 0), δ(1 // 2, 0), δ(0, 1 // 2), δ(1 // 2, 1 // 2)))
    @test occursin("○────▷────○", stencil2d)
    @test occursin("│         │", stencil2d)
    @test occursin("▼    ■    ▽", stencil2d)
    @test occursin("●────▶────○", stencil2d)
    @test occursin("0         1", stencil2d)
    @test occursin("i₂", stencil2d)

    stencil1d = sprint(show, MIME"text/plain"(), Stencil(δ(1 // 2)))
    @test occursin("○────▶────○", stencil1d)

    empty_cell = sprint(show, MIME"text/plain"(), Stencil(δ(0, 0), δ(1 // 2, 0), δ(0, 1 // 2)))
    @test occursin("▼    □    ▽", empty_cell)

    stencil3d = sprint(show, MIME"text/plain"(), Stencil(δ(0, 0, 1 // 2), δ(1 // 2, 1 // 2, 3 // 2)))
    @test occursin("i₃ = 1//2", stencil3d)
    @test occursin("i₃ = 3//2", stencil3d)
end
