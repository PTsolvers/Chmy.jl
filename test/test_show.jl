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

    stencil2d = sprint(show, MIME"text/plain"(), Stencil((Segment(), Point()), δ(0, 0), δ(1, 0)))
    @test occursin("location (Segment(), Point())", stencil2d)
    @test occursin("○────▶────○────▶────○", stencil2d)
    @test occursin("0         1", stencil2d)
    @test occursin("i₂", stencil2d)

    stencil1d = sprint(show, MIME"text/plain"(), Stencil(Segment(), δ(0)))
    @test occursin("○────▶────○", stencil1d)

    cell = sprint(show, MIME"text/plain"(), Stencil((Segment(), Segment()), δ(0, 0)))
    @test occursin("▽    ■    ▽", cell)

    stencil3d = sprint(show, MIME"text/plain"(), Stencil((Point(), Point(), Segment()), δ(0, 0, 0), δ(0, 0, 1)))
    @test occursin("i₃ = 1//2", stencil3d)
    @test occursin("i₃ = 3//2", stencil3d)

    colored_stencil = sprint(show,
                             MIME"text/plain"(),
                             Stencil((Segment(), Point()), δ(0, 0), δ(1, 0));
                             context=:color => true)
    @test occursin("\e[34m\e[1m▶", colored_stencil)

    colored_cell = sprint(show,
                          MIME"text/plain"(),
                          Stencil((Segment(), Segment()), δ(0, 0));
                          context=:color => true)
    @test occursin("\e[34m\e[1m■", colored_cell)

    i, j = SIndex(1), SIndex(2)
    colored_nu = sprint(show,
                        MIME"text/plain"(),
                        nonuniforms(a[Segment(), Point()][i, j]);
                        context=:color => true)
    @test occursin("\e[1ma", colored_nu)
    @test occursin("i₁", colored_nu)
    @test occursin("i₂", colored_nu)
end
