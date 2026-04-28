using Test
using Chmy
import Chmy: makeop, render_stencil

struct UnknownShowTerm <: Chmy.STerm
    x::Int
end

struct StyledShowTerm <: Chmy.STerm
    x::Int
end
Chmy.styled(t::StyledShowTerm) = Base.annotatedstring("styled(", string(t.x), ")")

struct CustomBaseShowTerm <: Chmy.STerm
    x::Int
end
Base.show(io::IO, t::CustomBaseShowTerm) = print(io, "base(", t.x, ")")

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
    @test occursin(r"\e\[31m(?:\e\[[0-9;]*m)*\)\e\[39m", colored)

    @test endswith(sprint(show, UnknownShowTerm(1)), "UnknownShowTerm(1)")
    @test sprint(show, StyledShowTerm(2)) == "styled(2)"
    @test sprint(show, CustomBaseShowTerm(3)) == "base(3)"

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

    direct_colored = sprint(print,
                            render_stencil(Stencil((Segment(), Point()), δ(0, 0), δ(1, 0)));
                            context=:color => true)
    @test occursin("\e[34m\e[1m▶", direct_colored)

    default_stencil_opts = Chmy.StencilRenderOptions()
    rendered = render_stencil(Stencil((Segment(), Point()), δ(0, 0), δ(1, 0)))
    @test rendered isa Base.AnnotatedString
    @test Chmy.stencil_render_width(Stencil((Segment(), Point()), δ(0, 0), δ(1, 0)), default_stencil_opts) == maximum(textwidth, split(string(rendered), '\n'))

    rendered3d = render_stencil(Stencil((Point(), Point(), Segment()), δ(0, 0, 0), δ(0, 0, 1)))
    @test Chmy.stencil_render_width(Stencil((Point(), Point(), Segment()), δ(0, 0, 0), δ(0, 0, 1)), default_stencil_opts) ==
          maximum(textwidth, split(string(rendered3d), '\n'))

    no_xticks = sprint(print, render_stencil(Stencil((Segment(), Point()), δ(0, 0)); x_ticks=false))
    @test !occursin("0         1", no_xticks)
    no_yticks = sprint(print, render_stencil(Stencil((Segment(), Point()), δ(0, 0)); y_ticks=false))
    @test !occursin(" 0 ", no_yticks)
    no_xlabel = sprint(print, render_stencil(Stencil((Segment(), Point()), δ(0, 0)); x_axis_label=false))
    @test !occursin("i₁", no_xlabel)
    no_ylabel = sprint(print, render_stencil(Stencil((Segment(), Point()), δ(0, 0)); y_axis_label=false))
    @test !occursin("i₂", no_ylabel)

    boxed = render_stencil(Stencil((Segment(), Point()), δ(0, 0)); left_margin=2, right_margin=3, textwidth=40)
    @test all(==(40), textwidth.(split(string(boxed), '\n')))
    boxed_opts = Chmy.StencilRenderOptions(; left_margin=2, right_margin=3, textwidth=40)
    @test Chmy.stencil_render_width(Stencil((Segment(), Point()), δ(0, 0)), boxed_opts) == 40

    upper_vertical = split(string(render_stencil(Stencil(δ(0, 0), δ(0, 1)))), '\n')
    lower_vertical = split(string(render_stencil(Stencil(δ(0, 0), δ(0, -1)))), '\n')
    @test any(startswith("1 "), upper_vertical)
    @test any(startswith("-1 "), lower_vertical)

    i, j = SIndex(1), SIndex(2)
    colored_nu = sprint(show,
                        MIME"text/plain"(),
                        nonuniforms(a[Segment(), Point()][i, j]);
                        context=:color => true)
    @test occursin("\e[1ma", colored_nu)
    @test occursin("i₁", colored_nu)
    @test occursin("i₂", colored_nu)

    total_nu = sprint(show,
                      MIME"text/plain"(),
                      Nonuniforms(a[Segment(), Point()][i, j] => Stencil((Segment(), Point()), δ(0, 0)),
                                  b[Point(), Segment()][i, j] => Stencil((Point(), Segment()), δ(0, 1))))
    @test occursin("Full", total_nu)
    @test occursin("Fields", total_nu)
    @test findfirst("Full", total_nu) < findfirst("Fields", total_nu)

    mixed_total_nu = sprint(show,
                            MIME"text/plain"(),
                            Nonuniforms(a[Segment()][i] => Stencil(Segment(), δ(0)),
                                        b[Point(), Segment()][i, j] => Stencil((Point(), Segment()), δ(0, 1))))
    @test occursin("Full 1D", mixed_total_nu)
    @test occursin("Full 2D", mixed_total_nu)
    @test occursin("Fields", mixed_total_nu)
    @test findfirst("Full 1D", mixed_total_nu) < findfirst("Full 2D", mixed_total_nu) < findfirst("Fields", mixed_total_nu)
end
