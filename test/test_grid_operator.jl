using Test
using Chmy

struct GridShift{O} <: STerm end
struct GridReach <: STerm end

(op::GridShift)(arg::STerm) = SExpr(Call(), op, arg)
(op::GridReach)(arg::STerm) = SExpr(Call(), op, arg)

Chmy.tensorrank(::GridShift, ::STerm) = 0
Chmy.tensorrank(::GridReach, ::STerm) = 0

grid_shift_index(ind, offset) = offset == 0 ? ind : (offset > 0 ? ind + offset : ind - (-offset))

function Chmy.stencil_rule(::GridShift{O}, args::Tuple{STerm}, loc::NTuple{N,Space}, inds::NTuple{N,STerm}) where {O,N}
    arg = only(args)
    shifted = ntuple(i -> grid_shift_index(inds[i], O[i]), Val(N))
    return arg[loc...][shifted...]
end

function Chmy.stencil_rule(::GridReach, args::Tuple{STerm}, loc::NTuple{2,Space}, inds::NTuple{2,STerm})
    arg = only(args)
    i, j = inds
    return arg[loc...][i-2, j] +
           arg[loc...][i+1, j] +
           arg[loc...][i, j+2] +
           arg[loc...][i, j-3]
end

@testset "grid operators" begin
    @testset "shift arithmetic" begin
        @test @inferred(Shift(1) + Shift(2)) === Shift(3)
        @test @inferred(Shift(1) + Shift(2) - Shift(1)) === Shift(2)
        @test @inferred(Shift(1) - Shift(3)) === Shift(-2)
        @test @inferred(-Shift(1)) === Shift(-1)
        @test_throws ArgumentError Shift(1 // 2)
        @test_throws ArgumentError δ(1 // 2)
        @test offset(Point()) === SLiteral(0)
        @test offset(Segment()) === SLiteral(1 // 2)
    end

    @testset "adjacent_faces" begin
        @test @inferred(adjacent_faces(Face(Lower()))) === (Face(Span()),)
        @test @inferred(adjacent_faces(Face(Upper()))) === (Face(Span()),)
        @test @inferred(adjacent_faces(Face(Span()))) === ()

        @test @inferred(adjacent_faces(Face(Lower(), Lower()))) === (Face(Span(), Lower()),
                                                                     Face(Lower(), Span()))

        @test @inferred(adjacent_faces(Face(Lower(), Span()))) === (Face(Span(), Span()),)

        @test @inferred(adjacent_faces(Face(Span(), Upper()))) === (Face(Span(), Span()),)

        @test @inferred(adjacent_faces(Face(Lower(), Lower(), Lower()))) === (Face(Span(), Lower(), Lower()),
                                                                              Face(Lower(), Span(), Lower()),
                                                                              Face(Lower(), Lower(), Span()))

        @test @inferred(adjacent_faces(Face(Lower(), Lower(), Span()))) === (Face(Span(), Lower(), Span()),
                                                                             Face(Lower(), Span(), Span()))

        @test @inferred(adjacent_faces(Face(Span(), Span(), Span()))) === ()
    end

    @testset "facets" begin
        @test @inferred(facet(Lower(), Val(2), Val(1))) === Face(Lower(), Span())
        @test @inferred(facet(Upper(), Val(2), Val(2))) === Face(Span(), Upper())
        @test @inferred(facets(Val(2))) === (Face(Lower(), Span()),
                                             Face(Upper(), Span()),
                                             Face(Span(), Lower()),
                                             Face(Span(), Upper()))
    end

    @testset "nonuniform staggered shifts" begin
        @scalars a b
        i, j = SIndex(1), SIndex(2)

        @test Stencil(δ(0, 0)).location === (Point(), Point())
        @test sprint(show, Stencil(δ(0))) == "Stencil(Point(), δ(0))"

        nu1 = nonuniforms(a[Segment()][i])
        @test sprint(show, Chmy.stencil(nu1, a[Segment()])) == "Stencil(Segment(), δ(0))"

        nu1_merged = nonuniforms(a[Point(), Segment()][i, j] + a[Point(), Segment()][i, j+1])
        @test sprint(show, Chmy.stencil(nu1_merged, a[Point(), Segment()])) == "Stencil((Point(), Segment()), δ(0, 0), δ(0, 1))"

        nu2 = nonuniforms(a[Segment(), Point()][i+1, j] + b[Point(), Segment()][i, j-1])
        @test sprint(show, Chmy.stencil(nu2, a[Segment(), Point()])) == "Stencil((Segment(), Point()), δ(1, 0))"
        @test sprint(show, Chmy.stencil(nu2, b[Point(), Segment()])) == "Stencil((Point(), Segment()), δ(0, -1))"
    end

    @testset "reach" begin
        @scalars a b

        point_stencil = Stencil((Point(), Point()),
                                δ(-2, 0),
                                δ(1, 0),
                                δ(0, 2),
                                δ(0, -3))
        @test @inferred(reach(point_stencil, Face(Lower(), Span()))) === SLiteral(2)
        @test @inferred(reach(point_stencil, Face(Span(), Upper()))) === SLiteral(2)

        point_reach = @inferred reach(point_stencil)
        @test point_reach[Face(Lower(), Span())] === SLiteral(2)
        @test point_reach[Face(Upper(), Span())] === SLiteral(1)
        @test point_reach[Face(Span(), Lower())] === SLiteral(3)
        @test point_reach[Face(Span(), Upper())] === SLiteral(2)

        touching_stencil = Stencil((Point(),), δ(0))
        touching_reach = @inferred reach(touching_stencil)
        @test touching_reach[Face(Lower())] === SLiteral(0)
        @test touching_reach[Face(Upper())] === SLiteral(0)

        nu = Nonuniforms(a[Point(), Point()] => point_stencil,
                         b[Point(), Point()] => Stencil((Point(), Point()), δ(-1, 0), δ(0, 4)))
        merged_reach = @inferred reach(nu)
        @test merged_reach[Face(Lower(), Span())] === SLiteral(2)
        @test merged_reach[Face(Upper(), Span())] === SLiteral(1)
        @test merged_reach[Face(Span(), Lower())] === SLiteral(3)
        @test merged_reach[Face(Span(), Upper())] === SLiteral(4)
    end
end
