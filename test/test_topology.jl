using Test
using Chmy
using Chmy: candidates, BoundaryClause

@testset verbose = true "FacePattern" begin
    @testset "Face" begin
        @test ndims(Face(Span())) == 1
        @test ndims(Face(Lower(), Span())) == 2
        @test ndims(Face(Lower(), Span(), Upper())) == 3
        @test dim(Face(Lower(), Upper())) == 0
        @test dim(Face(Lower(), Span())) == 1
        @test dim(Face(Span(), Span())) == 2
        @test codim(Face(Lower(), Upper())) == 2
        @test codim(Face(Lower(), Span())) == 1
        @test codim(Face(Span(), Span())) == 0
        @test matches(Face(Lower(), Span()), Face(Lower(), Span()))
        @test !matches(Face(Lower(), Span()), Face(Lower(), Upper()))
    end
    @testset "Wall" begin
        @test codim(Wall(3, Lower()), 1) == 1
        @test codim(Wall(3, Lower()), 2) == 1
        @test codim(Wall(3, Lower()), 3) == 1
        @test matches(Wall(3, Lower()), Face(Span(), Span(), Lower()))
        @test !matches(Wall(3, Lower()), Face(Span(), Span(), Upper()))
        @test !matches(Wall(3, Lower()), Face(Span(), Lower(), Span()))
        @test !matches(Wall(3, Lower()), Face(Span(), Lower(), Lower()))
        @test !matches(Wall(3, Lower()), Face(Span(), Lower()))
    end
    @testset "AlongAxis" begin
        @test codim(AlongAxis(2), 1) == 1
        @test codim(AlongAxis(2), 2) == 1
        @test codim(AlongAxis(2), 3) == 1
        @test matches(AlongAxis(2), Face(Span(), Lower(), Span()))
        @test matches(AlongAxis(2), Face(Span(), Upper(), Span()))
        @test !matches(AlongAxis(2), Face(Lower(), Span(), Span()))
        @test !matches(AlongAxis(2), Face(Lower(), Upper(), Span()))
        @test !matches(AlongAxis(2), Face(Lower()))
    end
    @testset "AnyWall" begin
        @test codim(AnyWall(), 1) == 1
        @test codim(AnyWall(), 2) == 1
        @test codim(AnyWall(), 3) == 1
        @test matches(AnyWall(), Face(Lower()))
        @test matches(AnyWall(), Face(Span(), Upper()))
        @test matches(AnyWall(), Face(Span(), Span(), Lower()))
        @test !matches(AnyWall(), Face(Span(), Span()))
        @test !matches(AnyWall(), Face(Lower(), Upper()))
    end
    @testset "AnyCorner" begin
        @test codim(AnyCorner(), 1) == 1
        @test codim(AnyCorner(), 2) == 2
        @test codim(AnyCorner(), 3) == 3
        @test matches(AnyCorner(), Face(Lower()))
        @test matches(AnyCorner(), Face(Lower(), Upper()))
        @test matches(AnyCorner(), Face(Upper(), Lower(), Upper()))
        @test !matches(AnyCorner(), Face(Lower(), Span()))
        @test !matches(AnyCorner(), Face(Lower(), Lower(), Span()))
    end
    @testset "Left" begin
        @test codim(Left(), 1) == 1
        @test codim(Left(), 2) == 1
        @test codim(Left(), 3) == 1
        @test matches(Left(), Face(Lower()))
        @test matches(Left(), Face(Lower(), Span()))
        @test matches(Left(), Face(Lower(), Span(), Span()))
        @test !matches(Left(), Face(Upper(), Span()))
        @test !matches(Left(), Face(Span(), Lower()))
        @test !matches(Left(), Face(Lower(), Lower()))
    end
    @testset "Right" begin
        @test codim(Right(), 1) == 1
        @test codim(Right(), 2) == 1
        @test codim(Right(), 3) == 1
        @test matches(Right(), Face(Upper()))
        @test matches(Right(), Face(Upper(), Span()))
        @test matches(Right(), Face(Upper(), Span(), Span()))
        @test !matches(Right(), Face(Lower(), Span()))
        @test !matches(Right(), Face(Span(), Upper()))
        @test !matches(Right(), Face(Upper(), Upper()))
    end
    @testset "Bottom" begin
        @test codim(Bottom(), 1) == 1
        @test codim(Bottom(), 2) == 1
        @test codim(Bottom(), 3) == 1
        @test matches(Bottom(), Face(Span(), Lower()))
        @test matches(Bottom(), Face(Span(), Lower(), Span()))
        @test !matches(Bottom(), Face(Span(), Upper()))
        @test !matches(Bottom(), Face(Lower(), Span()))
        @test !matches(Bottom(), Face(Lower(), Lower()))
        @test !matches(Bottom(), Face(Lower()))
    end
    @testset "Top" begin
        @test codim(Top(), 1) == 1
        @test codim(Top(), 2) == 1
        @test codim(Top(), 3) == 1
        @test matches(Top(), Face(Span(), Upper()))
        @test matches(Top(), Face(Span(), Upper(), Span()))
        @test !matches(Top(), Face(Span(), Lower()))
        @test !matches(Top(), Face(Upper(), Span()))
        @test !matches(Top(), Face(Upper(), Upper()))
        @test !matches(Top(), Face(Upper()))
    end
    @testset "Back" begin
        @test codim(Back(), 1) == 1
        @test codim(Back(), 2) == 1
        @test codim(Back(), 3) == 1
        @test matches(Back(), Face(Span(), Span(), Lower()))
        @test !matches(Back(), Face(Span(), Span(), Upper()))
        @test !matches(Back(), Face(Lower(), Span(), Span()))
        @test !matches(Back(), Face(Span(), Lower(), Lower()))
        @test !matches(Back(), Face(Span(), Lower()))
        @test !matches(Back(), Face(Lower()))
    end
    @testset "Front" begin
        @test codim(Front(), 1) == 1
        @test codim(Front(), 2) == 1
        @test codim(Front(), 3) == 1
        @test matches(Front(), Face(Span(), Span(), Upper()))
        @test !matches(Front(), Face(Span(), Span(), Lower()))
        @test !matches(Front(), Face(Upper(), Span(), Span()))
        @test !matches(Front(), Face(Span(), Upper(), Upper()))
        @test !matches(Front(), Face(Span(), Upper()))
        @test !matches(Front(), Face(Upper()))
    end
    @testset "Either" begin
        @test codim(Left() | Right(), 2) == 1
        @test codim(Face(Lower(), Lower()) | Face(Upper(), Upper()), 2) == 2
        @test_throws ArgumentError codim(Left() | Face(Lower(), Lower()), 2)
        @test matches(Left() | Right(), Face(Lower(), Span()))
        @test matches(Left() | Right(), Face(Upper(), Span()))
        @test !matches(Left() | Right(), Face(Span(), Lower()))
        @test !matches(Left() | Right(), Face(Lower(), Lower()))
    end
    @testset "Both" begin
        @test codim(AlongAxis(1) & Left(), 2) == 1
        @test_throws ArgumentError codim(Left() & Face(Lower(), Lower()), 2)
        @test matches(AlongAxis(1) & Left(), Face(Lower(), Span()))
        @test !matches(AlongAxis(1) & Left(), Face(Upper(), Span()))
        @test !matches(Left() & Bottom(), Face(Lower(), Span()))
        @test !matches(Left() & Bottom(), Face(Span(), Lower()))
    end
    @testset "Not" begin
        @test codim(!Left(), 2) == 1
        @test codim(!Face(Lower(), Lower()), 2) == 2
        @test matches(!Left(), Face(Upper(), Span()))
        @test matches(!Left(), Face(Span(), Lower()))
        @test !matches(!Left(), Face(Lower(), Span()))
        @test !matches(!Left(), Face(Lower(), Lower()))
        @test !matches(!Left(), Face(Span(), Span()))
        @test matches(!Face(Lower(), Lower()), Face(Upper(), Lower()))
        @test !matches(!Face(Lower(), Lower()), Face(Lower(), Lower()))
        @test !matches(!Face(Lower(), Lower()), Face(Upper(), Span()))
    end
    @testset "composition" begin
        @test matches(!(Left() | Right()), Face(Span(), Lower()))
        @test !matches(!(Left() | Right()), Face(Lower(), Span()))
        @test matches(AnyWall() & !Top(), Face(Span(), Lower()))
        @test !matches(AnyWall() & !Top(), Face(Span(), Upper()))
        @test matches((Left() | Right()) & !Wall(1, Upper()), Face(Lower(), Span()))
        @test !matches((Left() | Right()) & !Wall(1, Upper()), Face(Upper(), Span()))
        # De Morgan
        for f in [Face(Lower(), Span()), Face(Upper(), Span()), Face(Span(), Lower()), Face(Span(), Upper())]
            @test matches(!(Left() | Top()), f) == matches(!Left() & !Top(), f)
        end
        @test matches(!!Left(), Face(Lower(), Span()))
        @test !matches(!!Left(), Face(Upper(), Span()))
    end
    @testset "edge cases" begin
        # corners are walls in 1D
        @test codim(Left() | AnyCorner(), 1) == 1
        @test_throws ArgumentError codim(Left() | AnyCorner(), 2)
        # a face pattern is bound to its number of dimensions
        @test_throws ArgumentError codim(Face(Lower(), Lower()), 3)
        @test_throws ArgumentError codim(!Face(Lower(), Lower()), 3)
        # Front() holds on nothing in 2D, so its complement is every wall
        @test matches(!Front(), Face(Lower(), Span()))
        @test matches(!Front(), Face(Span(), Upper()))
    end
end

@testset "regions" begin
    # ξ is the signed distance along the outward normal, positive in the ghost region
    @test 1 // 2 ∈ Ghost()
    @test 0 // 1 ∉ Ghost()
    @test 0 // 1 ∈ OnBoundary()
    @test -1 // 2 ∉ OnBoundary()
    @test 1 // 2 ∉ OnBoundary()
    @test 0 // 1 ∉ NearBoundary(2)
    @test -1 // 2 ∈ NearBoundary(2)
    @test -2 // 1 ∈ NearBoundary(2)
    @test -5 // 2 ∉ NearBoundary(2)
end

@testset "position" begin
    # positive offsets move inward from the lower wall and outward from the upper one
    @test Chmy.position(Near(Lower(), -2), 0) == Near(Lower(), -2)
    @test Chmy.position(Near(Lower(), -1), 1 // 2) == Near(Lower(), -3 // 2)
    @test Chmy.position(Near(Upper(), -2), 1) == Near(Upper(), -1)
    @test Chmy.position(Near(Upper(), -1 // 2), 1) == Near(Upper(), 1 // 2)
    @test Chmy.position(Far(), 3) == Far()
end

@testset "candidates" begin
    L, U = Lower(), Upper()
    clauses(pairs...) = map(p -> BoundaryClause(first(p), last(p)), pairs)
    # no near axes, or near axes outside the region
    @test isempty(candidates((Far(), Far()), clauses(Left() => Ghost())))
    @test isempty(candidates((Near(L, -3), Far()), clauses(Left() => NearBoundary(2))))
    # higher codimension first, then the lower axis on a tie
    on = (Near(L, 0), Near(L, 0))
    @test candidates(on, clauses(AnyWall() => OnBoundary(), AnyCorner() => OnBoundary())) == [[1, 2] => 2, [1] => 1, [2] => 1]
    # ghost axes before boundary axes, whatever the declaration order
    ghost_on = (Near(L, 0), Near(L, 1 // 2))
    @test candidates(ghost_on, clauses(Left() => OnBoundary(), Bottom() => Ghost())) == [[2] => 2, [1] => 1]
    # a corner key needs every axis of the corner in its region
    @test candidates(ghost_on, clauses(Left() => OnBoundary(), Bottom() => Ghost(), AnyCorner() => Ghost())) == [[2] => 2, [1] => 1]
    # the first matching clause of a face acts, later ones are shadowed; faces are ordered by their clause
    ghosts = (Near(L, 1 // 2), Near(L, 1 // 2))
    @test candidates(ghosts, clauses(Left() => Ghost(), AnyWall() => Ghost())) == [[1] => 1, [2] => 2]
    # upper sides
    near = (Near(U, -1 // 2), Near(L, -1 // 2))
    @test candidates(near, clauses(AnyCorner() => NearBoundary(1), Right() => NearBoundary(1))) == [[1, 2] => 1, [1] => 2]
    # 3D: corner, then the one named edge, then the wall
    on3 = (Near(L, 0), Near(L, 0), Near(L, 0))
    rules3 = clauses(Left() => OnBoundary(), Face(L, L, Span()) => OnBoundary(), AnyCorner() => OnBoundary())
    @test candidates(on3, rules3) == [[1, 2, 3] => 3, [1, 2] => 2, [1] => 1]
    # malformed rules throw even without near axes
    @test_throws ArgumentError candidates((Far(), Far()), clauses((Left() | AnyCorner()) => Ghost()))
    @test_throws ArgumentError candidates(on, clauses(Face(L, L, Span()) => OnBoundary()))
    @test (@inferred candidates(on3, rules3)) == [[1, 2, 3] => 3, [1, 2] => 2, [1] => 1]
end
