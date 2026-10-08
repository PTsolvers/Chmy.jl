using Chmy

@testset verbose = true "topology" begin
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
    end
end
