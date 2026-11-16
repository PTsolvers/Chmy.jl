using Test
using Chmy
using Chmy: matches, dim, codim, distance, frame, wall, mirror

@testset "faces" begin
    @test dim((↓, :)) == 1
    @test codim((↓, :)) == 1
    @test dim((↓, ↑)) == 0
    @test codim((:, :, :)) == 0
    @test Chmy.iswall((:, ↑))
    @test Chmy.iscorner((↓, ↑))
    @test Chmy.isinterior((:, :))
end

@testset "positions" begin
    # the outward distance from the wall of the frame, positive in the ghost region, offsets point along the axis
    @test distance(Coord(↓, 1 // 2)) == -1 // 2
    @test distance(Coord(↓, -1 // 2)) == 1 // 2
    @test distance(Coord(↑, 1 // 2)) == 1 // 2
    @test isnothing(distance(Coord(:, 1 // 2)))
    @test isnothing(distance(Trace(↓)))
    # the sides of the face next to the positions, traces are never next to a wall
    @test map(frame, (Coord(↓, 1), Coord(:, 0), Trace(↑))) == (↓, :, :)
    # coordinates move along the axis, and traces are constant along it
    @test Coord(↑, -1 // 2) + 1 == Coord(↑, 1 // 2)
    @test Coord(:, 0) - 1 // 2 == Coord(:, -1 // 2)
    @test Trace(↓) + 1 // 2 == Trace(↓)
    # the point on the wall, and the mirror image through it
    @test wall(Coord(↑, -3 // 2)) == Coord(↑, 0)
    @test mirror(Coord(↓, -1 // 2)) == Coord(↓, 1 // 2)
    # offsets are multiples of 1//2
    @test_throws ArgumentError Coord(:, 1 // 3)
end

@testset "wall patterns" begin
    @test codim(AnyWall, 3) == 1
    @test dim(AnyWall, 3) == 2
    @test matches(AnyWall, (↓,))
    @test matches(AnyWall, (:, ↑))
    @test !matches(AnyWall, (:, :))
    @test !matches(AnyWall, (↓, ↑))
    @test matches(AlongAxis(2), (:, ↑, :))
    @test !matches(AlongAxis(2), (↓, :, :))
    @test matches(Wall(3, ↓), (:, :, ↓))
    @test !matches(Wall(3, ↓), (:, :, ↑))
    @test !matches(Wall(3, ↓), (:, ↓))
    @test matches(LeftWall, (↓,))
    @test matches(LeftWall, (↓, :, :))
    @test matches(RightWall, (↑, :))
    @test !matches(LeftWall, (↑, :))
    # the last axis points up
    @test matches(BottomWall, (:, ↓))
    @test matches(TopWall, (:, :, ↑))
    @test !matches(TopWall, (:, ↑, :))
    @test matches(BackWall, (:, ↓, :))
    @test matches(FrontWall, (:, ↑, :))
    # bottom and top walls only exist in 2D and 3D, back and front walls in 3D
    @test !matches(BottomWall, (↓,))
    @test !matches(BackWall, (↓, :))
    @test !matches(FrontWall, (:, ↑))
end

@testset "corner pattern" begin
    @test dim(AnyCorner, 3) == 0
    @test codim(AnyCorner, 3) == 3
    @test matches(AnyCorner, (↓,))
    @test matches(AnyCorner, (↓, ↑))
    @test !matches(AnyCorner, (↓, :))
end

@testset "wall pattern combinators" begin
    @test matches(LeftWall | RightWall, (↓, :))
    @test matches(LeftWall | RightWall, (↑, :))
    @test !matches(LeftWall | RightWall, (:, ↓))
    @test matches(AlongAxis(1) & LeftWall, (↓, :))
    @test !matches(AlongAxis(1) & LeftWall, (↑, :))
    @test matches(!LeftWall, (↑, :))
    @test matches(!LeftWall, (:, ↓))
    @test !matches(!LeftWall, (↓, :))
    # the complement only holds on walls
    @test !matches(!LeftWall, (↓, ↓))
    @test matches(AnyWall & !TopWall, (:, ↓))
    @test !matches(AnyWall & !TopWall, (:, ↑))
    @test !matches(!(LeftWall | RightWall), (↑, :))
end

@testset "region patterns" begin
    # ξ is the outward distance from the wall, positive in the ghost region
    @test matches(Ghost, 1 // 2)
    @test !matches(Ghost, 0 // 1)
    @test matches(Boundary, 0 // 1)
    @test !matches(Boundary, -1 // 2)
    @test matches(Interior, -1 // 2)
    @test !matches(Interior, 0 // 1)
    @test matches(Within(1), -1 // 1)
    @test matches(Within(1), 1 // 2)
    @test !matches(Within(1), -3 // 2)
    @test matches(Ghost | Boundary, 0 // 1)
    @test !matches(Ghost | Boundary, -1 // 2)
    @test matches(Within(1) & Interior, -1 // 2)
    @test !matches(Within(1) & Interior, 1 // 2)
    @test matches(!Ghost, 0 // 1)
    @test !matches(!Ghost, 1 // 2)
end

@testset "tensor patterns" begin
    @test matches((:, :), (1, 2), 2)
    @test matches((𝐧,), (2,), 2)
    @test !matches((𝐧,), (1,), 2)
    @test matches((𝐧, 𝐭), (2, 1), 2)
    @test !matches((𝐧, 𝐭), (2, 2), 2)
    @test matches((), (), 1)
end
