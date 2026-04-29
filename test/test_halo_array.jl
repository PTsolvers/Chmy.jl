using Test
using Chmy

@testset "halo array" begin
    @testset "wrapped parent indexing" begin
        parent_data = reshape(collect(1:42), 6, 7)
        A = HaloArray(parent_data, ((1, 2), (2, 1)))

        @test size(A) == (3, 4)
        @test axes(A) == (Base.OneTo(3), Base.OneTo(4))
        @test parent(A) === parent_data
        @test halowidths(A) == ((1, 2), (2, 1))

        @test A[1, 1] == parent_data[2, 3]
        @test A[0, 1] == parent_data[1, 3]
        @test A[5, 5] == parent_data[6, 7]
        @test_throws BoundsError A[-1, 1]
        @test_throws BoundsError A[1, 6]

        A[1, 1] = -1
        A[0, 1] = -2
        @test parent_data[2, 3] == -1
        @test parent_data[1, 3] == -2
    end

    @testset "interior and halo views" begin
        parent_data = reshape(collect(1:42), 6, 7)
        A = HaloArray(parent_data, (1, 2), (2, 1))

        inside = interior(A)
        @test inside == @view parent_data[2:4, 3:6]
        inside[1, 1] = -10
        @test parent_data[2, 3] == -10

        lower_x = halo(A, Face(Lower(), Span()))
        upper_y = halo(A, Face(Span(), Upper()))
        lower_corner = halo(A, Face(Lower(), Lower()))

        @test lower_x == @view parent_data[1:1, 3:6]
        @test upper_y == @view parent_data[2:4, 7:7]
        @test lower_corner == @view parent_data[1:1, 1:2]
        @test_throws ArgumentError halo(A, Face(Lower()))
    end

    @testset "zero-width halos" begin
        parent_data = reshape(collect(1:12), 3, 4)
        A = HaloArray(parent_data, ((0, 0), (1, 0)))

        @test size(A) == (3, 3)
        @test isempty(halo(A, Face(Lower(), Span())))
        @test isempty(halo(A, Face(Span(), Upper())))
        @test interior(A) == @view parent_data[1:3, 2:4]
    end

    @testset "allocation and validation" begin
        A = HaloArray{Float64}(undef, (3, 4), ((1, 2), (2, 1)))
        B = HaloArray(zeros(5), (1, 2))
        C = HaloArray(zeros(5), ((1, 2),))
        D = HaloArray{Float64}(undef, (2,), (1, 1))

        @test A isa HaloArray{Float64,2}
        @test size(A) == (3, 4)
        @test size(parent(A)) == (6, 7)
        @test halowidths(A) == ((1, 2), (2, 1))
        @test size(B) == (2,)
        @test size(C) == (2,)
        @test size(parent(D)) == (4,)

        @test_throws ArgumentError HaloArray(zeros(2, 2), ((2, 1), (0, 0)))
        @test_throws ArgumentError HaloArray(zeros(2, 2), ((-1, 0), (0, 0)))
        @test_throws ArgumentError HaloArray(zeros(2, 2), ((0, 0),))
        @test_throws ArgumentError HaloArray{Float64}(undef, (3, -1), ((0, 0), (0, 0)))
    end

    @testset "adapt" begin
        parent_data = reshape(collect(1:12), 3, 4)
        A = HaloArray(parent_data, ((1, 1), (1, 1)))
        adapted = Chmy.Adapt.adapt(identity, A)

        @test adapted isa HaloArray
        @test parent(adapted) === parent_data
        @test halowidths(adapted) == halowidths(A)
        @test adapted[1, 1] == A[1, 1]
    end
end
