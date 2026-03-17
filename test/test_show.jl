@testset "show" begin
    a = SScalar(:a)
    b = SScalar(:b)

    @test sprint(show, -a + b) == "-a + b"
    @test sprint(show, (-a)^b) == "(-a) ^ b"
end
