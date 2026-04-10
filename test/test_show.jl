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
end
