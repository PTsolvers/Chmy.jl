import Chmy: StaticCoef

@testset "StaticCoef" begin
    @test StaticCoef(2.0) === StaticCoef{2}()
    @test StaticCoef(6 // 3) === StaticCoef{2}()
    @test StaticCoef(3 // 2) === StaticCoef{3 // 2}()

    @test Chmy.isnegative(StaticCoef(-1))
    @test !Chmy.isnegative(StaticCoef(0))

    @test iszero(StaticCoef(0))
    @test !iszero(StaticCoef(1 // 2))
    @test isone(StaticCoef(1))
    @test !isone(StaticCoef(2))
    @test isinteger(StaticCoef(2))
    @test !isinteger(StaticCoef(1 // 2))
    @test abs(StaticCoef(-3)) === StaticCoef(3)
    @test abs(StaticCoef(-3 // 2)) === StaticCoef(3 // 2)

    @test StaticCoef(2) + StaticCoef(3) === StaticCoef(5)
    @test StaticCoef(2) - StaticCoef(3) === StaticCoef(-1)
    @test StaticCoef(2) * StaticCoef(3) === StaticCoef(6)

    @test StaticCoef(4) / StaticCoef(2) === StaticCoef(2)
    @test StaticCoef(1) / StaticCoef(2) === StaticCoef(1 // 2)
    @test StaticCoef(3 // 2) / StaticCoef(3) === StaticCoef(1 // 2)
end
