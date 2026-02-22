@inline sameterm(a::STerm, b::STerm) = a === b

@testset "rewriters" begin
    a = SScalar(:a)
    b = SScalar(:b)
    c = SScalar(:c)
    i = SIndex(1)
    j = SIndex(2)
    seg = Segment()
    pt = Point()

    @testset "AbstractRule and Passthrough" begin
        struct NoopRule <: AbstractRule end
        @test isnothing(NoopRule()(a))

        replace_a = t -> t === a ? b : nothing
        passthrough = Passthrough(replace_a)

        @test Passthrough(passthrough) === passthrough
        @test sameterm(passthrough(a), b)
        @test sameterm(passthrough(c), c)
    end

    @testset "Chain" begin
        replace_a = t -> t === a ? b : nothing
        replace_a_again = t -> t === a ? c : nothing
        never_reached = _ -> error("Chain should stop at the first match")

        chain = Chain(replace_a, replace_a_again, never_reached)
        safe_chain = Chain(replace_a, replace_a_again)

        @test Chain(chain) === chain
        @test sameterm(chain(a), b)
        @test sameterm(Chain((replace_a, replace_a_again))(a), b)
        @test isnothing(safe_chain(c))
        @test isnothing(Chain()(a))

        @inferred Union{Nothing,STerm} safe_chain(a)
        @inferred Union{Nothing,STerm} safe_chain(c)
    end

    @testset "Prewalk vs Postwalk order" begin
        expr = a[i]

        rule = function (t)
            t === a && return b
            if isind(t) && argument(t) === b
                return c
            end
            return nothing
        end

        @test sameterm(Prewalk(rule)(expr), b[i])
        @test sameterm(Postwalk(rule)(expr), c)
    end

    @testset "Fixpoint" begin
        chain = t -> t === a ? b : (t === b ? c : nothing)
        @test sameterm(Fixpoint(chain)(a), c)
        @test sameterm(Fixpoint(chain)(c), c)
    end

    @testset "stencil_rule and lower_stencil" begin
        @test sameterm(stencil_rule(SRef(:+), (a, b), (i, j)),
                       a[i, j] + b[i, j])

        @test sameterm(stencil_rule(SRef(:+), (a, b), (seg, pt), (i, j)),
                       a[seg, pt][i, j] + b[seg, pt][i, j])

        @test sameterm(lower_stencil((a+b)[i, j]),
                       a[i, j] + b[i, j])

        @test sameterm(lower_stencil((a[seg, pt]+b)[i, j]),
                       b[i, j] + a[seg, pt][i, j])

        @test sameterm(lower_stencil(a[seg][i]), a[seg][i])
    end

    @testset "lift" begin
        @test sameterm(lift(SRef(:+), (a, b), (i, j), Val(1)),
                       a[i, j] + b[i, j])

        @test sameterm(lift(SRef(:+), (a, b), (i, j), Val(2)),
                       a[i, j] + b[i, j])

        @test sameterm(lift(SRef(:+), (a, b), (seg, pt), (i, j), Val(2)),
                       a[seg, pt][i, j] + b[seg, pt][i, j])
    end

    @testset "subs" begin
        @test sameterm(subs(a, a => c), c)
        @test sameterm(subs(a + b, a => c), b + c)
        @test sameterm(subs((a+b)[i], a => c), (b+c)[i])
    end
end
