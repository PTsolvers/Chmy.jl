using Test
using Chmy

struct NoopRule <: AbstractRule end

@testset "rewriters" begin
    @scalars a b c
    replace_a = t -> t ==ₛ a ? b : nothing

    @test isnothing(@inferred NoopRule()(a))
    @test (@inferred Passthrough(replace_a)(a)) ==ₛ b
    @test (@inferred Passthrough(replace_a)(c)) ==ₛ c

    chain = Chain(replace_a, t -> t ==ₛ a ? c : nothing)
    @test (@inferred Union{Nothing, DTerm} chain(a)) ==ₛ b
    @test isnothing(@inferred Union{Nothing, DTerm} chain(c))
    @test (@inferred Union{Nothing, DTerm} Chain(NoopRule(), replace_a)(a)) ==ₛ b

    rule = function (t)
        t ==ₛ a && return b
        isinds(t) && argument(t) ==ₛ b && return c
        return nothing
    end
    @test (@inferred Prewalk(rule)(a[𝑖])) ==ₛ b[𝑖]
    @test (@inferred Postwalk(rule)(a[𝑖])) ==ₛ c

    expr = sin(a + Literal(2.0))
    literal_rule = t -> isliteral(t) ? Literal(Int(value(t))) : nothing
    for walk in (Prewalk, Postwalk)
        @test (@inferred walk(NoopRule())(expr)) === expr
        result = @inferred walk(literal_rule)(expr)
        @test result ==ₛ expr
        @test value(arguments(only(arguments(result)))[2]) === 2
    end

    successive = t -> t ==ₛ a ? b : (t ==ₛ b ? c : nothing)
    @test (@inferred Fixpoint(successive)(a)) ==ₛ c
end
