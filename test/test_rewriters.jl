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
    @test (@inferred Prewalk(rule)(a[(:, 1)])) ==ₛ b[(:, 1)]
    @test (@inferred Postwalk(rule)(a[(:, 1)])) ==ₛ c

    # expressions with rewritten children are rebuilt in canonical form
    expr = sin(a + b)
    for walk in (Prewalk, Postwalk)
        @test (@inferred walk(NoopRule())(expr)) === expr
        @test (@inferred walk(t -> t ==ₛ b ? -a : nothing)(expr)) ==ₛ Literal(0)
    end
    @tensors 2 A B
    @test (@inferred subs(a + b, a => -b)) ==ₛ Literal(0)
    @test (@inferred subs(A ⋅ B, A => inv(B))) ==ₛ ℐ(2)
    @test (@inferred subs(A ⋅ B, A => components(A, 2), B => components(B, 2))) ==ₛ components(A, 2) ⋅ components(B, 2)

    successive = t -> t ==ₛ a ? b : (t ==ₛ b ? c : nothing)
    @test (@inferred Fixpoint(successive)(a)) ==ₛ c
end
