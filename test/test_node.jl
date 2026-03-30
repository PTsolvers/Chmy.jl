import Chmy: makeop

@testset "node" begin
    @testset "simplify canonicalizes inside node expressions" begin
        a = SScalar(:a)
        b = SScalar(:b)
        raw = makeop(:+, b, a)
        wrapped = node(raw)
        simplified = node(makeop(:+, a, b))

        @test simplify(wrapped) === simplified
        @test simplify(cos(wrapped)) === cos(simplified)
    end

    @testset "subs descends into node expressions" begin
        a = SScalar(:a)
        b = SScalar(:b)
        c = SScalar(:c)
        d = SScalar(:d)
        inner = makeop(:+, b, a)
        expr = node(makeop(:*, inner, d))

        @test subs(expr, a => c) === node(makeop(:*, d, makeop(:+, b, c)))
        @test subs(expr, inner => c) === node(makeop(:*, c, d))
    end

    @testset "node_unwrap removes wrappers and reevaluates" begin
        a = SScalar(:a)
        b = SScalar(:b)
        expr = a + node(a + b)

        @test node_unwrap(expr) === makeop(:+, makeop(:*, SUniform(2), a), b)
        @test node_unwrap(node(a + b)) === a + b
    end

    @testset "tensor expansion inherits nodes" begin
        v = SVec(:v)
        tv = Tensor{2}(node(v))

        @test tv isa Vec{2}
        @test node(v)[1] isa SExpr{Node}
        @test node(v)[2] isa SExpr{Node}
        @test tv[1] === node(v[1])
        @test tv[2] === node(v[2])
    end

    @testset "node wraps tensor components" begin
        a = SScalar(:a)
        b = SScalar(:b)
        t = Tensor{2,1}(a, b)
        wrapped = node(t)

        @test wrapped isa Vec{2}
        @test wrapped[1] === node(a)
        @test wrapped[2] === node(b)
    end

    @testset "substitution matches simplified node subexpressions" begin
        a = SScalar(:a)
        b = SScalar(:b)
        c = SScalar(:c)
        q = node(makeop(:+, b, a))
        expr = simplify(q + c + q)

        @test simplify(q) === node(makeop(:+, a, b))
        @test simplify(subs(expr, simplify(q) => SUniform(0))) === c
    end

    @testset "compute ignores the wrapper" begin
        f = SScalar(:f)
        g = SScalar(:g)
        expr = node(f + g)
        binding = Binding(f => 2, g => 3)

        @test compute(expr, binding) == 5
    end
end
