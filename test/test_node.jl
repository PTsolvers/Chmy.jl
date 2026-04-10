import Chmy: makeop

@testset "node" begin
    @testset "simplify treats nodes as black boxes" begin
        a = SScalar(:a)
        b = SScalar(:b)
        raw = makeop(:+, b, a)
        wrapped = node(raw)

        @test simplify(wrapped) === wrapped
        @test simplify(cos(wrapped)) === cos(wrapped)
    end

    @testset "subs does not descend into node expressions" begin
        a = SScalar(:a)
        b = SScalar(:b)
        c = SScalar(:c)
        d = SScalar(:d)
        inner = makeop(:+, b, a)
        expr = node(makeop(:*, inner, d))

        @test subs(expr, a => c) === expr
        @test subs(expr, inner => c) === expr
        @test subs(expr, expr => c) === c
    end

    @testset "unwrap removes wrappers and reevaluates" begin
        a = SScalar(:a)
        b = SScalar(:b)
        raw = makeop(:+, b, a)
        expr = a + node(raw)

        @test unwrap(expr) === makeop(:+, makeop(:*, SLiteral(2), a), b)
        @test unwrap(node(raw)) === a + b
    end

    @testset "nodes keep tensor-valued terms opaque" begin
        v = SVec(:v)
        wrapped = node(v)
        comp = wrapped[1]
        tv = Tensor{2}(wrapped)

        @test wrapped isa SNode
        @test tensorrank(wrapped) == 1
        @test comp isa SNode
        @test unwrap(comp) === v[1]
        @test tv isa Vec{2}
        @test tv[1] isa SNode
        @test tv[2] isa SNode
        @test unwrap(tv[1]) === v[1]
        @test unwrap(tv[2]) === v[2]
    end

    @testset "nodes protect tensor components without contaminating parents" begin
        f = SScalar(:f)
        p = Point()
        s = Segment()
        D = StaggeredCentralDifference()
        grad = Gradient(D)
        divg = Divergence(D)
        i = SIndex(1)
        j = SIndex(2)

        q = node(grad(f))
        q1 = q[1][p, s][i, j]
        expr = divg(q)[s, s][i, j]
        expected = -q[1][p, s][i, j] + q[1][p, s][i+1, j] - q[2][s, p][i, j] + q[2][s, p][i, j+1]

        @test q1 === node(grad.op[1](f))[p, s][i, j]
        @test expr === expected
        @test unwrap(expr) === divg(grad(f))[s, s][i, j]
    end

    @testset "component propagation materializes tensor contents without lowering indices" begin
        V = SVec(:V)
        s = Segment()
        D = StaggeredCentralDifference()
        grad = Gradient(D)
        i = SIndex(1)
        j = SIndex(2)

        expr = node(grad(V))[1, 1][s, s][i, j]
        expected = node(grad.op[1](V[1]))[s, s][i, j]

        @test expr === expected
        @test unwrap(expr) === grad(V)[1, 1][s, s][i, j]
    end

    @testset "boundary substitutions match protected flux components" begin
        f = SScalar(:f)
        p = Point()
        s = Segment()
        D = StaggeredCentralDifference()
        grad = Gradient(D)
        divg = Divergence(D)
        i = SIndex(1)
        j = SIndex(2)

        q = node(-grad(f))
        expr = (-divg(q))[s, s][i, j]
        bc = q[1][p, s][i, j] => SLiteral(0)

        @test subs(expr, bc) === -q[1][p, s][i+1, j] + q[2][s, p][i, j] - q[2][s, p][i, j+1]
    end

    @testset "nodes wrap concrete tensors componentwise" begin
        a = SScalar(:a)
        b = SScalar(:b)
        t = Tensor{2,1}(a, b)
        wrapped = node(t)
        tw = Tensor{2}(wrapped)

        @test wrapped isa Vec{2}
        @test wrapped[1] isa SNode
        @test wrapped[2] isa SNode
        @test unwrap(wrapped[1]) === a
        @test unwrap(wrapped[2]) === b
        @test tw === wrapped
    end

    @testset "substitution still matches whole node subexpressions" begin
        a = SScalar(:a)
        b = SScalar(:b)
        c = SScalar(:c)
        q = node(makeop(:+, b, a))
        expr = simplify(q + c + q)

        @test simplify(q) === q
        @test simplify(subs(expr, q => SLiteral(0))) === c
    end

    @testset "compute ignores the wrapper" begin
        f = SScalar(:f)
        g = SScalar(:g)
        expr = node(makeop(:+, g, f))
        binding = Binding(f => 2, g => 3)

        @test compute(expr, binding) == 5
    end
end
