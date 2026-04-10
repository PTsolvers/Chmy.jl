@testset "node" begin
    @testset "simplify treats nodes as black boxes" begin
        @scalars a b
        raw = makeop(:+, b, a)
        wrapped = node(raw)

        @test simplify(wrapped) === wrapped
        @test simplify(cos(wrapped)) === cos(wrapped)
    end

    @testset "subs does not descend into node expressions" begin
        @scalars a b c d
        inner = makeop(:+, b, a)
        expr = node(makeop(:*, inner, d))

        @test subs(expr, a => c) === expr
        @test subs(expr, inner => c) === expr
        @test subs(expr, expr => c) === c
    end

    @testset "unwrap removes wrappers and reevaluates" begin
        @scalars a b
        raw = makeop(:+, b, a)
        expr = a + node(raw)

        @test unwrap(expr) === makeop(:+, makeop(:*, SLiteral(2), a), b)
        @test unwrap(node(raw)) === a + b
    end

    @testset "nodes keep tensor-valued terms opaque" begin
        @vectors u
        wrapped = node(u)
        comp = wrapped[1]
        tu = Tensor{2}(wrapped)

        @test wrapped isa SNode
        @test tensorrank(wrapped) == 1
        @test comp isa SNode
        @test unwrap(comp) === u[1]
        @test tu isa Vec{2}
        @test tu[1] isa SNode
        @test tu[2] isa SNode
        @test unwrap(tu[1]) === u[1]
        @test unwrap(tu[2]) === u[2]
    end

    @testset "nodes protect tensor components without contaminating parents" begin
        @scalars a
        p, s = Point(), Segment()
        diff = StaggeredCentralDifference()
        grad = Gradient(diff)
        divg = Divergence(diff)
        i, j = SIndex(1), SIndex(2)

        q = node(grad(a))
        q1 = q[1][p, s][i, j]
        expr = divg(q)[s, s][i, j]
        expected = -q[1][p, s][i, j] + q[1][p, s][i + 1, j] - q[2][s, p][i, j] + q[2][s, p][i, j + 1]

        @test q1 === node(grad.op[1](a))[p, s][i, j]
        @test expr === expected
        @test unwrap(expr) === divg(grad(a))[s, s][i, j]
    end

    @testset "component propagation materializes tensor contents without lowering indices" begin
        @vectors u
        s = Segment()
        diff = StaggeredCentralDifference()
        grad = Gradient(diff)
        i, j = SIndex(1), SIndex(2)

        expr = node(grad(u))[1, 1][s, s][i, j]
        expected = node(grad.op[1](u[1]))[s, s][i, j]

        @test expr === expected
        @test unwrap(expr) === grad(u)[1, 1][s, s][i, j]
    end

    @testset "boundary substitutions match protected flux components" begin
        @scalars a
        p, s = Point(), Segment()
        diff = StaggeredCentralDifference()
        grad = Gradient(diff)
        divg = Divergence(diff)
        i, j = SIndex(1), SIndex(2)

        q = node(-grad(a))
        expr = (-divg(q))[s, s][i, j]
        bc = q[1][p, s][i, j] => SLiteral(0)

        @test subs(expr, bc) === -q[1][p, s][i + 1, j] + q[2][s, p][i, j] - q[2][s, p][i, j + 1]
    end

    @testset "nodes wrap concrete tensors componentwise" begin
        @scalars a b
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
        @scalars a b c
        q = node(makeop(:+, b, a))
        expr = simplify(q + c + q)

        @test simplify(q) === q
        @test simplify(subs(expr, q => SLiteral(0))) === c
    end

    @testset "compute ignores the wrapper" begin
        @scalars a b
        expr = node(makeop(:+, b, a))
        binding = Binding(a => 2, b => 3)

        @test compute(expr, binding) == 5
    end
end
