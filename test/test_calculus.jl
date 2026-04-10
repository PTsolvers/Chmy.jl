@testset "calculus" begin
    @testset "generic calculus term indexing fallbacks" begin
        i = SIndex(1)
        p = Point()

        lit = SLiteral(2)
        ref = SRef(:f)
        fun = SFun(sin)
        diff = CentralDifference()

        @test lit[i] === lit
        @test lit[p] === lit
        @test ref[i] === ref
        @test ref[p] === ref
        @test fun[i] === fun
        @test fun[p] === fun

        ind = diff[i]
        loc = diff[p]

        @test isind(ind)
        @test argument(ind) === diff
        @test only(indices(ind)) === i

        @test isloc(loc)
        @test argument(loc) === diff
        @test only(location(loc)) === p
    end

    @testset "uniform fields under calculus" begin
        @uniform @scalars a
        @uniform @vectors u
        @uniform @tensors 2 T
        p, s = Point(), Segment()
        i, j = SIndex(1), SIndex(2)
        diff = CentralDifference()
        grad = Gradient(diff)
        divg = Divergence(diff)
        curl = Curl(diff)

        @test !isuniform(diff(a))
        @test !isuniform(grad(a))
        @test diff(a)[i] === SLiteral(0)
        @test diff(sin(a))[i] === SLiteral(0)
        @test grad(a)[1][i] === SLiteral(0)
        @test grad(u)[1, 1][i] === SLiteral(0)
        @test grad(T)[1, 1, 1][i] === SLiteral(0)
        @test divg(u)[i] === SLiteral(0)
        @test divg(T)[1][i] === SLiteral(0)
        @test Tensor{2}(curl(u))[i, j] === SLiteral(0)
        @test a[p, s][i, j] === a
    end

    @testset "calculus tensor expression expansion" begin
        @scalars a
        @vectors u w
        @tensors 2 @sym(S)
        diff = CentralDifference()
        grad = Gradient(diff)
        divg = Divergence(diff)
        curl = Curl(diff)

        gu = Tensor{2}(grad(u))
        @test gu isa Tensor{2,2}
        @test gu[1, 2] === grad.op[1](u[2])
        @test gu[2, 1] === grad.op[2](u[1])
        @test Tensor{2}(grad(u)[1, 2]) === grad.op[1](u[2])

        gS = Tensor{2}(grad(S))
        @test gS isa Tensor{2,3}
        @test gS[1, 1, 2] === grad.op[1](S[1, 2])
        @test gS[2, 1, 2] === grad.op[2](S[1, 2])

        divS = Tensor{2}(divg(S))
        @test divS isa Vec{2}
        @test divS[1] === divg.op[1](S[1, 1]) + divg.op[2](S[2, 1])
        @test divS[2] === divg.op[1](S[1, 2]) + divg.op[2](S[2, 2])

        curlw = Tensor{3}(curl(w))
        @test curlw isa Vec{3}
        @test curlw[1] === curl.op[2](w[3]) - curl.op[3](w[2])
        @test curlw[2] === curl.op[3](w[1]) - curl.op[1](w[3])
        @test curlw[3] === curl.op[1](w[2]) - curl.op[2](w[1])

        curlu = Tensor{2}(curl(u))
        @test curlu === curl.op[1](u[2]) - curl.op[2](u[1])

        @test_throws ArgumentError Tensor{4}(curl(w))

        q = -grad(a)
        @test Tensor{3}(q[1]) === -grad.op[1](a)
    end

    @testset "automatic scalar indexing" begin
        @scalars a
        p, s = Point(), Segment()
        i, j = SIndex(1), SIndex(2)
        diff = StaggeredCentralDifference()
        grad = Gradient(diff)
        divg = Divergence(diff)

        expr = divg(grad(a))
        @test expr[s, s][i, j] === Tensor{2}(expr)[s, s][i, j]
        @test (-grad(a))[1][p, s][i, j] === -a[s, s][i, j] + a[s, s][i - 1, j]
    end

    @testset "immediate lowering inference" begin
        @scalars a
        @vectors u
        @uniform @scalars b c

        p, s = Point(), Segment()
        i, j = SIndex(1), SIndex(2)
        diff = StaggeredCentralDifference()
        grad = Gradient(diff)
        divg = Divergence(diff)

        r_a = -divg(u) / b
        r_u = -grad(a) / c

        lower_scalar(r_a, s, i, j) = r_a[s, s][i, j]
        lower_vector(r_u, p, s, i, j) = r_u[1][p, s][i, j]

        @inferred lower_scalar(r_a, s, i, j)
        @inferred lower_vector(r_u, p, s, i, j)

        lower_scalar(r_a, s, i, j)
        lower_vector(r_u, p, s, i, j)
        @test @allocated(lower_scalar(r_a, s, i, j)) == 0
        @test @allocated(lower_vector(r_u, p, s, i, j)) == 0
    end
end
