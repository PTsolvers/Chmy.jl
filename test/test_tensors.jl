import Chmy: makeop, ncomponents, linear_index, dimensions
import Chmy: NoKind, SymKind, AltKind, DiagKind

@testset "tensors" begin
    @testset "symbolic tensors" begin
        a = SScalar(:a)
        v = SVec(:v)
        s = SSymTensor{2}(:S)
        A = SAltTensor{2}(:A)
        d = SDiagTensor{2}(:D)

        @test tensorrank(a) == 0
        @test tensorrank(v) == 1
        @test tensorrank(s) == 2
        @test tensorkind(a) === NoKind
        @test tensorkind(s) === SymKind
        @test tensorkind(A) === AltKind
        @test tensorkind(d) === DiagKind
        @test name(v) === :v

        @test a[] === a
        @test SZeroTensor{0}() === SUniform(0)
        @test SIdTensor{0}() === SUniform(1)

        @test SZeroTensor{2}()[1, 2] === SUniform(0)
        @test SIdTensor{2}()[1, 1] === SUniform(1)
        @test SIdTensor{2}()[1, 2] === SUniform(0)

        @test s[2, 1] === s[1, 2]
        @test d[1, 2] === SUniform(0)
        @test d[2, 2] === d[SUniform(2), SUniform(2)]
        @test A[1, 1] === SUniform(0)
        @test A[2, 1] === makeop(:-, A[1, 2])
    end

    @testset "tensor metadata and helper indices" begin
        @test ncomponents(NoKind, Val(3), Val(2)) == 9
        @test ncomponents(SymKind, Val(3), Val(2)) == 6
        @test ncomponents(AltKind, Val(3), Val(2)) == 3
        @test ncomponents(DiagKind, Val(3), Val(2)) == 3

        @test linear_index(NoKind, Val(2), 1, 1) == 1
        @test linear_index(NoKind, Val(2), 2, 1) == 2
        @test linear_index(SymKind, Val(3), 2, 1) == linear_index(SymKind, Val(3), 1, 2)
        @test linear_index(AltKind, Val(3), 3, 1) == linear_index(AltKind, Val(3), 1, 3)
        @test linear_index(DiagKind, Val(4), 3, 3) == 3
    end

    @testset "concrete tensor construction and indexing" begin
        a = SScalar(:a)
        b = SScalar(:b)
        c = SScalar(:c)
        d = SScalar(:d)

        t = Tensor{2,2}(a, b, c, d)
        @test t isa Tensor{2,2,NoKind}
        @test length(t) == 4
        @test ndims(t) == 2
        @test dimensions(t) == 2
        @test tensorrank(t) == 2
        @test tensorkind(t) === NoKind
        @test t[1, 1] === a
        @test t[2, 1] === b
        @test t[1, 2] === c
        @test t[2, 2] === d

        sym = Tensor{2,2}(a, b, b, c)
        @test sym isa SymTensor{2,2}
        @test length(sym) == 3
        @test sym[1, 2] === b
        @test sym[2, 1] === b

        diag = Tensor{2,2}(a, SUniform(0), SUniform(0), c)
        @test diag isa DiagTensor{2,2}
        @test diag[1, 1] === a
        @test diag[1, 2] === SUniform(0)
        @test diag[2, 2] === c

        alt = Tensor{3,2}(SUniform(0), -a, -b, a, SUniform(0), -c, b, c, SUniform(0))
        @test alt isa AltTensor{3,2}
        @test alt[1, 2] === a
        @test alt[2, 1] === -a
        @test alt[1, 1] === SUniform(0)

        @test Tensor{2,2}(SUniform(1), SUniform(0), SUniform(0), SUniform(1)) isa IdTensor{2,2}
        @test Tensor{2,2,DiagKind}(SUniform(0), SUniform(0)) isa ZeroTensor{2,2}

        @test_throws ErrorException Tensor{2,2,SymKind}(a, b)
    end

    @testset "symbolic tensor expansion" begin
        s = SSymTensor{2}(:S)
        A = SAltTensor{2}(:A)
        d = SDiagTensor{2}(:D)

        ts = Tensor{2}(s)
        ta = Tensor{3}(A)
        td = Tensor{3}(d)

        @test ts isa SymTensor{2,2}
        @test ta isa AltTensor{3,2}
        @test td isa DiagTensor{3,2}

        @test ts[2, 1] === s[1, 2]
        @test ta[1, 2] === A[1, 2]
        @test ta[2, 1] === makeop(:-, A[1, 2])
        @test ta[1, 1] === SUniform(0)
        @test td[1, 2] === SUniform(0)
        @test td[3, 3] === d[3, 3]
    end

    @testset "symbolic tensor expression expansion" begin
        u = SVec(:u)
        v = SVec(:v)
        T = SSymTensor{2}(:T)
        d = CentralDifference()
        grad = Gradient(d)
        divg = Divergence(d)
        curl = Curl(d)

        expr = T ⋅ u + 2 * v
        tex = Tensor{2}(expr)

        @test tex isa Vec{2}
        @test tex[1] === T[1, 1] * u[1] + T[1, 2] * u[2] + 2 * v[1]
        @test tex[2] === T[1, 2] * u[1] + T[2, 2] * u[2] + 2 * v[2]

        tau = SSymTensor{2}(:tau)
        @test Tensor{3}(tau[1, 1]) === tau[1, 1]
        @test Tensor{3}(tau[1, 1] + tau[2, 2]) === tau[1, 1] + tau[2, 2]

        @test Tensor{4}(SUniform(3)) === SUniform(3)
        @test Tensor{4}(SScalar(:p)) === SScalar(:p)

        p = SScalar(:p)
        I = SIdTensor{2}()
        negI = @inferred Tensor{2}(-p * I)
        @test negI isa DiagTensor{2,2}
        @test negI[1, 1] === -p
        @test negI[2, 2] === -p

        σ = -p * I + tau
        tσ = Tensor{2}(σ)
        @test tσ isa SymTensor{2,2}
        @test tσ[1, 1] === -p + tau[1, 1]
        @test tσ[1, 2] === tau[1, 2]
        @test tσ[2, 2] === -p + tau[2, 2]

        rawI = IdTensor{2,2}()
        left_scaled = @inferred(p * rawI)
        right_scaled = @inferred(rawI * p)
        @test left_scaled isa DiagTensor{2,2}
        @test left_scaled[1, 1] === p
        @test left_scaled[2, 2] === p
        @test right_scaled isa DiagTensor{2,2}
        @test right_scaled[1, 1] === p
        @test right_scaled[2, 2] === p

        zero_broadcast = @inferred Base.Broadcast.broadcasted(sin, ZeroTensor{2,2}())
        @test zero_broadcast isa ZeroTensor{2,2}

        id_broadcast = @inferred Base.Broadcast.broadcasted(sin, rawI)
        @test id_broadcast isa DiagTensor{2,2}
        @test !(id_broadcast isa IdTensor{2,2})
        @test id_broadcast[1, 1] === sin(SUniform(1))
        @test id_broadcast[1, 2] === SUniform(0)

        widened = @inferred Base.Broadcast.broadcasted(exp, ZeroTensor{2,2}())
        @test widened isa SymTensor{2,2}
        @test !(widened isa ZeroTensor{2,2})
        @test widened[1, 1] === exp(SUniform(0))
        @test widened[1, 2] === exp(SUniform(0))

        st = sin.(2tau)
        tst = Tensor{2}(st)
        @test tst[1, 1] === sin(2tau[1, 1])
        @test tst[1, 2] === sin(2tau[1, 2])
        @test tst[2, 2] === sin(2tau[2, 2])

        gu = Tensor{2}(grad(u))
        @test gu isa Tensor{2,2}
        @test gu[1, 2] === grad.op[1](u[2])
        @test gu[2, 1] === grad.op[2](u[1])

        gT = Tensor{2}(grad(T))
        @test gT isa Tensor{2,3}
        @test gT[1, 1, 2] === grad.op[1](T[1, 2])
        @test gT[2, 1, 2] === grad.op[2](T[1, 2])

        divT = Tensor{2}(divg(T))
        @test divT isa Vec{2}
        @test divT[1] === divg.op[1](T[1, 1]) + divg.op[2](T[2, 1])
        @test divT[2] === divg.op[1](T[1, 2]) + divg.op[2](T[2, 2])

        w = SVec(:w)
        curlw = Tensor{3}(curl(w))
        @test curlw isa Vec{3}
        @test curlw[1] === curl.op[2](w[3]) - curl.op[3](w[2])
        @test curlw[2] === curl.op[3](w[1]) - curl.op[1](w[3])
        @test curlw[3] === curl.op[1](w[2]) - curl.op[2](w[1])

        curlu = Tensor{2}(curl(u))
        @test curlu === curl.op[1](u[2]) - curl.op[2](u[1])

        @test_throws ArgumentError Tensor{4}(curl(SVec(:z)))
    end

    @testset "expression tensor rank inference" begin
        a = SScalar(:a)
        u = SVec(:u)
        v = SVec(:v)
        T = SSymTensor{2}(:T)

        @test tensorrank(u ⊗ v) == 2
        @test tensorrank(u ⋅ v) == 0
        @test tensorrank(T ⋅ u) == 1
        @test tensorrank(diag(T)) == 1
        @test tensorrank(gram(T)) == 2
        @test tensorrank(a * u) == 1
    end
end
