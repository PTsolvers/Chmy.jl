using Test
using Chmy
import Chmy: ncomponents, linear_index, dimensions
import Chmy: NoKind, SymKind, AltKind, DiagKind

@testset "tensors" begin
    @testset "symbolic tensors" begin
        @scalars a
        @vectors u
        @tensors 2 @sym(S) @diag(D) @alt(A)

        @test tensorrank(a) == 0
        @test tensorrank(u) == 1
        @test tensorrank(S) == 2
        @test tensorkind(a) === NoKind
        @test tensorkind(S) === SymKind
        @test tensorkind(A) === AltKind
        @test tensorkind(D) === DiagKind
        @test name(u) === :u

        @test a[] === a
        @test SZeroTensor{0}() === SLiteral(0)
        @test SIdTensor{0}() === SLiteral(1)

        @test SZeroTensor{2}()[1, 2] === SLiteral(0)
        @test SIdTensor{2}()[1, 1] === SLiteral(1)
        @test SIdTensor{2}()[1, 2] === SLiteral(0)

        @test S[2, 1] === S[1, 2]
        @test D[1, 2] === SLiteral(0)
        @test D[2, 2] === D[SLiteral(2), SLiteral(2)]
        @test A[1, 1] === SLiteral(0)
        @test A[2, 1] === -A[1, 2]
    end

    @testset "uniform symbolic tensors" begin
        @uniform @scalars a
        @uniform @vectors u
        @uniform @tensors 2 T @sym(S) @diag(D) @alt(A)
        @scalars b
        p, s = Point(), Segment()
        i, j = SIndex(1), SIndex(2)

        @test !isuniform(STensor{0,NoKind}(:a))
        @test STensor{1,NoKind,true}(:u) === SUVec(:u)

        @test isuniform(a)
        @test isuniform(u)
        @test isuniform(T)
        @test isuniform(S)
        @test isuniform(A)
        @test isuniform(D)
        @test isuniform(SZeroTensor{2}())
        @test isuniform(SIdTensor{2}())
        @test isuniform(sin(a) + 1)
        @test isuniform(u[1])
        @test !isuniform(b)
        @test !isuniform(sin(b))
        @test !isuniform(b[p][i])

        @test a[p, s][i, j] === a
        @test sin(a)[p, s][i, j] === sin(a)
        @test u[1][p, s][i, j] === u[1]
        @test_throws ArgumentError u[p]
        @test_throws ArgumentError u[i]
        @test_throws ArgumentError (u + u)[p]
        @test_throws ArgumentError (u + u)[i]
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
        @scalars a b c d

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
        @test_throws ArgumentError t[]
        @test t[SLiteral(2), SLiteral(1)] === b
        @test t[2, 1] === t[SLiteral(2), SLiteral(1)]

        sym = Tensor{2,2}(a, b, b, c)
        @test sym isa SymTensor{2,2}
        @test length(sym) == 3
        @test sym[1, 2] === b
        @test sym[2, 1] === b

        diag = Tensor{2,2}(a, SLiteral(0), SLiteral(0), c)
        @test diag isa DiagTensor{2,2}
        @test diag[1, 1] === a
        @test diag[1, 2] === SLiteral(0)
        @test diag[2, 2] === c

        alt = Tensor{3,2}(SLiteral(0), -a, -b, a, SLiteral(0), -c, b, c, SLiteral(0))
        @test alt isa AltTensor{3,2}
        @test alt[1, 2] === a
        @test alt[2, 1] === -a
        @test alt[1, 1] === SLiteral(0)

        @test Tensor{2,2}(SLiteral(1), SLiteral(0), SLiteral(0), SLiteral(1)) isa IdTensor{2,2}
        @test Tensor{2,2,DiagKind}(SLiteral(0), SLiteral(0)) isa ZeroTensor{2,2}

        wide_zero_sym = Chmy.tensor_with_kind(Tensor{2,2,SymKind}, SZeroTensor{2}())
        @test wide_zero_sym isa SymTensor{2,2}
        @test !(wide_zero_sym isa ZeroTensor{2,2})
        @test wide_zero_sym.components === (SLiteral(0), SLiteral(0), SLiteral(0))

        wide_id_sym = Chmy.tensor_with_kind(Tensor{2,2,SymKind}, SIdTensor{2}())
        @test wide_id_sym isa SymTensor{2,2}
        @test wide_id_sym.components === (SLiteral(1), SLiteral(0), SLiteral(1))

        diag_data = DiagTensor{2,2}(a, d)
        wide_diag_sym = Chmy.tensor_with_kind(Tensor{2,2,SymKind}, diag_data)
        @test wide_diag_sym isa SymTensor{2,2}
        @test wide_diag_sym.components === (a, SLiteral(0), d)

        wide_zero_alt = Chmy.tensor_with_kind(Tensor{3,2,AltKind}, SZeroTensor{2}())
        @test wide_zero_alt isa AltTensor{3,2}
        @test wide_zero_alt.components === (SLiteral(0), SLiteral(0), SLiteral(0))

        @test_throws ErrorException Tensor{2,2,SymKind}(a, b)
    end

    @testset "symbolic tensor expansion" begin
        @tensors 2 @sym(S) @diag(D) @alt(A)

        ts = Tensor{2}(S)
        ta = Tensor{3}(A)
        td = Tensor{3}(D)

        @test ts isa SymTensor{2,2}
        @test ta isa AltTensor{3,2}
        @test td isa DiagTensor{3,2}

        @test ts[2, 1] === S[1, 2]
        @test ta[1, 2] === A[1, 2]
        @test ta[2, 1] === -A[1, 2]
        @test ta[1, 1] === SLiteral(0)
        @test td[1, 2] === SLiteral(0)
        @test td[3, 3] === D[3, 3]
    end

    @testset "tensor expression expansion" begin
        @scalars a
        @vectors u v
        @tensors 2 @sym(S)

        expr = S ⋅ u + 2 * v
        tex = Tensor{2}(expr)

        @test tex isa Vec{2}
        @test tex[1] === S[1, 1] * u[1] + S[1, 2] * u[2] + 2 * v[1]
        @test tex[2] === S[1, 2] * u[1] + S[2, 2] * u[2] + 2 * v[2]

        @test Tensor{3}(S[1, 1]) === S[1, 1]
        @test Tensor{3}(S[1, 1] + S[2, 2]) === S[1, 1] + S[2, 2]

        @test Tensor{4}(SLiteral(3)) === SLiteral(3)
        @test Tensor{4}(a) === a

        I = SIdTensor{2}()
        negI = @inferred Tensor{2}(-a * I)
        @test negI isa DiagTensor{2,2}
        @test negI[1, 1] === -a
        @test negI[2, 2] === -a

        stress = -a * I + S
        @test stress[1, 1] === -a + S[1, 1]
        @test stress[1, 2] === S[1, 2]
        @test (v / a)[1] === v[1] / a
        tstress = Tensor{2}(stress)
        @test tstress isa SymTensor{2,2}
        @test tstress[1, 1] === -a + S[1, 1]
        @test tstress[1, 2] === S[1, 2]
        @test tstress[2, 2] === -a + S[2, 2]

        rawI = IdTensor{2,2}()
        left_scaled = @inferred(a * rawI)
        right_scaled = @inferred(rawI * a)
        @test left_scaled isa DiagTensor{2,2}
        @test left_scaled[1, 1] === a
        @test left_scaled[2, 2] === a
        @test right_scaled isa DiagTensor{2,2}
        @test right_scaled[1, 1] === a
        @test right_scaled[2, 2] === a

        zero_broadcast = @inferred Base.Broadcast.broadcasted(sin, ZeroTensor{2,2}())
        @test zero_broadcast isa ZeroTensor{2,2}

        id_broadcast = @inferred Base.Broadcast.broadcasted(sin, rawI)
        @test id_broadcast isa DiagTensor{2,2}
        @test !(id_broadcast isa IdTensor{2,2})
        @test id_broadcast[1, 1] === sin(SLiteral(1))
        @test id_broadcast[1, 2] === SLiteral(0)

        widened = @inferred Base.Broadcast.broadcasted(exp, ZeroTensor{2,2}())
        @test widened isa SymTensor{2,2}
        @test !(widened isa ZeroTensor{2,2})
        @test widened[1, 1] === exp(SLiteral(0))
        @test widened[1, 2] === exp(SLiteral(0))

        st = sin.(2S)
        tst = Tensor{2}(st)
        @test tst[1, 1] === sin(2S[1, 1])
        @test tst[1, 2] === sin(2S[1, 2])
        @test tst[2, 2] === sin(2S[2, 2])
    end

    @testset "uniform compute bindings" begin
        @uniform @scalars a
        @uniform @vectors u
        p, s = Point(), Segment()
        i, j = SIndex(1), SIndex(2)

        expr = (a + 1)[p, s][i, j]
        component = u[1][p][i]

        @test expr === a + 1
        @test component === u[1]
        @test compute(a[p, s][i, j], Binding(a => 2.0), 3, 4) == 2.0
        @test compute(component, Binding(component => 5.0), 3) == 5.0
    end

    @testset "expression tensor rank inference" begin
        @scalars a
        @vectors u v
        @tensors 2 @sym(S)

        @test tensorrank(u ⊗ v) == 2
        @test tensorrank(u ⋅ v) == 0
        @test tensorrank(S ⋅ u) == 1
        @test tensorrank(diag(S)) == 1
        @test tensorrank(gram(S)) == 2
        @test tensorrank(a * u) == 1
    end
end
