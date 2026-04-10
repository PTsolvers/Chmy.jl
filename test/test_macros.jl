@testset "macros" begin
    @testset "scalar and vector declarations" begin
        @scalars p T
        @vectors V q

        @test p === SScalar(:p)
        @test T === SScalar(:T)
        @test V === SVec(:V)
        @test q === SVec(:q)
    end

    @testset "tensor declarations" begin
        @tensors 2 A @sym(tau, eps) @diag(G, H) @alt(W, Z) @id(I, J) @zero(O, P)

        @test A === STensor{2}(:A)
        @test tau === SSymTensor{2}(:tau)
        @test eps === SSymTensor{2}(:eps)
        @test G === SDiagTensor{2}(:G)
        @test H === SDiagTensor{2}(:H)
        @test W === SAltTensor{2}(:W)
        @test Z === SAltTensor{2}(:Z)
        @test I === SIdTensor{2}()
        @test J === SIdTensor{2}()
        @test O === SZeroTensor{2}()
        @test P === SZeroTensor{2}()
    end

    @testset "expansion shape" begin
        ex = @macroexpand @uniform @tensors 2 @sym τ
        @test ex == Expr(:block, Expr(:(=), :τ, Expr(:call, Expr(:curly, GlobalRef(Chmy, :STensor), 2, GlobalRef(Chmy, :SymKind), true, QuoteNode(:τ)))))
    end

    @testset "uniform declarations" begin
        @uniform @scalars dx dy
        @uniform @vectors u v
        @uniform @tensors 2 H @sym(S) @diag(D) @alt(B) @id(I) @zero(O)

        @test dx === SUScalar(:dx)
        @test dy === SUScalar(:dy)
        @test u === SUVec(:u)
        @test v === SUVec(:v)
        @test H === STensor{2,Chmy.NoKind,true}(:H)
        @test S === SUSymTensor{2}(:S)
        @test D === SUDiagTensor{2}(:D)
        @test B === SUAltTensor{2}(:B)
        @test I === SIdTensor{2}()
        @test O === SZeroTensor{2}()
    end

    @testset "inline uniform groups" begin
        @scalars a b @uniform(c, d)
        @vectors x @uniform(y)
        @tensors 2 T @sym(sigma) @uniform(U, @uniform(V), @sym(Su, Sv), @uniform(@diag(Du, Dv)), @alt(Au, Av), @id(Iu), @zero(Ou))

        @test a === SScalar(:a)
        @test b === SScalar(:b)
        @test c === SUScalar(:c)
        @test d === SUScalar(:d)
        @test x === SVec(:x)
        @test y === SUVec(:y)
        @test T === STensor{2}(:T)
        @test sigma === SSymTensor{2}(:sigma)
        @test U === STensor{2,Chmy.NoKind,true}(:U)
        @test V === STensor{2,Chmy.NoKind,true}(:V)
        @test Su === SUSymTensor{2}(:Su)
        @test Sv === SUSymTensor{2}(:Sv)
        @test Du === SUDiagTensor{2}(:Du)
        @test Dv === SUDiagTensor{2}(:Dv)
        @test Au === SUAltTensor{2}(:Au)
        @test Av === SUAltTensor{2}(:Av)
        @test Iu === SIdTensor{2}()
        @test Ou === SZeroTensor{2}()
    end

    @testset "uniform blocks" begin
        @uniform begin
            @scalars x y
            @vectors v q
            @tensors 2 A @sym(B, C) @diag(D) @alt(E) @id(I) @zero(O)
        end

        @test x === SUScalar(:x)
        @test y === SUScalar(:y)
        @test v === SUVec(:v)
        @test q === SUVec(:q)
        @test A === STensor{2,Chmy.NoKind,true}(:A)
        @test B === SUSymTensor{2}(:B)
        @test C === SUSymTensor{2}(:C)
        @test D === SUDiagTensor{2}(:D)
        @test E === SUAltTensor{2}(:E)
        @test I === SIdTensor{2}()
        @test O === SZeroTensor{2}()
    end

    @testset "local scope" begin
        let
            @tensors 2 F @uniform(UF)
            @test F === STensor{2}(:F)
            @test UF === STensor{2,Chmy.NoKind,true}(:UF)
        end

        let
            @uniform @tensors 2 @sym(localS)
            @test localS === SUSymTensor{2}(:localS)
        end
    end

    @testset "invalid usage" begin
        @test_throws ArgumentError @macroexpand @scalars
        @test_throws ArgumentError @macroexpand @vectors
        @test_throws ArgumentError @macroexpand @tensors
        @test_throws ArgumentError @macroexpand @tensors 2
        @test_throws ArgumentError @macroexpand @scalars @sym(a)
        @test_throws ArgumentError @macroexpand @vectors @diag(v)
        @test_throws ArgumentError @macroexpand @scalars x 1
        @test_throws ArgumentError @macroexpand @tensors 2 1
        @test_throws ArgumentError @macroexpand @tensors R F
        @test_throws ArgumentError @macroexpand @tensors -1 T
        @test_throws ArgumentError @macroexpand @tensors 2.0 T
        @test_throws ArgumentError @macroexpand @tensors 2 @sym(@diag(T))
        @test_throws ArgumentError @macroexpand @uniform(x, y)
        @test_throws ArgumentError @macroexpand @uniform x y
        @test_throws ArgumentError @macroexpand @uniform @sym(T)
        @test_throws ArgumentError @macroexpand @uniform begin
            x = 1
        end
        @test_throws ArgumentError @macroexpand @sym(T)
        @test_throws ArgumentError @macroexpand @diag(T)
        @test_throws ArgumentError @macroexpand @alt(T)
        @test_throws ArgumentError @macroexpand @id(T)
        @test_throws ArgumentError @macroexpand @zero(T)
    end
end
