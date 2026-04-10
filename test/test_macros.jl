@testset "macros" begin
    @testset "scalar and vector declarations" begin
        @scalars a b
        @vectors u v

        @test a === SScalar(:a)
        @test b === SScalar(:b)
        @test u === SVec(:u)
        @test v === SVec(:v)
    end

    @testset "tensor declarations" begin
        @tensors 2 T @sym(S, R) @diag(D, E) @alt(A, B) @id(I, J) @zero(O, P)

        @test T === STensor{2}(:T)
        @test S === SSymTensor{2}(:S)
        @test R === SSymTensor{2}(:R)
        @test D === SDiagTensor{2}(:D)
        @test E === SDiagTensor{2}(:E)
        @test A === SAltTensor{2}(:A)
        @test B === SAltTensor{2}(:B)
        @test I === SIdTensor{2}()
        @test J === SIdTensor{2}()
        @test O === SZeroTensor{2}()
        @test P === SZeroTensor{2}()
    end

    @testset "expansion shape" begin
        ex = @macroexpand @uniform @tensors 2 @sym S
        @test ex == Expr(:block, Expr(:(=), :S, Expr(:call, Expr(:curly, GlobalRef(Chmy, :STensor), 2, GlobalRef(Chmy, :SymKind), true, QuoteNode(:S)))))
    end

    @testset "uniform declarations" begin
        @uniform @scalars a b
        @uniform @vectors u v
        @uniform @tensors 2 T @sym(S) @diag(D) @alt(A) @id(I) @zero(O)

        @test a === SUScalar(:a)
        @test b === SUScalar(:b)
        @test u === SUVec(:u)
        @test v === SUVec(:v)
        @test T === STensor{2,Chmy.NoKind,true}(:T)
        @test S === SUSymTensor{2}(:S)
        @test D === SUDiagTensor{2}(:D)
        @test A === SUAltTensor{2}(:A)
        @test I === SIdTensor{2}()
        @test O === SZeroTensor{2}()
    end

    @testset "inline uniform groups" begin
        @scalars a b @uniform(c, d)
        @vectors u @uniform(v)
        @tensors 2 T @sym(S) @uniform(U, @uniform(V), @sym(R, Q), @uniform(@diag(D, E)), @alt(A, B), @id(I), @zero(O))

        @test a === SScalar(:a)
        @test b === SScalar(:b)
        @test c === SUScalar(:c)
        @test d === SUScalar(:d)
        @test u === SVec(:u)
        @test v === SUVec(:v)
        @test T === STensor{2}(:T)
        @test S === SSymTensor{2}(:S)
        @test U === STensor{2,Chmy.NoKind,true}(:U)
        @test V === STensor{2,Chmy.NoKind,true}(:V)
        @test R === SUSymTensor{2}(:R)
        @test Q === SUSymTensor{2}(:Q)
        @test D === SUDiagTensor{2}(:D)
        @test E === SUDiagTensor{2}(:E)
        @test A === SUAltTensor{2}(:A)
        @test B === SUAltTensor{2}(:B)
        @test I === SIdTensor{2}()
        @test O === SZeroTensor{2}()
    end

    @testset "uniform blocks" begin
        @uniform begin
            @scalars a b
            @vectors u v
            @tensors 2 T @sym(S, R) @diag(D) @alt(A) @id(I) @zero(O)
        end

        @test a === SUScalar(:a)
        @test b === SUScalar(:b)
        @test u === SUVec(:u)
        @test v === SUVec(:v)
        @test T === STensor{2,Chmy.NoKind,true}(:T)
        @test S === SUSymTensor{2}(:S)
        @test R === SUSymTensor{2}(:R)
        @test D === SUDiagTensor{2}(:D)
        @test A === SUAltTensor{2}(:A)
        @test I === SIdTensor{2}()
        @test O === SZeroTensor{2}()
    end

    @testset "local scope" begin
        let
            @tensors 2 T @uniform(U)
            @test T === STensor{2}(:T)
            @test U === STensor{2,Chmy.NoKind,true}(:U)
        end

        let
            @uniform @tensors 2 @sym(S)
            @test S === SUSymTensor{2}(:S)
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
