using Test
using Chmy
using Chmy: Kind
import LinearAlgebra

# a scalar operator rewritten depending on the number of dimensions
struct Scaled <: Operator end
Chmy.evaluate(::Scaled, args::Tuple{DTerm}, dims::Integer) = only(args) / dims

@testset "tensor components" begin
    @scalars a b c
    @vectors u v w
    @tensors 2 A B @sym(S) @alt(K) @diag(D)

    @testset "storage and indexing" begin
        uc = @inferred components(u, 2)
        @test isequal(entries(uc), (u[1], u[2]))
        for (leaf, kind, data) in (
                (A, Kind.None(), (A[1, 1], A[2, 1], A[1, 2], A[2, 2])),
                (S, Kind.Sym(), (S[1, 1], S[1, 2], S[2, 2])),
                (K, Kind.Alt(), (K[1, 2],)),
                (D, Kind.Diag(), (D[1, 1], D[2, 2])),
            )
            tc = @inferred components(leaf, 2)
            @test tensorkind(tc) == kind
            @test isequal(entries(tc), data)
            @test (@inferred tc[1, 1]) ==ₛ leaf[1, 1]
            @test (@inferred tc[2, 1]) ==ₛ leaf[2, 1]
        end
        identity = @inferred components(ℐ(2), 2)
        @test identity[1, 1] ==ₛ Literal(1)
        @test identity[1, 2] ==ₛ Literal(0)
        zero = @inferred components(𝒪(1), 2)
        @test isequal(entries(zero), (Literal(0), Literal(0)))
        @test_throws ArgumentError components(A, 2)[1]
        @test_throws BoundsError uc[3]
    end

    @testset "kind promotion and detection" begin
        @test tensorkind(components(S + D, 2)) == Kind.Sym()
        @test tensorkind(components(S + K, 2)) == Kind.None()
        z = Literal(0)
        for (data, kind) in (
                (DTerm[a, b, b, c], Kind.Sym()),
                (DTerm[z, -a, a, z], Kind.Alt()),
                (DTerm[a, z, z, c], Kind.Diag()),
                (DTerm[a, b, c, a], Kind.None()),
            )
            tc = @inferred simplify(pack(reshape(data, 2, 2)))
            @test tensorkind(tc) == kind
            @test isequal(DTerm[tc[i, j] for i in 1:2, j in 1:2], reshape(data, 2, 2))
        end
    end

    @testset "arithmetic and contractions" begin
        for (input, expected) in (
                (-v, -v[1]),
                (u - v, u[1] - v[1]),
                (u + v + w, u[1] + v[1] + w[1]),
                (a * v, a * v[1]),
                (v / a, v[1] / a),
                (v // a, v[1] // a),
                (v ÷ a, v[1] ÷ a),
                ((A + B) ⋅ v, (A[1, 1] + B[1, 1]) * v[1] + (A[1, 2] + B[1, 2]) * v[2]),
                (u ⋅ A, u[1] * A[1, 1] + u[2] * A[2, 1]),
            )
            tc = @inferred components(input, 2)
            @test tc[1] ==ₛ expected
        end
        @test (@inferred components(u ⋅ v, 2)) ==ₛ u[1] * v[1] + u[2] * v[2]
        product = @inferred components(A ⋅ B, 2)
        @test product[1, 2] ==ₛ A[1, 1] * B[1, 2] + A[1, 2] * B[2, 2]
        outer = @inferred components(u ⊗ v, 2)
        @test outer[1, 2] ==ₛ u[1] * v[2]
        @test (@inferred components(A ⊡ B, 2)) ==ₛ A[1, 1] * B[1, 1] + A[2, 1] * B[2, 1] + A[1, 2] * B[1, 2] + A[2, 2] * B[2, 2]

        cross = @inferred components(u × v, 3)
        @test cross[1] ==ₛ u[2] * v[3] - u[3] * v[2]
        @test cross[2] ==ₛ u[3] * v[1] - u[1] * v[3]
        @test cross[3] ==ₛ u[1] * v[2] - u[2] * v[1]
        @test_throws ArgumentError components(u × v, 2)
    end

    @testset "idempotence" begin
        for input in (sin.(A .+ a) .* B .^ 2, A ⋅ B, u ⊗ v)
            tc = components(input, 2)
            @test (@inferred components(tc[1, 2], 2)) ==ₛ tc[1, 2]
        end
        scalar = sin(a) + b * u[1]
        @test (@inferred components(scalar, 2)) ==ₛ scalar
        @test components(scalar, 2) === scalar
    end

    @testset "dimension-dependent scalar operators" begin
        @test (@inferred components(Scaled()(a), 2)) ==ₛ a / 2
        @test components(sin(Scaled()(a)), 3) ==ₛ sin(a / 3)
    end

    @testset "matrix operations" begin
        matrix = [4 1; 2 3]
        tc = pack(matrix)
        @test value(simplify(@inferred tr(tc))) == LinearAlgebra.tr(matrix)
        diagonal = @inferred diag(tc)
        @test [value(simplify(diagonal[i])) for i in 1:2] == LinearAlgebra.diag(matrix)
        for (op, expected) in (
                (transpose, transpose(matrix)),
                (sym, (matrix + transpose(matrix)) / 2),
                (asym, (matrix - transpose(matrix)) / 2),
                (gram, transpose(matrix) * matrix),
                (cogram, matrix * transpose(matrix)),
                (inv, LinearAlgebra.inv(matrix)),
            )
            result = @inferred op(tc)
            @test [value(simplify(result[i, j])) for i in 1:2, j in 1:2] ≈ expected
        end

        complex_matrix = [1 + im 2 - im; 3 + 2im 4 - im]
        complex_tc = pack(complex_matrix)
        transposed = @inferred adjoint(complex_tc)
        @test value(simplify(transposed[1, 2])) == complex_matrix[2, 1]
    end

    @testset "dimension-specific determinant, cofactor, and adjugate" begin
        matrix = [4 1 2; 0 3 1; 2 0 5]
        for dims in 1:3
            m = matrix[1:dims, 1:dims]
            tc = pack(m)
            @test value(simplify(@inferred det(tc))) ≈ LinearAlgebra.det(m)
            cofactor = @inferred cof(tc)
            @test [value(simplify(cofactor[i, j])) for i in 1:dims, j in 1:dims] ≈ LinearAlgebra.det(m) * transpose(LinearAlgebra.inv(m))
            adjugate = @inferred adj(tc)
            @test [value(simplify(adjugate[i, j])) for i in 1:dims, j in 1:dims] ≈ LinearAlgebra.det(m) * LinearAlgebra.inv(m)
        end
    end
end
