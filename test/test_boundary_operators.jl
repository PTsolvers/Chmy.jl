using Test
using Chmy

struct BoundaryShift{O} <: STerm end
struct BoundaryShiftPair{O} <: STerm end
struct MixedLocationShift <: STerm end
struct BoundaryNormalCornerShift <: STerm end

(op::BoundaryShift)(arg::STerm) = SExpr(Call(), op, arg)
(op::BoundaryShiftPair)(args::Vararg{STerm}) = SExpr(Call(), op, args...)
(op::MixedLocationShift)(args::Vararg{STerm}) = SExpr(Call(), op, args...)
(op::BoundaryNormalCornerShift)(args::Vararg{STerm}) = SExpr(Call(), op, args...)

Chmy.tensorrank(::BoundaryShift, ::STerm) = 0
Chmy.tensorrank(::BoundaryShiftPair, ::STerm, ::STerm) = 0
Chmy.tensorrank(::MixedLocationShift, ::STerm, ::STerm) = 0
Chmy.tensorrank(::BoundaryNormalCornerShift, ::STerm, ::STerm) = 0

boundary_shift_index(ind, offset) = offset == 0 ? ind : (offset > 0 ? ind + offset : ind - (-offset))

function Chmy.stencil_rule(::BoundaryShift{O}, args::Tuple{STerm}, loc::NTuple{N,Space}, inds::NTuple{N,STerm}) where {O,N}
    arg = only(args)
    shifted = ntuple(i -> boundary_shift_index(inds[i], O[i]), Val(N))
    return arg[loc...][shifted...]
end

function Chmy.stencil_rule(::BoundaryShiftPair{O}, args::Tuple{STerm,STerm}, loc::NTuple{N,Space}, inds::NTuple{N,STerm}) where {O,N}
    a, b = args
    shifted = ntuple(i -> boundary_shift_index(inds[i], O[i]), Val(N))
    return a[loc...][shifted...] + b[loc...][shifted...]
end

function Chmy.stencil_rule(::MixedLocationShift, args::Tuple{STerm,STerm}, loc::NTuple{2,Space}, inds::NTuple{2,STerm})
    a, b = args
    i, j = inds
    return a[Point(), Segment()][i+1, j-1] + b[Segment(), Point()][i+1, j+1]
end

function Chmy.stencil_rule(::BoundaryNormalCornerShift, args::Tuple{STerm,STerm}, loc::NTuple{2,Space}, inds::NTuple{2,STerm})
    Vx, Vy = args
    i, j = inds
    return Vx[loc...][i+1, j] + Vy[loc...][i, j+1]
end

@testset "boundary operators" begin
    @testset "shift arithmetic" begin
        @test @inferred(Shift(1) + Shift(2)) === Shift(3)
        @test @inferred(Shift(1) + Shift(2) - Shift(1)) === Shift(2)
        @test @inferred(Shift(1) - Shift(3)) === Shift(-2)
        @test @inferred(-Shift(1)) === Shift(-1)
        @test_throws ArgumentError Shift(1 // 2)
        @test_throws ArgumentError δ(1 // 2)
        @test offset(Point()) === SLiteral(0)
        @test offset(Segment()) === SLiteral(1 // 2)
    end

    @testset "adjacent_faces" begin
        @test @inferred(adjacent_faces(Face(Lower()))) === (Face(Span()),)
        @test @inferred(adjacent_faces(Face(Upper()))) === (Face(Span()),)
        @test @inferred(adjacent_faces(Face(Span()))) === ()

        @test @inferred(adjacent_faces(Face(Lower(), Lower()))) === (Face(Span(), Lower()),
                                                                     Face(Lower(), Span()))

        @test @inferred(adjacent_faces(Face(Lower(), Span()))) === (Face(Span(), Span()),)

        @test @inferred(adjacent_faces(Face(Span(), Upper()))) === (Face(Span(), Span()),)

        @test @inferred(adjacent_faces(Face(Lower(), Lower(), Lower()))) === (Face(Span(), Lower(), Lower()),
                                                                              Face(Lower(), Span(), Lower()),
                                                                              Face(Lower(), Lower(), Span()))

        @test @inferred(adjacent_faces(Face(Lower(), Lower(), Span()))) === (Face(Span(), Lower(), Span()),
                                                                             Face(Lower(), Span(), Span()))

        @test @inferred(adjacent_faces(Face(Span(), Span(), Span()))) === ()
    end

    @testset "nonuniform staggered shifts" begin
        @scalars a b
        i, j = SIndex(1), SIndex(2)

        nu1 = nonuniforms(a[Segment()][i])
        @test sprint(show, Chmy.stencil(nu1, a[Segment()])) == "Stencil(Segment(), δ(0))"

        nu1_merged = nonuniforms(a[Point(), Segment()][i, j] + a[Point(), Segment()][i, j+1])
        @test sprint(show, Chmy.stencil(nu1_merged, a[Point(), Segment()])) == "Stencil((Point(), Segment()), δ(0, 0), δ(0, 1))"

        nu2 = nonuniforms(a[Segment(), Point()][i+1, j] + b[Point(), Segment()][i, j-1])
        @test sprint(show, Chmy.stencil(nu2, a[Segment(), Point()])) == "Stencil((Segment(), Point()), δ(1, 0))"
        @test sprint(show, Chmy.stencil(nu2, b[Point(), Segment()])) == "Stencil((Point(), Segment()), δ(0, -1))"
    end

    @testset "extension constructors" begin
        @scalars a b
        @vectors V

        @test @inferred(PolynomialReconstruction()) === LinearReconstruction()
        @test @inferred(PolynomialReconstruction(1)) === LinearReconstruction()
        @test @inferred(tensorrank(BoundaryNormal())) == 1
        @test isuniform(BoundaryNormal())

        value_spec = @inferred ExtensionSpec(b)
        @test value_spec.data.value === b
        @test value_spec.op === LinearReconstruction()

        data_spec = @inferred ExtensionSpec(ValueData(b))
        @test data_spec.data.value === b
        @test data_spec.op === LinearReconstruction()

        derivative_spec = @inferred ExtensionSpec(DerivativeData(b))
        @test derivative_spec.data.value === b
        @test derivative_spec.op === LinearReconstruction()

        explicit_spec = @inferred ExtensionSpec(ValueData(b), PolynomialReconstruction())
        @test explicit_spec.data.value === b
        @test explicit_spec.op === LinearReconstruction()

        rule = @inferred ExtensionRule(a => b)
        @test rule.specs[a].data.value === b
        @test rule.specs[a].op === LinearReconstruction()

        tensor_rule = @inferred ExtensionRule(V => SZeroTensor{1}())
        @test tensor_rule.specs[V].data.value === SZeroTensor{1}()
        @test tensor_rule.specs[V].op === LinearReconstruction()
    end

    @testset "extension value reconstruction" begin
        @scalars a b c
        i = SIndex(1)
        D = CentralDifference()

        point_rule = ExtensionRule(a => b)

        @test @inferred(boundary_rule(point_rule, D(a), Face(Upper()), (Point(),), (Shift(0),))) ===
              1 // 2 * (2 * b[Point()][i] - 2 * a[Point()][i-1])

        @test boundary_rule(point_rule, D(a), Face(Lower()), (Point(),), (Shift(0),)) ===
              1 // 2 * (2 * a[Point()][i+1] - 2 * b[Point()][i])

        @test boundary_rule(point_rule, a + D(a), Face(Upper()), (Point(),), (Shift(0),)) ===
              b[Point()][i] + 1 // 2 * (2 * b[Point()][i] - 2 * a[Point()][i-1])

        expr_rule = ExtensionRule(a => b + c)
        @test boundary_rule(expr_rule, BoundaryShift{(1,)}()(a), Face(Upper()), (Point(),), (Shift(0),)) ===
              2 * (b[Point()][i] + c[Point()][i]) - a[Point()][i-1]

        segment_rule = ExtensionRule(a => b)

        @test boundary_rule(segment_rule, D(a), Face(Upper()), (Segment(),), (Shift(-1),)) ===
              1 // 2 * (2 * b[Point()][i+1] - a[Segment()][i] - a[Segment()][i-1])

        @test boundary_rule(segment_rule, D(a), Face(Upper()), (Segment(),), (Shift(-2),)) ===
              1 // 2 * (a[Segment()][i+1] - a[Segment()][i-1])
    end

    @testset "extension derivative reconstruction" begin
        @scalars a b
        i = SIndex(1)
        D = CentralDifference()

        point_rule = ExtensionRule(a => ExtensionSpec(DerivativeData(b)))

        @test @inferred(boundary_rule(point_rule, D(a), Face(Upper()), (Point(),), (Shift(0),))) === b[Point()][i]
        @test boundary_rule(point_rule, D(a), Face(Lower()), (Point(),), (Shift(0),)) === -b[Point()][i]

        segment_rule = ExtensionRule(a => ExtensionSpec(DerivativeData(b)))

        @test boundary_rule(segment_rule, D(a), Face(Upper()), (Segment(),), (Shift(-1),)) ===
              1 // 2 * (b[Point()][i+1] + a[Segment()][i] - a[Segment()][i-1])
    end

    @testset "extension multiple fields" begin
        @scalars a b c d
        i, j = SIndex(1), SIndex(2)
        D = CentralDifference()

        rule = ExtensionRule(a => c, b => d)

        @test @inferred(boundary_rule(rule, D(a) + D(b), Face(Upper()), (Point(),), (Shift(0),))) ===
              1 // 2 * (2 * c[Point()][i] - 2 * a[Point()][i-1]) +
              1 // 2 * (2 * d[Point()][i] - 2 * b[Point()][i-1])

        @test boundary_rule(rule, D(a), Face(Upper()), (Point(),), (Shift(0),)) ===
              1 // 2 * (2 * c[Point()][i] - 2 * a[Point()][i-1])

        shifted = MixedLocationShift()(a, b)
        @test @inferred(boundary_rule(rule, shifted, Face(Upper(), Span()),
                                      (Point(), Segment()), (Shift(0), Shift(0)))) ===
              2 * c[Point(), Segment()][i, j-1] - a[Point(), Segment()][i-1, j-1] +
              4 * d[Point(), Point()][i, j+1] - 3 * b[Segment(), Point()][i-1, j+1]
    end

    @testset "extension tensor boundary conditions" begin
        @vectors V
        @scalars b c
        @tensors 2 σ
        @tensors 2 @sym(S) @diag(D) @alt(A)
        i, j, k = SIndex(1), SIndex(2), SIndex(3)
        Z = SZeroTensor{1}()
        N = BoundaryNormal()

        projected = @inferred project_boundary(V ⋅ N, Chmy.FaceNormal{1}())
        @test @inferred(Tensor{2}(projected)) === V[1]

        projected_rule = @inferred project_boundary(ExtensionRule(V ⋅ N => b), Chmy.FaceNormal{1}())
        @test projected_rule isa ExtensionRule
        @test @inferred(Tensor{2}(projected_rule)).specs[V[1]].data.value === b

        scalarized_rule = @inferred Tensor{2}(ExtensionRule(V => Z))
        @test scalarized_rule.specs[V[1]].data.value === SLiteral(0)
        @test scalarized_rule.specs[V[2]].data.value === SLiteral(0)

        sym_rule = @inferred Tensor{2}(ExtensionRule(S => SZeroTensor{2}()))
        sym_pairs = pairstuple(sym_rule.specs)
        @test map(first, sym_pairs) === (S[1, 1], S[1, 2], S[2, 2])
        @test map(p -> p.second.data.value, sym_pairs) === (SLiteral(0), SLiteral(0), SLiteral(0))

        diag_rule = @inferred Tensor{2}(ExtensionRule(D => SZeroTensor{2}()))
        diag_pairs = pairstuple(diag_rule.specs)
        @test map(first, diag_pairs) === (D[1, 1], D[2, 2])
        @test map(p -> p.second.data.value, diag_pairs) === (SLiteral(0), SLiteral(0))

        alt_rule = @inferred Tensor{3}(ExtensionRule(A => SZeroTensor{2}()))
        alt_pairs = pairstuple(alt_rule.specs)
        @test map(first, alt_pairs) === (A[1, 2], A[1, 3], A[2, 3])
        @test map(p -> p.second.data.value, alt_pairs) === (SLiteral(0), SLiteral(0), SLiteral(0))

        normal_projected = @inferred Tensor{2}(project_boundary(ExtensionRule(V ⋅ N => b), Chmy.FaceNormal{1}()))
        @test normal_projected.specs[V[1]].data.value === b

        zero_vec_rule = ExtensionRule(V => Z)
        shifted2 = BoundaryShiftPair{(1, 0)}()(V[1], V[2])
        @test @inferred(boundary_rule(zero_vec_rule, shifted2, Face(Upper(), Span()),
                                      (Point(), Point()), (Shift(0), Shift(0)))) ===
              -V[1][Point(), Point()][i-1, j] - V[2][Point(), Point()][i-1, j]

        shifted3 = BoundaryShift{(0, 0, 1)}()(V[3])
        @test boundary_rule(zero_vec_rule, shifted3, Face(Span(), Span(), Upper()),
                            (Point(), Point(), Point()), (Shift(0), Shift(0), Shift(0))) ===
              -V[3][Point(), Point(), Point()][i, j, k-1]

        normal_value = ExtensionRule(V ⋅ N => SLiteral(0))
        @test @inferred(boundary_rule(normal_value, BoundaryShift{(1, 0)}()(V[1]), Face(Upper(), Span()),
                                      (Point(), Point()), (Shift(0), Shift(0)))) ===
              -V[1][Point(), Point()][i-1, j]
        @test boundary_rule(normal_value, BoundaryShift{(-1, 0)}()(V[1]), Face(Lower(), Span()),
                            (Point(), Point()), (Shift(0), Shift(0))) ===
              -V[1][Point(), Point()][i+1, j]

        traction = ExtensionRule(σ ⋅ N => Z)
        shifted_traction = BoundaryShiftPair{(1, 0)}()(σ[1, 1], σ[2, 1])
        @test @inferred(boundary_rule(traction, shifted_traction, Face(Upper(), Span()),
                                      (Point(), Point()), (Shift(0), Shift(0)))) ===
              -σ[1, 1][Point(), Point()][i-1, j] - σ[2, 1][Point(), Point()][i-1, j]

        shifted_lower_traction = BoundaryShiftPair{(-1, 0)}()(σ[1, 1], σ[2, 1])
        @test boundary_rule(traction, shifted_lower_traction, Face(Lower(), Span()),
                            (Point(), Point()), (Shift(0), Shift(0))) ===
              -σ[1, 1][Point(), Point()][i+1, j] - σ[2, 1][Point(), Point()][i+1, j]

        corner_rules = Binding(Face(Upper(), Span()) => ExtensionRule(V ⋅ N => b),
                               Face(Span(), Upper()) => ExtensionRule(V ⋅ N => c))
        corner = @inferred Chmy.combine_rules(corner_rules, Face(Upper(), Upper()))
        @test @inferred(boundary_rule(corner, BoundaryNormalCornerShift()(V[1], V[2]), Face(Upper(), Upper()),
                                      (Point(), Point()), (Shift(0), Shift(0)))) ===
              2 * b[Point(), Point()][i, j] - V[1][Point(), Point()][i-1, j] +
              2 * c[Point(), Point()][i, j] - V[2][Point(), Point()][i, j-1]
    end

    @testset "extension lifting and combined rules" begin
        @scalars a b c d e f q
        i, j, k = SIndex(1), SIndex(2), SIndex(3)

        xrule = ExtensionRule(a => b)
        yrule = ExtensionRule(a => c)

        shifted_tangent = BoundaryShift{(1, -1)}()(a)
        @test @inferred(boundary_rule(xrule, shifted_tangent, Face(Upper(), Span()),
                                      (Point(), Point()), (Shift(0), Shift(-1)))) ===
              2 * b[Point(), Point()][i, j-1] - a[Point(), Point()][i-1, j-1]

        rules = Binding(Face(Upper(), Span()) => xrule,
                        Face(Span(), Upper()) => yrule)
        combined = @inferred Chmy.combine_rules(rules, Face(Upper(), Upper()))

        shifted_corner = BoundaryShift{(1, 1)}()(a)
        @test @inferred(boundary_rule(combined, shifted_corner, Face(Upper(), Upper()),
                                      (Point(), Point()), (Shift(0), Shift(0)))) ===
              -2 * b[Point(), Point()][i, j-1] +
              2 * c[Point(), Point()][i+1, j] +
              a[Point(), Point()][i-1, j-1]

        direct = ExtensionRule(a => d)
        direct_rules = Binding(Face(Upper(), Upper()) => direct,
                               Face(Upper(), Span()) => xrule,
                               Face(Span(), Upper()) => yrule)
        @test Chmy.combine_rules(direct_rules, Face(Upper(), Upper())) === direct

        rules3 = Binding(Face(Upper(), Span(), Span()) => ExtensionRule(a => b),
                         Face(Span(), Upper(), Span()) => ExtensionRule(a => c),
                         Face(Span(), Span(), Upper()) => ExtensionRule(a => d))
        combined3 = @inferred Chmy.combine_rules(rules3, Face(Upper(), Upper(), Upper()))
        @test length(combined3.rules) == 3

        shifted3 = BoundaryShift{(1, 0, 1)}()(a)
        @test boundary_rule(combined3, shifted3, Face(Upper(), Upper(), Upper()),
                            (Point(), Point(), Point()), (Shift(0), Shift(0), Shift(0))) ===
              -2 * b[Point(), Point(), Point()][i, j, k-1] +
              c[Point(), Point(), Point()][i-1, j, k-1] +
              2 * d[Point(), Point(), Point()][i+1, j, k]

        mxrule = ExtensionRule(a => b, q => e)
        myrule = ExtensionRule(a => c, q => f)
        mrules = Binding(Face(Upper(), Span()) => mxrule,
                         Face(Span(), Upper()) => myrule)
        mcombined = @inferred Chmy.combine_rules(mrules, Face(Upper(), Upper()))
        mshifted = BoundaryShiftPair{(1, 1)}()(a, q)
        @test @inferred(boundary_rule(mcombined, mshifted, Face(Upper(), Upper()),
                                      (Point(), Point()), (Shift(0), Shift(0)))) ===
              -2 * b[Point(), Point()][i, j-1] +
              2 * c[Point(), Point()][i+1, j] +
              a[Point(), Point()][i-1, j-1] -
              2 * e[Point(), Point()][i, j-1] +
              2 * f[Point(), Point()][i+1, j] +
              q[Point(), Point()][i-1, j-1]
    end

    @testset "direct reconstruction API" begin
        @scalars a b

        @test @inferred(reconstruct(PolynomialReconstruction(), ValueData(b), (a,), SLiteral(1), SLiteral(1))) === b
        @test reconstruct(PolynomialReconstruction(), ValueData(b), (a,), SLiteral(1), SLiteral(2)) === 2 * b - a
        @test reconstruct(PolynomialReconstruction(), ValueData(b), (a,), SLiteral(1 // 2), SLiteral(1)) === 2 * b - a
        @test reconstruct(PolynomialReconstruction(), DerivativeData(b), (a,), SLiteral(1), SLiteral(2)) === a + 2 * b
    end

    @testset "extension validation" begin
        @scalars a b
        @vectors V
        i = SIndex(1)

        bad_rule = ExtensionRule(a[Point()] => b)
        @test_throws ArgumentError boundary_rule(bad_rule, CentralDifference()(a), Face(Upper()), (Point(),), (Shift(0),))

        bad_tensor_rule = ExtensionRule(V => b)
        @test_throws ArgumentError boundary_rule(bad_tensor_rule, CentralDifference()(a), Face(Upper()), (Point(),), (Shift(0),))

        bad_tensor_data = ExtensionRule(V => SZeroTensor{2}())
        @test_throws ArgumentError boundary_rule(bad_tensor_data, CentralDifference()(a), Face(Upper()), (Point(),), (Shift(0),))

        zero_target_rule = ExtensionRule(BoundaryNormal()[2] => b)
        @test_throws ArgumentError boundary_rule(zero_target_rule, BoundaryShift{(1, 0)}()(a),
                                                 Face(Upper(), Span()), (Point(), Point()), (Shift(0), Shift(0)))

        missing_rule = ExtensionRule(b => b)
        @test boundary_rule(missing_rule, CentralDifference()(a), Face(Upper()), (Point(),), (Shift(0),)) ===
              1 // 2 * (a[Point()][i+1] - a[Point()][i-1])

        corner_rule = ExtensionRule(a => b)
        @test_throws ArgumentError boundary_rule(corner_rule, a, Face(Upper(), Upper()),
                                                 (Point(), Point()), (Shift(0), Shift(0)))
    end
end
