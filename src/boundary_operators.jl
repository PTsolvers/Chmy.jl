"""
    Shift(i)

Represents a static integer shift from a reference grid index.
"""
struct Shift{I} <: STerm end
function Shift(I::Integer)
    return Shift{I}()
end
Shift(I) = throw(ArgumentError("Shift must be constructed from an integer, got $I"))
Shift(::SLiteral{I}) where {I} = Shift{I}()
SLiteral(::Shift{I}) where {I} = SLiteral{I}()

tensorrank(::Shift) = 0
value(::Shift{I}) where {I} = I
isless_lex(::Shift{I}, ::Shift{J}) where {I,J} = isless(I, J)

Base.:+(::Shift{A}, ::Shift{B}) where {A,B} = Shift(A + B)
Base.:+(shift::Shift) = shift
Base.:-(::Shift{A}, ::Shift{B}) where {A,B} = Shift(A - B)
Base.:-(::Shift{A}) where {A} = Shift(-A)

"""
    CartesianShift(shifts...)
    δ(shifts...)

Represent an N-dimensional tuple of [`Shift`](@ref)s.

`δ` is the concise constructor intended for stencil index shifts, for example `δ(-1, 0)`.
"""
struct CartesianShift{N,S}
    shifts::S
    CartesianShift(shifts::NTuple{N,Shift}) where {N} = new{N,typeof(shifts)}(shifts)
end
CartesianShift(shifts::Vararg{Shift}) = CartesianShift(shifts)

"""
    δ(shifts...)

Construct a [`CartesianShift`](@ref) from integer shifts.
"""
δ(inds::Vararg{Integer,N}) where {N} = CartesianShift(ntuple(i -> Shift(inds[i]), Val(N)))
δ(inds::Vararg{Real,N}) where {N} = throw(ArgumentError("δ shifts must be integers, got $inds"))

Base.isless(a::CartesianShift, b::CartesianShift) = isless_tuple(a.shifts, b.shifts)

Base.getindex(s::STerm, o::CartesianShift) = getindex(s, o.shifts...)

get_shift(::SIndex) = Shift(0)
get_shift(shift::Shift) = shift
function get_shift(ind::SExpr{Call})
    op = operation(ind)
    i, o = arguments(ind)
    if !(op === SRef(:+) || op === SRef(:-)) || !isa(i, SIndex) || !isa(o, SLiteral)
        error("only indexing expression of the format `(i + c)` or `(i - c)` are supported, where c is `SLiteral` are supported, got '$ind'")
    end
    return operation(ind) === SRef(:+) ? Shift(o) : Shift(-o)
end

nonuniform_location(_) = nothing
nonuniform_location(arg::SExpr{Loc}) = location(arg)
nonuniform_shift(_, inds) = CartesianShift(map(get_shift, inds))

"""
    AxisFace
    Lower()
    Upper()
    Span()

Axis markers used to build a [`Face`](@ref).

`Lower()` and `Upper()` select the lower or upper boundary in one coordinate
direction. `Span()` means that the face spans that coordinate direction.
"""
abstract type AxisFace end

"""
    Lower()

Marker for the lower boundary of one coordinate direction in a [`Face`](@ref).
"""
struct Lower <: AxisFace end

"""
    Upper()

Marker for the upper boundary of one coordinate direction in a [`Face`](@ref).
"""
struct Upper <: AxisFace end

"""
    Span()

Marker for a coordinate direction spanned by a [`Face`](@ref).
"""
struct Span <: AxisFace end

"""
    Face(axes...)

Returns a face of an n-dimensional hypercube. Axes can be either `Lower()`, `Upper()`, or `Span()`.

For example, in 2D hypercube (square), Face(Lower(), Span()) represents a left edge, and Face(Upper(), Upper()) represents a top right vertex.
"""
struct Face{E}
    axes::E
end
Face(axes::Vararg{AxisFace,N}) where {N} = Face(axes)

"""
    ndims(face::Face)

Return the number of dimensions of a hypercube to which the face is attached.
"""
Base.ndims(f::Face) = length(f.axes)

"""
    dim(face)

Return the dimensionality of a face of n-dimensional hypercube.
"""
dim(f::Face) = count(==(Span()), f.axes)

"""
    codim(face)

Return codimension of a face inside an n-dimensional hypercube.
"""
codim(f::Face) = ndims(f) - dim(f)

"""
    adjacent_faces(face)

Return a tuple of codim-1 faces of the enclosing hypercube that are adjacent to `face`.

Each non-`Span()` axis of `face` is relaxed to `Span()` in turn, while all other axes are
kept unchanged.
"""
adjacent_faces(f::Face) = _adjacent_faces(f.axes)

_adjacent_faces(::Tuple{}) = ()
function _adjacent_faces(axes::Tuple{Span,Vararg{AxisFace}})
    # A leading `Span()` does not create a new adjacent face on its own, so we recurse on the
    # tail and restore the same leading axis on every face returned from deeper levels.
    return _prepend_axis(Span(), _adjacent_faces(Base.tail(axes)))
end
function _adjacent_faces(axes::Tuple{Vararg{AxisFace}})
    face = first(axes)
    tail = Base.tail(axes)
    # A boundary axis contributes one adjacent face by turning just that axis into `Span()`.
    # All deeper adjacent faces are preserved, with the current leading axis prepended back.
    return (Face(Span(), tail...), _prepend_axis(face, _adjacent_faces(tail))...)
end

_prepend_axis(axis::AxisFace, faces::Tuple{Vararg{Face}}) = map(face -> Face(axis, face.axes...), faces)

"""
    Stencil(location, shifts...)
    Stencil(shifts...)

Store the integer shifts read by a nonuniform symbolic field.

For located fields, `location` is the field staggering and `shifts` are pure
integer index displacements. Rendering combines the two to recover where the
read lies on the staggered grid.
"""
struct Stencil{N,L,S}
    location::L
    shifts::S
    function Stencil(location, shifts::NTuple{R,CartesianShift{N}}) where {R,N}
        loc = normalize_location(location, Val(N))
        return new{N,typeof(loc),typeof(shifts)}(loc, shifts)
    end
end
Stencil(shifts::CartesianShift...) = Stencil(nothing, shifts)
Stencil(location, shifts::CartesianShift...) = Stencil(location, shifts)

normalize_location(::Nothing, ::Val) = nothing
function normalize_location(loc::Space, ::Val{N}) where {N}
    N == 1 || throw(ArgumentError("single Space location can only be used with a 1D stencil"))
    return (loc,)
end
function normalize_location(loc, ::Val{N}) where {N}
    throw(ArgumentError("stencil location must be nothing, a Space, or an $N-tuple of Space values, got $loc"))
end
normalize_location(loc::NTuple{N,Space}, ::Val{N}) where {N} = loc

Base.ndims(::Stencil{N}) where {N} = N

function Base.merge(s1::Stencil, s2::Stencil)
    same_location(s1.location, s2.location) || throw(ArgumentError("cannot merge stencils with different locations"))
    return Stencil(s1.location, merge_sorted_unique(s1.shifts, s2.shifts))
end

same_location(::Nothing, ::Nothing) = true
same_location(::Nothing, _) = false
same_location(_, ::Nothing) = false
function same_location(a::Tuple, b::Tuple)
    length(a) == length(b) || return false
    return all(map((x, y) -> x === y, a, b))
end

merge_sorted_unique(::Tuple{}, ::Tuple{}) = ()
merge_sorted_unique(::Tuple{}, b::Tuple) = b
merge_sorted_unique(a::Tuple, ::Tuple{}) = a
function merge_sorted_unique(a::Tuple, b::Tuple)
    ah = first(a)
    bh = first(b)
    if ah === bh
        return (ah, merge_sorted_unique(Base.tail(a), Base.tail(b))...)
    elseif isless(ah, bh)
        return (ah, merge_sorted_unique(Base.tail(a), b)...)
    else
        return (bh, merge_sorted_unique(a, Base.tail(b))...)
    end
end

"""
    Nonuniforms(stencils...)

Mapping from nonuniform fields to their collected [`Stencil`](@ref)s.

Use [`nonuniforms`](@ref) to construct this from an expression.
"""
struct Nonuniforms{S}
    stencils::S
    Nonuniforms(bnd::Binding) = new{typeof(bnd)}(bnd)
end
Nonuniforms(stencils::Vararg{Pair{<:STerm,<:Stencil}}) = Nonuniforms(Binding(stencils...))

stencils(nu::Nonuniforms) = nu.stencils

stencil(nu::Nonuniforms, t::STerm) = nu.stencils[t]

Base.mergewith(combine, nu::Nonuniforms, others::Vararg{Nonuniforms}) = Nonuniforms(mergewith(combine, nu.stencils, map(stencils, others)...))

"""
    nonuniforms(expr)

Returns a Binding containing pairs of nonuniform fields and their corresponding stencils. 
"""
nonuniforms(::STerm) = Nonuniforms()
Base.@assume_effects :foldable nonuniforms(expr::SExpr{Call}) = mergewith(merge, map(nonuniforms, arguments(expr))...)
function nonuniforms(expr::SExpr{Ind})
    shift = nonuniform_shift(argument(expr), indices(expr))
    stencil = Stencil(nonuniform_location(argument(expr)), shift)
    return Nonuniforms(argument(expr) => stencil)
end

struct GridOperator{E,R}
    expr::E
    rules::R
end
GridOperator(expr, rules...) = GridOperator(expr, rules)

"""
    boundary_operator(expr, rules...)
    boundary_operator(expr, rules::Binding)

Construct a grid operator from an interior expression and boundary rules.

Rules are stored as a `Binding` from [`Face`](@ref) to [`BoundaryRule`](@ref).
When an operator is requested on a lower-dimensional face, Chmy first tries a
direct rule for that face and otherwise combines adjacent face rules.
"""
boundary_operator(expr, rules::Binding) = GridOperator(expr, rules)
boundary_operator(expr, rules::Pair...) = GridOperator(expr, Binding(rules...))

"""
    BoundaryRule

Abstract base type for symbolic boundary rules.

Boundary rules transform an already-defined interior Chmy expression into the
expression that should be evaluated on or near a boundary.
"""
abstract type BoundaryRule end

struct CombinedRule{R} <: BoundaryRule
    rules::R
end

# `operator` is the user-facing dispatcher for `GridOperator`. It keeps the
# interior expression path explicit too: even without a boundary rule the
# expression is lowered at `loc` with canonical static indices.
operator(op::GridOperator, ::Face{NTuple{N,Span}}, loc::NTuple{N,Space}, _) where {N} = op.expr[loc...][_canonical_indices(Val(N))...]
function operator(op::GridOperator, f::Face, loc::NTuple{N,Space}, shifts) where {N}
    rule = combine_rules(op.rules, f)
    isnothing(rule) && return op.expr[loc...][_canonical_indices(Val(N))...]
    return boundary_rule(rule, op.expr, f, loc, shifts)
end

# Choose the rule for a face. Direct user-provided rules take precedence; lower
# dimensional faces are otherwise reconstructed from adjacent faces.
function combine_rules(rules, f::Face)
    haskey(rules, f) && return rules[f]
    codim(f) == 1 && return nothing
    pairs = combine_adjacent_rules(rules, adjacent_faces(f))
    isnothing(pairs) && return nothing
    return CombinedRule(pairs)
end

# Recursively walk adjacent faces. The result is either `nothing` if any required
# adjacent rule is missing, or a tuple of `Face => BoundaryRule` pairs.
combine_adjacent_rules(_, ::Tuple{}) = ()
function combine_adjacent_rules(rules, faces::Tuple)
    face = first(faces)
    rule = combine_rules(rules, face)
    isnothing(rule) && return nothing
    rest = combine_adjacent_rules(rules, Base.tail(faces))
    isnothing(rest) && return nothing
    return merge_rule_pairs(rule_pairs(face, rule), rest)
end

# A nested combined rule already contains the codim-1 face identities that must
# be applied, so flatten it rather than applying an intermediate face wrapper.
rule_pairs(face::Face, rule) = (face => rule,)
rule_pairs(::Face, rule::CombinedRule) = rule.rules

# Deduplicate codim-1 face applications while combining higher-codimension
# faces. In a 3D corner the same face can be discovered through more than one
# adjacent edge.
merge_rule_pairs(::Tuple{}, pairs::Tuple) = pairs
function merge_rule_pairs(pairs::Tuple, rest::Tuple)
    pair = first(pairs)
    merged = has_rule_pair(rest, pair.first) ? rest : (pair, rest...)
    return merge_rule_pairs(Base.tail(pairs), merged)
end

has_rule_pair(pairs::Tuple, face::Face) = any(map(pair -> pair.first === face, pairs))

"""
    BoundaryNormal()

Symbolic outward unit normal vector for boundary-condition expressions.

`BoundaryNormal()` is resolved to the normal of the codim-1 face when a
boundary rule is applied.
"""
struct BoundaryNormal <: AbstractSTensor{1,NoKind,true} end

struct FaceNormal{I,A} <: AbstractSTensor{1,NoKind,true} end

FaceNormal{I}(axis::AxisFace) where {I} = FaceNormal{I,typeof(axis)}()

Base.getindex(n::BoundaryNormal, i::IntegerOrSLiteral) = n[STerm(i)]
Base.getindex(n::BoundaryNormal, i::SLiteral) = SExpr(Comp(), n, i)

Base.getindex(n::FaceNormal, i::SLiteral) = face_normal_component(n, i)

face_normal_component(::FaceNormal{I,Upper}, ::SLiteral{J}) where {I,J} = SLiteral(I == J ? 1 : 0)
face_normal_component(::FaceNormal{I,Lower}, ::SLiteral{J}) where {I,J} = SLiteral(I == J ? -1 : 0)

function Tensor{D}(n::FaceNormal) where {D}
    return Vec{D}(ntuple(i -> n[SLiteral(i)], Val(D))...)
end

"""
    ExtensionSpec(data[, op])
    ExtensionSpec(value[, op])

Boundary data and reconstruction method for one field.

Passing [`ValueData`](@ref) or [`DerivativeData`](@ref) keeps that boundary data
kind. Passing a plain Chmy expression stores it as [`ValueData`](@ref). If `op`
is omitted, [`LinearReconstruction`](@ref) is used.
"""
struct ExtensionSpec{D,O}
    data::D
    op::O
end

"""
    ExtensionRule(condition => ExtensionSpec(data, op), ...)
    ExtensionRule(condition => value, ...)

Boundary rule that extends fields across a codim-1 face.

Each key may be an unlocated scalar field identity, such as `u` or `V[1]`, or a
tensor boundary condition such as `V` or `V ⋅ BoundaryNormal()`. Tensor
conditions are projected to the concrete face and scalarized once
[`boundary_rule`](@ref) knows the dimension. When the interior expression is
lowered, the rule finds located occurrences such as `u[Point()]` or
`V[1][Point(), Segment()]` and uses each occurrence's location to reconstruct
boundary and ghost reads. Each value is an [`ExtensionSpec`](@ref) or a Chmy
expression, which is treated as value data with linear reconstruction.
Nonuniform boundary data is lowered at the boundary: the normal location is
always `Point()`, and tangential locations are copied from the matched field
read.
"""
struct ExtensionRule{B} <: BoundaryRule
    specs::B
end
ExtensionRule(specs::Pair...) = ExtensionRule(Binding(map(extension_rule_pair, specs)...))

"""
    BoundaryData

Abstract base type for boundary data used by [`ExtensionSpec`](@ref).
"""
abstract type BoundaryData end

"""
    ValueData(value)

Boundary value data for field reconstruction.

This represents the value of the extended field at the boundary.
"""
struct ValueData{V} <: BoundaryData
    value::V
end

"""
    DerivativeData(value)

Outward-normal derivative data for field reconstruction.
"""
struct DerivativeData{V} <: BoundaryData
    value::V
end

"""
    ExtensionOperator{O}

Abstract base type for one-dimensional boundary extension algorithms.

The type parameter `O` is the number of interior samples the operator requires
to reconstruct one boundary or ghost value.
"""
abstract type ExtensionOperator{O} end

# Small trait used by the rewrite path to construct the tuple of interior
# samples without threading the operator type parameter through every helper.
npoints(::ExtensionOperator{O}) where {O} = O

"""
    PolynomialReconstruction([order])

Polynomial extension operator for reconstructing boundary and ghost values.

The reconstruction is defined in one boundary-normal coordinate and is lifted back
to the full expression by [`boundary_rule`](@ref).
"""
struct PolynomialReconstruction{O} <: ExtensionOperator{O} end

"""
    LinearReconstruction

Alias for the one-inner-point linear polynomial reconstruction used by default.
"""
const LinearReconstruction = PolynomialReconstruction{1}

PolynomialReconstruction() = LinearReconstruction()
function PolynomialReconstruction(order::Integer)
    order == 1 || throw(ArgumentError("only linear polynomial reconstruction is supported"))
    return LinearReconstruction()
end

ExtensionSpec(data::BoundaryData) = ExtensionSpec(data, LinearReconstruction())
ExtensionSpec(value::STerm) = ExtensionSpec(ValueData(value), LinearReconstruction())
ExtensionSpec(value::STerm, op::ExtensionOperator) = ExtensionSpec(ValueData(value), op)

_canonical_indices(::Val{N}) where {N} = ntuple(SIndex, Val(N))

"""
    boundary_rule(rule, expr, face, loc, shifts)

Transform an interior expression for evaluation on or near a boundary.

The expression is first lowered at `loc` with canonical static indices
`SIndex(1:N)`. Boundary rules then replace any field reads whose physical
outward-normal coordinate is non-negative. `shifts` gives the integer index
displacement of the evaluation location from the boundary node; `loc` supplies
the staggering needed to convert those shifts to coordinates.
"""
function boundary_rule(rule, expr::STerm, face::Face, loc::NTuple{N,Space}, shifts::CartesianShift{N}) where {N}
    return boundary_rule(rule, expr, face, loc, shifts.shifts)
end
function boundary_rule(rule, expr::STerm, face::Face, loc::NTuple{N,Space}, shifts::NTuple{N,<:Integer}) where {N}
    return boundary_rule(rule, expr, face, loc, tuplemap(Shift, shifts))
end
Base.@assume_effects :foldable function boundary_rule(rule, expr::STerm, face::Face, loc::NTuple{N,Space}, shifts::NTuple{N,Shift}) where {N}
    ndims(face) == N || throw(ArgumentError("face dimension $(ndims(face)) does not match location dimension $N"))
    lowered = expr[loc...][_canonical_indices(Val(N))...]
    return apply_boundary_rule(rule, lowered, face, loc, shifts)
end

# A single extension rule is deliberately one-dimensional. Higher-codimension
# regions are handled by `CombinedRule`, which folds several codim-1 rules.
Base.@assume_effects :foldable function apply_boundary_rule(rule::ExtensionRule, expr::STerm, face::Face, loc::NTuple{N,Space}, shifts) where {N}
    codim(face) == 1 || throw(ArgumentError("ExtensionRule can only be applied to codim-1 faces, got codim $(codim(face))"))
    I = normal_axis(face)
    axis = face.axes[I]
    normal = FaceNormal{I}(axis)
    # Extension application is intentionally staged: first resolve
    # BoundaryNormal() for this face, then expand tensor conditions to scalar
    # component specs, then normalize scalar signs, then run the rewrite pass.
    scalar_rule = normalize_rule(Tensor{N}(project_boundary(rule, normal)))
    return apply_extension_specs(pairstuple(scalar_rule.specs), expr, ExtensionApplication{I}(axis, shifts[I]))
end

# Combined rules store flattened `Face => rule` pairs. Each pair carries the
# actual codim-1 face needed to recover the normal axis and lower/upper side.
Base.@assume_effects :foldable function apply_boundary_rule(rule::CombinedRule, expr::STerm, face::Face, loc, shifts)
    return apply_combined_rules(rule.rules, expr, loc, shifts)
end

apply_combined_rules(::Tuple{}, expr::STerm, _, _) = expr
function apply_combined_rules(rules::Tuple, expr::STerm, loc, shifts)
    pair = first(rules)
    next = apply_boundary_rule(pair.second, expr, pair.first, loc, shifts)
    return apply_combined_rules(Base.tail(rules), next, loc, shifts)
end

# Return the single non-`Span()` axis of a codim-1 face
normal_axis(face::Face) = findfirst(!=(Span()), face.axes)

"""
    project_boundary(obj, normal)

Project boundary-condition symbolic objects to a concrete face normal.

Custom [`BoundaryData`](@ref) implementations can overload this method to
replace [`BoundaryNormal`](@ref) inside their stored data.
"""
project_boundary(term::STerm, normal) = subs(term, BoundaryNormal() => normal)
project_boundary(data::ValueData, normal) = ValueData(project_boundary(data.value, normal))
project_boundary(data::DerivativeData, normal) = DerivativeData(project_boundary(data.value, normal))
project_boundary(spec::ExtensionSpec, normal) = ExtensionSpec(project_boundary(spec.data, normal), spec.op)
project_boundary(rule::ExtensionRule, normal) = ExtensionRule(project_boundary_pairs(pairstuple(rule.specs), normal)...)

project_boundary_pairs(pairs::Tuple, normal) = map(pair -> project_boundary(pair.first, normal) => project_boundary(pair.second, normal), pairs)

function Tensor{D}(rule::ExtensionRule) where {D}
    return ExtensionRule(scalarize_rule_pairs(pairstuple(rule.specs), Val(D))...)
end

scalarize_rule_pairs(::Tuple{}, ::Val) = ()
function scalarize_rule_pairs(pairs::Tuple, ::Val{D}) where {D}
    pair = first(pairs)
    component_pairs = scalarize_rule_pair(pair.first, pair.second, Val(D))
    return (component_pairs..., scalarize_rule_pairs(Base.tail(pairs), Val(D))...)
end

function scalarize_rule_pair(lhs, spec::ExtensionSpec, ::Val{D}) where {D}
    rank = tensorrank(lhs)
    if rank != boundary_data_rank(spec)
        throw(ArgumentError("boundary data rank does not match lhs rank"))
    end

    expanded_lhs = Tensor{D}(lhs)
    rank == 0 && return (expanded_lhs => spec,)
    return scalarize_tensor_rule_pair(expanded_lhs, spec)
end

function scalarize_tensor_rule_pair(lhs::Tensor{D,R,K}, spec::ExtensionSpec) where {D,R,K}
    if length(lhs.components) == 0
        throw(ArgumentError("boundary condition lhs has no scalar field components"))
    end
    specs = tuplemap(data -> ExtensionSpec(data, spec.op), scalarize_boundary_data(spec.data, lhs))
    return component_pairs(lhs.components, specs)
end

component_pairs(fields::Tuple, specs::Tuple) = map((field, spec) -> field => spec, fields, specs)

function scalarize_tensor_rule_pair(lhs, ::ExtensionSpec)
    throw(ArgumentError("tensor boundary condition must expand to a concrete Tensor, got $lhs"))
end

boundary_data_rank(spec::ExtensionSpec) = boundary_data_rank(spec.data)
boundary_data_rank(data::ValueData) = tensorrank(data.value)
boundary_data_rank(data::DerivativeData) = tensorrank(data.value)

function scalarize_boundary_data(data::ValueData, lhs::Tensor{D,R,K}) where {D,R,K}
    return tuplemap(ValueData, tensor_with_kind(Tensor{D,R,K}, data.value).components)
end
function scalarize_boundary_data(data::DerivativeData, lhs::Tensor{D,R,K}) where {D,R,K}
    return tuplemap(DerivativeData, tensor_with_kind(Tensor{D,R,K}, data.value).components)
end

struct ExtensionApplication{I,A,S}
    axis::A
    shift::S
end
function ExtensionApplication{I}(axis, shift) where {I}
    return ExtensionApplication{I,typeof(axis),typeof(shift)}(axis, shift)
end

apply_extension_specs(::Tuple{}, expr::STerm, ::ExtensionApplication) = expr
function apply_extension_specs(pairs::Tuple, expr::STerm, app::ExtensionApplication)
    pair = first(pairs)
    next = apply_scalar_extension(pair.first, pair.second, expr, app)
    return apply_extension_specs(Base.tail(pairs), next, app)
end

function apply_scalar_extension(field, spec::ExtensionSpec, expr::STerm, app::ExtensionApplication{I}) where {I}
    er = ExtensionRewriteRule{I}(spec, field, app.axis, app.shift)
    return evaluate(Postwalk(er)(expr))
end

function normalize_rule(rule::ExtensionRule)
    npairs = map(normalize_pair, pairstuple(rule.specs))
    return ExtensionRule(npairs...)
end

function normalize_pair(pair::Pair)
    lhs, rhs = pair
    if isunaryminus(lhs)
        arg = only(arguments(lhs))
        return extension_field(arg) => ExtensionSpec(negate(rhs.data), rhs.op)
    end
    return extension_field(lhs) => rhs
end

negate(data::ValueData) = ValueData(-data.value)
negate(data::DerivativeData) = DerivativeData(-data.value)

extension_field(field::SExpr{Loc}) = throw(ArgumentError("ExtensionRule keys must be unlocated scalar fields or tensor expressions, got $field"))
function extension_field(field::SExpr{Comp})
    tensorrank(argument(field)) > 0 || throw(ArgumentError("ExtensionRule component keys must index tensor fields, got $field"))
    return field
end
extension_field(field::AbstractSTensor{0}) = field
function extension_field(field)
    throw(ArgumentError("ExtensionRule scalar conditions must reduce to an unlocated scalar field or component, got $field"))
end

extension_rule_spec(spec::ExtensionSpec) = spec
extension_rule_spec(value::STerm) = ExtensionSpec(value)
extension_rule_spec(spec) = throw(ArgumentError("ExtensionRule values must be ExtensionSpec objects, got $spec"))

extension_rule_pair(pair::Pair) = pair.first => extension_rule_spec(pair.second)

# Carries the static context needed to decide whether an indexed read is inside
# the domain or requires reconstruction.
struct ExtensionRewriteRule{I,SP,F,A,S} <: AbstractRule
    spec::SP
    field::F
    axis::A
    shift::S
end
function ExtensionRewriteRule{I}(spec, field, axis, shift) where {I}
    return ExtensionRewriteRule{I,typeof(spec),typeof(field),typeof(axis),typeof(shift)}(spec, field, axis, shift)
end

# For a candidate read, convert its index displacement into the outward-positive
# coordinate system of the boundary. Non-negative coordinates are reconstructed;
# negative coordinates remain ordinary interior reads.
(er::ExtensionRewriteRule)(term::SExpr{Ind}) = rewrite_occurrence(er, term, argument(term))

rewrite_occurrence(::ExtensionRewriteRule, ::SExpr{Ind}, field) = nothing
function rewrite_occurrence(er::ExtensionRewriteRule{I}, term::SExpr{Ind}, field::SExpr{Loc}) where {I}
    argument(field) === er.field || return nothing
    loc = location(field)[I]
    inds = indices(term)
    x = SLiteral(er.shift + get_shift(inds[I])) + offset(loc)
    xn = normal_coordinate(er.axis, x)
    value(xn) < 0 && return nothing
    v = interior_values(er, field, inds, loc)
    Δb = boundary_delta(loc)
    data = lower_boundary_data(er.spec.data, field, inds, er.shift, Val(I))
    return reconstruct(er.spec.op, data, v, Δb, xn + Δb)
end

# Boundary data lives on the boundary itself, independent of the field
# staggering in the normal direction. Tangential staggering and indices follow
# the matched read so data like `g` can vary along the boundary.
function lower_boundary_data(data::ValueData, field::SExpr{Loc}, inds, shift, ::Val{I}) where {I}
    loc = boundary_data_location(location(field), Val(I))
    bnd_inds = boundary_data_indices(inds, shift, Val(I))
    return ValueData(data.value[loc...][bnd_inds...])
end
function lower_boundary_data(data::DerivativeData, field::SExpr{Loc}, inds, shift, ::Val{I}) where {I}
    loc = boundary_data_location(location(field), Val(I))
    bnd_inds = boundary_data_indices(inds, shift, Val(I))
    return DerivativeData(data.value[loc...][bnd_inds...])
end

boundary_data_location(loc::NTuple{N,Space}, ::Val{I}) where {N,I} = ntuple(j -> j == I ? Point() : loc[j], Val(N))
boundary_data_indices(inds::NTuple{N,STerm}, shift::Shift, ::Val{I}) where {N,I} = ntuple(j -> j == I ? shifted_index(SIndex(I), -shift) : inds[j], Val(N))

shifted_index(ind::STerm, ::Shift{0}) = ind
shifted_index(ind::STerm, ::Shift{O}) where {O} = O > 0 ? ind + SLiteral(O) : ind - SLiteral(-O)

# Orient a boundary-normal coordinate so positive values point out of the domain.
normal_coordinate(::Upper, coord) = coord
normal_coordinate(::Lower, coord) = -coord

boundary_delta(::Point) = SLiteral(1)
boundary_delta(::Segment) = SLiteral(1 // 2)

function interior_values(er::ExtensionRewriteRule, field, inds, loc)
    return interior_values(er, field, inds, loc, Val(npoints(er.spec.op)))
end
function interior_values(er::ExtensionRewriteRule, field, inds, loc, ::Val{O}) where {O}
    return ntuple(k -> interior_value(er, field, inds, loc, Val(k)), Val(O))
end

# Reinsert the normal index for each requested inner sample, preserving the
# tangential indices of the original N-dimensional read.
function interior_value(er::ExtensionRewriteRule{I}, field, inds, loc, ::Val{K}) where {I,K}
    ishift = inner_shift(er.axis, er.shift, loc, Val(K))
    ind = SIndex(I) + SLiteral(ishift)
    return field[replace_index(inds, ind, Val(I))...]
end

inner_shift(::Upper, ::Shift{E}, ::Space, ::Val{K}) where {E,K} = Shift(-E - K)
inner_shift(::Lower, ::Shift{E}, ::Point, ::Val{K}) where {E,K} = Shift(K - E)
inner_shift(::Lower, ::Shift{E}, ::Segment, ::Val{K}) where {E,K} = Shift(K - E - 1)

"""
    reconstruct(op, bc, v, Δb, Δg)

Reconstruct a one-dimensional boundary or ghost value.

`v` contains the interior samples required by `op`. `Δb` is the distance
from the nearest inner sample to the boundary data, and `Δg` is the distance
from that same inner sample to the reconstructed ghost value.
"""
function reconstruct(::PolynomialReconstruction{1}, bc::ValueData, v::Tuple{STerm}, Δb, Δg)
    x = only(v)
    λ = Δg / Δb
    return (1 - λ) * x + λ * bc.value
end
function reconstruct(::PolynomialReconstruction{1}, bc::DerivativeData, v::Tuple{STerm}, Δb, Δg)
    x = only(v)
    return x + Δg * bc.value
end
