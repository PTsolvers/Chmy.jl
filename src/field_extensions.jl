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
ExtensionRule(specs::Pair...) = ExtensionRule(Binding(map(extension_pair, specs)...))

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

# A single extension rule is deliberately one-dimensional. Higher-codimension
# regions are handled by `CombinedRule`, which folds several codim-1 rules.
function apply_boundary_rule(rule::ExtensionRule, expr::STerm, face::Face, loc::NTuple{N,Space}, shifts) where {N}
    codim(face) == 1 || throw(ArgumentError("ExtensionRule can only be applied to codim-1 faces, got codim $(codim(face))"))
    I = normal_axis(face)
    axis = face.axes[I]
    pairs = extension_field_pairs(pairstuple(project_boundary(rule, face).specs))
    return apply_extension_specs(pairs, expr, ExtensionApplication{I}(axis, only(shifts)))
end

"""
    project_boundary(rule, face)

Project boundary-condition symbolic objects to a concrete codim-1 face.

Custom [`BoundaryData`](@ref) implementations can overload this method to
replace [`BoundaryNormal`](@ref) and [`BoundaryTangent`](@ref) inside their
stored data.
"""
function project_boundary(rule::ExtensionRule, face::Face)
    codim(face) == 1 || throw(ArgumentError("ExtensionRule can only be projected to codim-1 faces, got codim $(codim(face))"))
    N = ndims(face)
    I = normal_axis(face)
    pairs = project_pairs(pairstuple(rule.specs), Val(N), Val(I))
    return ExtensionRule(Binding(pairs...))
end

function project_boundary(term::STerm, ::Val{D}, ::Val{I}, ::Val{J}) where {D,I,J}
    Tensor{D}(subs(term, BoundaryNormal() => BasisVector{I}(), BoundaryTangent() => BasisVector{J}()))
end
project_boundary(data::ValueData, ::Val{D}, ::Val{I}, ::Val{J}) where {D,I,J} = ValueData(project_boundary(data.value, Val(D), Val(I), Val(J)))
project_boundary(data::DerivativeData, ::Val{D}, ::Val{I}, ::Val{J}) where {D,I,J} = DerivativeData(project_boundary(data.value, Val(D), Val(I), Val(J)))
project_boundary(spec::ExtensionSpec, ::Val{D}, ::Val{I}, ::Val{J}) where {D,I,J} = ExtensionSpec(project_boundary(spec.data, Val(D), Val(I), Val(J)), spec.op)

project_normal(term::STerm, ::Val{I}) where {I} = subs(term, BoundaryNormal() => BasisVector{I}())
project_normal(spec::ExtensionSpec, ::Val{I}) where {I} = ExtensionSpec(project_normal(spec.data, Val(I)), spec.op)
project_normal(data::DerivativeData, ::Val{I}) where {I} = DerivativeData(project_normal(data.value, Val(I)))
project_normal(data::ValueData, ::Val{I}) where {I} = ValueData(project_normal(data.value, Val(I)))

# `Tensor{D}` materializes symbolic tensor algebra, but boundary values are
# wrapped in `ValueData`/`DerivativeData` and `ExtensionSpec`. Preserve those
# wrappers while collapsing projected expressions such as `V ⋅ e₂` to `V[2]`.
materialize_projected(term::STerm, ::Val{D}) where {D} = Tensor{D}(term)
materialize_projected(spec::ExtensionSpec, ::Val{D}) where {D} = ExtensionSpec(materialize_projected(spec.data, Val(D)), spec.op)
materialize_projected(data::ValueData, ::Val{D}) where {D} = ValueData(materialize_projected(data.value, Val(D)))
materialize_projected(data::DerivativeData, ::Val{D}) where {D} = DerivativeData(materialize_projected(data.value, Val(D)))

project_pairs(::Tuple{}, ::Val, ::Val) = ()
function project_pairs(pairs::Tuple, ::Val{D}, ::Val{I}) where {D,I}
    pair = first(pairs)
    lhs = project_normal(pair.first, Val(I))
    spec = project_normal(pair.second, Val(I))
    component_pairs = project_pair(lhs, spec, Val(D), Val(I))
    return (component_pairs..., project_pairs(Base.tail(pairs), Val(D), Val(I))...)
end

function project_pair(lhs, spec::ExtensionSpec, ::Val{D}, ::Val{I}) where {D,I}
    return project_pair(lhs, spec, has_tangent(lhs), has_tangent(spec), Val(D), Val(I))
end
function project_pair(lhs, spec::ExtensionSpec, ::Val{false}, ::Val{false}, ::Val{D}, ::Val{I}) where {D,I}
    return scalarize_pair(materialize_projected(lhs, Val(D)), materialize_projected(spec, Val(D)), Val(D))
end
function project_pair(lhs, spec::ExtensionSpec, ::Val, ::Val, ::Val{D}, ::Val{I}) where {D,I}
    return project_tangent_pairs(lhs, spec, Val(D), Val(I))
end

@generated function project_tangent_pairs(lhs, spec, ::Val{D}, ::Val{I}) where {D,I}
    ex = Expr(:tuple)
    for J in 1:D
        J == I && continue
        projected = :(scalarize_pair(project_boundary(lhs, Val($D), Val($I), Val($J)),
                                     project_boundary(spec, Val($D), Val($I), Val($J)),
                                     Val($D)))
        push!(ex.args, Expr(:..., projected))
    end
    return ex
end

has_tangent(::BoundaryTangent) = Val(true)
has_tangent(expr::SExpr) = has_tangent(children(expr))
has_tangent(data::ValueData) = has_tangent(data.value)
has_tangent(data::DerivativeData) = has_tangent(data.value)
has_tangent(spec::ExtensionSpec) = has_tangent(spec.data)
has_tangent(::STerm) = Val(false)
has_tangent(::Tuple{}) = Val(false)
function has_tangent(args::Tuple)
    return has_tangent(has_tangent(first(args)), Base.tail(args))
end
has_tangent(::Val{true}, ::Tuple) = Val(true)
has_tangent(::Val{false}, args::Tuple) = has_tangent(args)

function Tensor{D}(rule::ExtensionRule) where {D}
    return ExtensionRule(scalarize_pairs(pairstuple(rule.specs), Val(D))...)
end

scalarize_pairs(::Tuple{}, ::Val) = ()
function scalarize_pairs(pairs::Tuple, ::Val{D}) where {D}
    pair = first(pairs)
    component_pairs = scalarize_pair(pair.first, pair.second, Val(D))
    return (component_pairs..., scalarize_pairs(Base.tail(pairs), Val(D))...)
end

function scalarize_pair(lhs, spec::ExtensionSpec, ::Val{D}) where {D}
    rank = tensorrank(lhs)
    if rank != tensorrank(spec)
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

tensorrank(spec::ExtensionSpec) = tensorrank(spec.data)
tensorrank(data::ValueData) = tensorrank(data.value)
tensorrank(data::DerivativeData) = tensorrank(data.value)

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

function apply_extension_specs(pairs::Tuple, expr::STerm, app::ExtensionApplication)
    rules = extension_rewrite_rules(pairs, app)
    return evaluate(ExtensionWalk(Chain(rules))(expr))
end

extension_rewrite_rules(::Tuple{}, ::ExtensionApplication) = ()
function extension_rewrite_rules(pairs::Tuple, app::ExtensionApplication{I}) where {I}
    pair = first(pairs)
    rule = ExtensionRewriteRule{I}(pair.second, pair.first, app.axis, app.shift)
    return (rule, extension_rewrite_rules(Base.tail(pairs), app)...)
end

extension_field_pairs(pairs::Tuple) = map(pair -> extension_field(pair.first) => pair.second, pairs)

extension_field(field::SExpr{Loc}) = throw(ArgumentError("ExtensionRule keys must be unlocated scalar fields or tensor expressions, got $field"))
function extension_field(field::SExpr{Comp})
    tensorrank(argument(field)) > 0 || throw(ArgumentError("ExtensionRule component keys must index tensor fields, got $field"))
    return field
end
extension_field(field::AbstractSTensor{0}) = field
function extension_field(field::SNode)
    tensorrank(field) == 0 || throw(ArgumentError("ExtensionRule node keys must be scalarized before application, got $field"))
    return field
end
function extension_field(field)
    throw(ArgumentError("ExtensionRule scalar conditions must reduce to an unlocated scalar field or component, got $field"))
end

extension_spec(spec::ExtensionSpec) = spec
extension_spec(value::STerm) = ExtensionSpec(value)
extension_spec(spec) = throw(ArgumentError("ExtensionRule values must be ExtensionSpec objects, got $spec"))

extension_pair(pair::Pair) = pair.first => extension_spec(pair.second)

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

# Boundary extension is a top-down walk: the first matching spec owns a term, and
# the replacement is visited again so nested/protected fields can still receive
# their own boundary treatment. Ordinary symbolic walks keep `SNode` opaque; this
# walker peels only lowered node reads, one wrapper at a time, once no spec
# matched the still-protected read.
struct ExtensionWalk{R} <: AbstractRule
    rule::R
end

function (walk::ExtensionWalk)(term::STerm)
    rewritten = walk.rule(term)
    isnothing(rewritten) || return walk(rewritten)

    peeled = peel_node(term)
    isnothing(peeled) || return walk(peeled)

    isexpr(term) || return term
    return SExpr(head(term), map(walk, children(term)))
end

peel_node(::STerm) = nothing
peel_node(term::SExpr{Ind}) = peel_node(argument(term), indices(term))
function peel_node(field::SExpr{Loc}, inds)
    protected = argument(field)
    protected isa SNode || return nothing
    return argument(protected)[location(field)...][inds...]
end

# For a candidate read, convert its index displacement into the outward-positive
# coordinate system of the boundary. Non-negative coordinates are reconstructed;
# negative coordinates remain ordinary interior reads.
(er::ExtensionRewriteRule)(term::SExpr{Ind}) = rewrite_occurrence(er, term, argument(term))

rewrite_occurrence(::ExtensionRewriteRule, ::SExpr{Ind}, field) = nothing
function rewrite_occurrence(er::ExtensionRewriteRule{I}, term::SExpr{Ind}, field::SExpr{Loc}) where {I}
    # match the lhs
    argument(field) === er.field || return nothing
    loc = location(field)[I]
    inds = indices(term)
    x = SLiteral(er.shift + get_shift(inds[I])) + offset(loc)
    xn = normal_coordinate(er.axis, x)
    # interior points don't need extending
    value(xn) < 0 && return nothing
    v = interior_values(er, field, inds, loc)
    # Δb is the distance from the last point to the boudnary
    Δb = boundary_delta(loc)
    data = lower_boundary_data(er.spec.data, field, inds, er.shift, Val(I))
    return reconstruct(er.spec.op, data, v, Δb, xn + Δb)
end

# Boundary data lives on the boundary itself, independent of the field
# staggering in the normal direction. Tangential staggering and indices follow
# the matched read so data like `g` can vary along the boundary.
function lower_boundary_data(data::ValueData, field::SExpr{Loc}, inds, shift, ::Val{I}) where {I}
    loc = boundary_location(location(field), Val(I))
    bnd_inds = boundary_indices(inds, shift, Val(I))
    return ValueData(data.value[loc...][bnd_inds...])
end
function lower_boundary_data(data::DerivativeData, field::SExpr{Loc}, inds, shift, ::Val{I}) where {I}
    loc = boundary_location(location(field), Val(I))
    bnd_inds = boundary_indices(inds, shift, Val(I))
    return DerivativeData(data.value[loc...][bnd_inds...])
end

boundary_location(loc::NTuple{N,Space}, ::Val{I}) where {N,I} = ntuple(j -> j == I ? Point() : loc[j], Val(N))
boundary_indices(inds::NTuple{N,STerm}, shift::Shift, ::Val{I}) where {N,I} = ntuple(j -> j == I ? shifted_index(SIndex(I), -shift) : inds[j], Val(N))

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
