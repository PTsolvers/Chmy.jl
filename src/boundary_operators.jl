struct Offset{O} <: STerm end
Offset(i::Integer) = Offset{i}()
Offset(::SLiteral{i}) where {i} = Offset{i}()

δ(inds::Vararg{Integer,N}) where {N} = ntuple(i -> Offset(inds[i]), Val(N))

tensorrank(::Offset) = 0
isless_lex(::Offset{I}, ::Offset{J}) where {I,J} = isless(I, J)

get_offset(::SIndex) = Offset(0)
function get_offset(ind::SExpr{Call})
    op = operation(ind)
    i, o = arguments(ind)
    if !(op === SRef(:+) || op === SRef(:-)) || !isa(i, SIndex) || !isa(o, SLiteral)
        error("only indexing expression of the format `(i + c)` or `(i - c)` are supported, where c is `SLiteral` are supported, got '$ind'")
    end
    return operation(ind) === SRef(:+) ? Offset(o) : Offset(-o)
end

struct Dirichlet <: STerm end
function boundary_rule(::Dirichlet, args::Tuple{STerm,STerm}, ::Segment, ::Offset{1})
    lhs, rhs = args
    return 2rhs - lhs[δ(-1)]
end
function boundary_rule(::Dirichlet, args::Tuple{STerm,STerm}, ::Point, ::Offset{0})
    _, rhs = args
    return rhs
end

struct Neumann <: STerm end
function boundary_rule(::Neumann, args::Tuple{STerm,STerm}, ::Segment, ::Offset{1})
    lhs, rhs = args
    return lhs[δ(-1)] + rhs
end
function boundary_rule(::Neumann, args::Tuple{STerm,STerm}, ::Point, ::Offset{1})
    lhs, rhs = args
    return lhs[δ(-1)] + 2rhs
end

abstract type AxisFace end

struct Lower <: AxisFace end
struct Upper <: AxisFace end
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

_prepend_axis(::AxisFace, ::Tuple{}) = ()
function _prepend_axis(axis::AxisFace, faces::Tuple{Vararg{Face}})
    # Rebuild each face with the current leading axis restored after recursing on the tail.
    face = first(faces)
    return (Face(axis, face.axes...), _prepend_axis(axis, Base.tail(faces))...)
end

struct Stencil{N,O}
    offsets::O
    Stencil(offsets::Tuple{Vararg{Tuple{Vararg{Offset,N}},R}}) where {N,R} = new{N,typeof(offsets)}(offsets)
end
Stencil(offsets...) = Stencil(offsets)

Base.ndims(::Stencil{N}) where {N} = N

Base.merge(s1::Stencil, s2::Stencil) = Stencil(merge_sorted_unique(s1.offsets, s2.offsets))
merge_sorted_unique(::Tuple{}, ::Tuple{}) = ()
merge_sorted_unique(::Tuple{}, b::Tuple) = b
merge_sorted_unique(a::Tuple, ::Tuple{}) = a
function merge_sorted_unique(a::Tuple, b::Tuple)
    ah = first(a)
    bh = first(b)
    if ah === bh
        return (ah, merge_sorted_unique(Base.tail(a), Base.tail(b))...)
    elseif isless_tuple(ah, bh)
        return (ah, merge_sorted_unique(Base.tail(a), b)...)
    else
        return (bh, merge_sorted_unique(a, Base.tail(b))...)
    end
end

struct Nonuniforms{S}
    stencils::S
    Nonuniforms(bnd::Binding) = new{typeof(bnd)}(bnd)
end
Nonuniforms(stencils::Vararg{Pair{<:STerm,<:Stencil}}) = Nonuniforms(Binding(stencils...))

stencils(nu::Nonuniforms) = nu.stencils

stencil(nu::Nonuniforms, t::STerm) = nu.stencils[t]

Base.mergewith(combine, nu::Nonuniforms, others::Vararg{Nonuniforms}) = Nonuniforms(mergewith(combine, nu.stencils, map(stencils, others)...))

nonuniforms(::STerm) = Nonuniforms()
Base.@assume_effects :foldable nonuniforms(expr::SExpr{Call}) = mergewith(merge, map(nonuniforms, arguments(expr))...)
function nonuniforms(expr::SExpr{Ind})
    offset = map(get_offset, indices(expr))
    stencil = Stencil(offset)
    return Nonuniforms(argument(expr) => stencil)
end

struct GridOperator{E,R}
    expr::E
    rules::R
end

struct CombinedRule{R}
    rules::R
end

operator(op::GridOperator, ::Face{NTuple{N,Span}}, loc) where {N} = op.expr[loc]
function operator(op::GridOperator, f::Face, loc)
    rule = construct_rule(op.rules, f)
    isnothing(rule) && return op.expr[loc]
    return boundary_operator(rule, op.expr, loc)
end

function construct_rule(rules, f::Face)
    haskey(rules, f) && return rules[f]
    codim(f) == 1 && return nothing
    af = adjacent_faces(f)
    ar = map(x -> construct_rule(rules, x), af)
    any(isnothing, ar) && return nothing
    return CombinedRule(ar)
end

# Substitution BC procedure
# 1. User must define custom boundary condition type and a method to match an expression and return a replacement expression.
# 2. The stencil width in a direction is defined as the largest offset out-of-grid in the corresponding direction. For this width, the interior expression is walked and matched againes this custom rule for as many points as needed.
# 3. The interior expression is passed to the matching function as is, and the returned expression must be already lowered. Whether the interior expression should come in a lower form or not depends on a specific boundary rule.

# Replacement BC procedure
# User must define custom boundary condition type and a method to return a replacement expression.
# 

# Automatic inference of faces of low codimension
# If there is no bc rule specified for a face, Chmy will attempt to construct it automatically from higher-dimensional faces.
# Maximum one high-dim rule can be a replacement rule, all other must be substitution rules. The replacement is run first, and then the substitution ones. The order is undefined, and this is the users responsibility to make sure subs rules are commutative.

# Reference implementations
# 1. Reconstruction rules. In a lowered interior expression, finds the specified subexpressions that are computed at the boundary or outside, and then a reconstruction rule is called that must be defined by the user. Examples are reconstruction from value, from gradient, or from weighted sum of the two.
