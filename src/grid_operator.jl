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
    isfacet(face)

Checks if the face is a facet, i.e. face of codimension 1.
"""
isfacet(face::Face) = codim(face) == 1

# Return the single non-`Span()` axis of a codim-1 face.
normal_axis(face::Face) = findfirst(!=(Span()), face.axes)

"""
    facet(side, n, d)
    facet(side, Val(n), Val(d))

Return facet (face of codimension 1) of an n-dimensional hypercube
on the side `side` (can be Lower() or Upper()) in the direction `d`.
"""
function facet(side::AxisFace, ::Val{N}, ::Val{D}) where {N,D}
    ntuple(Val(N)) do I
        I == D ? side : Span()
    end |> Face
end
facet(side, n::Integer, d::Integer) = facet(side, Val(n), Val(d))

"""
    facets(n)
    facets(Val(n))

Return tuple of all facets of n-dimensional hypercube.
"""
function facets(::Val{N}) where {N}
    ntuple(Val(N)) do D
        facet(Lower(), Val(N), Val(D)),
        facet(Upper(), Val(N), Val(D))
    end |> flatten
end
facets(n::Integer) = facets(Val(n))

"""
    Stencil(location, shifts...)
    Stencil(shifts...)

Store the integer shifts read by a nonuniform symbolic field.

If `location` is omitted, the stencil is assumed to live on [`Point`](@ref)s in
all coordinate directions. For located fields, `location` is the field
staggering and `shifts` are pure integer index displacements. Rendering combines
the two to recover where the read lies on the staggered grid.
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

normalize_location(::Nothing, ::Val{N}) where {N} = ntuple(_ -> Point(), Val(N))
function normalize_location(loc::Space, ::Val{N}) where {N}
    N == 1 || throw(ArgumentError("single Space location can only be used with a 1D stencil"))
    return (loc,)
end
function normalize_location(loc, ::Val{N}) where {N}
    throw(ArgumentError("stencil location must be a Space or an $N-tuple of Space values, got $loc"))
end
normalize_location(loc::NTuple{N,Space}, ::Val{N}) where {N} = loc

Base.ndims(::Stencil{N}) where {N} = N

function Base.merge(s1::Stencil, s2::Stencil)
    same_location(s1.location, s2.location) || throw(ArgumentError("cannot merge stencils with different locations"))
    return Stencil(s1.location, merge_sorted_unique(s1.shifts, s2.shifts))
end

function same_location(a::Tuple, b::Tuple)
    length(a) == length(b) || return false
    return all(map((x, y) -> x === y, a, b))
end

# merge two sorted tuples into a sorted tuple while keeping only unique entries
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

# Reach analysis -------------------------------------------------------------

"""
    reach(stencil, facet)

Return the reach of the stencil in the direction determined by the facet of a hypercube.
"""
function reach(s::Stencil, f::Face)
    ndims(s) == ndims(f) || throw(ArgumentError("stencil and face dimensions must match"))
    isfacet(f) || throw(ArgumentError("reach is defined for codimension-one faces, got codimension $(codim(f))"))
    D = normal_axis(f)
    mapreduce(max, s.shifts) do x
        v = value(x.shifts[D])
        f.axes[D] == Upper() ? v : -v
    end |> SLiteral
end

"""
    reach(stencil)

Return the boundary reach of stencil accesses.

The result is a [`Binding`](@ref) from codimension-one [`Face`](@ref)s to
[`SLiteral`](@ref) values.
"""
function reach(s::Stencil)
    fs = facets(Val(ndims(s)))
    rs = map(f -> reach(s, f), fs)
    return Binding(fs, rs)
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
    reach(nonuniforms)

Return the maximum reach of all the stencils in the object.
"""
reach(nu::Nonuniforms) = reach_nonuniforms(values(stencils(nu)))

reach_nonuniforms(::Tuple{}) = Binding()
reach_nonuniforms(stencils::Tuple) = mergewith(max, map(reach, stencils)...)

"""
    nonuniforms(expr)

Returns a Binding containing pairs of nonuniform fields and their corresponding stencils.
"""
nonuniforms(::STerm) = Nonuniforms()
nonuniforms(expr::SExpr{Call}) = mergewith(merge, map(nonuniforms, arguments(expr))...)
function nonuniforms(expr::SExpr{Ind})
    shift = nonuniform_shift(argument(expr), indices(expr))
    stencil = Stencil(nonuniform_location(argument(expr)), shift)
    return Nonuniforms(argument(expr) => stencil)
end

"""
    GridOperator(expr, rules)

Symbolic grid operator made from an interior expression and boundary rules.

Use [`boundary_operator`](@ref) for the public constructor that accepts
`Face => BoundaryRule` pairs.
"""
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

_canonical_indices(::Val{N}) where {N} = ntuple(SIndex, Val(N))

interior_face(::Val{N}) where {N} = Face(ntuple(_ -> Span(), Val(N))...)

lowered_interior(expr::STerm, loc::NTuple{N,Space}) where {N} = unwrap(expr[loc...][_canonical_indices(Val(N))...])

codim_shift_tuple(face::Face, shifts::CartesianShift) = codim_shift_tuple(face, shifts.shifts)
function codim_shift_tuple(face::Face, ::Tuple{})
    c = codim(face)
    c == 0 || throw(ArgumentError("face codimension $c requires $c shifts, got 0"))
    return ()
end
function codim_shift_tuple(face::Face, shifts::NTuple{C,<:Integer}) where {C}
    return codim_shift_tuple(face, tuplemap(Shift, shifts))
end
function codim_shift_tuple(face::Face, shifts::NTuple{C,Shift}) where {C}
    c = codim(face)
    C == c || throw(ArgumentError("face codimension $c requires $c shifts, got $C"))
    return shifts
end

"""
    boundary_axes(face)

Return the coordinate axes where `face` lies on a lower or upper boundary.

The order of this tuple defines the public shift order for boundary operators:
shift `k` belongs to `boundary_axes(face)[k]`.
"""
boundary_axes(face::Face) = _boundary_axes(face.axes, 1)

_boundary_axes(::Tuple{}, _) = ()
function _boundary_axes(axes::Tuple{Span,Vararg{AxisFace}}, i)
    return _boundary_axes(Base.tail(axes), i + 1)
end
function _boundary_axes(axes::Tuple{Vararg{AxisFace}}, i)
    return (i, _boundary_axes(Base.tail(axes), i + 1)...)
end

function shift_for_axis(face::Face, shifts::Tuple, ::Val{I}) where {I}
    return shift_for_axis(boundary_axes(face), shifts, Val(I))
end

shift_for_axis(::Tuple{}, ::Tuple{}, ::Val{I}) where {I} = throw(ArgumentError("axis $I is not a boundary axis"))
function shift_for_axis(axes::Tuple, shifts::Tuple, ::Val{I}) where {I}
    first(axes) == I && return first(shifts)
    return shift_for_axis(Base.tail(axes), Base.tail(shifts), Val(I))
end

subface_shifts(parent::Face, shifts::Tuple, face::Face) = _subface_shifts(parent, shifts, face.axes, 1)
_subface_shifts(::Face, ::Tuple, ::Tuple{}, _) = ()
function _subface_shifts(parent::Face, shifts::Tuple, axes::Tuple{Span,Vararg{AxisFace}}, i)
    return _subface_shifts(parent, shifts, Base.tail(axes), i + 1)
end
function _subface_shifts(parent::Face, shifts::Tuple, axes::Tuple{Vararg{AxisFace}}, i)
    return (shift_for_axis(parent, shifts, Val(i)), _subface_shifts(parent, shifts, Base.tail(axes), i + 1)...)
end

# `operator` is the user-facing dispatcher for `GridOperator`. It keeps the
# interior expression path explicit too: even without a boundary rule the
# expression is lowered at `loc` with canonical static indices.
"""
    operator(op::GridOperator, face, loc, shift)

Return the expression for `op` on `face` at `loc` and codimension-sized `shift`.

If no rule applies to `face`, this returns the unwrapped lowered interior
expression. For higher-codimension faces, only available adjacent boundary
rules are applied.
"""
function operator(op::GridOperator, f::Face, loc::NTuple{N,Space}, shifts) where {N}
    ndims(f) == N || throw(ArgumentError("face dimension $(ndims(f)) does not match location dimension $N"))
    shifts = codim_shift_tuple(f, shifts)
    rule = combine_rules(op.rules, f)
    isnothing(rule) && return lowered_interior(op.expr, loc)
    return boundary_rule(rule, op.expr, f, loc, shifts)
end

# Choose the rule for a face. Direct user-provided rules take precedence; lower
# dimensional faces are otherwise reconstructed from whichever adjacent face
# rules exist. Missing rules are deliberately ignored so halo or periodic axes
# can leave their out-of-domain reads untouched.
function combine_rules(rules, f::Face)
    haskey(rules, f) && return rules[f]
    codim(f) == 1 && return nothing
    pairs = combine_adjacent_rules(rules, adjacent_faces(f))
    isempty(pairs) && return nothing
    return CombinedRule(pairs)
end

# Recursively walk adjacent faces, keeping only the rules that are present.
combine_adjacent_rules(_, ::Tuple{}) = ()
function combine_adjacent_rules(rules, faces::Tuple)
    face = first(faces)
    rule = combine_rules(rules, face)
    rest = combine_adjacent_rules(rules, Base.tail(faces))
    isnothing(rule) && return rest
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

has_rule_pair(::Tuple{}, ::Face) = false
function has_rule_pair(pairs::Tuple, face::Face)
    first(pairs).first === face && return true
    return has_rule_pair(Base.tail(pairs), face)
end

"""
    boundary_rule(rule, expr, face, loc, shifts)

Transform an interior expression for evaluation on or near a boundary.

`shifts` has length `codim(face)` and is ordered by [`boundary_axes`](@ref).
The expression is lowered at `loc` with canonical static indices before
`rule` is applied. Boundary rules may leave unmatched out-of-domain reads
untouched, which is how halo and future periodic axes can coexist with
explicit boundary conditions on other axes.
"""
function boundary_rule(rule, expr::STerm, face::Face, loc::NTuple{N,Space}, shifts::CartesianShift) where {N}
    return boundary_rule(rule, expr, face, loc, shifts.shifts)
end
function boundary_rule(rule, expr::STerm, face::Face, loc::NTuple{N,Space}, shifts::NTuple{C,<:Integer}) where {N,C}
    return boundary_rule(rule, expr, face, loc, tuplemap(Shift, shifts))
end
function boundary_rule(rule, expr::STerm, face::Face, loc::NTuple{N,Space}, shifts::NTuple{C,Shift}) where {N,C}
    ndims(face) == N || throw(ArgumentError("face dimension $(ndims(face)) does not match location dimension $N"))
    shifts = codim_shift_tuple(face, shifts)
    lowered = expr[loc...][_canonical_indices(Val(N))...]
    return apply_boundary_rule(rule, lowered, face, loc, shifts)
end

# Combined rules store flattened `Face => rule` pairs. Each pair carries the
# face needed to recover the subtuple of the parent face shift it should see.
function apply_boundary_rule(rule::CombinedRule, expr::STerm, face::Face, loc, shifts)
    return apply_combined_rules(rule.rules, expr, face, loc, shifts)
end

apply_combined_rules(::Tuple{}, expr::STerm, ::Face, _, _) = expr
function apply_combined_rules(rules::Tuple, expr::STerm, face::Face, loc, shifts)
    pair = first(rules)
    next = apply_boundary_rule(pair.second, expr, pair.first, loc, subface_shifts(face, shifts, pair.first))
    return apply_combined_rules(Base.tail(rules), next, face, loc, shifts)
end

"""
    BoundaryNormal()

Symbolic boundary-normal selector for boundary-condition expressions.

`BoundaryNormal()` is resolved to a vector with positive unit component in the
normal axis of the codim-1 face when a boundary rule is applied.
"""
struct BoundaryNormal <: AbstractSTensor{1,NoKind,true} end

"""
    BoundaryTangent()

Symbolic boundary-tangent selector for boundary-condition expressions.

`BoundaryTangent()` is resolved to one positive coordinate-basis vector for each
tangent axis of that face, producing one scalarized extension spec per tangent
direction.
"""
struct BoundaryTangent <: AbstractSTensor{1,NoKind,true} end

"""
    BasisVector{I}()

Positive coordinate-basis vector in direction `I`.

`BasisVector{I}` materializes to a vector whose only non-zero component is
`1` at index `I`. Boundary projection uses it to resolve
[`BoundaryNormal`](@ref) and [`BoundaryTangent`](@ref) for a concrete face.
"""
struct BasisVector{I} <: AbstractSTensor{1,NoKind,true} end

Base.getindex(::BasisVector{I}, ::SLiteral{J}) where {I,J} = SLiteral(I == J ? 1 : 0)

function Tensor{D}(v::BasisVector) where {D}
    return Vec{D}(ntuple(i -> v[SLiteral(i)], Val(D))...)
end

"""
    CompiledGridOperator(op, loc)

Precompute a [`GridOperator`](@ref) at the grid location `loc`.

The compiled operator stores expressions in a two-level [`Binding`](@ref):
faces map to bindings whose keys are codimension-sized [`CartesianShift`](@ref)
objects and whose values are lowered, ready-to-compute symbolic expressions.
Only shifts that can actually require boundary treatment are compiled. Axes
without an applicable boundary rule are kept inactive, so lookups that vary only
along those axes share the same expression and preserve untouched halo reads.
"""
struct CompiledGridOperator{N,L,E,A}
    loc::L
    exprs::E
    active::A
end

Base.ndims(::CompiledGridOperator{N}) where {N} = N

function CompiledGridOperator(op::GridOperator, loc::NTuple{N,Space}) where {N}
    interior = lowered_interior(op.expr, loc)
    reach = stencil_reach(interior, loc)
    expr_pairs, active_pairs = compile_face_pairs(op, loc, reach, all_faces(Val(N)))
    exprs = Binding(interior_face(Val(N)) => Binding(δ() => interior), expr_pairs...)
    active_axes = Binding(active_pairs...)
    return CompiledGridOperator{N,typeof(loc),typeof(exprs),typeof(active_axes)}(loc, exprs, active_axes)
end

"""
    operator(compiled, face, shift)

Return the precomputed expression for `face` and codimension-sized `shift`.

If the face or shift does not require boundary treatment, the compiled interior
expression is returned. Shift coordinates for inactive axes are ignored.
"""
function operator(op::CompiledGridOperator{N}, face::Face, shift) where {N}
    ndims(face) == N || throw(ArgumentError("face dimension $(ndims(face)) does not match compiled operator dimension $N"))
    shifts = codim_shift_tuple(face, shift)

    if haskey(op.exprs, face)
        bnd = op.exprs[face]
        mask = get(op.active, face, ntuple(_ -> false, Val(codim(face))))
        key = CartesianShift(tuplemap(inactive_to_zero, shifts, mask))
        haskey(bnd, key) && return bnd[key]
    end

    return compiled_interior(op)
end

inactive_to_zero(shift::Shift, active::Bool) = active ? shift : Shift(0)

compiled_interior(op::CompiledGridOperator{N}) where {N} = op.exprs[interior_face(Val(N))][δ()]

all_faces(::Val{0}) = (Face(),)
function all_faces(::Val{N}) where {N}
    tail = all_faces(Val(N - 1))
    return (_prepend_axis(Lower(), tail)...,
            _prepend_axis(Span(), tail)...,
            _prepend_axis(Upper(), tail)...)
end

compile_face_pairs(::GridOperator, _, _, ::Tuple{}) = (), ()
function compile_face_pairs(op::GridOperator, loc, reach, faces::Tuple)
    face = first(faces)
    expr_tail, active_tail = compile_face_pairs(op, loc, reach, Base.tail(faces))
    codim(face) == 0 && return expr_tail, active_tail

    rule = combine_rules(op.rules, face)
    isnothing(rule) && return expr_tail, active_tail

    ranges, mask = compiled_shift_ranges(rule, face, reach)
    tuple_any(mask) || return expr_tail, active_tail

    expr_pair = face => compile_face(op, face, loc, ranges)
    active_pair = face => mask
    return (expr_pair, expr_tail...), (active_pair, active_tail...)
end

function compile_face(op::GridOperator, face::Face, loc, ranges::Tuple)
    return Binding(compile_shift_pairs(op, face, loc, ranges)...)
end

function compile_shift_pairs(op::GridOperator, face::Face, loc, ranges::Tuple)
    return compile_shift_pairs(op, face, loc, ranges, ())
end
function compile_shift_pairs(op::GridOperator, face::Face, loc, ::Tuple{}, prefix::Tuple)
    shift = CartesianShift(prefix...)
    return (shift => operator(op, face, loc, shift),)
end
function compile_shift_pairs(op::GridOperator, face::Face, loc, ranges::Tuple, prefix::Tuple)
    return compile_shift_range(op, face, loc, first(ranges), Base.tail(ranges), prefix)
end

compile_shift_range(::GridOperator, ::Face, _, ::Tuple{}, ::Tuple, ::Tuple) = ()
function compile_shift_range(op::GridOperator, face::Face, loc, range::Tuple, rest::Tuple, prefix::Tuple)
    shift = first(range)
    pairs = compile_shift_pairs(op, face, loc, rest, (prefix..., shift))
    return (pairs..., compile_shift_range(op, face, loc, Base.tail(range), rest, prefix)...)
end

function compiled_shift_ranges(rule, face::Face, reach)
    active_axes = rule_axes(rule, face)
    return compiled_shift_ranges(boundary_axes(face), active_axes, face, reach, Val(ndims(face)))
end

compiled_shift_ranges(::Tuple{}, ::Tuple, ::Face, _, ::Val) = (), ()
function compiled_shift_ranges(axes::Tuple, active_axes::Tuple, face::Face, reach, ::Val{N}) where {N}
    I = first(axes)
    side_face = codim1_face(Val(N), Val(I), face.axes[I])
    active = has_axis(active_axes, I) && haskey(reach, side_face)
    range = active ? reach[side_face] : (Shift(0),)
    rest_ranges, rest_mask = compiled_shift_ranges(Base.tail(axes), active_axes, face, reach, Val(N))
    return (range, rest_ranges...), (active, rest_mask...)
end

tuple_any(::Tuple{}) = false
function tuple_any(values::Tuple)
    first(values) && return true
    return tuple_any(Base.tail(values))
end

rule_axes(rule, face::Face) = boundary_axes(face)
function rule_axes(rule::CombinedRule, face::Face)
    return rule_pair_axes(rule.rules)
end

rule_pair_axes(::Tuple{}) = ()
function rule_pair_axes(pairs::Tuple)
    pair = first(pairs)
    return merge_axes(boundary_axes(pair.first), rule_pair_axes(Base.tail(pairs)))
end

merge_axes(::Tuple{}, axes::Tuple) = axes
function merge_axes(axes::Tuple, rest::Tuple)
    axis = first(axes)
    merged = has_axis(rest, axis) ? rest : (axis, rest...)
    return merge_axes(Base.tail(axes), merged)
end

has_axis(::Tuple{}, ::Integer) = false
function has_axis(axes::Tuple, axis::Integer)
    first(axes) == axis && return true
    return has_axis(Base.tail(axes), axis)
end

function codim1_face(::Val{N}, ::Val{I}, axis::A) where {N,I,A<:AxisFace}
    return Face(ntuple(j -> j == I ? axis : Span(), Val(N))...)
end

function stencil_reach(expr::STerm, loc::NTuple{N,Space}) where {N}
    nu = nonuniforms(expr)
    return Binding(stencil_reach_pairs(nu, loc, Val(1), Val(N))...)
end

function stencil_reach_pairs(nu::Nonuniforms, loc, ::Val{I}, ::Val{N}) where {I,N}
    I > N && return ()
    lower = axis_reach(nu, loc, Val(I), Lower())
    upper = axis_reach(nu, loc, Val(I), Upper())
    lower_pairs = isempty(lower) ? () : (codim1_face(Val(N), Val(I), Lower()) => lower,)
    upper_pairs = isempty(upper) ? () : (codim1_face(Val(N), Val(I), Upper()) => upper,)
    return (lower_pairs..., upper_pairs..., stencil_reach_pairs(nu, loc, Val(I + 1), Val(N))...)
end

function axis_reach(nu::Nonuniforms, loc, ::Val{I}, ::Lower) where {I}
    maxshift = lower_reach_stencils(values(stencils(nu)), loc, Val(I), -1)
    return maxshift < 0 ? () : shift_range(0, maxshift)
end

function axis_reach(nu::Nonuniforms, loc, ::Val{I}, ::Upper) where {I}
    start = upper_start_shift(loc[I])
    minshift = upper_reach_stencils(values(stencils(nu)), loc, Val(I), start, start + 1)
    return minshift > start ? () : shift_range(minshift, start)
end

lower_reach_stencils(::Tuple{}, _, ::Val, maxshift) = maxshift
function lower_reach_stencils(stencils::Tuple, loc, ::Val{I}, maxshift) where {I}
    stencil = first(stencils)
    read_loc = stencil_axis_location(stencil, loc, Val(I))
    next = lower_reach_shifts(stencil.shifts, value(offset(read_loc)), Val(I), maxshift)
    return lower_reach_stencils(Base.tail(stencils), loc, Val(I), next)
end

lower_reach_shifts(::Tuple{}, _, ::Val, maxshift) = maxshift
function lower_reach_shifts(shifts::Tuple, off, ::Val{I}, maxshift) where {I}
    shift = first(shifts)
    bound = floor(Int, -(value(shift.shifts[I]) + off))
    return lower_reach_shifts(Base.tail(shifts), off, Val(I), max(maxshift, bound))
end

upper_reach_stencils(::Tuple{}, _, ::Val, _, minshift) = minshift
function upper_reach_stencils(stencils::Tuple, loc, ::Val{I}, start, minshift) where {I}
    stencil = first(stencils)
    read_loc = stencil_axis_location(stencil, loc, Val(I))
    next = upper_reach_shifts(stencil.shifts, value(offset(read_loc)), Val(I), start, minshift)
    return upper_reach_stencils(Base.tail(stencils), loc, Val(I), start, next)
end

upper_reach_shifts(::Tuple{}, _, ::Val, _, minshift) = minshift
function upper_reach_shifts(shifts::Tuple, off, ::Val{I}, start, minshift) where {I}
    shift = first(shifts)
    bound = ceil(Int, -(value(shift.shifts[I]) + off))
    next = bound <= start ? min(minshift, bound) : minshift
    return upper_reach_shifts(Base.tail(shifts), off, Val(I), start, next)
end

function shift_range(first::Integer, last::Integer)
    first > last && return ()
    return (Shift(first), shift_range(first + 1, last)...)
end

stencil_axis_location(stencil::Stencil, loc, ::Val{I}) where {I} = stencil.location[I]

upper_start_shift(::Point) = 0
upper_start_shift(::Segment) = -1
