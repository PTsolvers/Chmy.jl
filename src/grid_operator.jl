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

CartesianShift(inds::NTuple{N,STerm}) where {N} = CartesianShift(map(Shift, inds))
CartesianShift(inds::Vararg{STerm}) = CartesianShift(inds)

"""
    δ(shifts...)

Construct a [`CartesianShift`](@ref) from integer shifts.
"""
δ(inds::Vararg{Integer,N}) where {N} = CartesianShift(ntuple(i -> Shift(inds[i]), Val(N)))
δ(inds::Vararg{Real,N}) where {N} = throw(ArgumentError("δ shifts must be integers, got $inds"))

Base.isless(a::CartesianShift, b::CartesianShift) = isless_tuple(a.shifts, b.shifts)

Base.getindex(s::STerm, o::CartesianShift) = getindex(s, o.shifts...)

Shift(::SIndex) = Shift(0)
Shift(shift::Shift) = shift
function Shift(ind::SExpr{SFun})
    op = operation(ind)
    i, o = arguments(ind)
    if !(op === SFun(+) || op === SFun(-)) || !isa(i, SIndex) || !isa(o, SLiteral)
        error("only indexing expression of the format `(i + c)` or `(i - c)` are supported, where c is `SLiteral` are supported, got '$ind'")
    end
    return operation(ind) === SFun(+) ? Shift(o) : Shift(-o)
end

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
reach_nonuniforms(stencils::Tuple) = mergewith(constval ∘ max, map(reach, stencils)...)

"""
    nonuniforms(expr)

Returns a Binding containing pairs of nonuniform fields and their corresponding stencils.
"""
nonuniforms(::STerm) = Nonuniforms()
nonuniforms(expr::SExpr{SFun}) = mergewith(merge, map(nonuniforms, arguments(expr))...)
function nonuniforms(expr::SExpr{SSub})
    inds = indices(expr)
    shift = CartesianShift(inds)
    arg = argument(expr)
    loc = nonuniform_location(arg, inds)
    stencil = Stencil(loc, shift)
    return Nonuniforms(arg => stencil)
end

nonuniform_location(_, inds::Tuple{Vararg{STerm,N}}) where {N} = ntuple(i -> Point(), Val(N))
nonuniform_location(arg::SExpr{SAt}, ::Tuple{Vararg{STerm,N}}) where {N} = locations(arg)
