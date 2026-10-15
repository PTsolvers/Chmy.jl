struct Lower end
struct Upper end
struct Span  end

const Side = Union{Lower, Upper}
const FaceMarker = Union{Span, Side}

abstract type FacePattern end

struct Face{N} <: FacePattern
    axes::NTuple{N, FaceMarker}
end
Face(axes::Vararg{FaceMarker}) = Face(axes)

Base.ndims(f::Face) = length(f.axes)

dim(f::Face) = count(==(Span()), f.axes)
codim(f::Face) = count(!=(Span()), f.axes)

function codim(f::Face, n)
    ndims(f) == n || throw(ArgumentError("number of dimensions != ndims(f)"))
    return codim(f)
end

matches(p::Face, f::Face) = p.axes == f.axes

abstract type WallPattern <: FacePattern end

codim(::WallPattern, n) = 1

struct Wall <: WallPattern
    axis::Int
    side::Side
end

struct AnyWall <: WallPattern end

struct AlongAxis <: WallPattern
    axis::Int
end

struct Left <: WallPattern end

struct Right <: WallPattern end

struct Bottom <: WallPattern end

struct Top <: WallPattern end

struct Back <: WallPattern end

struct Front <: WallPattern end

function matches(p::Wall, f::Face)
    return codim(f) == 1 && p.axis <= ndims(f) && f.axes[p.axis] == p.side
end
matches(::AnyWall, f::Face) = codim(f) == 1
function matches(p::AlongAxis, f::Face)
    return codim(f) == 1 && p.axis <= ndims(f) && f.axes[p.axis] isa Side
end
matches(::Left, f::Face) = codim(f) == 1 && f.axes[1] == Lower()
matches(::Right, f::Face) = codim(f) == 1 && f.axes[1] == Upper()
matches(::Bottom, f::Face) = codim(f) == 1 && ndims(f) > 1 && f.axes[2] == Lower()
matches(::Top, f::Face) = codim(f) == 1 && ndims(f) > 1 && f.axes[2] == Upper()
matches(::Back, f::Face) = codim(f) == 1 && ndims(f) > 2 && f.axes[3] == Lower()
matches(::Front, f::Face) = codim(f) == 1 && ndims(f) > 2 && f.axes[3] == Upper()

struct AnyCorner <: FacePattern end

codim(::AnyCorner, n) = n

matches(::AnyCorner, f::Face) = dim(f) == 0

struct Either{P, Q} <: FacePattern
    p::P
    q::Q
end

function codim(e::Either, n)
    c = codim(e.p, n)
    c == codim(e.q, n) || throw(ArgumentError("codimensions of patterns must be the same"))
    return c
end

matches(e::Either, f::Face) = matches(e.p, f) || matches(e.q, f)

Base.:|(p::FacePattern, q::FacePattern) = Either(p, q)

struct Both{P, Q} <: FacePattern
    p::P
    q::Q
end

function codim(e::Both, n)
    c = codim(e.p, n)
    c == codim(e.q, n) || throw(ArgumentError("codimensions of patterns must be the same"))
    return c
end

matches(b::Both, f::Face) = matches(b.p, f) && matches(b.q, f)

Base.:&(p::FacePattern, q::FacePattern) = Both(p, q)

struct Not{P} <: FacePattern
    p::P
end

codim(neg::Not, n) = codim(neg.p, n)

matches(neg::Not, f::Face) = codim(neg, ndims(f)) == codim(f) && !matches(neg.p, f)

Base.:!(p::FacePattern) = Not(p)

# grid regions

struct Ghost end
struct OnBoundary end
struct NearBoundary
    w::Rational{Int}
end
const GridRegion = Union{NearBoundary, OnBoundary, Ghost}

Base.in(ξ::Rational, ::Ghost) = ξ < zero(ξ)
Base.in(ξ::Rational, ::OnBoundary) = iszero(ξ)
Base.in(ξ::Rational, r::NearBoundary) = zero(ξ) < ξ <= r.w

# lowering context

struct Far end
struct At
    side::Side
    k::Int
end
const LowerAxes{N} = NTuple{N, Union{Far, At}}

offset(::Lower, ::Point, i) = i
offset(::Upper, ::Point, i) = -i

offset(::Lower, ::Segment, i) = i + 1 // 2
offset(::Upper, ::Segment, i) = -i - 1 // 2

position(s, loc, ::Far) = Far()
position(s, loc, x::At) = Near(x.side, x.k + offset(x.side, loc, s))

# means that the sample is near the side of the axis at distance ξ
struct Near
    side::Side
    ξ::Rational{Int}
end

# ghost axes are resolved first
# then boundary axes
# then near-boundary
severity(ξ) = ξ < 0 ? 0 : ξ == 0 ? 1 : 2

# check all combinations of Near axes
# create faces for them
# check which rules match
# sort the groups
function candidates(pos::NTuple{N, Union{Far, Near}}, rules::Vector{<:Pair{<:FacePattern, <:GridRegion}}) where {N}
    # precompute codimensions of rules for known N once
    # this step also throws for malformed rules
    codims = map(r -> codim(first(r), N), rules)
    # find all near axes of the position
    near = findall(x -> x isa Near, pos)
    # each group corresponds to a single face that could potentially contribute a rule to resolve the selector
    # we can have multiple rules in a group because some rules might return nothing, then the next one will be tried
    groups = Pair{Vector{Int}, Vector{Int}}[]
    # iterate over all faces spanning near axes of position
    # for p near axes, there are 2^p combinations of these axes
    # but the first one would exclude all near axes
    for mask in 1:(2^length(near) - 1)
        axes = [near[i] for i in eachindex(near) if !iszero(mask & 1 << (i - 1))]
        # create Face from the near axes
        face = ntuple(Val(N)) do i
            i ∈ axes ? pos[i].side : Span()
        end |> Face
        group = Int[]
        for (i, (pat, reg)) in enumerate(rules)
            # check codim
            codims[i] == length(axes) || continue
            # check if face matches
            matches(pat, face) || continue
            # check if region matches
            all(pos[k].ξ ∈ reg for k in axes) || continue
            # all checks passed, add the candidate rule's index to the group
            push!(group, i)
        end
        !isempty(group) && push!(groups, axes => group)
    end
    # we sort groups by codim first
    # then by "severity" (ghost axes come first, then boundary, then near-boundary)
    # then by the index of the first matching rule
    # tie-break by the axis index (lower first)
    function priority(kv)
        s = sort!([severity(pos[i].ξ) for i in kv[1]])
        return (-length(kv[1]), s, first(kv[2]), kv[1])
    end
    sort!(groups; by = priority)
    return groups
end

# struct SelectorPattern
#     facepat::FacePattern
#     region::GridRegion
# end

# struct SelectorRule
#     pattern::FacePattern
#     region::GridRegion
# end

# struct Selector <: Operator
#     rules::Vector{Rule}
#     default::DTerm
# end

# function lower(sel, locs, offsets, axes)
#     pos = map(position, offsets, locs, axes)
#     # TODO
#     return
# end
