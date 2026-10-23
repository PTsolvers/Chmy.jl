struct Lower end
struct Upper end
struct Span  end

const Side = Union{Lower, Upper}
const FaceMarker = Union{Span, Side}

abstract type FacePattern end

dim(p::FacePattern, n) = n - codim(p, n)

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
struct EitherRegion{P, Q}
    p::P
    q::Q
end
const GridRegion = Union{NearBoundary, OnBoundary, Ghost, EitherRegion}

Base.in(ξ::Rational, ::Ghost) = ξ > zero(ξ)
Base.in(ξ::Rational, ::OnBoundary) = iszero(ξ)
Base.in(ξ::Rational, r::NearBoundary) = -r.w <= ξ < zero(ξ)
Base.in(ξ::Rational, r::EitherRegion) = ξ ∈ r.p || ξ ∈ r.q

Base.:|(p::GridRegion, q::GridRegion) = EitherRegion(p, q)

# lowering context

struct Far end

# means that the sample is near the side of the axis, at the signed distance ξ from the wall along the outward
# normal, i.e. ξ > 0 in the ghost region, ξ = 0 on the wall, and ξ < 0 inside the domain
struct Near
    side::Side
    ξ::Rational{Int}
end

const Block{N} = NTuple{N, Union{Far, Near}}

position(::Far, o) = Far()
position(x::Near, o) = Near(x.side, x.ξ + outward(x.side) * o)

# the direction of the outward normal along the axis
outward(::Lower) = -1
outward(::Upper) = 1

# ghost axes are resolved first
# then boundary axes
# then near-boundary
severity(ξ) = ξ > 0 ? 0 : ξ == 0 ? 1 : 2
