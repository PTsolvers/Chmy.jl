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

function matches(p::Wall, f::Face)
    return codim(f) == 1 && p.axis <= ndims(f) && f.axes[p.axis] == p.side
end

struct AlongAxis <: WallPattern
    axis::Int
end

function matches(p::AlongAxis, f::Face)
    return codim(f) == 1 && p.axis <= ndims(f) && f.axes[p.axis] isa Side
end

struct AnyWall <: WallPattern end

matches(::AnyWall, f::Face) = codim(f) == 1

struct Left <: WallPattern end

matches(::Left, f::Face) = codim(f) == 1 && f.axes[1] == Lower()

struct Right <: WallPattern end

matches(::Right, f::Face) = codim(f) == 1 && f.axes[1] == Upper()

struct Bottom <: WallPattern end

matches(::Bottom, f::Face) = codim(f) == 1 && ndims(f) > 1 && f.axes[2] == Lower()

struct Top <: WallPattern end

matches(::Top, f::Face) = codim(f) == 1 && ndims(f) > 1 && f.axes[2] == Upper()

struct Back <: WallPattern end

matches(::Back, f::Face) = codim(f) == 1 && ndims(f) > 2 && f.axes[3] == Lower()

struct Front <: WallPattern end

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
