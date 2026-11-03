module Kind
    struct Sym end
    struct Alt end
    struct Diag end
    struct None end
end
const TensorKind = Union{Kind.Sym, Kind.Alt, Kind.Diag, Kind.None}

"""
Abstract supertype for all Chmy operators.
"""
abstract type Operator end

# locations

"""
Point location on a staggered grid.
"""
struct Point end

"""
Segment location on a staggered grid.
"""
struct Segment end

"""
Location of the lower wall along the axis.
"""
struct Lower end

"""
Location of the upper wall along the axis.
"""
struct Upper end

const 𝓅 = Point()
const 𝓈 = Segment()
const ↓ = Lower()
const ↑ = Upper()

"""
Location on a staggered grid.
"""
const Location = Union{Point, Segment, Lower, Upper, Colon}

# global positions along an axis: the traces on the lower and upper walls, and values independent of the axis
const Global = Union{Lower, Upper, Colon}

# a position along an axis: a lattice coordinate, or a global position
const Position = Union{Rational{Int}, Global}

# a position as given by users, offsets are converted to `Rational{Int}`
const PositionLike = Union{Integer, Rational, Global}

# head of a Chmy expression
@data HeadImpl begin
    export Call, Comp, Inds, Pack

    Call(Operator)
    Comp(Tuple{Vararg{Int}})
    Inds(Tuple{Vararg{Position}})
    # a tensor in component form, the children are the stored entries
    struct Pack
        rank::Int
        dims::Int
        kind::TensorKind
    end
end
using .HeadImpl
const Head = HeadImpl.Type

# DTerm definition
@data DTermImpl begin
    export Literal, Index, Coordinate, Tensor, IdTensor, ZeroTensor, DExpr

    Literal(Number)
    Index(Int)
    Coordinate(Int)
    struct Tensor
        rank::Int
        name::Symbol
        kind::TensorKind
        uniform::Bool
    end
    ZeroTensor(Int)
    IdTensor(Int)
    struct DExpr
        head::Head
        args::Tuple{Vararg{DTermImpl}}
        rank::Int
        hash::UInt
    end
end
using .DTermImpl
const DTerm = DTermImpl.Type

struct Fun{F} <: Operator
    f::F
    function Fun(f)
        if Base.issingletontype(typeof(f))
            return new{typeof(f)}(f)
        else
            throw(ArgumentError("function must be a singleton type"))
        end
    end
end

struct BroadcastedFun{F} <: Operator
    f::F
    function BroadcastedFun(f)
        Base.issingletontype(typeof(f)) || throw(ArgumentError("function must be a singleton type"))
        return new{typeof(f)}(f)
    end
end

"""
    AnyNormal()

The outward unit normal of the wall, which matches the normal component in a tensor pattern, and is resolved to the
normal vector of the matched wall in the data of override rules. See also `𝐧`.
"""
struct AnyNormal <: Operator end

"""
    AnyTangent(slot = 1)

The tangential components in a tensor pattern. In the data of override rules, it is resolved to the unit vector along
the component matched by the `slot`-th tangential index of the pattern. See also `𝐭`.
"""
struct AnyTangent <: Operator
    slot::Int
end
AnyTangent() = AnyTangent(1)

const 𝐧 = AnyNormal()
const 𝐭 = AnyTangent()
