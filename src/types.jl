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

# the elements of an axis: the lower wall, the upper wall, and the whole axis
const Global = Union{Lower, Upper, Colon}

# the offset of a position from the origin of a frame, in grid steps
const Offset = Union{Integer, Rational}

"""
    Coord(frame, offset)

The position at `offset` along an axis from the origin of the `frame`: the grid point of the kernel for `:`, the lower
wall for `↓`, and the upper wall for `↑`. Integer offsets are points and half-integer offsets are segments, e.g.
`Coord(↑, -1//2)` is the last segment. Written `(frame, offset)` in samples, or just `offset` for the frame `:`.
"""
struct Coord
    frame::Global
    offset::Rational{Int}
    Coord(frame::Global, offset::Offset) = new(frame, checkoffset(offset))
end

# offsets along an axis are multiples of 1//2, as the positions on a staggered grid
function checkoffset(x)
    y = Rational{Int}(x)
    denominator(y) <= 2 || throw(ArgumentError(LazyString("offsets must be multiples of 1//2, got ", y)))
    return y
end

"""
    Trace(element)

The position without a coordinate along an axis: the trace on the lower wall `↓` or the upper wall `↑`, or the value
independent of the axis `:`. Written `↓`, `↑` and `:` in samples.
"""
struct Trace
    element::Global
end

"""
The absolute position along an axis, a [`Coord`](@ref) or a [`Trace`](@ref).
"""
const Position = Union{Coord, Trace}

# a position as given by users: tuples `(frame, offset)` are coordinates, numbers are offsets from the origin, and the
# elements of the axis are traces
const PositionLike = Union{Global, Position, Offset, Tuple{Global, Offset}}

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
    export Literal, Coordinate, Tensor, DExpr, AddMul

    Literal(Number)
    Coordinate(Int)
    struct Tensor
        rank::Int
        name::Symbol
        kind::TensorKind
        uniform::Bool
    end
    struct DExpr
        head::Head
        args::Tuple{Vararg{DTermImpl}}
        rank::Int
        hash::UInt
    end
    # a sum `coeff + Σ v * k`, mapping the terms `k` to their literal coefficients `v`, or a product `coeff * Π k ^ v`,
    # mapping the bases `k` to their powers `v`, with the keys sorted by `<ₛ`
    struct AddMul
        op::Union{typeof(+), typeof(*)}
        coeff::Number
        dict::LittleDict{DTermImpl, DTermImpl, Vector{DTermImpl}, Vector{DTermImpl}}
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

# the zero tensor and the identity tensor of rank `rank`, nullary operators, see `𝒪` and `ℐ`
struct ZeroTensor <: Operator
    rank::Int
end

struct IdTensor <: Operator
    rank::Int
end

const 𝐧 = AnyNormal()
const 𝐭 = AnyTangent()
