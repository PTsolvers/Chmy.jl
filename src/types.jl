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
Location on a staggered grid.
"""
const Location = Union{Point, Segment}

# head of a Chmy expression
@data HeadImpl begin
    export Call, Comp, Locs, Inds

    Call(Operator)
    Comp(Tuple{Vararg{Int}})
    Locs(Tuple{Vararg{Location}})
    Inds()
end
using .HeadImpl
const Head = HeadImpl.Type

# DTerm definition
@data DTermImpl begin
    export Literal, Index, Tensor, IdTensor, ZeroTensor, DExpr

    Literal(Number)
    Index(Int)
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

struct TensorComponents
    dims::Int
    rank::Int
    kind::TensorKind
    data::Vector{DTerm}
end

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
