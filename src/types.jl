## DTerm definition

# Head of a Chmy expression

@data HeadImpl begin
    export Call, Comp, Locs, Inds

    Call(Operator)
    Comp(Tuple{Vararg{Int}})
    Locs(Tuple{Vararg{Location}})
    Inds()
end
using .HeadImpl
const Head = HeadImpl.Type

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
