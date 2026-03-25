abstract type AbstractAveraging <: STerm end

(a::AbstractAveraging)(args::Vararg{STerm}) = SExpr(Call(), a, args...)

abstract type AbstractPartialAveraging{I} <: STerm end

(pd::AbstractPartialAveraging)(arg::STerm) = SExpr(Call(), pd, arg)

struct LiftedPartialAveraging{I,Op} <: AbstractPartialAveraging{I}
    op::Op
end

LiftedPartialAveraging{I}(op::STerm) where {I} = LiftedPartialAveraging{I,typeof(op)}(op)

(∂::LiftedPartialAveraging)(arg::STerm) = SExpr(Call(), ∂, arg)

function stencil_rule(∂::LiftedPartialAveraging{I}, args, loc, inds) where {I}
    return lift(∂.op, args, loc, inds, Val(I))
end

function stencil_rule(∂::LiftedPartialAveraging{I}, args, inds) where {I}
    return lift(∂.op, args, inds, Val(I))
end

struct PartialAveraging{Op}
    op::Op
end

Base.getindex(∂::PartialAveraging, i::Integer) = LiftedPartialAveraging{i}(∂.op)

(∂::PartialAveraging)(arg::STerm, i::Integer) = ∂[i](arg)

struct StaggeredLinearAveraging <: AbstractAveraging end

function stencil_rule(::StaggeredLinearAveraging, args::Tuple{STerm}, loc::Tuple{Point}, inds::Tuple{STerm})
    f, i = only(args), only(inds)
    l = Segment()
    return (f[l][i] + f[l][i-1]) // 2
end

function stencil_rule(::StaggeredLinearAveraging, args::Tuple{STerm}, loc::Tuple{Segment}, inds::Tuple{STerm})
    f, i = only(args), only(inds)
    l = Point()
    return (f[l][i+1] + f[l][i]) // 2
end
