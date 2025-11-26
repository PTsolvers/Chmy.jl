abstract type AbstractDerivative <: STerm end

(d::AbstractDerivative)(args::Vararg{STerm}) = SExpr(Call(), d, args...)

abstract type AbstractPartialDerivative{I} <: STerm end

struct LiftedPartialDerivative{I,Op} <: AbstractPartialDerivative{I}
    op::Op
end

LiftedPartialDerivative{I}(op::STerm) where {I} = LiftedPartialDerivative{I,typeof(op)}(op)

function stencil_rule(∂::LiftedPartialDerivative{I}, args, loc, inds) where {I}
    return lift(∂.op, args, loc, inds, Val(I))
end

function stencil_rule(∂::LiftedPartialDerivative{I}, args, inds) where {I}
    return lift(∂.op, args, inds, Val(I))
end

struct PartialDerivative{Op}
    op::Op
end

(p::PartialDerivative)(arg::STerm, i::Integer) = SExpr(Call(), LiftedPartialDerivative{i}(p.op), arg)

struct CentralDifference <: AbstractDerivative end

function stencil_rule(::CentralDifference, args::Tuple{STerm}, loc::Tuple{Space}, inds::Tuple{STerm})
    f, l, i = only(args), only(loc), only(inds)
    return 0.5 * (f[l][i+1] - f[l][i-1])
end

function stencil_rule(::CentralDifference, args::Tuple{STerm}, inds::Tuple{STerm})
    f, i = only(args), only(inds)
    return 0.5 * (f[i+1] - f[i-1])
end

struct StaggeredCentralDifference <: AbstractDerivative end

function stencil_rule(::StaggeredCentralDifference, args::Tuple{STerm}, loc::Tuple{Point}, inds::Tuple{STerm})
    f, i = only(args), only(inds)
    l = Segment()
    return 0.5 * (f[l][i] + f[l][i-1])
end

function stencil_rule(::StaggeredCentralDifference, args::Tuple{STerm}, loc::Tuple{Segment}, inds::Tuple{STerm})
    f, i = only(args), only(inds)
    l = Point()
    return 0.5 * (f[l][i+1] + f[l][i])
end
