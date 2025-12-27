abstract type DifferentialOperator <: STerm end

(op::DifferentialOperator)(args::Vararg{STerm}) = SExpr(Call(), op, args...)

abstract type AbstractGradient <: DifferentialOperator end
abstract type AbstractDivergence <: DifferentialOperator end
abstract type AbstractCurl <: DifferentialOperator end

struct Gradient{Op} <: AbstractGradient
    op::Op
end

struct Divergence{Op} <: AbstractDivergence
    op::Op
end

struct Curl{Op} <: AbstractCurl
    op::Op
end
