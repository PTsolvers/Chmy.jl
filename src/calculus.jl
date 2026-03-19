abstract type DifferentialOperator <: STerm end

(op::DifferentialOperator)(args::Vararg{STerm}) = SExpr(Call(), op, args...)

struct Gradient{Op} <: DifferentialOperator
    op::Op
end
Gradient(op::AbstractDerivative) = Gradient(PartialDerivative(op))

struct Divergence{Op} <: DifferentialOperator
    op::Op
end
Divergence(op::AbstractDerivative) = Divergence(PartialDerivative(op))

struct Curl{Op} <: DifferentialOperator
    op::Op
end
Curl(op::AbstractDerivative) = Curl(PartialDerivative(op))
