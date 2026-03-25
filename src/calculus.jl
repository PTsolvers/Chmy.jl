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

tensorrank(::Gradient, t)   = tensorrank(t) + 1
tensorrank(::Divergence, t) = tensorrank(t) - 1
tensorrank(::Curl, t)       = 1

Tensor{D}(grad::Gradient, s::STerm) where {D} = Vec(ntuple(i -> grad.op[i](s), D)...)
@generated function Tensor{D}(grad::Gradient, t::Tensor{D,R}) where {D,R}
    ex = Expr(:call, :(Tensor{$D,$(R + 1)}))
    for idx in CartesianIndices(ntuple(_ -> D, Val(R + 1)))
        I = Tuple(idx)
        push!(ex.args, :(grad.op[$(I[1])](t[$(I[2:end]...)])))
    end
    quote
        @inline
        return $ex
    end
end
@generated function Tensor{D}(divg::Divergence, t::Tensor{D,R}) where {D,R}
    if R == 1
        ex = Expr(:call, :+)
        for i in 1:D
            push!(ex.args, :(divg.op[$i](t[$i])))
        end
    else
        ex = Expr(:call, :(Tensor{$D,$(R - 1)}))
        for idx in CartesianIndices(ntuple(_ -> D, Val(R - 1)))
            I = Tuple(idx)
            comp = Expr(:call, :+)
            for i in 1:D
                push!(comp.args, :(divg.op[$i](t[$i, $(I...)])))
            end
            push!(ex.args, comp)
        end
    end
    quote
        @inline
        return $ex
    end
end
function Tensor{D}(::Curl, ::Vec{D}) where {D}
    throw(ArgumentError("curl is only supported for 2D and 3D vector fields, got dimension $D"))
end
function Tensor{2}(curl::Curl, v::Vec{2})
    return curl.op[1](v[2]) - curl.op[2](v[1])
end
function Tensor{3}(curl::Curl, v::Vec{3})
    return Vec{3}(curl.op[2](v[3]) - curl.op[3](v[2]),
                  curl.op[3](v[1]) - curl.op[1](v[3]),
                  curl.op[1](v[2]) - curl.op[2](v[1]))
end
