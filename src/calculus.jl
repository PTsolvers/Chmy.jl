abstract type DifferentialOperator <: Operator end

"""
    Gradient(op)

Gradient differential operator.
"""
struct Gradient{Op} <: DifferentialOperator
    op::Op
end

"""
    Divergence(op)

Divergence differential operator.
"""
struct Divergence{Op} <: DifferentialOperator
    op::Op
end

"""
    Curl(op)

Curl differential operator.
"""
struct Curl{Op} <: DifferentialOperator
    op::Op
end

tensorrank(::Gradient, args) = tensorrank(only(args)) + 1
tensorrank(::Divergence, args) = tensorrank(only(args)) - 1
tensorrank(::Curl, args) = 1

function checkranks(::Gradient, ::NTuple{N, DTerm}) where {N}
    N == 1 || throw(ArgumentError("gradient requires exactly one argument"))
    return
end
function checkranks(::Divergence, args::NTuple{N, DTerm}) where {N}
    N == 1 || throw(ArgumentError("divergence requires exactly one argument"))
    tensorrank(only(args)) > 0 || throw(ArgumentError("divergence requires a tensor of positive rank"))
    return
end
function checkranks(::Curl, args::NTuple{N, DTerm}) where {N}
    N == 1 || throw(ArgumentError("curl requires exactly one argument"))
    tensorrank(only(args)) == 1 || throw(ArgumentError("curl requires a rank-1 tensor (vector)"))
    return
end

function evaluate(grad::Gradient, args::Tuple{DTerm}, dims::Integer)
    t = only(args)
    return pack(Kind.None(), tensorrank(t) + 1, dims) do I
        grad.op[I[1]](component(t, Base.tail(I)))
    end
end
function evaluate(divg::Divergence, args::Tuple{DTerm}, dims::Integer)
    t = only(args)
    entry(I) = component_sum(DTerm[divg.op[i](t[i, I...]) for i in 1:dims])
    tensorrank(t) == 1 && return entry(())
    return pack(entry, Kind.None(), tensorrank(t) - 1, dims)
end
function evaluate(curl::Curl, args::Tuple{DTerm}, dims::Integer)
    v = only(args)
    dims == 3 || throw(ArgumentError(LazyString("curl is only supported for 3D vector fields, got dimension ", dims)))
    return pack(Kind.None(), 1, dims) do (i,)
        j, k = mod1(i + 1, 3), mod1(i + 2, 3)
        curl.op[j](v[k]) - curl.op[k](v[j])
    end
end

# the component `I` of a tensor in component form, or the scalar itself
component(t::DTerm, ::Tuple{}) = t
component(t::DTerm, I::Tuple) = t[I...]
