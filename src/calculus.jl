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

# the rules read the components of the argument through its component form, which checks the dimensions of packs

# the gradient differentiates the component `tail(comps)` of the argument along the axis `first(comps)`
function lower_rule(grad::Gradient, args, comps, offs, block)
    t = components(only(args), length(offs))
    return here(grad.op[first(comps)](t[Base.tail(comps)...]), offs)
end

function lower_rule(divg::Divergence, args, comps, offs, block)
    t = components(only(args), length(offs))
    return here(component_sum(DTerm[divg.op[i](t[i, comps...]) for i in eachindex(offs)]), offs)
end

function lower_rule(curl::Curl, args, (i,), offs, block)
    length(offs) == 3 || throw(ArgumentError(LazyString("curl is only supported for 3D vector fields, got dimension ", length(offs))))
    v = components(only(args), 3)
    j, k = mod1(i + 1, 3), mod1(i + 2, 3)
    return here(curl.op[j](v[k]) - curl.op[k](v[j]), offs)
end
