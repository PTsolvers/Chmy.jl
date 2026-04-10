"""
    compute(expr, [binding, inds...])

Compute a Chmy expression against a concrete [`Binding`](@ref) and optional
runtime grid indices.

`compute` differs from [`evaluate`](@ref): it lowers the symbolic expression to a
plain Julia expression and executes it, replacing bound symbolic terms by the
concrete data stored in `binding`. Scalar bindings are read directly, while
indexed expressions can read from array-valued bindings using `inds...`.

Omitting `binding` uses an empty binding `Binding()`.
"""
Base.@propagate_inbounds compute(expr::STerm, binding::Binding, inds::Vararg{Integer,N}) where {N} = compute_unwrapped(unwrap(expr), binding, inds...)
Base.@propagate_inbounds compute(expr::STerm, inds::Vararg{Integer,N}) where {N} = compute(expr, Binding(), inds...)

Base.@propagate_inbounds compute_unwrapped(expr::STerm, binding::Binding, inds::Vararg{Integer,N}) where {N} = compute_expr(expr, binding, inds)

# `compute` is implemented as generated function so a fully static symbolic term and
# the concrete binding types can be turned into plain Julia code with no
# runtime overhead.
@generated function compute_expr(expr, b, I)
    expri = expr.instance
    bndi = Binding(b.types[1].instance, Tuple(b.types[2].types))
    return quote
        Base.@_propagate_inbounds_meta
        $(to_expr(expri, bndi))
    end
end

"""
    to_expr(expr, binding_types(binding))

Translate a Chmy expression to a Julia `Expr` using the provided binding type
information.

This is the lowering step used internally by [`compute`](@ref). Symbolic terms
that appear in `binding` are replaced by reads from `binding.data`; unbound
symbolic terms are left symbolic in the generated expression.
"""
function to_expr end

function to_expr(expr::STerm, bnd)
    idx = expr_idx(bnd, expr)
    isnothing(idx) && return expr
    dtype = bnd.data[idx]
    if dtype <: Number
        return :(b.data[$idx])
    else
        error("unsupported data type $dtype for expression $expr")
    end
end

to_expr(::SRef{F}, bnd) where {F} = F
to_expr(sf::SFun, bnd) = sf.f

# Calls are lowered structurally by recursively translating all children.
to_expr(expr::SExpr{Call}, bnd) = Expr(:call, map(arg -> to_expr(arg, bnd), children(expr))...)
to_expr(expr::SNode, bnd) = to_expr(unwrap(expr), bnd)
to_expr(::SLiteral{Value}, bnd) where {Value} = Value
to_expr(::SIndex{i}, bnd) where {i} = :(I[$i])
function to_expr(expr::SExpr{Ind}, bnd)
    arg = argument(expr)
    inds = indices(expr)
    idx = expr_idx(bnd, arg)
    isnothing(idx) && return expr
    dtype = bnd.data[idx]
    # Indexed terms are the point where `compute` switches from symbolic Chmy
    # indexing to concrete array access against bound storage.
    if dtype <: Number
        return :(b.data[$idx])
    elseif dtype <: AbstractArray
        return Expr(:ref, :(b.data[$idx]), map(i -> to_expr(i, bnd), inds)...)
    else
        error("unsupported data type $dtype for expression $expr")
    end
end
