"""
    compute(expr, [binding, inds...])

Compute a Chmy expression against a concrete [`Binding`](@ref) and optional
runtime grid indices.

`compute` differs from [`evaluate`](@ref): it lowers the symbolic expression to a
plain Julia expression and executes it, replacing bound symbolic terms by the
concrete data stored in `binding`. Scalar bindings are read directly, while
indexed expressions can read from array-valued bindings using `inds...`.

Calling `compute(expr)` uses an empty binding.
"""
Base.@propagate_inbounds function compute(expr::STerm, binding::Binding, inds::Vararg{Integer,N}) where {N}
    compute_expr(expr, binding, inds)
end
compute(expr::STerm) = compute(expr, Binding())

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
# Exact bindings for uniform expressions can be substituted directly because any
# location/grid indexing has already been erased symbolically.
function direct_bound_expr(term::STerm, bnd)
    idx = expr_idx(bnd, term)
    isnothing(idx) && return nothing
    dtype = bnd.data[idx]
    if isuniform(term)
        return :(b.data[$idx])
    elseif term isa STensor
        dtype <: Number || error("unsupported data type $dtype for term $term")
        return :(b.data[$idx])
    else
        return nothing
    end
end

to_expr(expr::STerm, bnd) = something(direct_bound_expr(expr, bnd), expr)
to_expr(::SRef{F}, bnd) where {F} = F
to_expr(sf::SFun, bnd) = sf.f

# Calls are lowered structurally by recursively translating all children.
function to_expr(expr::SExpr{Call}, bnd)
    direct = direct_bound_expr(expr, bnd)
    !isnothing(direct) && return direct
    return Expr(:call, map(arg -> to_expr(arg, bnd), children(expr))...)
end
function to_expr(expr::SExpr{Node}, bnd)
    direct = direct_bound_expr(expr, bnd)
    !isnothing(direct) && return direct
    return to_expr(argument(expr), bnd)
end
to_expr(::SLiteral{Value}, bnd) where {Value} = Value
to_expr(::SIndex{i}, bnd) where {i} = :(I[$i])
to_expr(term::STensor, bnd) = something(direct_bound_expr(term, bnd), term)
function to_expr(expr::SExpr{Ind}, bnd)
    arg = argument(expr)
    inds = indices(expr)
    direct = direct_bound_expr(arg, bnd)
    !isnothing(direct) && return direct
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
