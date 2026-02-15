Base.@propagate_inbounds function compute(expr::STerm, binding::Binding, inds::Vararg{Integer,N}) where {N}
    compute_expr(expr, binding, inds)
end
compute(expr::STerm) = compute(expr, Binding())

@generated function compute_expr(expr, b, I)
    expri = expr.instance
    bndi = Binding(b.types[1].instance, Tuple(b.types[2].types))
    return quote
        Base.@_propagate_inbounds_meta
        $(to_expr(expri, bndi))
    end
end

to_expr(expr::STerm, bnd) = expr

to_expr(::SRef{F}, bnd) where {F} = F
to_expr(sf::SFun, bnd) = sf.f

function to_expr(expr::SExpr{Call}, bnd)
    return Expr(:call, map(arg -> to_expr(arg, bnd), children(expr))...)
end

to_expr(::SUniform{Value}, bnd) where {Value} = Value
to_expr(::SIndex{i}, bnd) where {i} = :(I[$i])

function to_expr(expr::SExpr{Ind}, bnd)
    arg = argument(expr)
    inds = indices(expr)
    idx = _expr_idx(bnd, arg)
    isnothing(idx) && return expr
    dtype = bnd.data[idx]
    if dtype <: Number
        return :(b.data[$idx])
    elseif dtype <: AbstractArray
        return Expr(:ref, :(b.data[$idx]), map(i -> to_expr(i, bnd), inds)...)
    else
        error("unsupported data type $dtype for expression $expr")
    end
end
