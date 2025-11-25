struct Binding{Exprs,Data}
    exprs::Exprs
    data::Data
end

function Binding(kvs::Vararg{Pair,N}) where {N}
    exprs = ntuple(i -> kvs[i].first, Val(N))
    data = ntuple(i -> kvs[i].second, Val(N))
    return Binding(exprs, data)
end

_expr_idx(bnd::Binding, expr) = findfirst(Base.Fix2(===, expr), bnd.exprs)

Base.getindex(bnd::Binding, expr) = bnd.data[_expr_idx(bnd, expr)]

Base.haskey(bnd::Binding, expr) = !isnothing(_expr_idx(bnd, expr))

function push(bnd::Binding, kv::Pair)
    if !haskey(bnd, kv.first)
        exprs = (bnd.exprs..., kv.first)
        data = (bnd.data..., kv.second)
        return Binding(exprs, data)
    else
        idx = _expr_idx(bnd, kv.first)
        data = ntuple(i -> i == idx ? kv.second : bnd.data[i], Val(length(bnd.data)))
        return Binding(bnd.exprs, data)
    end
end

function push(bnd::Binding, kvs::Vararg{Pair,N}) where {N}
    return push(push(bnd, first(kvs)), Base.tail(kvs)...)
end

function binding_types(bnd::Binding)
    return Binding(bnd.exprs, ntuple(i -> typeof(bnd.data[i]), Val(length(bnd.data))))
end
