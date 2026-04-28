struct Binding{Exprs,Data}
    exprs::Exprs
    data::Data
end

"""
    Binding(pairs...)

Create an immutable dictionary mapping Chmy expressions to data.

`Binding()` constructs an empty binding.
"""
function Binding(kvs::Vararg{Pair,N}) where {N}
    kvsu = sunique(kvs)
    exprs = ntuple(i -> kvsu[i].first, Val(length(kvsu)))
    data = ntuple(i -> kvsu[i].second, Val(length(kvsu)))
    return Binding(exprs, data)
end

# deduplicate the pairs by key
sunique(kv::Tuple{}) = kv
function sunique(kvs::Tuple{Vararg{Pair}})
    kv1 = first(kvs)
    rest = Base.tail(kvs)
    idx = findfirst(kv2 -> kv1.first === kv2.first, rest)
    isnothing(idx) && return (kv1, sunique(rest)...)
    return sunique(rest)
end

expr_idx(bnd::Binding, expr) = findfirst(Base.Fix2(===, expr), bnd.exprs)

"""
    length(bnd::Binding)

Return the number of entries stored in `bnd`.
"""
Base.length(bnd::Binding) = length(bnd.exprs)

"""
    getindex(bnd::Binding, expr)

Return the value associated with `expr` in `bnd`.

Throws a `BoundsError` if `expr` is not present.
"""
Base.getindex(bnd::Binding, expr) = bnd.data[expr_idx(bnd, expr)]

"""
    haskey(bnd::Binding, expr)

Return `true` if `bnd` contains a value for `expr`, and `false` otherwise.
"""
Base.haskey(bnd::Binding, expr) = !isnothing(expr_idx(bnd, expr))

"""
    get(bnd::Binding, expr, default)

Return the value associated with `expr` in `bnd`, or `default` if `expr` is not present.
"""
Base.get(bnd::Binding, expr, default) = haskey(bnd, expr) ? bnd[expr] : default

"""
    keys(bnd::Binding)

Return the tuple of expressions stored as keys in `bnd`.
"""
Base.keys(bnd::Binding) = bnd.exprs

"""
    values(bnd::Binding)

Return the tuple of values stored in `bnd`.
"""
Base.values(bnd::Binding) = bnd.data

"""
    pairstuple(bnd::Binding)

Return the contents of `bnd` as a tuple of `expr => value` pairs.
"""
pairstuple(bnd::Binding) = (pairs(bnd)...,)

"""
    push(bnd::Binding, pairs...)

Return a new binding with the given pairs inserted.

If a key is already present, its value is replaced.
"""
function push(bnd::Binding, kv::Pair)
    if !haskey(bnd, kv.first)
        exprs = (bnd.exprs..., kv.first)
        data = (bnd.data..., kv.second)
        return Binding(exprs, data)
    else
        idx = expr_idx(bnd, kv.first)
        data = ntuple(i -> i == idx ? kv.second : bnd.data[i], Val(length(bnd.data)))
        return Binding(bnd.exprs, data)
    end
end
function push(bnd::Binding, kvs::Vararg{Pair,N}) where {N}
    return push(push(bnd, first(kvs)), Base.tail(kvs)...)
end

Base.mergewith(_, bnd::Binding) = bnd
Base.mergewith(combine, bnd::Binding, others::Vararg{Binding,N}) where {N} = mergewith(combine, mergewith(combine, bnd, first(others)), Base.tail(others)...)
Base.mergewith(combine, bnd1::Binding, bnd2::Binding) = merge_bindings(combine, bnd1, pairstuple(bnd2))
merge_bindings(_, bnd, ::Tuple{}) = bnd
function merge_bindings(combine, bnd, kvs)
    k, v1 = first(kvs)
    if haskey(bnd, k)
        v2 = bnd[k]
        return merge_bindings(combine, push(bnd, k => combine(v1, v2)), Base.tail(kvs))
    end
    return merge_bindings(combine, push(bnd, first(kvs)), Base.tail(kvs))
end

"""
    binding_types(bnd::Binding)

Return a binding with the same keys as `bnd`, replacing each value by its
concrete type.

Can be used to inspect Julia expressions generated from the Chmy expression.
"""
function binding_types(bnd::Binding)
    return Binding(bnd.exprs, ntuple(i -> typeof(bnd.data[i]), Val(length(bnd.data))))
end

# Implement Adapt.jl interface to allow passing bindngs to GPU kernels
Adapt.adapt_structure(to, bnd::Binding) = Binding(bnd.exprs, Adapt.adapt(to, bnd.data))
