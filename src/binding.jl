struct Binding{K,D}
    keys::K
    data::D
end

"""
    Binding(pairs...)

Create an immutable dictionary binding Chmy expressions to data.
"""
Binding(kvs::Vararg{Pair,N}) where {N} = Binding(zip(kvs...)...)
Binding() = Binding((), ())

"""
    findkey(binding, key)

Returns the index of the key in the binding, or nothing if the key is not in the binding.
"""
findkey(b::Binding, key) = findfirst(Base.Fix1(isequal, key), b.keys)

"""
    getindex(b::Binding, key)

Return the value associated with `key` in `b`.
"""
Base.getindex(b::Binding, key) = b.data[findkey(b, key)]

"""
    haskey(b::Binding, key)

Return `true` if `b` contains a value for `key`, and `false` otherwise.
"""
Base.haskey(b::Binding, key) = !isnothing(findkey(b, key))

"""
    get(b::Binding, key, default)

Return the value associated with `key` in `b`, or `default` if `key` is not present.
"""
Base.get(b::Binding, key, default) = haskey(b, key) ? b[key] : default

"""
    keys(b::Binding)

Return the tuple of keys in `b`.
"""
Base.keys(b::Binding) = b.keys

"""
    values(b::Binding)

Return the tuple of values stored in `b`.
"""
Base.values(b::Binding) = b.data

"""
    pairstuple(b::Binding)

Return the contents of `b` as a tuple of `key => value` pairs.
"""
pairstuple(b::Binding) = (pairs(b)...,)

"""
    push(b::Binding, pairs...)

Return a new binding with the given pairs inserted.

If a key is already present, its value is replaced.
"""
function push(b::Binding, kv::Pair)
    if !haskey(b, kv.first)
        keys = (b.keys..., kv.first)
        data = (b.data..., kv.second)
        return Binding(keys, data)
    else
        idx = findkey(b, kv.first)::Int
        data = ntuple(i -> i == idx ? kv.second : b.data[i], Val(length(b.data)))
        return Binding(b.keys, data)
    end
end
function push(b::Binding, kvs::Vararg{Pair,N}) where {N}
    return push(push(b, first(kvs)), Base.tail(kvs)...)
end

"""
    binding_types(b::Binding)

Return a binding with the same keys as `bnd`, replacing each value by its
concrete type.

Can be used to inspect Julia expressions generated from the Chmy expression.
"""
function binding_types(b::Binding)
    return Binding(b.keys, ntuple(i -> typeof(b.data[i]), Val(length(b.data))))
end
