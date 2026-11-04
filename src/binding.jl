# the key binding the samples of a scalar field at the `locations` to data, e.g. `u[𝓈, 𝓈]` for all samples at
# half-integer positions, or `g[↓, 𝓈]` for the samples of the trace on the lower wall of the first axis
struct Located
    term::DTerm
    locations::Tuple{Vararg{Location}}
    function Located(term::DTerm, locations::Tuple{Vararg{Location}})
        tensorrank(term) == 0 || throw(ArgumentError("only scalars can be located"))
        return new(term, locations)
    end
end

# keys compare by content, never with the symbolic `==` of DTerms
Base.isequal(a::Located, b::Located) = isequal(a.term, b.term) && a.locations === b.locations
Base.hash(l::Located, h::UInt) = hash(l.locations, hash(l.term, hash(:Located, h)))

# a binding key, at least one location is a point or a segment, otherwise the sample is the key itself
Base.getindex(term::DTerm, locs::Vararg{Location, N}) where {N} = Located(term, locs)

# the location of a position: integer positions are points and half-integer ones segments
location(p::Rational{Int}) = isinteger(p) ? Point() : Segment()
location(p::Global) = p

# the key of the sample of `term` at `positions`: samples at global positions only are their own keys
bindkey(term::DTerm, positions) = all(p -> p isa Global, positions) ? makeinds_unchecked(term, positions) : Located(term, map(location, positions))

"""
    bindings(term)

Returns the tuple of keys of the data that a lowered `term` reads, in the order of their first appearance: the
locations of the sampled fields, e.g. `u[𝓈, 𝓈]` for `u[1//2, 3//2]`, samples at global positions only, e.g. `g[↓, :]`,
and uniform terms.
"""
function bindings(term::DTerm)
    keys = Union{DTerm, Located}[]
    collect_bindings!(keys, term)
    return Tuple(unique(keys))
end

function collect_bindings!(keys, term::DTerm)
    @match term begin
        DExpr(Inds(positions), (arg,)) => push!(keys, bindkey(arg, positions))
        Tensor(_, _, _, _) || DExpr(Comp(_), _) => push!(keys, term)
        DExpr(_, args) => foreach(t -> collect_bindings!(keys, t), args)
        AddMul(_, _, dict) => foreach(t -> collect_bindings!(keys, t), Iterators.flatten((dict.keys, dict.vals)))
        _ => nothing
    end
    return keys
end

struct Binding{K, D}
    keys::K
    data::D
end

"""
    Binding(pairs...)

Create an immutable dictionary binding Chmy expressions to data.
"""
Binding(kvs::Vararg{Pair, N}) where {N} = Binding(zip(kvs...)...)
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
function push(b::Binding, kvs::Vararg{Pair, N}) where {N}
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
