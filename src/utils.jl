function inversion_count(t::NTuple{N}) where {N}
    s = 0
    for i in 1:N-1, j in i+1:N
        s += t[i] > t[j]
    end
    return s
end

@inline function foreach_nondecreasing(f, ::Val{D}, ::Val{R}) where {D,R}
    I0 = ntuple(_ -> 1, Val(R))
    _nondecreasing(f, Val(R), D, I0)
end

@inline function foreach_increasing(f, ::Val{D}, ::Val{R}) where {D,R}
    I0 = ntuple(identity, Val(R))
    _increasing(f, Val(R), D, I0)
end

_nondecreasing(f, ::Val{0}, ub, I) = f(I)
@inline function _nondecreasing(f, ::Val{r}, ub, I) where {r}
    for j in 1:ub
        Ij = Base.setindex(I, j, r)
        _nondecreasing(f, Val(r - 1), j, Ij)
    end
end

_increasing(f, ::Val{0}, ub, I) = f(I)
@inline function _increasing(f, ::Val{r}, ub, I) where {r}
    for j in 1:ub
        Ij = Base.setindex(I, j, r)
        _increasing(f, Val(r - 1), j - 1, Ij)
    end
end

tuplemap(f, xs) = map(f, xs)
tuplemap(f, xs, ys) = map(f, xs, ys)

"""
    replace_index(inds, v, Val(I))

Return a tuple equal to `inds`, except that position `I` is replaced by `v`.

This is the type-stable tuple analogue of `Base.setindex`, used when a
one-dimensional symbolic rule is lifted back into an N-dimensional index or
location tuple.
"""
replace_index(inds::Tuple, v, ::Val{I}) where {I} = ntuple(i -> i == I ? v : inds[i], Val(length(inds)))

@generated function tuplemap(f, xs::T) where {T<:Tuple}
    ex = Expr(:tuple)
    for i in 1:length(T.parameters)
        push!(ex.args, :(f(xs[$i])))
    end
    return ex
end

@generated function tuplemap(f, xs::TX, ys::TY) where {TX<:Tuple,TY<:Tuple}
    length(TX.parameters) == length(TY.parameters) || error("tuple lengths must match")
    ex = Expr(:tuple)
    for i in 1:length(TX.parameters)
        push!(ex.args, :(f(xs[$i], ys[$i])))
    end
    return ex
end

flatten(args::Tuple{Vararg{Tuple}}) = (Iterators.flatten(args)...,)
flatten(args::Vararg{Tuple})        = flatten(args)
