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
