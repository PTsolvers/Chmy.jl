# staggered grid operators
@add_cartesian function left(f::AbstractField, dim, I::Vararg{Integer,N}) where {N}
    loc  = location(f)
    from = flipped(loc, dim)
    left(f, loc, from, dim, I...)
end

@add_cartesian function right(f::AbstractField, dim, I::Vararg{Integer,N}) where {N}
    loc  = location(f)
    from = flipped(loc, dim)
    right(f, loc, from, dim, I...)
end

@add_cartesian function δ(f::AbstractField, dim, I::Vararg{Integer,N}) where {N}
    loc  = location(f)
    from = flipped(loc, dim)
    return δ(f, loc, from, dim, I...)
end

@add_cartesian function ∂(f::AbstractField, grid, dim, I::Vararg{Integer,N}) where {N}
    loc  = location(f)
    from = flipped(loc, dim)
    ∂(f, loc, from, grid, dim, I...)
end

@add_cartesian function ∂²(f::AbstractField, grid, dim, I::Vararg{Integer,N}) where {N}
    loc  = location(f)
    from = location(f)
    ∂²(f, loc, from, grid, dim, I...)
end

# covariant derivatives
@propagate_inbounds @generated function divg(V::NamedTuple{names,<:NTuple{N,AbstractField}}, grid::StructuredGrid{N}, I::Vararg{Integer,N}) where {names,N}
    quote
        @inline
        Base.Cartesian.@ncall $N (+) D -> ∂(V[D], grid, Dim(D), I...)
    end
end

@propagate_inbounds function divg(V::NamedTuple{names,<:NTuple{N,AbstractField}}, grid::StructuredGrid{N}, I::CartesianIndex{N}) where {names,N}
    return divg(V, grid, Tuple(I)...)
end

@propagate_inbounds @generated function vmag(V::NamedTuple{names,<:NTuple{N,AbstractField}}, grid::StructuredGrid{N}, I::Vararg{Integer,N}) where {names,N}
    quote
        @inline
        sqrt(Base.Cartesian.@ncall $N (+) D -> lerp(V[D], Center(), grid, I...)^2)
    end
end

@propagate_inbounds function vmag(V::NamedTuple{names,<:NTuple{N,AbstractField}}, grid::StructuredGrid{N}, I::CartesianIndex{N}) where {names,N}
    return vmag(V, grid, Tuple(I)...)
end
