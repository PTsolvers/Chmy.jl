# staggered grid operators
@add_cartesian left(f::AbstractField, dim, I::Vararg{Integer,N}) where {N} = left(f, location(f, dim), flip(location(f, dim)), dim, I...)

@add_cartesian right(f::AbstractField, dim, I::Vararg{Integer,N}) where {N} = right(f, location(f, dim), flip(location(f, dim)), dim, I...)

@add_cartesian δ(f::AbstractField, dim, I::Vararg{Integer,N}) where {N} = δ(f, location(f, dim), flip(location(f, dim)), dim, I...)

@add_cartesian ∂(f::AbstractField, grid, dim, I::Vararg{Integer,N}) where {N} = ∂(f, location(f, dim), flip(location(f, dim)), grid, dim, I...)

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
