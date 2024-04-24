# staggered grid operators
@add_cartesian left(f::AbstractField, ω::AbstractMask, dim, I::Vararg{Integer,N}) where {N} = left(f, location(f, dim), flip(location(f, dim)), ω, dim, I...)

@add_cartesian right(f::AbstractField, ω::AbstractMask, dim, I::Vararg{Integer,N}) where {N} = right(f, location(f, dim), flip(location(f, dim)), ω, dim, I...)

@add_cartesian δ(f::AbstractField, ω::AbstractMask, dim, I::Vararg{Integer,N}) where {N} = δ(f, location(f, dim), flip(location(f, dim)), ω, dim, I...)

@add_cartesian function ∂(f::AbstractField, ω::AbstractMask, grid, dim, I::Vararg{Integer,N}) where {N}
    return ∂(f, location(f, dim), flip(location(f, dim)), ω, grid, dim, I...)
end

# covariant derivatives
@propagate_inbounds @generated function divg(V::NamedTuple{names,<:NTuple{N,AbstractField}},
                                             ω::AbstractMask{T,N},
                                             grid::StructuredGrid{N},
                                             I::Vararg{Integer,N}) where {names,T,N}
    quote
        @inline
        Base.Cartesian.@ncall $N (+) D -> ∂(V[D], ω, grid, Dim(D), I...)
    end
end

@propagate_inbounds function divg(V::NamedTuple{names,<:NTuple{N,AbstractField}},
                                  ω::AbstractMask{T,N},
                                  grid::StructuredGrid{N},
                                  I::CartesianIndex{N}) where {names,T,N}
    return divg(V, ω, grid, Tuple(I)...)
end
