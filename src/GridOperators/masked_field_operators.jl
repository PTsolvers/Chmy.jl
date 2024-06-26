# staggered grid operators
@add_cartesian function left(f::AbstractField, ω::AbstractMask, dim, I::Vararg{Integer,N}) where {N}
    loc  = location(f)
    from = flipped(loc, dim)
    return left(f, loc, from, ω, dim, I...)
end

@add_cartesian function right(f::AbstractField, ω::AbstractMask, dim, I::Vararg{Integer,N}) where {N}
    loc  = location(f)
    from = flipped(loc, dim)
    return right(f, loc, from, ω, dim, I...)
end

@add_cartesian function δ(f::AbstractField, ω::AbstractMask, dim, I::Vararg{Integer,N}) where {N}
    loc  = location(f)
    from = flipped(loc, dim)
    δ(f, loc, from, ω, dim, I...)
end

@add_cartesian function ∂(f::AbstractField, ω::AbstractMask, grid, dim, I::Vararg{Integer,N}) where {N}
    loc  = location(f)
    from = flipped(loc, dim)
    return ∂(f, loc, from, ω, grid, dim, I...)
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
