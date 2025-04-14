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
    return δ(f, loc, from, ω, dim, I...)
end

@add_cartesian function ∂(f::AbstractField, ω::AbstractMask, grid, dim, I::Vararg{Integer,N}) where {N}
    loc  = location(f)
    from = flipped(loc, dim)
    return ∂(f, loc, from, ω, grid, dim, I...)
end

@add_cartesian function ∂²(f::AbstractField, ω::AbstractMask, grid, dim, I::Vararg{Integer,N}) where {N}
    loc  = location(f)
    from = location(f)
    return ∂²(f, loc, from, ω, grid, dim, I...)
end

@add_cartesian function ∂k∂(f::AbstractField, k::AbstractField, ω::AbstractMask, grid, dim, I::Vararg{Integer,N}) where {N}
    loc  = location(f)
    from = location(f)
    return ∂k∂(f, k, loc, from, ω, grid, dim, I...)
end

# covariant derivatives
"""
    divg(V, ω, grid, I)

Compute the divergence of a vector field `V` on a structured grid `grid`, using the mask `ω` to handle masked regions.
This operation is performed along all dimensions of the grid.

## Arguments:
- `V`: The vector field represented as a named tuple of fields.
- `ω`: The mask for the grid.
- `grid`: The structured grid on which the operation is performed.
- `I...`: The indices specifying the location on the grid.
"""
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

"""
    lapl(F, ω, grid, I...)

Compute the Laplacian of a field `F` on a structured grid `grid`.
This operation is performed along all dimensions of the grid.

## Arguments
- `F`: The field whose Laplacian is to be computed.
- `ω`: The mask for the grid.
- `grid`: The structured grid on which the operation is performed.
- `I...`: The indices specifying the location on the grid.
"""
@propagate_inbounds @generated function lapl(F::AbstractField,
                                             ω::AbstractMask{T,N},
                                             grid::StructuredGrid{N},
                                             I::Vararg{Integer,N}) where {T,N}
    quote
        @inline
        Base.Cartesian.@ncall $N (+) D -> ∂²(F, ω, grid, Dim(D), I...)
    end
end

@propagate_inbounds function lapl(F::AbstractField,
                                  ω::AbstractMask{T,N},
                                  grid::StructuredGrid{N},
                                  I::CartesianIndex{N}) where {T,N}
    return lapl(F, ω, grid, Tuple(I)...)
end

"""
    divg_grad(F, K, ω, grid, I...)

Compute the divergence of the diffusion flux of a field `F` weighted by a coefficient `K` on a structured grid `grid`.
This operation is performed along all dimensions of the grid.

## Arguments
- `F`: The field whose gradient is to be computed.
- `K`: The weighting field for the gradient.
- `ω`: The mask for the grid.
- `grid`: The structured grid on which the operation is performed.
- `I...`: The indices specifying the location on the grid.
"""
@propagate_inbounds @generated function divg_grad(F::AbstractField,
                                                  K::AbstractField,
                                                  ω::AbstractMask{T,N},
                                                  grid::StructuredGrid{N},
                                                  I::Vararg{Integer,N}) where {T,N}
    quote
        @inline
        Base.Cartesian.@ncall $N (+) D -> ∂k∂(F, K, ω, grid, Dim(D), I...)
    end
end

@propagate_inbounds function divg_grad(F::AbstractField,
                                       K::AbstractField,
                                       ω::AbstractMask{T,N},
                                       grid::StructuredGrid{N},
                                       I::CartesianIndex{N}) where {T,N}
    return divg_grad(F, K, ω, grid, Tuple(I)...)
end
