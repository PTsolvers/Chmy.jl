# staggered grid operators
@add_cartesian function left(f::AbstractField, dim, I::Vararg{Integer,N}) where {N}
    loc  = location(f)
    from = flipped(loc, dim)
    return left(f, loc, from, dim, I...)
end

@add_cartesian function right(f::AbstractField, dim, I::Vararg{Integer,N}) where {N}
    loc  = location(f)
    from = flipped(loc, dim)
    return right(f, loc, from, dim, I...)
end

@add_cartesian function δ(f::AbstractField, dim, I::Vararg{Integer,N}) where {N}
    loc  = location(f)
    from = flipped(loc, dim)
    return δ(f, loc, from, dim, I...)
end

@add_cartesian function ∂(f::AbstractField, grid, dim, I::Vararg{Integer,N}) where {N}
    loc  = location(f)
    from = flipped(loc, dim)
    return ∂(f, loc, from, grid, dim, I...)
end

@add_cartesian function ∂²(f::AbstractField, grid, dim, I::Vararg{Integer,N}) where {N}
    loc  = location(f)
    from = location(f)
    return ∂²(f, loc, from, grid, dim, I...)
end

@add_cartesian function ∂k∂(f::AbstractField, k::AbstractField, grid, dim, I::Vararg{Integer,N}) where {N}
    loc  = location(f)
    from = location(f)
    return ∂k∂(f, k, loc, from, grid, dim, I...)
end

# covariant derivatives
"""
    divg(V, grid, I...)

Compute the divergence of a vector field `V` on a structured grid `grid`.
This operation is performed along all dimensions of the grid.

## Arguments
- `V`: The vector field represented as a named tuple of fields.
- `grid`: The structured grid on which the operation is performed.
- `I...`: The indices specifying the location on the grid.
"""
@propagate_inbounds @generated function divg(V::NamedTuple{names,<:NTuple{N,AbstractField}}, grid::StructuredGrid{N}, I::Vararg{Integer,N}) where {names,N}
    quote
        @inline
        Base.Cartesian.@ncall $N (+) D -> ∂(V[D], grid, Dim(D), I...)
    end
end

@propagate_inbounds function divg(V::NamedTuple{names,<:NTuple{N,AbstractField}}, grid::StructuredGrid{N}, I::CartesianIndex{N}) where {names,N}
    return divg(V, grid, Tuple(I)...)
end

"""
    lapl(F, grid, I...)

Compute the Laplacian of a field `F` on a structured grid `grid`.
This operation is performed along all dimensions of the grid.

## Arguments
- `F`: The field whose Laplacian is to be computed.
- `grid`: The structured grid on which the operation is performed.
- `I...`: The indices specifying the location on the grid.
"""
@propagate_inbounds @generated function lapl(F::AbstractField, grid::StructuredGrid{N}, I::Vararg{Integer,N}) where {N}
    quote
        @inline
        Base.Cartesian.@ncall $N (+) D -> ∂²(F, grid, Dim(D), I...)
    end
end

@propagate_inbounds function lapl(F::AbstractField, grid::StructuredGrid{N}, I::CartesianIndex{N}) where {N}
    return lapl(F, grid, Tuple(I)...)
end

"""
    divg_grad(F, K, grid, I...)

Compute the divergence of the gradient of a field `F` weighted by a coefficient `K` on a structured grid `grid`.
This operation is performed along all dimensions of the grid.

## Arguments
- `F`: The field whose gradient is to be computed.
- `K`: The weighting field for the gradient.
- `grid`: The structured grid on which the operation is performed.
- `I...`: The indices specifying the location on the grid.
"""
@propagate_inbounds @generated function divg_grad(F::AbstractField, K::AbstractField, grid::StructuredGrid{N}, I::Vararg{Integer,N}) where {N}
    quote
        @inline
        Base.Cartesian.@ncall $N (+) D -> ∂k∂(F, K, grid, Dim(D), I...)
    end
end

@propagate_inbounds function divg_grad(F::AbstractField, K::AbstractField, grid::StructuredGrid{N}, I::CartesianIndex{N}) where {N}
    return divg_grad(F, K, grid, Tuple(I)...)
end

"""
    vmag(V, grid, I...)

Compute the magnitude of a vector field `V` at a given grid location `I` in a structured grid `grid`.

## Arguments
- `V`: A named tuple representing the vector field, where each component is an `AbstractField`.
- `grid`: The structured grid on which the vector field is defined.
- `I...`: The indices specifying the location on the grid.
"""
@propagate_inbounds @generated function vmag(V::NamedTuple{names,<:NTuple{N,AbstractField}}, grid::StructuredGrid{N}, I::Vararg{Integer,N}) where {names,N}
    quote
        @inline
        sqrt(Base.Cartesian.@ncall $N (+) D -> lerp(V[D], Center(), grid, I...)^2)
    end
end

@propagate_inbounds function vmag(V::NamedTuple{names,<:NTuple{N,AbstractField}}, grid::StructuredGrid{N}, I::CartesianIndex{N}) where {names,N}
    return vmag(V, grid, Tuple(I)...)
end
