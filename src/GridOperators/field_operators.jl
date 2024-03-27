# staggered grid operators
@add_cartesian left(f::AbstractField, dim, I::Vararg{Integer,N}) where {N} = left(f, location(f, dim), flip(location(f, dim)), dim, I...)

@add_cartesian right(f::AbstractField, dim, I::Vararg{Integer,N}) where {N} = right(f, location(f, dim), flip(location(f, dim)), dim, I...)

@add_cartesian δ(f::AbstractField, dim, I::Vararg{Integer,N}) where {N} = δ(f, location(f, dim), flip(location(f, dim)), dim, I...)

@add_cartesian ∂(f::AbstractField, grid, dim, I::Vararg{Integer,N}) where {N} = ∂(f, location(f, dim), flip(location(f, dim)), grid, dim, I...)

# staggered operators on Cartesian grids
for (dim, coord) in enumerate((:x, :y, :z))
    _l = Symbol(:left, coord)
    _r = Symbol(:right, coord)
    _δ = Symbol(:δ, coord)
    _∂ = Symbol(:∂, coord)

    @eval begin
        export $_δ, $_∂, $_l, $_r

        """
            $($_l)(f, I)

        "left side" of a field (`[1:end-1]`) in $($(string(coord))) direction.
        """
        @add_cartesian function $_l(f::AbstractField, I::Vararg{Integer,N}) where {N}
            left(f, Dim($dim), I...)
        end

        """
            $($_r)(f, I)

        "right side" of a field (`[2:end]`) in $($(string(coord))) direction.
        """
        @add_cartesian function $_r(f::AbstractField, I::Vararg{Integer,N}) where {N}
            right(f, Dim($dim), I...)
        end

        """
            $($_δ)(f, I)

        Finite difference in $($(string(coord))) direction.
        """
        @add_cartesian function $_δ(f::AbstractField, I::Vararg{Integer,N}) where {N}
            δ(f, Dim($dim), I...)
        end

        """
            $($_∂)(f, grid, I)

        Directional partial derivative in $($(string(coord))) direction.
        """
        @add_cartesian function $_∂(f::AbstractField, grid, I::Vararg{Integer,N}) where {N}
            ∂(f, grid, Dim($dim), I...)
        end
    end
end

# covariant derivatives
@propagate_inbounds @generated function divg(V::NamedTuple{names,<:NTuple{N,AbstractField}}, grid::StructuredGrid{N}, I::Vararg{Integer,N}) where {names,N}
    return :(Base.Cartesian.@ncall $N (+) D -> ∂(V[D], grid, Dim(D), I...))
end

@propagate_inbounds function divg(V::NamedTuple{names,<:NTuple{N,AbstractField}}, grid::StructuredGrid{N}, I::CartesianIndex{N}) where {names,N}
    return divg(V, grid, Tuple(I)...)
end
