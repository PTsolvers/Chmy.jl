for (dim, coord) in enumerate((:x, :y, :z))
    _l = Symbol(:left, coord)
    _r = Symbol(:right, coord)
    _δ = Symbol(:δ, coord)
    _∂ = Symbol(:∂, coord)
    _∂² = Symbol(:∂², coord)
    _∂k∂ = Symbol(:∂k∂, coord)

    @eval begin
        export $_δ, $_∂, $_∂², $_∂k∂, $_l, $_r

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

        """
            $($_∂²)(f, grid, I)

        Directional partial second derivative in $($(string(coord))) direction.
        """
        @add_cartesian function $_∂²(f::AbstractField, grid, I::Vararg{Integer,N}) where {N}
            ∂²(f, grid, Dim($dim), I...)
        end

        """
            $($_∂k∂)(f, k, grid, I)

        Directional divergence of gradient times coefficient `k` in $($(string(coord))) direction.
        """
        @add_cartesian function $_∂k∂(f::AbstractField, k::AbstractField, grid, I::Vararg{Integer,N}) where {N}
            ∂k∂(f, k, grid, Dim($dim), I...)
        end
    end
end
