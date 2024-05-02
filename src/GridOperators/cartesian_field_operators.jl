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
