for (dim, coord) in enumerate((:x, :y, :z))
    _l = Symbol(:left, coord)
    _r = Symbol(:right, coord)
    _δ = Symbol(:δ, coord)
    _∂ = Symbol(:∂, coord)
    _∂² = Symbol(:∂², coord)

    @eval begin
        export $_δ, $_∂, $_∂², $_l, $_r

        """
            $($_l)(f, ω, I)

        "left side" of a field (`[1:end-1]`) in $($(string(coord))) direction, masked with `ω`.
        """
        @add_cartesian function $_l(f::AbstractField, ω::AbstractMask, I::Vararg{Integer,N}) where {N}
            left(f, ω, Dim($dim), I...)
        end

        """
            $($_r)(f, ω, I)

        "right side" of a field (`[2:end]`) in $($(string(coord))) direction, masked with `ω`.
        """
        @add_cartesian function $_r(f::AbstractField, ω::AbstractMask, I::Vararg{Integer,N}) where {N}
            right(f, ω, Dim($dim), I...)
        end

        """
            $($_δ)(f, ω, I)

        Finite difference in $($(string(coord))) direction, masked with `ω`.
        """
        @add_cartesian function $_δ(f::AbstractField, ω::AbstractMask, I::Vararg{Integer,N}) where {N}
            δ(f, ω, Dim($dim), I...)
        end

        """
            $($_∂)(f, ω, grid, I)

        Directional partial derivative in $($(string(coord))) direction, masked with `ω`.
        """
        @add_cartesian function $_∂(f::AbstractField, ω::AbstractMask, grid, I::Vararg{Integer,N}) where {N}
            ∂(f, ω, grid, Dim($dim), I...)
        end

        """
        $($_∂²)(f, ω, grid, I)

        Directional partial second derivative in $($(string(coord))) direction, masked with `ω`.
        """
        @add_cartesian function $_∂²(f::AbstractField, ω::AbstractMask, grid, I::Vararg{Integer,N}) where {N}
            ∂²(f, ω, grid, Dim($dim), I...)
        end
    end
end
