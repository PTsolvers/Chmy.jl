module GridOperators

using Chmy.Grids
import Chmy.@add_cartesian

export left, right, δ, ∂, lerp

Base.@assume_effects :foldable p(::Val{D}, I::Vararg{Integer,N}) where {D,N} = ntuple(i -> i == D ? I[i] + oneunit(I[i]) : I[i], Val(N))
Base.@assume_effects :foldable m(::Val{D}, I::Vararg{Integer,N}) where {D,N} = ntuple(i -> i == D ? I[i] - oneunit(I[i]) : I[i], Val(N))

@add_cartesian left(f, ::Vertex, dim, I::Vararg{Integer,N}) where {N} = f[m(dim, I...)...]
@add_cartesian left(f, ::Center, dim, I::Vararg{Integer,N}) where {N} = f[I...]

@add_cartesian right(f, ::Vertex, dim, I::Vararg{Integer,N}) where {N} = f[I...]
@add_cartesian right(f, ::Center, dim, I::Vararg{Integer,N}) where {N} = f[p(dim, I...)...]

# finite difference
@add_cartesian δ(f, loc, dim, I::Vararg{Integer,N}) where {N} = right(f, loc, dim, I...) - left(f, loc, dim, I...)

# partial derivative
@add_cartesian ∂(f, loc, grid, dim, I::Vararg{Integer,N}) where {N} = δ(f, loc, dim, I...) * iΔ(grid, loc, dim, I...)

# interpolation
@add_cartesian lerp(f, ::Center, grid, dim, I::Vararg{Integer,N}) where {N} = eltype(f)(0.5) * (f[I...] + f[p(dim, I...)...])
@add_cartesian function lerp(f, ::Vertex, grid, dim, I::Vararg{Integer,N}) where {N}
    t = eltype(f)(0.5) * Δ(grid, Center(), dim, m(dim, I...)...) * iΔ(grid, Vertex(), dim, I...)
    a = f[I...]
    b = f[m(dim, I...)...]
    return fma(t, a - b, b)
end

# more efficient for uniform grids
@add_cartesian lerp(f, ::Vertex, grid::UniformGrid, dim, I::Vararg{Integer,N}) where {N} = eltype(f)(0.5) * (f[m(dim, I...)...] + f[I...])

# operators on Cartesian grids
for (dim, coord) in enumerate((:x, :y, :z))
    δ = Symbol(:δ, coord)
    ∂ = Symbol(:∂, coord)

    @eval begin
        export $δ, $∂

        """
            $($δ)(f, loc, I)

        Finite difference in $($(string(coord))) direction.
        """
        @add_cartesian $δ(f, loc, I::Vararg{Integer,N}) where {N} = δ(f, loc, Val($dim), I...)

        """
            $($∂)(f, loc, grid, I)

        Directional partial derivative in $($(string(coord))) direction.
        """
        @add_cartesian $∂(f, loc, grid, I::Vararg{Integer,N}) where {N} = ∂(f, loc, grid, Val($dim), I...)
    end
end

end
