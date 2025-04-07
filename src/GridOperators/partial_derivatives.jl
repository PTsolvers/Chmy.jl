# finite difference
@add_cartesian δ(f, loc, from, dim, I::Vararg{Integer,N}) where {N} = right(f, loc, from, dim, I...) - left(f, loc, from, dim, I...)

# partial derivative
@add_cartesian ∂(f, loc, from, grid, dim, I::Vararg{Integer,N}) where {N} = δ(f, loc, from, dim, I...) * iΔ(grid, loc, dim, I...)
@add_cartesian ∂²(f, loc, from, grid, dim, I::Vararg{Integer,N}) where {N} = ((right(f, loc, from, dim, I...) - f[I...]) * iΔ(grid, loc, dim, I...) -
                                                                              (f[I...] - left(f, loc, from, dim, I...) ) * iΔ(grid, loc, dim, I...)) *
                                                                             iΔ(grid, loc, dim, I...)
