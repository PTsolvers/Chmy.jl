# finite difference
@add_cartesian δ(f, loc, from, dim, I::Vararg{Integer,N}) where {N} = right(f, loc, from, dim, I...) - left(f, loc, from, dim, I...)

# partial derivative
@add_cartesian ∂(f, loc, from, grid, dim, I::Vararg{Integer,N}) where {N} = δ(f, loc, from, dim, I...) * iΔ(grid, loc, dim, I...)

@add_cartesian ∂²(f, loc, from, grid, dim::Dim{D}, I::Vararg{Integer,N}) where {N,D} = (∂(f, grid, dim, ir(flipped(loc, dim)[D], from[D], dim, I...)...) -
                                                                                        ∂(f, grid, dim, il(flipped(loc, dim)[D], from[D], dim, I...)...)) *
                                                                                       iΔ(grid, loc, dim, I...)
