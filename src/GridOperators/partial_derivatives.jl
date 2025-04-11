# finite difference
@add_cartesian δ(f, loc, from, dim, I::Vararg{Integer,N}) where {N} = right(f, loc, from, dim, I...) - left(f, loc, from, dim, I...)

# partial derivatives
@add_cartesian ∂(f, loc, from, grid, dim, I::Vararg{Integer,N}) where {N} = δ(f, loc, from, dim, I...) * iΔ(grid, loc, dim, I...)

@add_cartesian function ∂²(f, loc, from, grid, dim::Dim{D}, I::Vararg{Integer,N}) where {N,D}
    Ir = ir(flipped(loc, dim)[D], from[D], dim, I...)
    Il = il(flipped(loc, dim)[D], from[D], dim, I...)
    return (∂(f, grid, dim, Ir...) - ∂(f, grid, dim, Il...)) * iΔ(grid, loc, dim, I...)
end

@add_cartesian function ∂k∂(f, k, loc, from, grid, dim::Dim{D}, I::Vararg{Integer,N}) where {N,D}
    Ir = ir(flipped(loc, dim)[D], from[D], dim, I...)
    Il = il(flipped(loc, dim)[D], from[D], dim, I...)
    return (lerp(k, flipped(loc, dim), grid, Ir...) * ∂(f, grid, dim, Ir...) -
            lerp(k, flipped(loc, dim), grid, Il...) * ∂(f, grid, dim, Il...)) *
           iΔ(grid, loc, dim, I...)
end
