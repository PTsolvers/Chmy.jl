"""
    abstract type AbstractMask{T,N}

Abstract type representing the data transformation to be performed on elements
in a field of dimension `N`, where each element is of type`T`.
"""
abstract type AbstractMask{T,N} end

@propagate_inbounds at(ω::AbstractMask{T,N}, loc::NTuple{N,Location}, I::CartesianIndex{N}) where {T,N} = at(ω, loc, Tuple(I)...)

@add_cartesian function left(f, loc::NTuple{N,Location}, from::NTuple{N,Location}, ω::AbstractMask, dim::Dim{D}, I::Vararg{Integer,N}) where {N,D}
    Il = il(loc[D], from[D], dim, I...)
    return f[Il...] * at(ω, loc, Il...)
end

@add_cartesian function right(f, loc::NTuple{N,Location}, from::NTuple{N,Location}, ω::AbstractMask, dim::Dim{D}, I::Vararg{Integer,N}) where {N,D}
    Ir = ir(loc[D], from[D], dim, I...)
    return f[Ir...] * at(ω, loc, Ir...)
end

@add_cartesian δ(f, loc, from, ω::AbstractMask, dim, I::Vararg{Integer,N}) where {N} = right(f, loc, from, ω, dim, I...) - left(f, loc, from, ω, dim, I...)

@add_cartesian ∂(f, loc, from, ω::AbstractMask, grid, dim, I::Vararg{Integer,N}) where {N} = δ(f, loc, from, ω, dim, I...) * iΔ(grid, loc, dim, I...)

@add_cartesian function ∂²(f, loc, from, ω::AbstractMask, grid, dim::Dim{D}, I::Vararg{Integer,N}) where {N,D}
    Ir = ir(flipped(loc, dim)[D], from[D], dim, I...)
    Il = il(flipped(loc, dim)[D], from[D], dim, I...)
    return (∂(f, ω, grid, dim, Ir...) - ∂(f, ω, grid, dim, Il...)) * iΔ(grid, loc, dim, I...)
end

@add_cartesian function ∂k∂(f, k, loc, from, ω::AbstractMask, grid, dim::Dim{D}, I::Vararg{Integer,N}) where {N,D}
    Ir = ir(flipped(loc, dim)[D], from[D], dim, I...)
    Il = il(flipped(loc, dim)[D], from[D], dim, I...)
    return (lerp(k, flipped(loc, dim), grid, Ir...) * ∂(f, ω, grid, dim, Ir...) -
            lerp(k, flipped(loc, dim), grid, Il...) * ∂(f, ω, grid, dim, Il...)) *
           iΔ(grid, loc, dim, I...)
end
