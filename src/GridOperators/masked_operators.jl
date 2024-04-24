abstract type AbstractMask{T,N} end

@add_cartesian at(ω::AbstractMask{T,N}, loc::Location, I::Vararg{Integer,N}) where {T,N} = at(ω, expand_loc(Val(N), loc), I...)

@add_cartesian function left(f, loc, from, ω::AbstractMask, dim, I::Vararg{Integer,N}) where {N}
    Il = il(loc, from, dim, I...)
    return f[Il...] * at(ω, loc, Il...)
end

@add_cartesian function right(f, loc, from, ω::AbstractMask, dim, I::Vararg{Integer,N}) where {N}
    Ir = ir(loc, from, dim, I...)
    return f[Ir...] * at(ω, loc, Ir...)
end

@add_cartesian δ(f, loc, from, ω::AbstractMask, dim, I::Vararg{Integer,N}) where {N} = right(f, loc, from, ω, dim, I...) - left(f, loc, from, ω, dim, I...)

@add_cartesian ∂(f, loc, from, ω::AbstractMask, grid, dim, I::Vararg{Integer,N}) where {N} = δ(f, loc, from, ω, dim, I...) * iΔ(grid, loc, dim, I...)
