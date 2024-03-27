module GridOperators

export left, right, δ, ∂, itp

export InterpolationRule, Linear, HarmonicLinear
export itp_rule, itp_weight

using Chmy
using Chmy.Grids
using Chmy.Fields

import Chmy.@add_cartesian

import Base: @propagate_inbounds, front

p(::Dim{D}, I::Vararg{Integer,N}) where {D,N} = ntuple(i -> i == D ? I[i] + oneunit(I[i]) : I[i], Val(N))
m(::Dim{D}, I::Vararg{Integer,N}) where {D,N} = ntuple(i -> i == D ? I[i] - oneunit(I[i]) : I[i], Val(N))

@add_cartesian il(loc::Vertex, from::Center, dim, I::Vararg{Integer,N}) where {N} = I
@add_cartesian il(loc::Center, from::Vertex, dim, I::Vararg{Integer,N}) where {N} = m(dim, I...)

@add_cartesian ir(loc::Vertex, from::Center, dim, I::Vararg{Integer,N}) where {N} = p(dim, I...)
@add_cartesian ir(loc::Center, from::Vertex, dim, I::Vararg{Integer,N}) where {N} = I

@add_cartesian il(loc::L, from::L, dim, I::Vararg{Integer,N}) where {N,L<:Location} = m(dim, I...)
@add_cartesian ir(loc::L, from::L, dim, I::Vararg{Integer,N}) where {N,L<:Location} = p(dim, I...)

@add_cartesian left(f, loc, from, dim, I::Vararg{Integer,N}) where {N} = f[il(loc, from, dim, I...)...]
@add_cartesian right(f, loc, from, dim, I::Vararg{Integer,N}) where {N} = f[ir(loc, from, dim, I...)...]

include("partial_derivatives.jl")
include("interpolation.jl")
include("field_operators.jl")

end
