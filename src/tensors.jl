abstract type AbstractTensor{O,D,S} end

abstract type AbstractPermutationGroup{N} end

struct IdentityGroup{N} <: AbstractPermutationGroup{N} end
struct SymmetricGroup{N} <: AbstractPermutationGroup{N} end

order(::AbstractTensor{O}) where {O} = O
dimensions(::AbstractTensor{O,D}) where {O,D} = D
symmetry(::AbstractTensor{O,D,S}) where {O,D,S} = S

order(::Type{<:AbstractTensor{O}}) where {O} = O
dimensions(::Type{<:AbstractTensor{O,D}}) where {O,D} = D
symmetry(::Type{<:AbstractTensor{O,D,S}}) where {O,D,S} = S

struct Tensor{O,D,S,C} <: AbstractTensor{O,D,S}
    components::C
end

Base.length(t::Tensor) = length(t.components)

const AsymmetricTensor{O,D} = Tensor{O,D,IdentityGroup{O}}
const SymmetricTensor{O,D} = Tensor{O,D,SymmetricGroup{O}}

const Vec{D} = AsymmetricTensor{1,D}

ncomponents(t::AbstractTensor) = ncomponents(typeof(t))
ncomponents(tt::Type{<:AbstractTensor}) = ncomponents(symmetry(tt), Val(dimensions(tt)))

ncomponents(::Type{SymmetricGroup{O}}, ::Val{D}) where {O,D} = binomial(D + O - 1, O)
ncomponents(::Type{IdentityGroup{O}}, ::Val{D}) where {O,D} = D^O

function Tensor{O,D,S}(data::Vararg{Any,M}) where {O,D,S,M}
    N = ncomponents(S, Val(D))
    M == N || error("expected $N components to construct order-$O $(S) Tensor, got $M")
    Tensor{O,D,S,typeof(data)}(data)
end

Tensor{O,D}(data::Vararg{Any,M}) where {O,D,M} = Tensor{O,D,IdentityGroup{O}}(data...)

Vec(data::Vararg{Any,M}) where {M} = Vec{M}(data...)

Base.getindex(t::Tensor, i::Int) = t.components[i]

Base.getindex(t::Tensor{O}, I::Vararg{Int,O}) where {O} = t.components[canonical_index(typeof(t), I...)]

canonical_index(tt::Type{<:AbstractTensor{O}}, I::Vararg{Int,O}) where {O} = canonical_index(symmetry(tt), Val(dimensions(tt)), I...)

function canonical_index(::Type{SymmetricGroup{O}}, ::Val, I::Vararg{Int,O}) where {O}
    J = sort(SVector(I))
    return 1 + sum(ntuple(k -> binomial(J[k] + k - 2, k), Val(O)))
end

function canonical_index(::Type{IdentityGroup{O}}, ::Val{D}, I::Vararg{Int,O}) where {O,D}
    return LinearIndices(ntuple(_ -> D, Val(O)))[I...]
end

Base.:+(t::AbstractTensor) = t
Base.:-(t::Tensor{O,D,S}) where {O,D,S} = Tensor{O,D,S}(map(-, t.components)...)

Base.:+(t1::Tensor{O,D,S}, t2::Tensor{O,D,S}) where {O,D,S} = Tensor{O,D,S}(map(+, t1.components, t2.components)...)
Base.:-(t1::Tensor{O,D,S}, t2::Tensor{O,D,S}) where {O,D,S} = Tensor{O,D,S}(map(-, t1.components, t2.components)...)

const NumberOrTerm = Union{Number,STerm}

Base.:*(s::NumberOrTerm, t::Tensor{O,D,S}) where {O,D,S} = Tensor{O,D,S}(map(x -> s * x, t.components)...)
Base.:*(t::Tensor{O,D,S}, s::NumberOrTerm) where {O,D,S} = Tensor{O,D,S}(map(x -> x * s, t.components)...)

Base.:/(t::Tensor{O,D,S}, s::NumberOrTerm) where {O,D,S} = Tensor{O,D,S}(map(x -> x / s, t.components)...)
Base.://(t::Tensor{O,D,S}, s::NumberOrTerm) where {O,D,S} = Tensor{O,D,S}(map(x -> x // s, t.components)...)

@generated function dcontract(t1::Tensor{2,D}, t2::Tensor{2,D}) where {D}
    idx1(i, j) = canonical_index(t1, i, j)
    idx2(i, j) = canonical_index(t2, i, j)
    ex_p = Tuple(:(t1.components[$(idx1(i, j))] * t2.components[$(idx2(i, j))]) for i in 1:D, j in 1:D)
    ex_c = Expr(:call, :+, ex_p...)
    quote
        @inline
        return $ex_c
    end
end

const ⊡ = dcontract

LinearAlgebra.dot(v1::Vec{D}, v2::Vec{D}) where {D} = +(ntuple(i -> v1[i] * v2[i], Val(D))...)

@generated function LinearAlgebra.dot(t::Tensor{2,D}, v::Vec{D}) where {D}
    idx(i, j) = canonical_index(t, i, j)
    vc(i) = Expr(:call, :+, Tuple(:(t.components[$(idx(i, j))] * v.components[$j]) for j in 1:D)...)
    ex = Expr(:call, :(Vec{$D}), Tuple(vc(i) for i in 1:D)...)
    quote
        @inline
        return $ex
    end
end

@generated function LinearAlgebra.dot(t1::Tensor{2,D}, t2::Tensor{2,D}) where {D}
    idx1(i, j) = canonical_index(t1, i, j)
    idx2(i, j) = canonical_index(t2, i, j)
    vc(i, j) = Expr(:call, :+, Tuple(:(t1.components[$(idx1(i, k))] * t2.components[$(idx2(k, j))]) for k in 1:D)...)
    ex = Expr(:call, :(Tensor{2,$D}), Tuple(vc(i, j) for i in 1:D, j in 1:D)...)
    quote
        @inline
        return $ex
    end
end

Base.:*(t1::Tensor{2,D}, t2::Tensor{2,D}) where {D} = LinearAlgebra.dot(t1, t2)

Base.transpose(t::SymmetricTensor{2}) = t

@generated function Base.transpose(t::AsymmetricTensor{2,D}) where {D}
    idx(i, j) = canonical_index(t, i, j)
    ex = Expr(:call, :(Tensor{2,$D}), Tuple(:(t.components[$(idx(j, i))]) for i in 1:D, j in 1:D)...)
    quote
        @inline
        return $ex
    end
end

Base.adjoint(S::AbstractTensor) = transpose(S)
