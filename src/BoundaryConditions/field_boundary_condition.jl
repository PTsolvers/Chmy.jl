struct DirichletKind end
struct NeumannKind end
struct FieldBoundaryCondition{T,Kind}
    value::T
    FieldBoundaryCondition{Kind}(value::T) where {Kind,T} = new{T,Kind}(value)
end

const Dirichlet{T} = FieldBoundaryCondition{T,DirichletKind}
const Neumann{T}   = FieldBoundaryCondition{T,NeumannKind}

Dirichlet(value=nothing) = FieldBoundaryCondition{DirichletKind}(value)
Neumann(value=nothing)   = FieldBoundaryCondition{NeumannKind}(value)

Base.show(io::IO, ::Dirichlet{Nothing}) = print(io, "Dirichlet(0)")
Base.show(io::IO, ::Neumann{Nothing}) = print(io, "Neumann(0)")

const FBC          = FieldBoundaryCondition
const FBCOrNothing = Union{FBC,Nothing}
const SidesBCs     = Tuple{FBCOrNothing,FBCOrNothing}
const BCOrTuple    = Union{FBCOrNothing,SidesBCs}
const TupleBC      = NamedTuple{Names,<:Tuple{Vararg{BCOrTuple}}} where {Names}
const InputBC      = Union{FBCOrNothing,TupleBC}
const FieldAndBC   = Pair{<:Field,<:InputBC}

value(::FBC{<:Nothing}, grid::SG{N}, loc, dim, ::Vararg{Integer,N}) where {N} = zero(eltype(grid))

value(bc::FBC{<:Number}, ::SG{N}, loc, dim, ::Vararg{Integer,N}) where {N} = bc.value

@propagate_inbounds function value(bc::FBC{<:AbstractField}, ::SG{N}, loc, dim, I::Vararg{Integer,N}) where {N}
    bc.value[remove_dim(dim, I)...]
end

@propagate_inbounds function bc!(::Val, dim::Val, grid::SG{N}, f, loc::Vertex, bc::Dirichlet, I::Vararg{Integer,N}) where {N}
    f[I...] = value(bc, grid, loc, dim, I...)
    return
end

@propagate_inbounds neighbor_index(::Val{1}, ::Val{D}, I::Vararg{Integer,N}) where {D,N} = ntuple(i -> i == D ? I[i] + 1 : I[i], Val(N))
@propagate_inbounds neighbor_index(::Val{2}, ::Val{D}, I::Vararg{Integer,N}) where {D,N} = ntuple(i -> i == D ? I[i] - 1 : I[i], Val(N))

@propagate_inbounds function bc!(side::Val, dim::Val, grid::SG{N}, f, ::Center, bc::Dirichlet, I::Vararg{Integer,N}) where {N}
    I2 = neighbor_index(side, dim, I...)
    if grid isa UniformGrid
        t = eltype(f)(2)
    else
        t = eltype(f)(2) * Δ(grid, Center(), dim, I2...) * iΔ(grid, Vertex(), dim, I...)
    end
    a = value(bc, grid, Center(), dim, I...)
    b = f[I2...]
    f[I...] = fma(t, a - b, b)
    return
end

@add_cartesian function bc!(side::Val, dim::Val, grid::SG{N}, f, loc::Location, bc::Neumann, I::Vararg{Integer,N}) where {N}
    I2 = neighbor_index(side, dim, I...)
    f[I...] = f[I2...] + Δ(grid, flip(loc), dim, I...) * value(bc, grid, flip(loc), dim, I...)
    return
end

@inline halo_index(::Val{1}, ::Val{D}, f::Field, ::Center) where {D} = firstindex(f, D) - 1
@inline halo_index(::Val{2}, ::Val{D}, f::Field, ::Center) where {D} = lastindex(f, D) + 1

@inline halo_index(::Val{1}, ::Val{D}, f::Field, ::Vertex) where {D} = firstindex(f, D)
@inline halo_index(::Val{2}, ::Val{D}, f::Field, ::Vertex) where {D} = lastindex(f, D)
