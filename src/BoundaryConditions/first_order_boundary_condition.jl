struct DirichletKind end
struct NeumannKind end

struct FirstOrderBC{T,Kind} <: FieldBoundaryCondition
    value::T
    FirstOrderBC{Kind}(value::T) where {Kind,T} = new{T,Kind}(value)
end

const Dirichlet{T} = FirstOrderBC{T,DirichletKind}
const Neumann{T}   = FirstOrderBC{T,NeumannKind}

Dirichlet(value=nothing) = FirstOrderBC{DirichletKind}(value)
Neumann(value=nothing)   = FirstOrderBC{NeumannKind}(value)

Base.show(io::IO, ::Dirichlet{Nothing}) = print(io, "Dirichlet(0)")
Base.show(io::IO, ::Neumann{Nothing}) = print(io, "Neumann(0)")

value(::FirstOrderBC{<:Nothing}, grid::SG{N}, loc, dim, ::Vararg{Integer,N}) where {N} = zero(eltype(grid))

value(bc::FirstOrderBC{<:Number}, ::SG{N}, loc, dim, ::Vararg{Integer,N}) where {N} = bc.value

@propagate_inbounds function value(bc::FirstOrderBC{<:AbstractField}, ::SG{N}, loc, dim, I::Vararg{Integer,N}) where {N}
    bc.value[remove_dim(dim, I)...]
end

@propagate_inbounds function bc!(::Side, dim::Dim, grid::SG{N}, f, loc::Vertex, bc::Dirichlet, I::Vararg{Integer,N}) where {N}
    f[I...] = value(bc, grid, loc, dim, I...)
    return
end

@propagate_inbounds neighbor_index(::Side{1}, ::Dim{D}, I::Vararg{Integer,N}) where {D,N} = ntuple(i -> i == D ? I[i] + 1 : I[i], Val(N))
@propagate_inbounds neighbor_index(::Side{2}, ::Dim{D}, I::Vararg{Integer,N}) where {D,N} = ntuple(i -> i == D ? I[i] - 1 : I[i], Val(N))

@propagate_inbounds function bc!(side::Side, dim::Dim, grid::SG{N}, f, ::Center, bc::Dirichlet, I::Vararg{Integer,N}) where {N}
    I2 = neighbor_index(side, dim, I...)
    if grid isa UniformGrid
        t = eltype(f)(2)
    else
        t = eltype(f)(2) * Δ(grid, Center(), dim, I2...) * iΔ(grid, Vertex(), dim, I...)
    end
    a = value(bc, grid, Center(), dim, I...)
    b = f[I2...]
    f[I...] = muladd(t, a - b, b)
    return
end

@add_cartesian function bc!(side::Side, dim::Dim, grid::SG{N}, f, loc::Location, bc::Neumann, I::Vararg{Integer,N}) where {N}
    I2 = neighbor_index(side, dim, I...)
    f[I...] = f[I2...] + Δ(grid, flip(loc), dim, I...) * value(bc, grid, flip(loc), dim, I...)
    return
end

@inline halo_index(::Side{1}, ::Dim{D}, f::Field, ::Center) where {D} = firstindex(f, D) - 1
@inline halo_index(::Side{2}, ::Dim{D}, f::Field, ::Center) where {D} = lastindex(f, D) + 1

@inline halo_index(::Side{1}, ::Dim{D}, f::Field, ::Vertex) where {D} = firstindex(f, D)
@inline halo_index(::Side{2}, ::Dim{D}, f::Field, ::Vertex) where {D} = lastindex(f, D)
