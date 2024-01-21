module BoundaryConditions

export FieldBoundaryCondition, Dirichlet, Neumann, bc!, batch

using Chmy
using Chmy.Grids
using Chmy.Fields

import Chmy: @add_cartesian

using KernelAbstractions

import Base.@propagate_inbounds

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

const SG  = StructuredGrid
const FBC = FieldBoundaryCondition

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
    f[I...] = f[I2...] + Δ(grid, flip(loc), dim, I...) * value(bc, grid, loc, dim, I...)
    return
end

@inline halo_index(::Val{1}, ::Val{D}, f::Field, ::Center) where {D} = firstindex(f, D) - 1
@inline halo_index(::Val{2}, ::Val{D}, f::Field, ::Center) where {D} = lastindex(f, D) + 1

@inline halo_index(::Val{1}, ::Val{D}, f::Field, ::Vertex) where {D} = firstindex(f, D)
@inline halo_index(::Val{2}, ::Val{D}, f::Field, ::Vertex) where {D} = lastindex(f, D)

@inline bc!(side, dim, grid, f, ::Union{Nothing,NTuple{K,Nothing}}) where {K} = nothing

# single field, same BC
@kernel inbounds = true function bc_kernel!(side::Val, dim::Val, grid::SG{N}, f::Field{T,N}, bc::FBC) where {N,T}
    I   = @index(Global, NTuple)
    I   = I .- 1
    Ibc = insert_dim(dim, I, halo_index(side, dim, f, location(f, dim)))
    bc!(side, dim, grid, f, location(f, dim), bc, Ibc...)
end

function bc!(side::Val, dim::Val, grid::SG{N}, f::Field{T,N}, bc::FBC) where {N,T}
    worksize = remove_dim(dim, size(interior(f; with_halo=true)))
    backend  = get_backend(f)
    bc_kernel!(backend, 256, worksize)(side, dim, grid, f, bc)
    return
end

# batched version
@kernel function bc_kernel!(side::Val, dim::Val, grid::SG{N}, fs::NTuple{K,Field{T,N}}, bcs::NTuple{K,FBC}) where {N,T,K}
    I = @index(Global, NTuple)
    I = I .- 1
    ntuple(Val(K)) do ifield
        Base.@_inline_meta
        @inbounds begin
            f   = fs[ifield]
            bc  = bcs[ifield]
            Ibc = insert_dim(dim, I, halo_index(side, dim, f, location(f, dim)))
            bc!(side, dim, grid, f, location(f, dim), bc, Ibc...)
        end
    end
end

function bc!(side::Val, dim::Val, grid::SG{N}, fs::NTuple{K,Field{T,N}}, bcs::NTuple{K,FBC}) where {N,T,K}
    worksize = remove_dim(dim, size(grid, Vertex()))
    backend  = get_backend(first(fs))
    bc_kernel!(backend, 256, worksize)(side, dim, grid, fs, bcs)
    return
end

const FBCOrNothing = Union{FBC,Nothing}

const FieldAndBC{N} = Pair{<:Field{<:Any,N},<:FBCOrNothing} where {N}
const BCOrTuple = Union{FBC,Tuple{FBCOrNothing,FBCOrNothing}}

const TupleBC{names,K} = NamedTuple{names,<:NTuple{K,BCOrTuple}} where {names,K}
const FieldAndBCs{N,names,K} = Pair{<:Field{<:Any,N},<:TupleBC{names,K}} where {N,names,K}

# same bc for all sides
function bc!(grid::SG{N}, f_bc::FieldAndBC{N}) where {N}
    f, bc = f_bc
    ntuple(Val(N)) do J
        Base.@_inline_meta
        D = N - J + 1 # set bcs in reverse order for consistency with the hidecomm version
        bc!(Val(1), Val(D), grid, f, bc)
        bc!(Val(2), Val(D), grid, f, bc)
    end
    return
end

# bc per dim per side
function bc!(grid::SG{N}, f_bc::FieldAndBCs{N,names,K}) where {N,names,K}
    f, bcs = f_bc
    ntuple(Val(K)) do J
        Base.@_inline_meta
        I = K - J + 1 # set bcs in reverse order for consistency with the hidecomm version
        dir = direction(grid, Val(names[I]))
        bc!(Val(1), dir, grid, f, bcs[I][1])
        bc!(Val(2), dir, grid, f, bcs[I][2])
    end
    return
end

default_bc(grid::SG{N}) where {N} = NamedTuple{axes_names(grid)}(ntuple(_ -> (nothing, nothing), Val(N)))

expand(bc::Tuple{FBCOrNothing,FBCOrNothing}) = bc
expand(bc::FBC) = (bc, bc)

regularise(grid::SG{N}, bc::TupleBC) where {N} = merge(default_bc(grid), map(expand, bc))

@inline function prune(fields, bcs)
    f_bc   = (zip(fields, bcs)...,)
    pruned = filter(x -> !isnothing(last(x)), f_bc)
    return (zip(pruned...)...,)
end

@inline function batch(grid::SG{N}, f_bcs::NTuple{K,FieldAndBCs{N}}) where {N,K}
    fs, bcs = zip(f_bcs...)
    bcs_reg = map(x -> regularise(grid, x), bcs)
    ntuple(Val(N)) do D
        Base.@_inline_meta
        ntuple(Val(2)) do S
            batch = ntuple(J -> bcs_reg[J][D][S], Val(K))
            prune(fs, batch)
        end
    end
end

# batched version
function bc!(grid::SG{N}, batched::Tuple) where {N}
    ntuple(Val(N)) do D
        Base.@_inline_meta
        bc!(Val(1), Val(D), grid, batched[D][1]...)
        bc!(Val(2), Val(D), grid, batched[D][2]...)
    end
    return
end

bc!(grid::SG{N}, f_bc::Vararg{FieldAndBCs{N}}) where {N} = bc!(grid, batch(grid, f_bc))

end
