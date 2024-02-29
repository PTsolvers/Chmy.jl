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

delta_index(::Side{1}) = -1
delta_index(::Side{2}) = +1

halo_index(::Side{1}, ::Dim{D}, f::Field) where {D} = firstindex(f, D)
halo_index(::Side{2}, ::Dim{D}, f::Field) where {D} = lastindex(f, D)

itp_halo_index(side, dim, f) = halo_index(side, dim, f) + delta_index(side)

@propagate_inbounds function bc!(side::Side, dim::Dim, grid, f, loc::Vertex, bc::Dirichlet, I::Vararg{Integer})
    I_f = insert_dim(dim, I, halo_index(side, dim, f))
    f[I_f...] = value(bc, grid, loc, dim, I_f...)
    return
end

@propagate_inbounds neighbor_index(::Side{1}, ::Dim{D}, I::Vararg{Integer,N}) where {D,N} = ntuple(i -> i == D ? I[i] + 1 : I[i], Val(N))
@propagate_inbounds neighbor_index(::Side{2}, ::Dim{D}, I::Vararg{Integer,N}) where {D,N} = ntuple(i -> i == D ? I[i] - 1 : I[i], Val(N))

@add_cartesian function bc!(side::Side, dim::Dim, grid, f, ::Center, bc::Dirichlet, I::Vararg{Integer,N}) where {N}
    I_f  = insert_dim(dim, I, itp_halo_index(side, dim, f))
    I_f2 = neighbor_index(side, dim, I_f...)
    if grid isa UniformGrid
        t = eltype(f)(2)
    else
        t = eltype(f)(2) * Δ(grid, Center(), dim, I_f2...) * iΔ(grid, Vertex(), dim, I_f...)
    end
    a = value(bc, grid, Center(), dim, I_f...)
    b = f[I_f2...]
    f[I_f...] = muladd(t, a - b, b)
    return
end

flux_sign(q, ::Side{1}) = -q
flux_sign(q, ::Side{2}) = +q

@add_cartesian function bc!(side::Side, dim::Dim, grid, f, loc::Location, bc::Neumann, I::Vararg{Integer,N}) where {N}
    I_f       = insert_dim(dim, I, itp_halo_index(side, dim, f))
    I_f2      = neighbor_index(side, dim, I_f...)
    q         = value(bc, grid, flip(loc), dim, I_f...)
    q_s       = flux_sign(q, side)
    h         = Δ(grid, flip(loc), dim, I_f...)
    f[I_f...] = muladd(h, q_s, f[I_f2...])
    return
end
