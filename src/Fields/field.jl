"""
    struct Field{T,N,L,H,A} <: AbstractField{T,N,L}

Field represents a discrete scalar field with specified type, number of dimensions, location, and halo size.
"""
struct Field{T,N,L,H,A} <: AbstractField{T,N,L}
    data::A
    dims::NTuple{N,Int}
    Field{L,H}(data::AbstractArray{T,N}, dims::NTuple{N,Integer}) where {L,H,T,N} = new{T,N,L,H,typeof(data)}(data, dims)
end

Base.@assume_effects :foldable halo(::Field{T,N,A,H}) where {T,N,A,H} = H

# AbstractArray interface
Base.size(f::Field) = f.dims
Base.parent(f::Field) = f.data

@propagate_inbounds Base.getindex(f::Field{T,N,L,H}, I::Vararg{Int,N}) where {T,N,L,H} = f.data[(I .+ 2 .* H)...]

@propagate_inbounds function Base.setindex!(f::Field{T,N,L,H}, v, I::Vararg{Int,N}) where {T,N,L,H}
    f.data[(I .+ 2 .* H)...] = v
end

_expand(rng::AbstractUnitRange, h) = (first(rng)-h):(last(rng)+h)

function interior(f::Field; with_halo=false)
    ax      = with_halo ? _expand.(axes(f), halo(f)) : axes(f)
    indices = broadcast(.+, 2 .* halo(f), ax)
    view(f.data, indices...)
end

const LocOrLocs{N} = Union{Location,NTuple{N,Location}}

# adapt to GPU
Adapt.adapt_structure(to, f::Field{T,N,L,H}) where {T,N,L,H} = Field{L,H}(Adapt.adapt(to, f.data), f.dims)

# fields on grids

expand_loc(::Val{N}, locs::NTuple{N,Location}) where {N} = locs
expand_loc(::Val{N}, loc::Location) where {N} = ntuple(_ -> loc, Val(N))

"""
    Field(backend, grid, loc, type=eltype(grid); halo=1)

Constructs a field on a structured grid at the specified location.

Arguments:
- `backend`: The backend to use for memory allocation.
- `grid`: The structured grid on which the field is constructed.
- `loc`: The location or locations on the grid where the field is constructed.
- `type`: The element type of the field. Defaults to the element type of the grid.
- `halo`: The halo size for the field. Defaults to 1.
"""
function Field(backend::Backend, grid::StructuredGrid{N}, loc::LocOrLocs{N}, type=eltype(grid); halo=1) where {N}
    dims = size(grid, loc)
    data_size = size(grid, loc) .+ 4 .* halo
    data = KernelAbstractions.zeros(backend, type, data_size)
    loc = expand_loc(Val(N), loc)
    return Field{typeof(loc),halo}(data, dims)
end

Field(arch::Architecture, args...; kwargs...) = Field(Architectures.get_backend(arch), args...; kwargs...)

# set fields

set!(f::Field, other::Field) = (copy!(interior(f), interior(other)); nothing)
set!(f::Field, val::Number) = (fill!(interior(f), val); nothing)
set!(f::Field, A::AbstractArray) = (copy!(interior(f), A); nothing)

@kernel inbounds = true function _set_continuous!(dst, grid, loc, fun::F, args...) where {F}
    I = @index(Global, Cartesian)
    dst[I] = fun(coord(grid, loc, I)..., args...)
end

@kernel inbounds = true function _set_discrete!(dst, grid, loc, fun::F, args...) where {F}
    I = @index(Global, Cartesian)
    dst[I] = fun(grid, loc, I, args...)
end

function set!(f::Field{T,N}, grid::StructuredGrid{N}, fun::F; discrete=false, parameters=(), async=false) where {T,F,N}
    loc = location(f)
    dst = interior(f)
    backend = KernelAbstractions.get_backend(dst)
    if discrete
        _set_discrete!(backend, 256, size(dst))(dst, grid, loc, fun, parameters...)
    else
        _set_continuous!(backend, 256, size(dst))(dst, grid, loc, fun, parameters...)
    end
    async || KernelAbstractions.synchronize(backend)
    return
end

# tuples of field
set!(fs::NamedTuple{names,<:NTuple{N,Field}}, args...) where {names,N} = foreach(f -> set!(f, args...), fs)

# vector fields
vector_location(::Val{dim}, ::Val{N}) where {dim,N} = ntuple(i -> i == dim ? Vertex() : Center(), Val(N))

function VectorField(backend::Backend, grid::StructuredGrid{N}, args...; kwargs...) where {N}
    coord_names = axes_names(grid)
    names = ntuple(i -> coord_names[i], Val(N))
    values = ntuple(Val(N)) do D
        Base.@_inline_meta
        Field(backend, grid, vector_location(Val(D), Val(N)), args...; kwargs...)
    end
    return NamedTuple{names}(values)
end

# tensor fields
function TensorField(backend::Backend, grid::StructuredGrid{2}, args...; kwargs...)
    (xx = Field(backend, grid, Center(), args...; kwargs...),
     yy = Field(backend, grid, Center(), args...; kwargs...),
     xy = Field(backend, grid, Vertex(), args...; kwargs...))
end

function TensorField(backend::Backend, grid::StructuredGrid{3}, args...; kwargs...)
    (xx = Field(backend, grid, Center(), args...; kwargs...),
     yy = Field(backend, grid, Center(), args...; kwargs...),
     zz = Field(backend, grid, Center(), args...; kwargs...),
     xy = Field(backend, grid, (Vertex(), Vertex(), Center()), args...; kwargs...),
     xz = Field(backend, grid, (Vertex(), Center(), Vertex()), args...; kwargs...),
     yz = Field(backend, grid, (Center(), Vertex(), Vertex()), args...; kwargs...))
end

VectorField(arch::Architecture, args...; kwargs...) = VectorField(Architectures.get_backend(arch), args...; kwargs...)
TensorField(arch::Architecture, args...; kwargs...) = TensorField(Architectures.get_backend(arch), args...; kwargs...)
