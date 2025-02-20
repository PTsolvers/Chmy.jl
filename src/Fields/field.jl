"""
    struct Field{T,N,L,H,A} <: AbstractField{T,N,L}

Field represents a discrete scalar field with specified type, number of dimensions, location, and halo size.
"""
struct Field{T,N,L,H,A} <: AbstractField{T,N,L}
    data::A
    dims::NTuple{N,Int}
    Field{L,H}(data::AbstractArray{T,N}, dims::NTuple{N,Integer}) where {L,H,T,N} = new{T,N,L,H,typeof(data)}(data, dims)
end

halo(::Field{T,N,A,H}) where {T,N,A,H} = H

# AbstractArray interface
Base.size(f::Field) = f.dims
Base.parent(f::Field) = f.data

@propagate_inbounds Base.getindex(f::Field{T,N,L,H}, I::Vararg{Int,N}) where {T,N,L,H} = f.data[(I .+ 2 .* H)...]

@propagate_inbounds function Base.setindex!(f::Field{T,N,L,H}, v, I::Vararg{Int,N}) where {T,N,L,H}
    f.data[(I .+ 2 .* H)...] = v
end

_expand(rng::AbstractUnitRange, h) = (first(rng)-h):(last(rng)+h)

"""
    interior(f::Field; with_halo=false)

Displays the field on the interior of the grid on which it is defined on.
One could optionally specify to display the halo regions on the grid with
`with_halo=true`.
"""
function interior(f::Field; with_halo=false)
    ax      = with_halo ? _expand.(axes(f), halo(f)) : axes(f)
    indices = broadcast(.+, 2 .* halo(f), ax)
    view(f.data, indices...)
end

const LocOrLocs{N} = Union{Location,NTuple{N,Location}}

# adapt to GPU
Adapt.adapt_structure(to, f::Field{T,N,L,H}) where {T,N,L,H} = Field{L,H}(Adapt.adapt(to, f.data), f.dims)

"""
    Field(backend, grid, loc, type=eltype(grid); halo=1)

Constructs a field on a structured grid at the specified location.

## Arguments:
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
    disable_task_sync!(backend, data)
    loc = expand_loc(Val(N), loc)
    return Field{typeof(loc),halo}(data, dims)
end

"""
    Field(arch::Architecture, args...; kwargs...)

Create a `Field` object on the specified architecture.

## Arguments:
- `arch::Architecture`: The architecture for which to create the `Field`.
- `args...`: Additional positional arguments to pass to the `Field` constructor.
- `kwargs...`: Additional keyword arguments to pass to the `Field` constructor.
"""
Field(arch::Architecture, args...; kwargs...) = Field(Architectures.get_backend(arch), args...; kwargs...)

# set fields

"""
    set!(f::Field, val::Number)

Set all elements of the `Field` `f` to the specified numeric value `val`.

## Arguments:
- `f::Field`: The `Field` object to be modified.
- `val::Number`: The numeric value to set in the `Field`.
"""
set!(f::Field, val::Number) = (fill!(interior(f), val); nothing)

"""
    set!(f::Field, A::AbstractArray)

Set the elements of the `Field` `f` using the values from the `AbstractArray` `A`.

## Arguments:
- `f::Field`: The `Field` object to be modified.
- `A::AbstractArray`: The array whose values are to be copied to the `Field`.
"""
set!(f::Field, A::AbstractArray) = (copyto!(interior(f), A); nothing)

"""
    set!(f::Field, other::AbstractField)

Set the elements of the `Field` `f` using the values from another `AbstractField` `other`.

## Arguments:
- `f::Field`: The destination `Field` object to be modified.
- `other::AbstractField`: The source `AbstractField` whose values are to be copied to `f`.
"""
function set!(f::Field, other::AbstractField)
    dst = interior(f)
    src = interior(other)
    backend = KernelAbstractions.get_backend(dst)
    _set_field!(backend, 256, size(dst))(dst, src)
end

@kernel inbounds = true function _set_field!(dst, src)
    I = @index(Global, NTuple)
    dst[I...] = src[I...]
end

@kernel inbounds = true function _set_continuous!(dst, grid, loc, fun::F, args::Vararg{Any, N}) where {F, N}
    I = @index(Global, NTuple)
    dst[I...] = fun(coord(grid, loc, I...)..., args...)
end

@kernel inbounds = true function _set_discrete!(dst, grid, loc, fun::F, args::Vararg{Any, N}) where {F, N}
    I = @index(Global, NTuple)
    dst[I...] = fun(grid, loc, I..., args...)
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

"""
    VectorField(backend::Backend, grid::StructuredGrid{N}, args...; kwargs...) where {N}

Create a vector field in the form of a `NamedTuple` on the given `grid` using the specified `backend`. With each component being a `Field`.

## Arguments:
- `backend::Backend`: The backend to be used for computation.
- `grid::StructuredGrid{N}`: The structured grid defining the computational domain.
- `args...`: Additional positional arguments to pass to the `Field` constructor.
- `kwargs...`: Additional keyword arguments to pass to the `Field` constructor.
"""
function VectorField(backend::Backend, grid::StructuredGrid{N}, args...; kwargs...) where {N}
    coord_names = axes_names(grid)
    names = ntuple(i -> coord_names[i], Val(N))
    values = ntuple(Val(N)) do D
        Base.@_inline_meta
        Field(backend, grid, vector_location(Val(D), Val(N)), args...; kwargs...)
    end
    return NamedTuple{names}(values)
end

"""
    TensorField(backend::Backend, grid::StructuredGrid{2}, args...; kwargs...)

Create a 2D tensor field in the form of a named tuple on the given `grid` using the specified `backend`, with components `xx`, `yy`, and `xy` each being a `Field`.

## Arguments:
- `backend::Backend`: The backend to be used for computation.
- `grid::StructuredGrid{2}`: The 2D structured grid defining the computational domain.
- `args...`: Additional positional arguments to pass to the `Field` constructor.
- `kwargs...`: Additional keyword arguments to pass to the `Field` constructor.
"""
function TensorField(backend::Backend, grid::StructuredGrid{2}, args...; kwargs...)
    (xx = Field(backend, grid, Center(), args...; kwargs...),
     yy = Field(backend, grid, Center(), args...; kwargs...),
     xy = Field(backend, grid, Vertex(), args...; kwargs...))
end

"""
    TensorField(backend::Backend, grid::StructuredGrid{3}, args...; kwargs...)

Create a 3D tensor field in the form of a named tuple on the given `grid` using the specified `backend`, with components `xx`, `yy`, `zz`, `xy`, `xz`, and `yz` each being a `Field`.

## Arguments:
- `backend::Backend`: The backend to be used for computation.
- `grid::StructuredGrid{3}`: The 3D structured grid defining the computational domain.
- `args...`: Additional positional arguments to pass to the `Field` constructor.
- `kwargs...`: Additional keyword arguments to pass to the `Field` constructor.
"""
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
