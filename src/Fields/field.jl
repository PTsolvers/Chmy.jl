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

@kernel inbounds = true function _set_continuous!(dst, grid, loc, fun::F, args::Vararg{Any,N}) where {F,N}
    I = @index(Global, NTuple)
    dst[I...] = fun(coord(grid, loc, I...)..., args...)
end

@kernel inbounds = true function _set_discrete!(dst, grid, loc, fun::F, args::Vararg{Any,N}) where {F,N}
    I = @index(Global, NTuple)
    dst[I...] = fun(grid, loc, I..., args...)
end

function set!(f::Field{T,N}, grid::StructuredGrid{N}, fun::F; discrete=false, parameters=()) where {T,F,N}
    loc = location(f)
    dst = interior(f)
    backend = KernelAbstractions.get_backend(dst)
    if discrete
        _set_discrete!(backend, 256, size(dst))(dst, grid, loc, fun, parameters...)
    else
        _set_continuous!(backend, 256, size(dst))(dst, grid, loc, fun, parameters...)
    end
    KernelAbstractions.synchronize(backend)
    return
end

# tuples of field
set!(fs::NamedTuple{names,<:NTuple{N,Field}}, args...) where {names,N} = foreach(f -> set!(f, args...), fs)

# vector fields
vector_location(::Val{dim}, ::Val{N}) where {dim,N} = ntuple(i -> i == dim ? Vertex() : Center(), Val(N))

"""
    struct VectorField{N, C <: Tuple}

A vector field of N components, each a `Field` at its staggered location.
Components are accessible via dot notation (`v.x`, `v.y`, `v.z`) or integer indexing (`v[1]`, `v[2]`, `v[3]`).
"""
struct VectorField{N,C<:Tuple}
    components::C
    VectorField{N}(components::C) where {N,C<:Tuple} = new{N,C}(components)
end

"""
    struct TensorField{N, C <: Tuple}

A symmetric tensor field for N-dimensional space.
Components are accessible via dot notation (`t.xx`, `t.xy`, etc.) or
two-index notation (`t[1,1]`, `t[1,2]`, etc., with symmetry: `t[i,j] == t[j,i]`).

2D components (in order): `xx, yy, xy`
3D components (in order): `xx, yy, zz, xy, xz, yz`
"""
struct TensorField{N,C<:Tuple}
    components::C
    TensorField{N}(components::C) where {N,C<:Tuple} = new{N,C}(components)
end

# --- VectorField accessors ---

function Base.getproperty(v::VectorField, s::Symbol)
    s === :components && return getfield(v, :components)
    s === :x && return getfield(v, :components)[1]
    s === :y && return getfield(v, :components)[2]
    s === :z && return getfield(v, :components)[3]
    error("VectorField has no property $s")
end

Base.propertynames(::VectorField{1}) = (:x,)
Base.propertynames(::VectorField{2}) = (:x, :y)
Base.propertynames(::VectorField{3}) = (:x, :y, :z)

Base.getindex(v::VectorField, i::Int) = getfield(v, :components)[i]

Base.iterate(v::VectorField, state...) = iterate(getfield(v, :components), state...)
Base.length(::VectorField{N}) where {N} = N

# --- TensorField accessors ---

function Base.getproperty(t::TensorField{N}, s::Symbol) where {N}
    s === :components && return getfield(t, :components)
    s === :xx && return getfield(t, :components)[1]
    s === :yy && return getfield(t, :components)[2]
    if N == 3
        s === :zz && return getfield(t, :components)[3]
        s === :xy && return getfield(t, :components)[4]
        s === :xz && return getfield(t, :components)[5]
        s === :yz && return getfield(t, :components)[6]
    else
        s === :xy && return getfield(t, :components)[3]
    end
    error("TensorField has no property $s")
end

Base.propertynames(::TensorField{2}) = (:xx, :yy, :xy)
Base.propertynames(::TensorField{3}) = (:xx, :yy, :zz, :xy, :xz, :yz)

function Base.getindex(t::TensorField{N}, i::Int, j::Int) where {N}
    a, b = minmax(i, j)
    if N == 2
        a == 1 && b == 1 && return getfield(t, :components)[1]
        a == 2 && b == 2 && return getfield(t, :components)[2]
        a == 1 && b == 2 && return getfield(t, :components)[3]
    else
        a == 1 && b == 1 && return getfield(t, :components)[1]
        a == 2 && b == 2 && return getfield(t, :components)[2]
        a == 3 && b == 3 && return getfield(t, :components)[3]
        a == 1 && b == 2 && return getfield(t, :components)[4]
        a == 1 && b == 3 && return getfield(t, :components)[5]
        a == 2 && b == 3 && return getfield(t, :components)[6]
    end
    error("invalid tensor index ($i, $j) for $(N)D TensorField")
end

Base.iterate(t::TensorField, state...) = iterate(getfield(t, :components), state...)
Base.length(::TensorField{N}) where {N} = N == 2 ? 3 : 6

# --- set! ---

set!(v::VectorField, args...) = foreach(f -> set!(f, args...), getfield(v, :components))
set!(t::TensorField, args...) = foreach(f -> set!(f, args...), getfield(t, :components))

# --- GPU adaptation ---

Adapt.adapt_structure(to, v::VectorField{N}) where {N} = VectorField{N}(map(f -> Adapt.adapt(to, f), getfield(v, :components)))

Adapt.adapt_structure(to, t::TensorField{N}) where {N} = TensorField{N}(map(f -> Adapt.adapt(to, f), getfield(t, :components)))

# --- factory functions ---

"""
    VectorField(backend::Backend, grid::StructuredGrid{N}, args...; kwargs...) where {N}

Create a `VectorField` on the given `grid`. Each of the N components is a `Field` staggered
at `Vertex()` along its own dimension and `Center()` along all others.
Components are accessible as `v.x`/`v.y`/`v.z` or `v[1]`/`v[2]`/`v[3]`.
"""
function VectorField(backend::Backend, grid::StructuredGrid{N}, args...; kwargs...) where {N}
    components = ntuple(Val(N)) do D
        Base.@_inline_meta
        Field(backend, grid, vector_location(Val(D), Val(N)), args...; kwargs...)
    end
    return VectorField{N}(components)
end

"""
    TensorField(backend::Backend, grid::StructuredGrid{2}, args...; kwargs...)

Create a 2D symmetric `TensorField`. Components `(xx, yy, xy)` are stored in order.
Access via `t.xx`/`t.yy`/`t.xy` or `t[1,1]`/`t[2,2]`/`t[1,2]` (symmetric: `t[i,j] == t[j,i]`).
"""
function TensorField(backend::Backend, grid::StructuredGrid{2}, args...; kwargs...)
    components = (Field(backend, grid, Center(), args...; kwargs...),
                  Field(backend, grid, Center(), args...; kwargs...),
                  Field(backend, grid, Vertex(), args...; kwargs...))
    return TensorField{2}(components)
end

"""
    TensorField(backend::Backend, grid::StructuredGrid{3}, args...; kwargs...)

Create a 3D symmetric `TensorField`. Components `(xx, yy, zz, xy, xz, yz)` are stored in order.
Access via `t.xx` etc. or `t[1,1]`/`t[1,2]` etc. (symmetric: `t[i,j] == t[j,i]`).
"""
function TensorField(backend::Backend, grid::StructuredGrid{3}, args...; kwargs...)
    components = (Field(backend, grid, Center(), args...; kwargs...),
                  Field(backend, grid, Center(), args...; kwargs...),
                  Field(backend, grid, Center(), args...; kwargs...),
                  Field(backend, grid, (Vertex(), Vertex(), Center()), args...; kwargs...),
                  Field(backend, grid, (Vertex(), Center(), Vertex()), args...; kwargs...),
                  Field(backend, grid, (Center(), Vertex(), Vertex()), args...; kwargs...))
    return TensorField{3}(components)
end

VectorField(arch::Architecture, args...; kwargs...) = VectorField(Architectures.get_backend(arch), args...; kwargs...)
TensorField(arch::Architecture, args...; kwargs...) = TensorField(Architectures.get_backend(arch), args...; kwargs...)
