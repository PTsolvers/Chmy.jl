"""
    StructuredGrid

Represents a structured grid with orthogonal axes.
"""
struct StructuredGrid{N,T,C,A}
    axes::A
    StructuredGrid{C}(axes::Vararg{AbstractAxis{T},N}) where {N,T,C} = new{N,T,C,typeof(axes)}(axes)
end

StructuredGrid{C}(::SingleDeviceArchitecture, axes::Vararg{AbstractAxis}) where {C} = StructuredGrid{C}(axes...)

const UniformGrid{N,T,C} = StructuredGrid{N,T,C,NTuple{N,UniformAxis{T}}}

"""
    UniformGrid(arch; origin, extent, dims, topology=nothing)

Constructs a uniform grid with specified origin, extent, dimensions, and topology.

## Arguments
- `arch::Architecture`: The associated architecture.
- `origin::NTuple{N,Number}`: The origin of the grid.
- `extent::NTuple{N,Number}`: The extent of the grid.
- `dims::NTuple{N,Integer}`: The dimensions of the grid.
- `topology=nothing`: The topology of the grid. If not provided, a default `Bounded` topology is used.
"""
function UniformGrid(arch::Architecture; origin::NTuple{N,Number}, extent::NTuple{N,Number}, dims::NTuple{N,Integer}, topology=nothing) where {N}
    if isnothing(topology)
        topology = ntuple(_ -> (Bounded(), Bounded()), Val(N))
    end

    extent = float.(extent)

    axes = UniformAxis.(promote(origin...),
                        promote(extent...),
                        promote(dims...))

    return StructuredGrid{typeof(topology)}(arch, axes...)
end

const SG = StructuredGrid
const Locs{N} = NTuple{N,Location}
const LocOrLocs{N} = Union{Location,Locs{N}}

Base.@assume_effects :foldable Base.eltype(::StructuredGrid{N,T}) where {N,T} = T

Base.@assume_effects :foldable Base.ndims(::StructuredGrid{N}) where {N} = N

Base.size(grid::SG{N}, loc::LocOrLocs{N}) where {N} = length.(grid.axes, loc)
Base.size(grid::SG, loc::Location, ::Val{dim}) where {dim} = length(grid.axes[dim], loc)

"""
    connectivity(grid, dim::Dim, side::Side)

Return the connectivity of the structured grid `grid` for the given dimension `dim` and side `side`.
"""
connectivity(::SG{N,T,C}, ::Dim{D}, ::Side{S}) where {N,T,C,D,S} = C.instance[D][S]

"""
    bounds(grid, loc, [dim::Dim])

Return the bounds of a structured grid at the specified location(s).
"""
bounds(grid::SG{N}, loc::LocOrLocs{N}) where {N} = bounds.(grid.axes, loc)
bounds(grid::SG, loc::Location, ::Dim{dim}) where {dim} = bounds(grid.axes[dim], loc)
bounds(grid::SG{N}, locs::Locs{N}, ::Dim{dim}) where {N,dim} = bounds(grid.axes[dim], locs[dim])

"""
    extent(grid, loc, [dim::Dim])

Return the extent of a structured grid at the specified location(s).
"""
extent(grid::SG{N}, loc::LocOrLocs{N}) where {N} = extent.(grid.axes, loc)
extent(grid::SG, loc::Location, ::Dim{dim}) where {dim} = extent(grid.axes[dim], loc)
extent(grid::SG{N}, locs::Locs{N}, ::Dim{dim}) where {N,dim} = extent(grid.axes[dim], locs[dim])

"""
    origin(grid, loc, [dim::Dim])

Return the origin of a structured grid at the specified location(s).
"""
origin(grid::SG{N}, loc::LocOrLocs{N}) where {N} = origin.(grid.axes, loc)
origin(grid::SG, loc::Location, ::Dim{dim}) where {dim} = origin(grid.axes[dim], loc)
origin(grid::SG{N}, locs::Locs{N}, ::Dim{dim}) where {N,dim} = origin(grid.axes[dim], locs[dim])

"""
    axis(grid, dim::Dim)

Return the axis corresponding to the spatial dimension `dim`.
"""
axis(grid::SG, ::Dim{dim}) where {dim} = grid.axes[dim]

# coordinates

"""
    coord(grid, loc, I...)

Return a tuple of spatial coordinates of a grid point at location `loc` and indices `I`.

For vertex locations, first grid point is at the origin.
For center locations, first grid point at half-spacing distance from the origin.
"""
@add_cartesian coord(grid::SG{N}, loc::Location, I::Vararg{Integer,N}) where {N} =
    ntuple(Val(N)) do D
        Base.@_inline_meta
        coord(grid.axes[D], loc, I[D])
    end

@add_cartesian coord(grid::SG{N}, locs::Locs{N}, I::Vararg{Integer,N}) where {N} =
    ntuple(Val(N)) do D
        Base.@_inline_meta
        coord(grid.axes[D], locs[D], I[D])
    end

@add_cartesian coord(grid::SG{N}, loc::Location, ::Dim{dim}, I::Vararg{Integer,N}) where {N,dim} = coord(grid.axes[dim], loc, I[dim])
@add_cartesian coord(grid::SG{N}, loc::Locs{N}, ::Dim{dim}, I::Vararg{Integer,N}) where {N,dim} = coord(grid.axes[dim], loc[dim], I[dim])

@propagate_inbounds coord(grid::SG, loc::Location, ::Dim{dim}, i::Integer) where {dim} = coord(grid.axes[dim], loc, i)
@propagate_inbounds coord(grid::SG{N}, loc::Locs{N}, ::Dim{dim}, i::Integer) where {N,dim} = coord(grid.axes[dim], loc[dim], i)

@propagate_inbounds vertex(grid::SG, ::Dim{dim}, i::Integer) where {dim} = vertex(grid.axes[dim], i)
@add_cartesian vertex(grid::SG{N}, ::Dim{dim}, I::Vararg{Integer,N}) where {N,dim} = vertex(grid.axes[dim], I[dim])

@propagate_inbounds center(grid::SG, ::Dim{dim}, i::Integer) where {dim} = center(grid.axes[dim], i)
@add_cartesian center(grid::SG{N}, ::Dim{dim}, I::Vararg{Integer,N}) where {N,dim} = center(grid.axes[dim], I[dim])

# spacing

for (sp, desc) in ((:spacing, "grid spacings"), (:inv_spacing, "inverse grid spacings"))
    @eval begin
        """
            $($sp)(grid, loc, I...)

        Return a tuple of $($desc) at location `loc` and indices `I`.
        """
        @add_cartesian $sp(grid::SG{N}, loc::Location, I::Vararg{Integer,N}) where {N} =
            ntuple(Val(N)) do D
                Base.@_inline_meta
                $sp(grid.axes[D], loc, I[D])
            end

        @add_cartesian $sp(grid::SG{N}, locs::Locs{N}, I::Vararg{Integer,N}) where {N} =
            ntuple(Val(N)) do D
                Base.@_inline_meta
                $sp(grid.axes[D], locs[D], I[D])
            end

        @add_cartesian $sp(grid::SG{N}, loc::Location, ::Dim{dim}, I::Vararg{Integer,N}) where {N,dim} = $sp(grid.axes[dim], loc, I[dim])
        @add_cartesian $sp(grid::SG{N}, loc::Locs{N}, ::Dim{dim}, I::Vararg{Integer,N}) where {N,dim} = $sp(grid.axes[dim], loc[dim], I[dim])

        @propagate_inbounds $sp(grid::SG, loc::Location, ::Dim{dim}, i::Integer) where {dim} = $sp(grid.axes[dim], loc, i)
        @propagate_inbounds $sp(grid::SG{N}, loc::Locs{N}, ::Dim{dim}, i::Integer) where {N,dim} = $sp(grid.axes[dim], loc[dim], i)
    end
end

"""
    spacing(grid::UniformGrid)

Return a tuple of grid spacing for a uniform grid `grid`.
"""
spacing(grid::UniformGrid) = getfield.(grid.axes, :spacing)
spacing(grid::UniformGrid, ::Dim{dim}) where {dim} = grid.axes[dim].spacing

"""
    inv_spacing(grid::UniformGrid)

Return a tuple of inverse grid spacing for a uniform grid `grid`.
"""
inv_spacing(grid::UniformGrid) = getfield.(grid.axes, :inv_spacing)
inv_spacing(grid::UniformGrid, ::Dim{dim}) where {dim} = grid.axes[dim].inv_spacing

# coordinate lists

coords(grid::SG{N}, loc::LocOrLocs{N}) where {N} = coords.(grid.axes, loc)
coords(grid::SG{N}, loc::Location, ::Dim{dim}) where {N,dim} = coords(grid.axes[dim], loc)
coords(grid::SG{N}, loc::Locs{N}, ::Dim{dim}) where {N,dim} = coords(grid.axes[dim], loc[dim])

vertices(grid::SG) = vertices.(grid.axes)
vertices(grid::SG, ::Dim{dim}) where {dim} = vertices(grid.axes[dim])

centers(grid::SG) = centers.(grid.axes)
centers(grid::SG, ::Dim{dim}) where {dim} = centers(grid.axes[dim])

# Cartesian coordinate systems

for (dim, c) in enumerate((:x, :y, :z))
    _Δ      = Symbol(:Δ, c)
    _coord  = Symbol(c, :coord)
    _coords = Symbol(c, :coords)

    _vertex = Symbol(c, :vertex)
    _center = Symbol(c, :center)

    _vertices = Symbol(c, :vertices)
    _centers  = Symbol(c, :centers)

    @eval begin
        export $_Δ, $_coord, $_coords, $_vertex, $_center, $_vertices, $_centers

        @propagate_inbounds $_Δ(grid::UniformGrid) = spacing(grid, Dim($dim))

        @propagate_inbounds $_Δ(grid::SG{N}, loc, I::Vararg{Integer,N}) where {N} = spacing(grid, loc, Dim($dim), I...)
        @propagate_inbounds $_Δ(grid::SG, loc, I) = spacing(grid, loc, Dim($dim), I)

        @propagate_inbounds $_coord(grid::SG{N}, loc, I::Vararg{Integer,N}) where {N} = coord(grid, loc, Dim($dim), I...)
        @propagate_inbounds $_coord(grid::SG, loc, I) = coord(grid, loc, Dim($dim), I)

        @propagate_inbounds $_vertex(grid::SG{N}, I::Vararg{Integer,N}) where {N} = vertex(grid, Dim($dim), I...)
        @propagate_inbounds $_vertex(grid::SG, I) = vertex(grid, Dim($dim), I)

        @propagate_inbounds $_center(grid::SG{N}, I::Vararg{Integer,N}) where {N} = center(grid, Dim($dim), I...)
        @propagate_inbounds $_center(grid::SG, I) = center(grid, Dim($dim), I)

        @propagate_inbounds $_coords(grid::SG, loc) = coords(grid, loc, Dim($dim))

        @propagate_inbounds $_vertices(grid::SG) = vertices(grid, Dim($dim))
        @propagate_inbounds $_centers(grid::SG) = centers(grid, Dim($dim))
    end
end

# coordinate names and directions
direction(::SG, ::Val{:x}) = Dim(1)
direction(::SG, ::Val{:y}) = Dim(2)
direction(::SG, ::Val{:z}) = Dim(3)

axes_names(::SG{1}) = (:x,)
axes_names(::SG{2}) = (:x, :y)
axes_names(::SG{3}) = (:x, :y, :z)
