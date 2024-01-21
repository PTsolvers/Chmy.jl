struct StructuredGrid{N,T,C,A}
    axes::A
    StructuredGrid{C}(axes::Vararg{<:AbstractAxis{T},N}) where {N,T,C} = new{N,T,C,typeof(axes)}(axes)
end

const UniformGrid{N,T,C} = StructuredGrid{N,T,C,NTuple{N,UniformAxis{T}}}

function UniformGrid(; origin::NTuple{N,Number}, extent::NTuple{N,Number}, dims::NTuple{N,Integer}, topology=nothing) where {N}
    if isnothing(topology)
        topology = ntuple(_ -> Bounded(), Val(N))
    end

    extent = float.(extent)

    axes = UniformAxis.(promote(origin...),
                        promote(extent...),
                        promote(dims...))

    return StructuredGrid{typeof(topology)}(axes...)
end

const SG = StructuredGrid
const Locs{N} = NTuple{N,Location}
const LocOrLocs{N} = Union{Location,Locs{N}}

Base.@assume_effects :foldable Base.eltype(::StructuredGrid{N,T}) where {N,T} = T

Base.@assume_effects :foldable Base.ndims(::StructuredGrid{N}) where {N} = N

Base.size(grid::StructuredGrid{N}, loc::LocOrLocs{N}) where {N} = length.(grid.axes, loc)
Base.size(grid::StructuredGrid, loc::Location, ::Val{dim}) where {dim} = length(grid.axes[dim], loc)

bounds(grid::StructuredGrid{N}, loc::LocOrLocs{N}) where {N} = bounds.(grid.axes, loc)

"""
    axis(grid::RegularGrid, ::Val{dim}) where {dim}

Return the axis corresponding to the spatial dimension `dim`.
"""
axis(grid::StructuredGrid, ::Val{dim}) where {dim} = grid.axes[dim]

# coordinates

"""
    coord(grid::RegularGrid{N}, loc::[Location, NTuple{N,Location}], I...) where {N}

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

@add_cartesian coord(grid::SG{N}, loc::Location, ::Val{dim}, I::Vararg{Integer,N}) where {N,dim} = coord(grid.axes[dim], loc, I[dim])
@add_cartesian coord(grid::SG{N}, loc::Locs{N}, ::Val{dim}, I::Vararg{Integer,N}) where {N,dim} = coord(grid.axes[dim], loc[dim], I[dim])

@propagate_inbounds coord(grid::SG, loc::Location, ::Val{dim}, i::Integer) where {dim} = coord(grid.axes[dim], loc, i)
@propagate_inbounds coord(grid::SG{N}, loc::Locs{N}, ::Val{dim}, i::Integer) where {N,dim} = coord(grid.axes[dim], loc[dim], i)

@propagate_inbounds vertex(grid::SG, ::Val{dim}, i::Integer) where {dim} = vertex(grid.axes[dim], i)
@add_cartesian vertex(grid::SG{N}, ::Val{dim}, I::Vararg{Integer,N}) where {N,dim} = vertex(grid.axes[dim], I[dim])

@propagate_inbounds center(grid::SG, ::Val{dim}, i::Integer) where {dim} = center(grid.axes[dim], i)
@add_cartesian center(grid::SG{N}, ::Val{dim}, I::Vararg{Integer,N}) where {N,dim} = center(grid.axes[dim], I[dim])

# spacing

for (sp, desc) in ((:spacing, "grid spacings"), (:inv_spacing, "inverse grid spacings"))
    @eval begin
        """
            $($sp)(grid::RegularGrid{N}, loc::[Location, NTuple{N,Location}], I...) where {N}

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

        @add_cartesian $sp(grid::SG{N}, loc::Location, ::Val{dim}, I::Vararg{Integer,N}) where {N,dim} = $sp(grid.axes[dim], loc, I[dim])
        @add_cartesian $sp(grid::SG{N}, loc::Locs{N}, ::Val{dim}, I::Vararg{Integer,N}) where {N,dim} = $sp(grid.axes[dim], loc[dim], I[dim])

        @propagate_inbounds $sp(grid::SG, loc::Location, ::Val{dim}, i::Integer) where {dim} = $sp(grid.axes[dim], loc, i)
        @propagate_inbounds $sp(grid::SG{N}, loc::Locs{N}, ::Val{dim}, i::Integer) where {N,dim} = $sp(grid.axes[dim], loc[dim], i)
    end
end

# coordinate lists

coords(grid::SG{N}, loc::LocOrLocs{N}) where {N} = coords.(grid.axes, loc)
coords(grid::SG{N}, loc::Location, ::Val{dim}) where {N,dim} = coords(grid.axes[dim], loc)
coords(grid::SG{N}, loc::Locs{N}, ::Val{dim}) where {N,dim} = coords(grid.axes[dim], loc[dim])

vertices(grid::SG) = vertices.(grid.axes)
vertices(grid::SG, ::Val{dim}) where {dim} = vertices(grid.axes[dim])

centers(grid::SG) = centers.(grid.axes)
centers(grid::SG, ::Val{dim}) where {dim} = centers(grid.axes[dim])

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

        @propagate_inbounds $_Δ(grid::SG{N}, loc, I::Vararg{Integer,N}) where {N} = spacing(grid, loc, Val($dim), I...)
        @propagate_inbounds $_Δ(grid::SG, loc, I) = spacing(grid, loc, Val($dim), I)

        @propagate_inbounds $_coord(grid::SG{N}, loc, I::Vararg{Integer,N}) where {N} = coord(grid, loc, Val($dim), I...)
        @propagate_inbounds $_coord(grid::SG, loc, I) = coord(grid, loc, Val($dim), I)

        @propagate_inbounds $_vertex(grid::SG{N}, I::Vararg{Integer,N}) where {N} = vertex(grid, Val($dim), I...)
        @propagate_inbounds $_vertex(grid::SG, I) = vertex(grid, Val($dim), I)

        @propagate_inbounds $_center(grid::SG{N}, I::Vararg{Integer,N}) where {N} = center(grid, Val($dim), I...)
        @propagate_inbounds $_center(grid::SG, I) = center(grid, Val($dim), I)

        @propagate_inbounds $_coords(grid::SG, loc) = coords(grid, loc, Val($dim))

        @propagate_inbounds $_vertices(grid::SG) = vertices(grid, Val($dim))
        @propagate_inbounds $_centers(grid::SG) = centers(grid, Val($dim))
    end
end

# coordinate names and directions
direction(::SG, ::Val{:x}) = Val(1)
direction(::SG, ::Val{:y}) = Val(2)
direction(::SG, ::Val{:z}) = Val(3)

axes_names(::SG{1}) = (:x,)
axes_names(::SG{2}) = (:x, :y)
axes_names(::SG{3}) = (:x, :y, :z)
