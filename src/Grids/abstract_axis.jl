abstract type AbstractAxis{T} end

ncenters(ax::AbstractAxis{T}) where {T} = nvertices(ax) - 1

Base.length(ax::AbstractAxis, ::Vertex) = nvertices(ax)
Base.length(ax::AbstractAxis, ::Center) = ncenters(ax)

extent(ax::AbstractAxis, ::Vertex) = vertex(ax, nvertices(ax)) - vertex(ax, 1)
extent(ax::AbstractAxis, ::Center) = center(ax, ncenters(ax)) - center(ax, 1)

@propagate_inbounds center(ax::AbstractAxis{T}, i::Integer) where {T} = T(0.5) * (vertex(ax, i) + vertex(ax, i + oneunit(i)))

@propagate_inbounds coord(ax::AbstractAxis, ::Vertex, i::Integer) = vertex(ax, i)
@propagate_inbounds coord(ax::AbstractAxis, ::Center, i::Integer) = center(ax, i)

origin(ax::AbstractAxis, ::Vertex) = @inbounds vertex(ax, 1)
origin(ax::AbstractAxis, ::Center) = @inbounds center(ax, 1)

@propagate_inbounds spacing(ax::AbstractAxis, ::Center, i::Integer) = vertex(ax, i + 1) - vertex(ax, i)
@propagate_inbounds spacing(ax::AbstractAxis, ::Vertex, i::Integer) = center(ax, i) - center(ax, i - 1)

@propagate_inbounds inv_spacing(ax::AbstractAxis{T}, loc, i) where {T} = one(T) / spacing(ax, loc, i)

coords(ax::AbstractAxis, loc::Location) = @inbounds [coord(ax, loc, i) for i in 1:length(ax, loc)]

centers(ax::AbstractAxis)  = coords(ax, Center())
vertices(ax::AbstractAxis) = coords(ax, Vertex())
