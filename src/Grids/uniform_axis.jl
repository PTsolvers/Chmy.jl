struct UniformAxis{T} <: AbstractAxis{T}
    origin::T
    extent::T
    spacing::T
    inv_spacing::T
    length::Int
    function UniformAxis(origin::T, extent::T, len::Integer) where {T}
        spacing = extent / len
        inv_spacing = one(T) / spacing
        new{T}(origin, extent, spacing, inv_spacing, len)
    end
end

UniformAxis(origin, extent, len) = UniformAxis(promote(origin, extent)..., len)

nvertices(ax::UniformAxis) = ax.length + 1

vertex(ax::UniformAxis, i::Integer) = muladd(i - oneunit(i), ax.spacing, ax.origin)
center(ax::UniformAxis{T}, i::Integer) where {T} = muladd(i - oneunit(i), ax.spacing, muladd(T(0.5), ax.spacing, ax.origin))

origin(ax::UniformAxis, ::Vertex) = ax.origin
origin(ax::UniformAxis{T}, ::Center) where {T} = muladd(T(0.5), ax.spacing, ax.origin)

extent(ax::UniformAxis, ::Vertex) = ax.extent
extent(ax::UniformAxis, ::Center) = ax.extent - ax.spacing

spacing(ax::UniformAxis, ::Vertex, ::Integer) = ax.spacing
spacing(ax::UniformAxis, ::Center, ::Integer) = ax.spacing

inv_spacing(ax::UniformAxis, ::Vertex, ::Integer) = ax.inv_spacing
inv_spacing(ax::UniformAxis, ::Center, ::Integer) = ax.inv_spacing

function coords(ax::UniformAxis, loc::Location)
    start = coord(ax, loc, 1)
    stop  = coord(ax, loc, length(ax, loc))
    return LinRange(start, stop, length(ax, loc))
end
