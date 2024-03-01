function subaxis(ax::UniformAxis{T}, offset::Integer, len::Integer) where {T}
    new_origin = vertex(ax, offset + oneunit(offset))
    new_extent = ax.spacing * len
    return UniformAxis(new_origin, new_extent, len)
end

function subaxis(ax::FunctionAxis{T,F}, offset::Integer, len::Integer) where {T,F}
    @inline new_f(i::Integer) = ax.vertex_fun(offset + i)
    return FunctionAxis(new_f, len)
end

function overwrite_connectivity(conn, topo::CartesianTopology{N}) where {N}
    ntuple(Val(N)) do D
        Base.@_inline_meta
        ntuple(S -> has_neighbor(topo, D, S) ? Connected() : conn[D][S], Val(2))
    end
end

function Grids.StructuredGrid{C}(arch::DistributedArchitecture, axes::Vararg{AbstractAxis,N}; kwargs...) where {C,N}
    global_dims = ntuple(Val(N)) do D
        Base.@_inline_meta
        length(axes[D], Center())
    end

    local_dims = cld.(global_dims, dims(arch.topology))
    offsets = coords(arch.topology) .* local_dims

    local_axes = ntuple(Val(N)) do D
        Base.@_inline_meta
        subaxis(axes[D], offsets[D], local_dims[D])
    end

    conn = overwrite_connectivity(C.instance, arch.topology)

    return StructuredGrid{typeof(conn)}(local_axes...; kwargs...)
end
