struct FunctionAxis{T,F} <: AbstractAxis{T}
    vertex_fun::F
    length::Int
    FunctionAxis(f::F, len::Integer) where {F} = new{typeof(float(f(1))),F}(f, len)
end

nvertices(ax::FunctionAxis) = ax.length + 1

@propagate_inbounds vertex(ax::FunctionAxis, i::Integer) = ax.vertex_fun(i)
