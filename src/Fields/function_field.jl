struct Continuous end
struct Discrete end

"""
    FunctionField <: AbstractField

Continuous or discrete field with values computed at runtime.

## Constructors
- `FunctionField(func, grid, loc; [discrete], [parameters])`: Create a new `FunctionField` object.
"""
struct FunctionField{T,N,L,CD,F,G,P} <: AbstractField{T,N,L}
    func::F
    grid::G
    parameters::P
    function FunctionField{CD,L}(func::F, grid::G, parameters::P) where {CD,L,F,G,P}
        N = ndims(grid)
        T = eltype(grid)
        return new{T,N,L,CD,F,G,P}(func, grid, parameters)
    end
end

"""
    FunctionField(func::F, grid::StructuredGrid{N}, loc; discrete=false, parameters=nothing) where {F,N}

Create a `FunctionField` on the given `grid` using the specified function `func`.

## Arguments:
- `func::F`: The function used to generate the field values.
- `grid::StructuredGrid{N}`: The structured grid defining the computational domain.
- `loc`: The nodal location on the grid grid where the function field is defined on.
- `discrete::Bool=false`: A flag indicating whether the field should be discrete. Defaults to `false`.
- `parameters=nothing`: Additional parameters to be used by the function. Defaults to `nothing`.
"""
function FunctionField(func::F, grid::StructuredGrid{N}, loc; discrete=false, parameters=nothing) where {F,N}
    loc = expand_loc(Val(N), loc)
    L   = typeof(loc)
    CD  = discrete ? Discrete : Continuous
    return FunctionField{CD,L}(func, grid, parameters)
end

Base.size(f::FunctionField) = size(f.grid, location(f))

@inline func_type(::FunctionField{T,N,L,CD}) where {T,N,L,CD} = CD

@inline _params(::Nothing) = ()
@inline _params(p) = p

@propagate_inbounds function call_func(func::F, grid, loc, params, ::Type{Continuous}, I::Vararg{Integer,N}) where {F,N}
    func(coord(grid, loc, I...)..., params...)
end

@propagate_inbounds function call_func(func::F, grid, loc, params, ::Type{Discrete}, I::Vararg{Integer,N}) where {F,N}
    func(grid, loc, I..., params...)
end

@add_cartesian function Base.getindex(f::FunctionField{T,N}, I::Vararg{Integer,N}) where {T,N}
    call_func(f.func, f.grid, location(f), _params(f.parameters), func_type(f), I...)
end

function Adapt.adapt_structure(to, f::FunctionField{T,N,L,CD}) where {T,N,L,CD}
    FunctionField{CD,L}(Adapt.adapt(to, f.func),
                        Adapt.adapt(to, f.grid),
                        Adapt.adapt(to, f.parameters))
end
