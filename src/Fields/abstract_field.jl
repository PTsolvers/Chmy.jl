abstract type AbstractField{T,N,L} <: AbstractArray{T,N} end

Base.@assume_effects :foldable halo(::AbstractArray) = 0

Base.IndexStyle(::Type{<:AbstractField}) = IndexCartesian()

Base.@assume_effects :foldable location(::AbstractField{T,N,L}) where {T,N,L} = L.instance
Base.@assume_effects :foldable location(::AbstractField{T,N,L}, ::Val{D}) where {T,N,L,D} = L.instance[D]

interior(f::AbstractField) = f

# linear algebra
LinearAlgebra.norm(f::AbstractField) = LinearAlgebra.norm(interior(f))
LinearAlgebra.norm(f::AbstractField, p::Real) = LinearAlgebra.norm(interior(f), p)

_loc_string(::Center) = "Center"
_loc_string(::Vertex) = "Vertex"
_loc_string(locs::NTuple{N,Location}) where {N} = join(_loc_string.(locs), ", ")

# pretty printing
function Base.show(io::IO, ::MIME"text/plain", field::AbstractField{T,N,L}) where {T,N,L}
    print(io, "$(N)D $(join(size(field), "×")) $(nameof(typeof(field))){$T} located at ($(_loc_string(location(field))))\n")
end

function Base.show(io::IO, field::AbstractField{T,N,L}) where {T,N,L}
    print(io, "$(N)D $(join(size(field), "×")) $(nameof(typeof(field))){$T}")
end

# grid operators
@propagate_inbounds @add_cartesian function GridOperators.left(f::AbstractField, dim, I::Vararg{Integer,N}) where {N}
    GridOperators.left(f, flip(location(f, dim)), dim, I...)
end

@propagate_inbounds @add_cartesian function GridOperators.right(f::AbstractField, dim, I::Vararg{Integer,N}) where {N}
    GridOperators.right(f, flip(location(f, dim)), dim, I...)
end

@propagate_inbounds @add_cartesian function GridOperators.δ(f::AbstractField, dim, I::Vararg{Integer,N}) where {N}
    GridOperators.δ(f, flip(location(f, dim)), dim, I...)
end

@propagate_inbounds @add_cartesian function GridOperators.∂(f::AbstractField, grid, dim, I::Vararg{Integer,N}) where {N}
    GridOperators.∂(f, flip(location(f, dim)), grid, dim, I...)
end

# operators on Cartesian grids
for (dim, coord) in enumerate((:x, :y, :z))
    left = Symbol(:left, coord)
    right = Symbol(:right, coord)
    δ = Symbol(:δ, coord)
    ∂ = Symbol(:∂, coord)

    @eval begin
        @propagate_inbounds @add_cartesian function GridOperators.$left(f::AbstractField, I::Vararg{Integer,N}) where {N}
            GridOperators.left(f, flip(location(f, Val($dim))), Val($dim), I...)
        end

        @propagate_inbounds @add_cartesian function GridOperators.$right(f::AbstractField, I::Vararg{Integer,N}) where {N}
            GridOperators.right(f, flip(location(f, Val($dim))), Val($dim), I...)
        end

        @propagate_inbounds @add_cartesian function GridOperators.$δ(f::AbstractField, I::Vararg{Integer,N}) where {N}
            GridOperators.δ(f, flip(location(f, Val($dim))), Val($dim), I...)
        end

        @propagate_inbounds @add_cartesian function GridOperators.$∂(f::AbstractField, grid, I::Vararg{Integer,N}) where {N}
            GridOperators.∂(f, flip(location(f, Val($dim))), grid, Val($dim), I...)
        end
    end
end

const SG = StructuredGrid

# TODO: proper boundscheck
@add_cartesian function divg(V::NamedTuple{names,<:NTuple{N,AbstractField}}, grid::SG{N}, I::Vararg{Integer,N}) where {names,N}
    ntuple(Val(N)) do D
        Base.@_propagate_inbounds_meta
        @inbounds GridOperators.∂(V[D], grid, Val(D), I...)
    end |> sum
end
