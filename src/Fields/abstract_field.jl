abstract type AbstractField{T,N,L} <: AbstractArray{T,N} end

Base.@assume_effects :foldable halo(::AbstractArray) = 0

Base.IndexStyle(::Type{<:AbstractField}) = IndexCartesian()

Base.@assume_effects :foldable location(::AbstractField{T,N,L}) where {T,N,L} = L.instance
Base.@assume_effects :foldable location(::AbstractField{T,N,L}, ::Dim{D}) where {T,N,L,D} = L.instance[D]

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
