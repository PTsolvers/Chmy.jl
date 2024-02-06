"Scalar field with a constant value"
abstract type ConstantField{T} <: AbstractField{T,0,Nothing} end

@inline Base.size(::ConstantField) = ()

"Constant field with values equal to zero(T)"
struct ZeroField{T} <: ConstantField{T} end

@inline Base.getindex(::ZeroField{T}, inds...) where {T} = zero(T)

Base.broadcastable(::ZeroField{T}) where {T} = zero(T)

Base.show(io::IO, ::ZeroField{T}) where {T} = print(io, "ZeroField{$T}")
Base.show(io::IO, ::MIME"text/plain", ::ZeroField{T}) where {T} = println(io, "ZeroField{$T}")

"Constant field with values equal to one(T)"
struct OneField{T} <: ConstantField{T} end

@inline Base.getindex(::OneField{T}, inds...) where {T} = one(T)

Base.show(io::IO, ::OneField{T}) where {T} = print(io, "OneField{$T}")
Base.show(io::IO, ::MIME"text/plain", ::OneField{T}) where {T} = println(io, "OneField{$T}")

Base.broadcastable(::OneField{T}) where {T} = one(T)

"Field with a constant value"
struct ValueField{T} <: ConstantField{T}
    value::T
end

@inline Base.getindex(f::ValueField, inds...) = f.value

Base.show(io::IO, cf::ValueField{T}) where {T} = print(io, "ValueField{$T}($(cf.value))")
Base.show(io::IO, ::MIME"text/plain", cf::ValueField{T}) where {T} = println(io, "ValueField{$T}($(cf.value))")

Base.broadcastable(f::ValueField) = f.value
