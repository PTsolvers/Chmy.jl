"""
    Dim(D)

Return Dim{D}() which contains no run-time data like `Val`.
Used to statically dispatch on the spatial dimension of the computational grid.
"""
struct Dim{D} end
Dim(D::Integer) = Dim{D}()

"""
    Side(S)

Return Side{S}() which contains no run-time data like `Val`.
Used to statically dispatch on the left (Side(1)) or right (Side(2)) sides of the computational domain.
"""
struct Side{S} end
Side(S::Integer) = Side{S}()

const Left  = Side{1}()
const Right = Side{2}()

"""
    remove_dim(dim::Dim, A::NTuple)

Remove the dimension specified by `dim` from the tuple `A`.
"""
@inline function remove_dim(::Dim{D}, A::NTuple{N}) where {D,N}
    ntuple(Val(N - 1)) do I
        Base.@_inline_meta
        I < D ? A[I] : A[I+1]
    end
end

@inline remove_dim(::Dim{1}, I::NTuple{1}) = 1

"""
    remove_dim(dim::Dim, I::CartesianIndex)

Remove the dimension specified by `dim` from the CartesianIndex `I`.
"""
@inline remove_dim(dim, I::CartesianIndex) = remove_dim(dim, Tuple(I)) |> CartesianIndex

"""
    insert_dim(dim::Dim, A::NTuple, i)

Takes a tuple `A` and inserts a new element `i` at position specified by `dim`.
"""
@inline insert_dim(::Dim{D}, A::NTuple{N}, i) where {D,N} =
    ntuple(Val(N + 1)) do I
        Base.@_inline_meta
        @inbounds (I < D) ? A[I] : (I == D) ? i : A[I-1]
    end

@inline insert_dim(::Dim{1}, A::NTuple{1}, i) = A

"""
    insert_dim(dim::Dim, I::CartesianIndex, i)

Takes a CartesianIndex `I` and inserts a new element `i` at position specified by `dim`.
"""
@inline insert_dim(dim, I::CartesianIndex, i) = insert_dim(dim, Tuple(I), i) |> CartesianIndex

struct Offset{O} end

Offset(o::Vararg{Integer}) = Offset{o}()
Offset(o::Tuple{Vararg{Integer}}) = Offset{o}()
Offset(o::CartesianIndex) = Offset{o.I}()
Offset() = Offset{0}()

Base.:+(::Offset{O1}, ::Offset{O2}) where {O1,O2} = Offset((O1 .+ O2)...)
Base.:+(::Offset{O}, tp::Tuple{Vararg{Integer}}) where {O} = O .+ tp
Base.:+(::Offset{O}, tp::CartesianIndex) where {O} = CartesianIndex(O .+ Tuple(tp))

Base.:+(tp, off::Offset) = off + tp
