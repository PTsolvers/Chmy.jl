# Return a copy of the tuple `A` with element in position `D` removed
@inline function remove_dim(::Val{D}, A::NTuple{N}) where {D,N}
    ntuple(Val(N - 1)) do I
        Base.@_inline_meta
        I < D ? A[I] : A[I+1]
    end
end

@inline remove_dim(::Val{1}, I::NTuple{1}) = 1

# Same as `remove_dim`, but for `CartesianIndex`
@inline remove_dim(dim, I::CartesianIndex) = remove_dim(dim, Tuple(I)) |> CartesianIndex

# Returns a copy of tuple `A`, but inserts `i` into position `D`
@inline insert_dim(::Val{D}, A::NTuple{N}, i) where {D,N} =
    ntuple(Val(N + 1)) do I
        Base.@_inline_meta
        @inbounds (I < D) ? A[I] : (I == D) ? i : A[I-1]
    end

# Same as `insert_dim`, but for `CartesianIndex`
@inline insert_dim(dim, A::CartesianIndex, i) = insert_dim(dim, Tuple(A), i) |> CartesianIndex
