"""
    remove_dim(::Val{D}, A::NTuple{N}) where {D,N}

Remove the dimension specified by `D` from the tuple `A`.
"""
@inline function remove_dim(::Val{D}, A::NTuple{N}) where {D,N}
    ntuple(Val(N - 1)) do I
        Base.@_inline_meta
        I < D ? A[I] : A[I+1]
    end
end

@inline remove_dim(::Val{1}, I::NTuple{1}) = 1

"""
    remove_dim(::Val{D}, A::CartesianIndex) where {D}

Remove the dimension specified by `D` from the CartesianIndex `I`.
"""
@inline remove_dim(dim, I::CartesianIndex) = remove_dim(dim, Tuple(I)) |> CartesianIndex

"""
    insert_dim(::Val{D}, A::NTuple{N}, i) where {D,N}

Inserts a dimension into a tuple.

This function takes a tuple `A` of length `N` and inserts a new element `i` at position `D`.
The resulting tuple has length `N + 1`.
"""
@inline insert_dim(::Val{D}, A::NTuple{N}, i) where {D,N} =
    ntuple(Val(N + 1)) do I
        Base.@_inline_meta
        @inbounds (I < D) ? A[I] : (I == D) ? i : A[I-1]
    end

"""
    insert_dim(::Val{D}, A::CartesianIndex{N}, i) where {D,N}

Inserts a dimension into a CartesianIndex.

This function takes a CartesianIndex `A` of length `N` and inserts a new element `i` at position `D`.
The resulting CartesianIndex has length `N + 1`.
"""
@inline insert_dim(dim, A::CartesianIndex, i) = insert_dim(dim, Tuple(A), i) |> CartesianIndex
