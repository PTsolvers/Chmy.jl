struct Grid{N,I}
    dims::NTuple{N,I}
end

Grid(dims::Vararg{Integer,N}) where {N} = Grid(promote(dims...))

Base.ndims(::Grid{N}) where {N} = N

dims(grid::Grid) = grid.dims

function dims(grid::Grid{N}, loc::Vararg{Space,N}) where {N}
    ntuple(Val(N)) do D
        Base.@inline
        dim(grid, loc[D], Val(D))
    end
end

dim(grid::Grid, ::Point, ::Val{D}) where {D} = grid.dims[D]
dim(grid::Grid, ::Segment, ::Val{D}) where {D} = grid.dims[D] - 1

indices(::Grid{N}) where {N} = ntuple(i -> SIndex(i), Val(N))
