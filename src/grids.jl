struct Grid{N,I}
    dims::NTuple{N,I}
end

Grid(dims::Vararg{Integer,N}) where {N} = Grid(promote(dims...))

Base.ndims(::Grid{N}) where {N} = N

dims(grid::Grid) = grid.dims

dims(grid::Grid{N}, loc::Vararg{Space,N}) where {N} = map((n, l) -> scale(l) * n + offset(l), dims(grid), loc)
