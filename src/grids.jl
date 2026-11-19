"""
Grids storing the components of tensors at the locations given by [`locations`](@ref), see
[`interpolate`](@ref).
"""
module Grids

using ..Chmy: 𝓅, 𝓈
import ..Chmy: locations

export MAC, FullyStaggered

struct MACGrid end

"""
The marker-and-cell grid, the Arakawa C grid in 2D: a component is at points along the axes appearing an odd number
of times in its indices, and at segments along the other axes, e.g. scalars at the cell centres, the component `i` of
vectors on the faces normal to the axis `i`, and the components `(i, j)` of rank-2 tensors at the cell centres for
`i = j`, otherwise at the vertices in 2D and on the edges in 3D. Staggered differences of the components land at the
locations of the components of the result.
"""
const MAC = MACGrid()

locations(::MACGrid, comps, dims) = (ntuple(a -> isodd(count(==(a), comps)) ? 𝓅 : 𝓈, dims),)

struct FullyStaggeredGrid end

"""
The fully staggered grid, the Arakawa E grid in 2D: every component of a tensor is at all locations whose number of
points has the parity of the rank, e.g. scalars and rank-2 tensors at the cell centres and the vertices in 2D, and at
the cell centres and on the edges in 3D, and vectors on the faces and at the vertices in 3D.
"""
const FullyStaggered = FullyStaggeredGrid()

function locations(::FullyStaggeredGrid, comps, dims)
    lattice = Iterators.product(ntuple(_ -> (𝓅, 𝓈), dims)...)
    return Tuple(l for l in lattice if iseven(count(==(𝓅), l) + length(comps)))
end

end # module Grids
