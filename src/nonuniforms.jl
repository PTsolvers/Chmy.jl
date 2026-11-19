# the samples of the nonuniform scalar fields read by an expression lowered at the output position
struct Nonuniforms{N, D}
    samples::D
    position::NTuple{N, Position}
end

"""
    nonuniforms(term, positions...)

Lower the scalar `term` at the output `positions`, one per axis as in `term[positions...]`, and gather the positions
of the samples of every nonuniform scalar field, e.g. `f` or `V[1]`, in the order of their first appearance. Returns a
`Nonuniforms` with the fields `samples`, a dictionary from the fields to the vectors of their sample positions, and
`position`, the output position. Coordinates and uniform terms are not gathered. Displayed in `text/plain`, the
stencil of all fields and of each field is drawn on the staggered grid in 1D, 2D and 3D.
"""
function nonuniforms(term::DTerm, positions::Vararg{PositionLike, N}) where {N}
    x = map(asposition, positions)
    samples = LittleDict{DTerm, Vector{NTuple{N, Position}}}()
    foreach_read(makeinds(term, x)) do t
        isinds(t) && push!(get!(valtype(samples), samples, argument(t)), indices(t))
    end
    foreach(unique!, values(samples))
    return Nonuniforms(samples, x)
end
