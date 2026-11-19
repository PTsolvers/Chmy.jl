const Side = Union{Lower, Upper}
const Face{N} = NTuple{N, Global}

dim(f::Face) = count(==(:), f)
codim(f::Face) = count(!=(:), f)

iswall(face::Face) = codim(face) == 1
iscorner(face::Face) = dim(face) == 0
isinterior(face::Face) = codim(face) == 0

# the direction of the outward normal along the axis
outward(::Lower) = -1
outward(::Upper) = 1

# the side of the face next to a position: the frame of a coordinate, and `:` for traces, which are never next to a wall
frame(c::Coord) = c.frame
frame(::Trace) = (:)

# the outward distance of a position from the wall of its frame, positive in the ghost region, and `nothing` for the
# coordinates relative to the grid point and for traces, which have no distance from a wall
distance(c::Coord) = distance(c.frame, c.offset)
distance(::Trace) = nothing
distance(::Colon, p) = nothing
distance(side::Side, p::Rational{Int}) = outward(side) * p

# positions shifted along the axis: coordinates move, and traces are constant along the axis
Base.:+(c::Coord, s::Union{Integer, Rational}) = Coord(c.frame, c.offset + s)
Base.:+(t::Trace, ::Union{Integer, Rational}) = t
Base.:-(p::Position, s::Union{Integer, Rational}) = p + (-s)
