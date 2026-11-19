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
# no wall in the frame of the grid point
outward(::Colon) = 0

# the side of the face next to a position: the frame of a coordinate, and `:` for traces, which are never next to a wall
frame(c::Coord) = c.frame
frame(::Trace) = (:)

# the outward distance of a position from the wall of its frame, positive in the ghost region, and `nothing` for the
# coordinates relative to the grid point and for traces, which have no distance from a wall
distance(c::Coord) = distance(c.frame, c.offset)
distance(::Trace) = nothing
distance(::Colon, p) = nothing
distance(side::Side, p::Rational{Int}) = outward(side) * p

# whether a position lies outside the domain, beyond the wall of its frame; positions on the wall are inside, and
# coordinates relative to the grid point and traces never lie outside
isghost(p::Position) = something(distance(p), 0 // 1) > 0

# the location of a position: coordinates at integer offsets are points and at half-integer ones segments, and traces
# are located at their element
location(c::Coord) = isinteger(c.offset) ? Point() : Segment()
location(t::Trace) = t.element

# positions shifted along the axis: coordinates move, and traces are constant along the axis
Base.:+(c::Coord, s::Offset) = Coord(c.frame, c.offset + s)
Base.:+(t::Trace, ::Offset) = t
Base.:-(p::Position, s::Offset) = p + (-s)
