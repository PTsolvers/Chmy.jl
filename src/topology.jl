const Side = Union{Lower, Upper}
const AxisKind = Union{Colon, Side}

const Face{N} = NTuple{N, AxisKind}

dim(f::Face) = count(==(:), f)
codim(f::Face) = count(!=(:), f)

iswall(face::Face) = codim(face) == 1
iscorner(face::Face) = dim(face) == 0
isinterior(face::Face) = codim(face) == 0

# the direction of the outward normal along the axis
outward(::Lower) = -1
outward(::Upper) = 1

flip(::Colon) = (:)
flip(::Lower) = Upper()
flip(::Upper) = Lower()

# the outward distance of the position `p` from the wall on the side of a face, positive in the ghost region, and
# `nothing` along the spanning axes of the face and at global positions, which have no distance from the wall
distance(::Colon, p) = nothing
distance(side::Side, p::Rational{Int}) = outward(side) * p
distance(::Side, ::Global) = nothing

# the position `p` sampled at `s`: offsets add up, a global position absorbs offsets, and the innermost global
# position wins
shift(p::Rational{Int}, s::Rational{Int}) = p + s
shift(p::Position, s::Global) = s
shift(p::Global, s::Rational{Int}) = p

# the position in the frame of the reflected axis, and the position, or the side of a face, along the signed axis `a`,
# reflected if `a` is negative
mirror(p::Rational{Int}) = -p
mirror(p::Global) = flip(p)
mirror(p, a::Int) = a > 0 ? p : mirror(p)
