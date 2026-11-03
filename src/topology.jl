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

const Coord{N} = NTuple{N, Rational{Int}}

# the lowering context: the face next to which the expression is lowered, and the origin of the evaluation point,
# i.e. the outward distance from the wall on the sides of the face, and the lattice coordinate on its spanning axes
struct LoweringContext{N}
    face::Face{N}
    origin::Coord{N}
end

Base.ndims(::LoweringContext{N}) where {N} = N

# the coordinate of the sample at `coord` from the evaluation point along the axis `a`, relative to the grid point of
# the kernel, which is the wall point on the sides of the face
position(ctx::LoweringContext, coord, a) = position(ctx.face[a], ctx.origin[a], coord[a])
position(::Colon, origin, c) = origin + c
position(side::Side, origin, c) = outward(side) * origin + c

# the outward distance of the sample at `coord` from the wall on the side `a` of the face, positive in the ghost region
distance(ctx::LoweringContext, coord, a) = distance(ctx.face[a], ctx.origin[a], coord[a])
distance(side::Side, origin, c) = origin + outward(side) * c
