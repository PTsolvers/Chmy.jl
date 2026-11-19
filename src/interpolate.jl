# Fornberg recurrence for the weights of the derivatives up to the order `m` at zero
#
# the weights are derivatives at zero of the Lagrange basis polynomials over the nodes `x`; the recurrence builds them
# incrementally, adding one node per outer iteration, and `c[j, k + 1]` holds the weight of node `j` for the `k`-th
# derivative using nodes `1:i`; `c1` and `c2` are the products of distances from the previous and the new node to all
# preceding nodes, `c4` and `c5` are the new and the previous node positions
function fornberg(x, m)
    T = eltype(x)
    n = length(x)
    c = zeros(T, n, m + 1)
    c[1, 1] = 1
    c1 = one(T)
    c4 = x[1]
    for i in 2:n
        mn = min(i, m + 1)
        c2 = one(T)
        c5, c4 = c4, x[i]
        for j in 1:(i - 1)
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1
                for k in mn:-1:2
                    c[i, k] = c1 * ((k - 1) * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
                end
                c[i, 1] = -c1 * c5 * c[i - 1, 1] / c2
            end
            for k in mn:-1:2
                c[j, k] = (c4 * c[j, k] - (k - 1) * c[j, k - 1]) / c3
            end
            c[j, 1] = c4 * c[j, 1] / c3
        end
        c1 = c2
    end
    return c
end

# `n` nodes with unit spacing centred on the evaluation point, on half-integer offsets for even `n`
centred(n) = ((1 - n) // 2):((n - 1) // 2)

# the nodes as offsets from the evaluation point
function offsets(nodes)
    x = SVector(Tuple(checkoffset(n) for n in nodes))
    allunique(x) || throw(ArgumentError("the nodes must be unique"))
    return x
end

"""
    Interpolate(order, nodes)

The `order`-th derivative at the evaluation point of the polynomial interpolating the argument at `nodes`, the offsets
from the evaluation point in grid units, e.g. `Interpolate(2, (-1, 0, 1))`.
"""
struct Interpolate <: Operator
    order::Int
    nodes::SVector{<:Any, Rational{Int}}
    weights::SVector{<:Any, Rational{Int}}
    function Interpolate(order::Integer, nodes)
        order >= 0 || throw(ArgumentError("the order of the derivative must be nonnegative"))
        x = offsets(nodes)
        if length(x) <= order
            throw(ArgumentError(LazyString("a derivative of order ", order, " requires more than ", order, " nodes")))
        end
        weights = fornberg(x, order)[:, order + 1]
        return new(order, x, SVector{length(x)}(weights))
    end
end

"""
    Interpolate(nodes)

The value at the evaluation point of the polynomial interpolating the argument at `nodes`, see
[`Interpolate(order, nodes)`](@ref Interpolate).
"""
Interpolate(nodes) = Interpolate(0, nodes)

# whether every node `x` has the node `-x`; the interpolation then commutes with the reflection, so the weights satisfy
# `w(-x) = (-1)ᵏ w(x)` for the `k`-th derivative
ismirrored(nodes) = all(x -> -x in nodes, nodes)

# the weighted sum of the samples at the nonzero weights; stencils on mirrored nodes are grouped in pairs
# `w * (f[x] ± f[-x])` by the parity of the order, e.g. `(1//2) * (f[i + 1] - f[i - 1])`
function lower_rule(d::Interpolate, f::DTerm, i::Position)
    stencil = zip(d.nodes, d.weights)
    if ismirrored(d.nodes)
        s = iseven(d.order) ? 1 : -1
        pairs = ((iszero(x) ? w * f[i] : w * (f[i + x] + s * f[i - x])) for (x, w) in stencil if x >= 0 && !iszero(w))
        return sum(pairs; init = ZERO)
    end
    return sum((w * f[i + x] for (x, w) in stencil if !iszero(w)); init = ZERO)
end

"""
    Extrapolate(position, nodes)

Called as `Extrapolate(position, nodes)(u, g, a₀, a₁, …)`, the value at the evaluation point of the polynomial equal to
`u` at `nodes` and satisfying the constraint `Σₖ aₖ p⁽ᵏ⁾ = g` at `position`, with the derivatives along the axis. The
position and the nodes are offsets from the evaluation point in grid units, and `g` and the coefficients `aₖ` are
sampled at the position, e.g. the value `g` on a wall half a spacing away is `Extrapolate(1//2, (1, 2))(u, g, 1)`, the
derivative `(u, g, 0, 1)`, and the Robin condition `(u, g, a, b)`.
"""
struct Extrapolate <: Operator
    position::Rational{Int}
    nodes::SVector{<:Any, Rational{Int}}
    # the weights of the derivatives at the position, the rows for the evaluation point and then the nodes, the columns
    # for the derivative orders from zero
    weights::SMatrix{<:Any, <:Any, Rational{Int}}
    function Extrapolate(position, nodes)
        y = checkoffset(position)
        x = offsets(nodes)
        0 in x && throw(ArgumentError("the evaluation point cannot be a node of the extrapolation"))
        n = length(x) + 1
        c = fornberg(map(p -> p - y, [0 // 1; x]), n - 1)
        return new(y, x, SMatrix{n, n}(c))
    end
end

function checkranks(e::Extrapolate, args::Tuple{Vararg{DTerm}})
    3 <= length(args) <= length(e.nodes) + 3 ||
        throw(ArgumentError(LazyString("'", e, "' requires a field, the data, and 1 to ", length(e.nodes) + 1, " coefficients")))
    return checkscalar(e, args)
end

# the constraint applied to the Lagrange basis polynomials gives `r₀ p(0) + Σⱼ rⱼ u[i + xⱼ] = g`, solved for `p(0)`
function lower_rule(e::Extrapolate, (u, g, a...)::Tuple{DTerm, DTerm, Vararg{DTerm}}, i::Position)
    y = i + e.position
    coeffs = map(c -> c[y], a)
    row(j) = sum((coeffs[k] * e.weights[j, k] for k in eachindex(coeffs)); init = ZERO)
    r0 = row(1)
    isstaticzero(r0) && throw(ArgumentError(LazyString("the constraint of '", e, "' does not depend on the extrapolated value")))
    # nodes with zero weights in the constraint are not sampled
    rows = map(j -> row(j + 1), eachindex(e.nodes))
    known = sum((r / r0 * u[i + x] for (r, x) in zip(rows, e.nodes) if !isstaticzero(r)); init = ZERO)
    return g[y] / r0 - known
end

# the number of nodes of interpolations on the staggered grid, even so that they are centred on the evaluation point
function checkorder(order::Integer)
    if order < 2 || isodd(order)
        throw(ArgumentError(LazyString("the order of interpolation must be even and at least 2, got ", order)))
    end
    return Int(order)
end

"""
Abstract supertype of the policies choosing the nodes of [`InterpolateFrom`](@ref) next to the walls. A policy
implements `Chmy.wallnodes(policy, nodes, c)`, which returns the nodes used at the coordinate `c`, given the vector of
the `nodes` centred on `c`, as offsets from `c` to the positions of the argument.
"""
abstract type WallPolicy end

struct ReduceOrderPolicy <: WallPolicy end
struct KeepOrderPolicy <: WallPolicy end
struct ReadGhostsPolicy <: WallPolicy end

"Drops the nodes outside the domain, down to the nearest node inside, which lowers the order next to the walls."
const ReduceOrder = ReduceOrderPolicy()
"Shifts the nodes inside the domain, which keeps the order next to the walls."
const KeepOrder = KeepOrderPolicy()
"Keeps the nodes outside the domain, whose samples are read from the argument, e.g. from halos or an override."
const ReadGhosts = ReadGhostsPolicy()

wallnodes(::ReadGhostsPolicy, nodes, c) = nodes

# the nodes move inward until the outermost one is the outermost position of the argument inside the domain, as the
# nodes are on the lattice of these positions
function wallnodes(::KeepOrderPolicy, nodes, c)
    any(x -> isghost(c + x), nodes) || return nodes
    shift = ceil(Int, maximum(x -> distance(c + x), nodes))
    return map(x -> x - outward(c.frame) * shift, nodes)
end

# the nodes inside the domain, or the outermost position of the argument inside the domain if no node is inside
function wallnodes(::ReduceOrderPolicy, nodes, c)
    inside = filter(x -> !isghost(c + x), nodes)
    isempty(inside) || return inside
    return [argmax(x -> distance(c + x), wallnodes(KeepOrder, nodes, c))]
end

"""
    InterpolateFrom(location; order = 2, wall = ReduceOrder)

The value at the evaluation point of the polynomial interpolating the argument at the `order` positions nearest to
it at the `location`, `𝓅` or `𝓈`, and the argument itself at the `location`, e.g. `InterpolateFrom(𝓈)` interpolates
linearly from the segments. Next to the walls, the nodes are chosen by the [`WallPolicy`](@ref) `wall`. Traces are
read as they are, since they are constant along the axis.
"""
struct InterpolateFrom <: Operator
    location::Union{Point, Segment}
    order::Int
    wall::WallPolicy
    function InterpolateFrom(location::Union{Point, Segment}, order::Integer, wall::WallPolicy)
        return new(location, checkorder(order), wall)
    end
end
InterpolateFrom(location; order = 2, wall = ReduceOrder) = InterpolateFrom(location, order, wall)

# the interpolations take one scalar argument
function checkranks(d::Union{Interpolate, InterpolateFrom}, args::Tuple{Vararg{DTerm}})
    length(args) == 1 || throw(ArgumentError(LazyString("'", d, "' requires one argument")))
    return checkscalar(d, args)
end

# traces are at every location, since they are constant along the axis
athome(l, c::Coord) = location(c) == l
athome(l, ::Trace) = true

# away from the location, the nodes are the positions at the location centred on the coordinate, as chosen by the wall
# policy
function lower_rule(r::InterpolateFrom, f::DTerm, i::Position)
    athome(r.location, i) && return f[i]
    return lower_rule(Interpolate(wallnodes(r.wall, centred(r.order), i)), f, i)
end

# interpolation on grids

"""
    locations(grid, comps, dims)

The locations where the component `comps` of tensors is stored on the `grid` in `dims` dimensions, a tuple of
locations with one `𝓅` or `𝓈` per axis, e.g. `((𝓅, 𝓈),)` for the first component of vectors on [`Grids.MAC`](@ref)
in 2D. A grid is any object implementing this method: `𝓅` and `𝓈` store every component at this location along every
axis, and a tuple of them every component at these locations, in as many dimensions as there are locations.
"""
function locations end

locations(l::Union{Point, Segment}, comps, dims) = (ntuple(_ -> l, dims),)

function locations(l::Tuple{Vararg{Union{Point, Segment}}}, comps, dims)
    length(l) == dims || throw(DimensionMismatch(LazyString("the grid ", l, " has ", length(l), " dimensions, got ", dims)))
    return (l,)
end

# the interpolation of the component `comps` of the argument from its locations on the grid, see `interpolate`; the
# calls on tensors have no components, and their components interpolate the components of the argument
struct GridInterpolate <: Operator
    grid::Any
    comps::Tuple{Vararg{Int}}
    order::Int
    wall::WallPolicy
end

tensorrank(::GridInterpolate, args) = tensorrank(only(args))

checkranks(::GridInterpolate, args::Tuple{Vararg{DTerm}}) = length(args) == 1 || throw(ArgumentError("an interpolation requires one argument"))

# constants are not interpolated
evaluate(::GridInterpolate, (t,)::Tuple{DTerm}) = isconstant(t) ? t : nothing
component(op::GridInterpolate, (t,)::Tuple{DTerm}, I) = GridInterpolate(op.grid, I, op.order, op.wall)(t[I...])

function lower_rule(op::GridInterpolate, f::DTerm, x::NTuple{N, Position}) where {N}
    return reconstruct(locations(op.grid, op.comps, N), op, f, x)::DTerm
end

# a single location: the tensor product of the interpolations from the location along every axis
function reconstruct((home,)::Tuple{Any}, op, f, x)
    g = foldl((g, a) -> Lifted(InterpolateFrom(home[a], op.order, op.wall), a)(g), eachindex(x); init = f)
    return g[x...]
end

# several locations: the argument is read at one of them, and cannot be interpolated between them yet
function reconstruct(homes::Tuple, op, f, x)
    if !any(home -> all(map(athome, home, x)), homes)
        throw(ArgumentError(LazyString("cannot interpolate `", f, "` between the locations ", homes)))
    end
    return f[x...]
end

"""
    interpolate(u, grid; order = 2, wall = ReduceOrder)

The term `u` read only at the locations where it is stored on the `grid`, see [`locations`](@ref), and interpolated
elsewhere, every component of a tensor from its own locations. A component stored at one location is interpolated
along every axis with [`InterpolateFrom`](@ref) on `order` positions, 2 for linear and 4 for cubic interpolation, with
the nodes next to the walls chosen by the [`WallPolicy`](@ref) `wall`. A component stored at several locations is
only read at them. At its locations, `u` is read as it is.

```julia
interpolate(η, 𝓈)                        # a scalar at the cell centres
interpolate(v, Grids.MAC; order = 4)     # every component from its faces, cubic
interpolate(k, (𝓅, 𝓈); wall = ReadGhosts)
```
"""
function interpolate(u, grid; order = 2, wall = ReduceOrder)
    return GridInterpolate(grid, (), checkorder(order), wall)(asterm(u))
end
