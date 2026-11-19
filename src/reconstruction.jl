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

# the nodes as offsets from the evaluation point
function offsets(nodes)
    x = Tuple(checkoffset(n) for n in nodes)
    allunique(x) || throw(ArgumentError("the nodes must be unique"))
    return x
end

"""
    FiniteDifference(order, nodes)

The `order`-th derivative at the evaluation point of the polynomial interpolating the argument at `nodes`, the offsets
from the evaluation point in grid units, e.g. `FiniteDifference(2, (-1, 0, 1))`.
"""
struct FiniteDifference <: Operator
    order::Int
    nodes::Tuple{Vararg{Rational{Int}}}
    weights::Tuple{Vararg{Rational{Int}}}
    function FiniteDifference(order::Integer, nodes)
        order >= 0 || throw(ArgumentError("the order of the derivative must be nonnegative"))
        x = offsets(nodes)
        if length(x) <= order
            throw(ArgumentError(LazyString("a derivative of order ", order, " requires more than ", order, " nodes")))
        end
        weights = fornberg(x, order)[:, order + 1]
        return new(order, x, Tuple(weights))
    end
end

"""
    Interpolate(nodes)

The value at the evaluation point of the polynomial interpolating the argument at `nodes`, see [`FiniteDifference`](@ref).
"""
Interpolate(nodes) = FiniteDifference(0, nodes)

"""
    CentralDifference(n = 3)

The first derivative on `n` points centred on the evaluation point, `n` odd.
"""
function CentralDifference(n::Integer = 3)
    isodd(n) || throw(ArgumentError("a central difference requires an odd number of points"))
    return FiniteDifference(1, ((1 - n) // 2):((n - 1) // 2))
end

"""
    ForwardDifference(n = 2)

The first derivative on `n` points starting at the evaluation point.
"""
ForwardDifference(n::Integer = 2) = FiniteDifference(1, 0:(n - 1))

"""
    BackwardDifference(n = 2)

The first derivative on `n` points ending at the evaluation point.
"""
BackwardDifference(n::Integer = 2) = FiniteDifference(1, (1 - n):0)

"""
    StaggeredCentralDifference(n = 2)

The first derivative on `n` half-integer offsets centred on the evaluation point, `n` even.
"""
function StaggeredCentralDifference(n::Integer = 2)
    iseven(n) || throw(ArgumentError("a staggered central difference requires an even number of points"))
    return FiniteDifference(1, ((1 - n) // 2):((n - 1) // 2))
end

"""
    StaggeredForwardDifference(n = 2)

The first derivative on `n` half-integer offsets starting half a spacing after the evaluation point.
"""
StaggeredForwardDifference(n::Integer = 2) = FiniteDifference(1, (1 // 2):(n - 1 // 2))

"""
    StaggeredBackwardDifference(n = 2)

The first derivative on `n` half-integer offsets ending half a spacing before the evaluation point.
"""
StaggeredBackwardDifference(n::Integer = 2) = FiniteDifference(1, (1 // 2 - n):(-1 // 2))

function checkranks(d::FiniteDifference, args::Tuple{Vararg{DTerm}})
    length(args) == 1 || throw(ArgumentError(LazyString("'", d, "' requires one argument")))
    return checkscalar(d, args)
end

# whether every nonzero weight `w` at the node `x` has the weight `s * w` at the node `-x`
function ispaired(d::FiniteDifference, s)
    return all(zip(d.nodes, d.weights)) do (x, w)
        k = findfirst(==(-x), d.nodes)
        return iszero(w) || (!isnothing(k) && d.weights[k] == s * w)
    end
end

# the weighted sum of the samples at the nonzero weights; stencils that are symmetric for even orders and antisymmetric
# for odd orders are grouped in pairs `w * (f[x] ± f[-x])`, e.g. `(1//2) * (f[i + 1] - f[i - 1])`
function lower_rule(d::FiniteDifference, f::DTerm, i::Position)
    s = iseven(d.order) ? 1 : -1
    stencil = zip(d.nodes, d.weights)
    if ispaired(d, s)
        pairs = ((iszero(x) ? w * f[i] : w * (f[i + x] + s * f[i - x])) for (x, w) in stencil if x >= 0 && !iszero(w))
        return sum(pairs; init = ZERO)
    end
    return sum((w * f[i + x] for (x, w) in stencil if !iszero(w)); init = ZERO)
end

"""
    Extrapolate(nodes, position)

Called as `Extrapolate(nodes, position)(u, g, a₀, a₁, …)`, the value at the evaluation point of the polynomial equal to
`u` at `nodes` and satisfying the constraint `Σₖ aₖ p⁽ᵏ⁾ = g` at `position`, with the derivatives along the axis. The
nodes and the position are offsets from the evaluation point in grid units, and `g` and the coefficients `aₖ` are
sampled at the position, e.g. the value `g` on a wall half a spacing away is `Extrapolate((1, 2), 1//2)(u, g, 1)`, the
derivative `(u, g, 0, 1)`, and the Robin condition `(u, g, a, b)`.
"""
struct Extrapolate <: Operator
    nodes::Tuple{Vararg{Rational{Int}}}
    position::Rational{Int}
    # the weights of the derivatives at the position, of the evaluation point first and then of the nodes
    weights::Tuple{Vararg{Tuple{Vararg{Rational{Int}}}}}
    function Extrapolate(nodes, position)
        x = offsets(nodes)
        y = checkoffset(position)
        0 in x && throw(ArgumentError("the evaluation point cannot be a node of the extrapolation"))
        c = fornberg(map(p -> p - y, (0 // 1, x...)), length(x))
        return new(x, y, Tuple(map(Tuple, eachrow(c))))
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
    row(w) = sum((coeffs[k] * w[k] for k in eachindex(coeffs)); init = ZERO)
    r0 = row(first(e.weights))
    isstaticzero(r0) && throw(ArgumentError(LazyString("the constraint of '", e, "' does not depend on the extrapolated value")))
    known = sum((row(w) / r0 * u[i + x] for (x, w) in zip(e.nodes, Base.tail(e.weights))); init = ZERO)
    return g[y] / r0 - known
end
