"""
Abstract supertype for finite difference operators.
"""
abstract type AbstractDifference <: Operator end

Base.getindex(d::AbstractDifference, i::Integer) = Lifted(d, i)

"""
    Difference{N,O}()
    Difference{N}()
    Difference()

First derivative on a unit-spaced grid using `N` consecutive points with midpoint `O`.
`N` includes zero-weight samples. The midpoint must be integer or rational.
Default parameters are `N = 3` and `O = 0`.
"""
struct Difference{N, O} <: AbstractDifference
    function Difference{N, O}() where {N, O}
        check_difference(N, O, false)
        return new{N, O}()
    end
end

Difference() = Difference{3, 0}()
Difference{N}() where {N} = Difference{N, 0}()

"""
    StaggeredDifference{N,O}()
    StaggeredDifference{N}()
    StaggeredDifference()

First derivative on a unit-spaced grid using `N` consecutive points on the
opposite grid location, with midpoint `O`. Default parameters are `N = 2` and `O = 0`.
"""
struct StaggeredDifference{N, O} <: AbstractDifference
    function StaggeredDifference{N, O}() where {N, O}
        check_difference(N, O, true)
        return new{N, O}()
    end
end

StaggeredDifference() = StaggeredDifference{2, 0}()
StaggeredDifference{N}() where {N} = StaggeredDifference{N, 0}()

"""
    CentralDifference{N}()
    CentralDifference()

Centered first difference using an odd number `N` of points, defaulting to `N = 3`.
"""
const CentralDifference{N} = Difference{N, 0}
CentralDifference() = CentralDifference{3}()

"""
    StaggeredCentralDifference{N}()
    StaggeredCentralDifference()

Centered staggered first difference using `N` points, defaulting to `N = 2`.
"""
const StaggeredCentralDifference{N} = StaggeredDifference{N, 0}
StaggeredCentralDifference() = StaggeredCentralDifference{2}()

"""
    ForwardDifference{N}()
    ForwardDifference()

Collocated forward difference using `N` points, defaulting to `N = 2`.
"""
struct ForwardDifference{N}
    ForwardDifference{N}() where {N} = Difference{N, (N - 1) // 2}()
end
ForwardDifference() = ForwardDifference{2}()

"""
    BackwardDifference{N}()
    BackwardDifference()

Collocated backward difference using `N` points, defaulting to `N = 2`.
"""
struct BackwardDifference{N}
    BackwardDifference{N}() where {N} = Difference{N, -(N - 1) // 2}()
end
BackwardDifference() = BackwardDifference{2}()

"""
    StaggeredForwardDifference{N}()
    StaggeredForwardDifference()

Staggered forward difference using `N` points, defaulting to `N = 2`.
"""
struct StaggeredForwardDifference{N}
    StaggeredForwardDifference{N}() where {N} = StaggeredDifference{N, N // 2}()
end
StaggeredForwardDifference() = StaggeredForwardDifference{2}()

"""
    StaggeredBackwardDifference{N}()
    StaggeredBackwardDifference()

Staggered backward difference using `N` points, defaulting to `N = 2`.
"""
struct StaggeredBackwardDifference{N}
    StaggeredBackwardDifference{N}() where {N} = StaggeredDifference{N, -N // 2}()
end
StaggeredBackwardDifference() = StaggeredBackwardDifference{2}()

function check_difference(n, offset, staggered)
    n isa Integer && n >= 2 || throw(ArgumentError("a difference requires an integer number of points of at least two"))
    offset isa Union{Integer, Rational} || throw(ArgumentError("the stencil midpoint must be integer or rational"))
    first_offset = offset - (n - 1) // 2
    if staggered
        denominator(first_offset) == 2 || throw(ArgumentError("staggered differences require half-integer offsets: the midpoint must be integer for even N and half-integer for odd N"))
    else
        isinteger(first_offset) || throw(ArgumentError("standard differences require integer offsets: the midpoint must be integer for odd N and half-integer for even N"))
    end
    return
end

function difference_nodes(n, offset)
    first = Rational(offset) - (n - 1) // 2
    return ntuple(k -> first + k - 1, n)
end

# Fornberg recurrence for the zeroth and first derivative weights at zero
function difference_weights(x)
    n = length(x)
    c = zeros(eltype(x), n, 2)
    c[1, 1] = 1
    c1 = one(eltype(x))
    c4 = x[1]
    for i in 2:n
        c2 = one(eltype(x))
        c5, c4 = c4, x[i]
        for j in 1:(i - 1)
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1
                c[i, 2] = c1 * (c[i - 1, 1] - c5 * c[i - 1, 2]) / c2
                c[i, 1] = -c1 * c5 * c[i - 1, 1] / c2
            end
            c[j, 2] = (c4 * c[j, 2] - c[j, 1]) / c3
            c[j, 1] = c4 * c[j, 1] / c3
        end
        c1 = c2
    end
    return Tuple(c[:, 2])
end

stencil_constant(x) = isone(denominator(x)) ? Int(numerator(x)) : Rational{Int}(x)

"""
    stencil_nodes(op)

Return the stencil offsets in grid-spacing units in ascending order.
Staggered offsets are half-integers.
"""
@generated function stencil_nodes(::Union{Difference{N, O}, StaggeredDifference{N, O}}) where {N, O}
    return QuoteNode(map(stencil_constant, difference_nodes(N, O)))
end

"""
    stencil_weights(op)

Return the full tuple of exact first-derivative coefficients aligned with
`stencil_nodes(op)`, including zeros.
"""
@generated function stencil_weights(::Union{Difference{N, O}, StaggeredDifference{N, O}}) where {N, O}
    return QuoteNode(map(stencil_constant, difference_weights(difference_nodes(N, O))))
end

function field_sample(x, staggered, loc)
    offset = Int(staggered ? x + (loc === Point ? -1 // 2 : 1 // 2) : x)
    index = iszero(offset) ? :i : offset > 0 ? :(i + $offset) : :(i - $(-offset))
    field = staggered ? (loc === Point ? :(f[𝓈]) : :(f[𝓅])) : isnothing(loc) ? :f : :(f[l])
    return :($field[$index])
end

function difference_expr(x, weights, staggered, loc)
    nonzero = findall(!iszero, weights)
    paired = all(nonzero) do k
        opposite = findfirst(==(-x[k]), x)
        return !isnothing(opposite) && weights[opposite] == -weights[k]
    end
    result = nothing
    # symmetric pairs are ordered from nearest to farthest
    points = paired ? sort!(filter(k -> x[k] > 0, nonzero); by = k -> x[k]) : sortperm(collect(weights); by = c -> c < 0)
    for k in points
        c = weights[k]
        iszero(c) && continue
        term = field_sample(x[k], staggered, loc)
        if paired
            opposite = field_sample(-x[k], staggered, loc)
            term = :($term - $opposite)
        end
        coefficient = abs(c)
        isone(coefficient) || (term = :($coefficient * $term))
        result = if isnothing(result)
            c > 0 ? term : :(-$term)
        else
            c > 0 ? :($result + $term) : :($result - $term)
        end
    end
    return something(result, :ZERO)
end

@generated function stencil_rule(d::Difference, f::DTerm, i::DTerm)
    return difference_expr(stencil_nodes(d()), stencil_weights(d()), false, nothing)
end

@generated function stencil_rule(d::Difference, f::DTerm, l::L, i::DTerm) where {L <: Location}
    return difference_expr(stencil_nodes(d()), stencil_weights(d()), false, L)
end

@generated function stencil_rule(d::StaggeredDifference, f::DTerm, ::L, i::DTerm) where {L <: Location}
    return difference_expr(stencil_nodes(d()), stencil_weights(d()), true, L)
end
