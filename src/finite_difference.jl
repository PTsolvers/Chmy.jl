"""
    CentralDifference(n = 3)

The first derivative on `n` points centred on the evaluation point, `n` odd, see [`Interpolate`](@ref).
"""
function CentralDifference(n::Integer = 3)
    isodd(n) || throw(ArgumentError("a central difference requires an odd number of points"))
    return Interpolate(1, centred(n))
end

"""
    ForwardDifference(n = 2)

The first derivative on `n` points starting at the evaluation point, see [`Interpolate`](@ref).
"""
ForwardDifference(n::Integer = 2) = Interpolate(1, 0:(n - 1))

"""
    BackwardDifference(n = 2)

The first derivative on `n` points ending at the evaluation point, see [`Interpolate`](@ref).
"""
BackwardDifference(n::Integer = 2) = Interpolate(1, (1 - n):0)

"""
    StaggeredCentralDifference(n = 2)

The first derivative on `n` half-integer offsets centred on the evaluation point, `n` even, see [`Interpolate`](@ref).
"""
function StaggeredCentralDifference(n::Integer = 2)
    iseven(n) || throw(ArgumentError("a staggered central difference requires an even number of points"))
    return Interpolate(1, centred(n))
end

"""
    StaggeredForwardDifference(n = 2)

The first derivative on `n` half-integer offsets starting half a spacing after the evaluation point, see
[`Interpolate`](@ref).
"""
StaggeredForwardDifference(n::Integer = 2) = Interpolate(1, (1 // 2):(n - 1 // 2))

"""
    StaggeredBackwardDifference(n = 2)

The first derivative on `n` half-integer offsets ending half a spacing before the evaluation point, see
[`Interpolate`](@ref).
"""
StaggeredBackwardDifference(n::Integer = 2) = Interpolate(1, (1 // 2 - n):(-1 // 2))
