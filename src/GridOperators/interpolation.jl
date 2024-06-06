# 1D interpolation rules

"""
    abstract type InterpolationRule

A type representing an interpolation rule that specifies how
the interpolant `f` should be reconstructed using a data set on a given grid.
"""
abstract type InterpolationRule end

struct Linear <: InterpolationRule end
struct HarmonicLinear <: InterpolationRule end

itp_rule(::Linear, t, a, b)         = muladd(t, b - a, a)
itp_rule(::HarmonicLinear, t, a, b) = inv(muladd(t, inv(b) - inv(a), inv(a)))

# recursive rule application

itp_impl(r, t, v0, v1) = itp_impl(Dim(length(t)), r, t, v0, v1)

itp_impl(::Dim{D}, r, t, v0, v1) where {D} = itp_rule(r, last(t),
                                                      itp_impl(Dim(D - 1), r, front(t), v0...),
                                                      itp_impl(Dim(D - 1), r, front(t), v1...))

itp_impl(::Dim{1}, r, t, v0, v1) = itp_rule(r, last(t), v0, v1)

# interpolation weights

itp_weight(ax::AbstractAxis, ::Vertex, ::Center, i) = convert(eltype(ax), 0.5)
itp_weight(ax::AbstractAxis, ::Center, ::Vertex, i) = convert(eltype(ax), 0.5) * Δ(ax, Center(), i - oneunit(i)) * iΔ(ax, Vertex(), i)

# special rule for efficient interpolation on uniform axes
itp_weight(ax::UniformAxis, ::Center, ::Vertex, i) = convert(eltype(ax), 0.5)

# dimensions to interpolate

@generated function itp_dims(from::NTuple{N,Location}, to::NTuple{N,Location}) where {N}
    dims = Tuple(Dim(D) for D in 1:N)
    locs = zip(dims, from.instance, to.instance)

    pred = (_, l1, l2) -> l1 !== l2
    filtered_locs = Iterators.filter(splat(pred), locs)

    dims_r, from_r, to_r = (zip(filtered_locs...)...,)

    return :($dims_r, $from_r, $to_r)
end

# interpolation knots

@propagate_inbounds itp_knots(f, from, to, dims, I) = itp_knots_impl(f, from, to, dims, I)

@propagate_inbounds itp_knots_impl(f, from, to, dims, I) = (itp_knots_impl(f, front(from), front(to), front(dims), il(last(from), last(to), last(dims), I...)),
                                                            itp_knots_impl(f, front(from), front(to), front(dims), ir(last(from), last(to), last(dims), I...)))

@propagate_inbounds itp_knots_impl(f, from::Tuple{}, to::Tuple{}, dims::Tuple{}, I) = f[I...]

# interpolation interface

"""
    itp(f, to, r, grid, I...)

Interpolates the field `f` from its current location to the specified location(s) `to` using the given interpolation rule `r`.
The indices specify the position within the grid at location(s) `to`.
"""
@add_cartesian function itp(f::AbstractField, to::NTuple{N,Location}, r::InterpolationRule, grid::StructuredGrid{N}, I::Vararg{Integer,N}) where {N}
    from = location(f)
    if typeof(from) === typeof(to)
        return f[I...]
    end

    dims_r, from_r, to_r = itp_dims(from, to)

    t = ntuple(Val(length(dims_r))) do D
        ax = axis(grid, dims_r[D])
        itp_weight(ax, from_r[D], to_r[D], I[D])
    end

    v = itp_knots(f, from_r, to_r, dims_r, I)

    return itp_impl(r, t, v...)
end

# version for repeated locations
@add_cartesian itp(f, to::Location, r, grid, I::Vararg{Integer,N}) where {N} = itp(f, expand_loc(Val(N), to), r, grid, I...)

# shortcuts for common use cases

"""
    lerp(f, to, grid, I...)

Linearly interpolate values of a field `f` to location `to`.
"""
@add_cartesian lerp(f, to, grid, I::Vararg{Integer,N}) where {N} = itp(f, to, Linear(), grid, I...)

"""
    hlerp(f, to, grid, I...)

Interpolate a field `f` to location to using harmonic linear interpolation rule.

`rule(t, v0, v1) = 1/(1/v0 + t * (1/v1 - 1/v0))`
"""
@add_cartesian hlerp(f, to, grid, I::Vararg{Integer,N}) where {N} = itp(f, to, HarmonicLinear(), grid, I...)
