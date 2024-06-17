module Grids

export Location, Center, Vertex, flip
export Connectivity, Bounded, Connected, Periodic, Flat
export AbstractAxis, UniformAxis, FunctionAxis
export StructuredGrid, UniformGrid

export nvertices, ncenters, spacing, inv_spacing, Δ, iΔ, volume, inv_volume, coord, coords, center, vertex, centers, vertices
export origin, extent, bounds, axis
export direction, axes_names
export expand_loc
export connectivity

using Chmy
using Chmy.Architectures

import Chmy: @add_cartesian

import Base.@propagate_inbounds

"""
    abstract type Location

Abstract type representing a location in a grid cell.
"""
abstract type Location end

"""
    struct Center <: Location

The `Center` struct represents a location at the center along a dimension of a grid cell.
"""
struct Center <: Location end

"""
    struct Vertex <: Location

The `Vertex` struct represents a location at the vertex along a dimension of a grid cell.
"""
struct Vertex <: Location end

Base.@assume_effects :total flip(::Center) = Vertex()
Base.@assume_effects :total flip(::Vertex) = Center()

Base.broadcastable(o::Location) = Ref(o)

"""
    abstract type Connectivity

Abstract type representing the connectivity of grid elements.
"""
abstract type Connectivity end

struct Bounded <: Connectivity end
struct Periodic <: Connectivity end
struct Connected <: Connectivity end
struct Flat <: Connectivity end

expand_loc(::Val{N}, locs::NTuple{N,Location}) where {N} = locs
expand_loc(::Val{N}, loc::Location) where {N} = ntuple(_ -> loc, Val(N))

include("abstract_axis.jl")
include("uniform_axis.jl")
include("function_axis.jl")
include("structured_grid.jl")

"""
    Δ

Alias for the `spacing` method that returns the spacing between grid points.
"""
const Δ  = spacing

"""
    iΔ

Alias for the `inv_spacing` method that returns the reciprocal of the spacing between grid points.
"""
const iΔ = inv_spacing

end # module Grids
