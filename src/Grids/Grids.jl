module Grids

export Location, Center, Vertex, flip
export Connectivity, Bounded, Periodic, Flat
export AbstractAxis, UniformAxis, FunctionAxis
export StructuredGrid, UniformGrid

export nvertices, ncenters, spacing, inv_spacing, Δ, iΔ, coord, coords, center, vertex, centers, vertices
export origin, extent, bounds
export direction, axes_names
export connectivity

using Chmy.Architectures

import Chmy: @add_cartesian

import Base.@propagate_inbounds

abstract type Location end

struct Center <: Location end
struct Vertex <: Location end

Base.@assume_effects :total flip(::Center) = Vertex()
Base.@assume_effects :total flip(::Vertex) = Center()

Base.broadcastable(o::Location) = Ref(o)

abstract type Connectivity end

struct Bounded <: Connectivity end
struct Periodic <: Connectivity end
struct Flat <: Connectivity end

include("abstract_axis.jl")
include("uniform_axis.jl")
include("function_axis.jl")
include("structured_grid.jl")

const Δ  = spacing
const iΔ = inv_spacing

end # module Grids
