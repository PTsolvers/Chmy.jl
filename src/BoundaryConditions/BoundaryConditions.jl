module BoundaryConditions

export FieldBoundaryCondition, FirstOrderBC, Dirichlet, Neumann, bc!
export BoundaryFunction
export DimSide
export AbstractBatch, FieldBatch, ExchangeBatch, EmptyBatch, BatchSet, batch

using Chmy
using Chmy.Grids
using Chmy.Fields
using Chmy.Architectures

import Chmy: @add_cartesian

using KernelAbstractions

import Base.@propagate_inbounds

const SDA = SingleDeviceArchitecture
const SG  = StructuredGrid

"""
    FieldBoundaryCondition

Abstract supertype for all boundary conditions that are specified per-field.
"""
abstract type FieldBoundaryCondition end

const FBC          = FieldBoundaryCondition
const FBCOrNothing = Union{FBC,Nothing}
const SidesBCs     = Tuple{FBCOrNothing,FBCOrNothing}
const BCOrTuple    = Union{FBCOrNothing,SidesBCs}
const TupleBC      = NamedTuple{Names,<:Tuple{Vararg{BCOrTuple}}} where {Names}
const PerFieldBC   = Union{FBCOrNothing,TupleBC}
const FieldAndBC   = Pair{<:Field,<:PerFieldBC}

include("first_order_boundary_condition.jl")
include("batch.jl")
include("boundary_function.jl")

end
